#!/usr/bin/env python
"""Offline evaluation harness for semantic, lexical, and dual-channel modes.

Correctness stays offline: pass@k is derived from generated code grouped per task and
matched against local reference solutions via normalized AST equality.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.common.dataset_loader import load_reference_solutions
from wfcllm.common.offline_code_eval import annotate_correctness_from_references
from wfcllm.common.offline_code_eval import build_perturbation_corpus
from wfcllm.common.offline_code_eval import compute_average_latency
from wfcllm.common.offline_code_eval import compute_pass_at_k
from wfcllm.common.offline_code_eval import compute_retry_rate
from wfcllm.common.offline_code_eval import compute_roc_auc
from wfcllm.common.offline_code_eval import compute_tpr_at_fpr
from wfcllm.common.offline_code_eval import extract_scores
from wfcllm.common.offline_code_eval import load_jsonl_records
from wfcllm.common.offline_code_eval import mean_or_zero
from wfcllm.common.offline_code_eval import relative_delta
from wfcllm.common.offline_code_eval import write_jsonl_records

MODES = ("semantic-only", "lexical-only", "dual-channel")
PERTURBATIONS = ("formatting", "comments", "rename", "light-rewrite")
TARGET_FPR = 0.01


@dataclass(frozen=True)
class CommandRunResult:
    """Process execution result used by the evaluation harness."""

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    elapsed_seconds: float = 0.0


CommandRunner = Callable[[list[str], dict[str, str] | None], CommandRunResult]


def run_evaluation(
    dataset: str,
    config_path: str,
    output_dir: str,
    *,
    candidate_count: int = 10,
    reference_records: list[dict[str, Any]] | None = None,
    command_runner: CommandRunner | None = None,
) -> dict[str, Any]:
    """Run the offline dual-channel evaluation harness and write metric artifacts."""
    if candidate_count <= 0:
        raise ValueError("candidate_count must be > 0")

    runner = command_runner or _run_command
    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env.setdefault("HF_HUB_OFFLINE", "1")
    resolved_reference_records = _resolve_reference_records(
        dataset=dataset,
        config_path=config_path,
        reference_records=reference_records,
    )

    negative_output = output_root / "negative" / f"{dataset}_negative.jsonl"
    _ensure_negative_corpus(dataset, config_path, negative_output, runner, env)

    results_by_mode: dict[str, dict[str, Any]] = {}
    baseline_metrics: dict[str, float] | None = None

    for mode in MODES:
        mode_dir = output_root / mode
        mode_dir.mkdir(parents=True, exist_ok=True)

        watermark_paths: list[Path] = []
        positive_detail_paths: list[Path] = []
        elapsed_total = 0.0
        watermarked_records: list[dict[str, Any]] = []
        positive_detail_records: list[dict[str, Any]] = []

        for candidate_index in range(candidate_count):
            watermarked_path, watermark_timing = _run_watermark_phase(
                dataset=dataset,
                config_path=config_path,
                mode=mode,
                output_dir=mode_dir / f"watermarked_candidate_{candidate_index + 1}",
                runner=runner,
                env=env,
            )
            positive_details_path = _run_extract_phase(
                config_path=config_path,
                mode=mode,
                input_file=watermarked_path,
                output_dir=mode_dir / f"positive_extract_candidate_{candidate_index + 1}",
                runner=runner,
                env=env,
            )
            candidate_records = load_jsonl_records(watermarked_path)
            for record in candidate_records:
                record["candidate_index"] = candidate_index
            watermarked_records.extend(candidate_records)
            positive_detail_records.extend(load_jsonl_records(positive_details_path))
            watermark_paths.append(watermarked_path)
            positive_detail_paths.append(positive_details_path)
            elapsed_total += watermark_timing.elapsed_seconds

        watermarked_records = annotate_correctness_from_references(
            watermarked_records,
            resolved_reference_records,
        )
        combined_watermarked_path = mode_dir / "watermarked_candidates.jsonl"
        write_jsonl_records(combined_watermarked_path, watermarked_records)
        combined_positive_details_path = mode_dir / "positive_details.jsonl"
        write_jsonl_records(combined_positive_details_path, positive_detail_records)

        negative_details_path = _run_extract_phase(
            config_path=config_path,
            mode=mode,
            input_file=negative_output,
            output_dir=mode_dir / "negative_extract",
            runner=runner,
            env=env,
        )

        negative_detail_records = load_jsonl_records(negative_details_path)

        sample_count = len(watermarked_records)
        pass_at_1 = compute_pass_at_k(watermarked_records, k=1)
        pass_at_10 = compute_pass_at_k(watermarked_records, k=10)
        retry_rate = compute_retry_rate(watermarked_records)
        latency = compute_average_latency(elapsed_total, sample_count)
        positive_scores = extract_scores(positive_detail_records, mode)
        negative_scores = extract_scores(negative_detail_records, mode)
        roc_auc = compute_roc_auc(positive_scores, negative_scores)
        tpr_at_1pct_fpr = compute_tpr_at_fpr(positive_scores, negative_scores, target_fpr=TARGET_FPR)

        metric_payload: dict[str, Any] = {
            "mode": mode,
            "artifacts": {
                "watermarked_candidates": str(combined_watermarked_path),
                "candidate_watermarked_runs": [str(path) for path in watermark_paths],
                "positive_details": str(combined_positive_details_path),
                "candidate_positive_extract_runs": [str(path) for path in positive_detail_paths],
                "negative_details": str(negative_details_path),
            },
            "generation": {
                "pass_at_1": pass_at_1,
                "pass_at_10": pass_at_10,
                "retry_rate": retry_rate,
                "latency_seconds": latency,
            },
            "detection": {
                "roc_auc": roc_auc,
                "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
                "positive_mean_score": mean_or_zero(positive_scores),
                "negative_mean_score": mean_or_zero(negative_scores),
            },
        }

        if baseline_metrics is None:
            baseline_metrics = {
                "pass_at_1": pass_at_1,
                "pass_at_10": pass_at_10,
                "retry_rate": retry_rate,
                "latency_seconds": latency,
                "roc_auc": roc_auc,
                "tpr_at_1pct_fpr": tpr_at_1pct_fpr,
            }

        metric_payload["deltas_vs_semantic_only"] = {
            "pass_at_1": pass_at_1 - baseline_metrics["pass_at_1"],
            "pass_at_10": pass_at_10 - baseline_metrics["pass_at_10"],
            "retry_delta": retry_rate - baseline_metrics["retry_rate"],
            "latency_delta": latency - baseline_metrics["latency_seconds"],
            "joint_uplift_roc_auc": roc_auc - baseline_metrics["roc_auc"],
            "joint_uplift_tpr_at_1pct_fpr": (
                tpr_at_1pct_fpr - baseline_metrics["tpr_at_1pct_fpr"]
            ),
        }
        metric_payload["thresholds"] = _build_threshold_report(metric_payload, baseline_metrics)
        metric_payload["perturbations"] = _run_perturbation_checks(
            config_path=config_path,
            mode=mode,
            mode_dir=mode_dir,
            source_records=watermarked_records,
            baseline_scores=positive_scores,
            runner=runner,
            env=env,
        )

        metrics_path = mode_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metric_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        results_by_mode[mode] = metric_payload

    summary = {
        "dataset": dataset,
        "config_path": str(Path(config_path)),
        "target_fpr": TARGET_FPR,
        "candidate_count": candidate_count,
        "modes": results_by_mode,
    }
    summary_path = output_root / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for the offline evaluation harness."""
    parser = argparse.ArgumentParser(
        description="Run offline dual-channel evaluation across semantic-only, lexical-only, and dual-channel modes.",
    )
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--config", default="configs/base_config.json", help="JSON config path")
    parser.add_argument(
        "--output-dir",
        default="data/eval/dual_channel",
        help="directory for evaluation artifacts",
    )
    parser.add_argument(
        "--num-candidates",
        type=int,
        default=10,
        help="number of generated candidates per task for pass@k evaluation",
    )
    return parser


def main() -> None:
    """CLI entrypoint."""
    args = build_argument_parser().parse_args()
    result = run_evaluation(
        dataset=args.dataset,
        config_path=args.config,
        output_dir=args.output_dir,
        candidate_count=args.num_candidates,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def _resolve_reference_records(
    *,
    dataset: str,
    config_path: str,
    reference_records: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    if reference_records is not None:
        return reference_records
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    watermark_config = config.get("watermark") or {}
    dataset_path = str(watermark_config.get("dataset_path", "data/datasets"))
    return load_reference_solutions(dataset, dataset_path)


def _ensure_negative_corpus(
    dataset: str,
    config_path: str,
    output_path: Path,
    runner: CommandRunner,
    env: dict[str, str],
) -> None:
    if output_path.exists():
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "run.py",
        "--config",
        config_path,
        "--phase",
        "generate-negative",
        "--dataset",
        dataset,
        "--negative-output",
        str(output_path),
    ]
    _require_success(runner(command, env))


def _run_watermark_phase(
    *,
    dataset: str,
    config_path: str,
    mode: str,
    output_dir: Path,
    runner: CommandRunner,
    env: dict[str, str],
) -> tuple[Path, CommandRunResult]:
    output_dir.mkdir(parents=True, exist_ok=True)
    before = set(output_dir.glob("*.jsonl"))
    command = [
        "python",
        "run.py",
        "--config",
        config_path,
        "--phase",
        "watermark",
        "--dataset",
        dataset,
        "--output-dir",
        str(output_dir),
        *list(_mode_cli_args(mode)),
    ]
    result = runner(command, env)
    _require_success(result)
    return _newest_artifact(output_dir, before), result


def _run_extract_phase(
    *,
    config_path: str,
    mode: str,
    input_file: Path,
    output_dir: Path,
    runner: CommandRunner,
    env: dict[str, str],
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "python",
        "run.py",
        "--config",
        config_path,
        "--phase",
        "extract",
        "--input-file",
        str(input_file),
        "--extract-output-dir",
        str(output_dir),
        *list(_mode_cli_args(mode)),
    ]
    _require_success(runner(command, env))
    return output_dir / f"{input_file.stem}_details.jsonl"


def _run_perturbation_checks(
    *,
    config_path: str,
    mode: str,
    mode_dir: Path,
    source_records: list[dict[str, Any]],
    baseline_scores: list[float],
    runner: CommandRunner,
    env: dict[str, str],
) -> dict[str, Any]:
    perturbation_root = mode_dir / "perturbations"
    perturbation_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    baseline_mean_score = mean_or_zero(baseline_scores)

    for perturbation in PERTURBATIONS:
        corpus_path = perturbation_root / f"{perturbation}.jsonl"
        build_info = build_perturbation_corpus(source_records, corpus_path, perturbation)
        details_path = _run_extract_phase(
            config_path=config_path,
            mode=mode,
            input_file=corpus_path,
            output_dir=perturbation_root / perturbation,
            runner=runner,
            env=env,
        )
        detail_records = load_jsonl_records(details_path)
        perturb_scores = extract_scores(detail_records, mode)
        results[perturbation] = {
            **build_info,
            "mean_score": mean_or_zero(perturb_scores),
            "mean_score_delta": mean_or_zero(perturb_scores) - baseline_mean_score,
        }

    return results


def _build_threshold_report(
    metric_payload: dict[str, Any],
    baseline_metrics: dict[str, float],
) -> dict[str, Any]:
    mode = str(metric_payload["mode"])
    generation = metric_payload["generation"]
    detection = metric_payload["detection"]

    thresholds = {
        "pass_at_1_within_2pp": baseline_metrics["pass_at_1"] - generation["pass_at_1"] <= 0.02,
        "pass_at_10_within_3pp": baseline_metrics["pass_at_10"] - generation["pass_at_10"] <= 0.03,
        "retry_within_25pct": relative_delta(generation["retry_rate"], baseline_metrics["retry_rate"]) <= 0.25,
        "latency_within_35pct": relative_delta(
            generation["latency_seconds"],
            baseline_metrics["latency_seconds"],
        ) <= 0.35,
    }

    if mode == "lexical-only":
        thresholds["roc_auc_gte_0_65"] = detection["roc_auc"] >= 0.65
        thresholds["tpr_at_1pct_fpr_gte_0_20"] = detection["tpr_at_1pct_fpr"] >= 0.20
    if mode == "dual-channel":
        thresholds["joint_roc_auc_uplift_gte_0_02"] = (
            detection["roc_auc"] - baseline_metrics["roc_auc"] >= 0.02
        )
        thresholds["joint_tpr_uplift_gte_0_05"] = (
            detection["tpr_at_1pct_fpr"] - baseline_metrics["tpr_at_1pct_fpr"] >= 0.05
        )

    return thresholds


def _mode_cli_args(mode: str) -> tuple[str, ...]:
    enabled = "false" if mode == "semantic-only" else "true"
    return (
        "--token-channel-enabled",
        enabled,
        "--token-channel-mode",
        mode,
    )


def _newest_artifact(output_dir: Path, before: set[Path]) -> Path:
    candidates = [path for path in output_dir.glob("*.jsonl") if path not in before]
    if not candidates:
        candidates = list(output_dir.glob("*.jsonl"))
    if not candidates:
        raise ValueError(f"no JSONL artifact produced under {output_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _require_success(result: CommandRunResult) -> None:
    if result.exit_code != 0:
        message = result.stderr or result.stdout or "command failed"
        raise RuntimeError(message)


def _run_command(command: list[str], env: dict[str, str] | None = None) -> CommandRunResult:
    start = time.perf_counter()
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.perf_counter() - start
    return CommandRunResult(
        exit_code=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
        elapsed_seconds=elapsed,
    )


if __name__ == "__main__":
    main()
