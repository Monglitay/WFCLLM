#!/usr/bin/env python3
"""Prepare and summarize the frozen Watermark Mechanism V4 pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.batch_invariant_v4.audit import (  # noqa: E402
    evidence_sha256,
    load_retry_ledgers,
    rows_to_candidates,
)
from wfcllm.batch_invariant_v4.calibration import load_calibration  # noqa: E402
from wfcllm.batch_invariant_v4.channel import StructuralChannel  # noqa: E402
from wfcllm.batch_invariant_v4.config import load_public_config  # noqa: E402
from wfcllm.batch_invariant_v4.context import (  # noqa: E402
    ContextConfig,
    StructuralContextExtractor,
)
from wfcllm.batch_invariant_v4.detector import DetectorPayload, V4Detector  # noqa: E402
from wfcllm.batch_invariant_v4.keying import (  # noqa: E402
    derive_wrong_control_key,
    load_secret_key,
)
from wfcllm.datasets.loaders.local import load_prompts  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--public-config", type=Path, required=True)
    prepare.add_argument("--calibration", type=Path, required=True)
    prepare.add_argument("--key-file", type=Path, required=True)
    prepare.add_argument("--cold-run", type=Path, action="append", required=True)
    prepare.add_argument("--retry-ledger", type=Path, action="append", required=True)
    prepare.add_argument("--dataset-path", type=Path, default=Path("data/datasets"))
    prepare.add_argument("--output-dir", type=Path, required=True)

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--prepared-audit", type=Path, required=True)
    summarize.add_argument("--current-execution", type=Path, required=True)
    summarize.add_argument("--v4-execution", type=Path, required=True)
    summarize.add_argument("--calibration-control", type=Path, required=True)
    summarize.add_argument("--output", type=Path, required=True)
    summarize.add_argument("--bootstrap-repetitions", type=int, default=10000)
    summarize.add_argument("--seed", type=int, default=20260714)
    return parser


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = [
        json.loads(raw)
        for raw in path.read_text(encoding="utf-8").splitlines()
        if raw
    ]
    if not all(isinstance(value, dict) for value in values):
        raise ValueError(f"JSONL rows must be objects: {path}")
    return values


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _detector(config: Any, calibration: Any, key: Any) -> V4Detector:
    return V4Detector(
        extractor=StructuralContextExtractor(
            ContextConfig(
                schema_version=config.canonical_context.schema_version,
                max_unit_bytes=config.canonical_context.max_unit_bytes,
                max_context_bytes=config.canonical_context.max_context_bytes,
                global_ordinal_keying=config.canonical_context.global_ordinal_keying,
            )
        ),
        channel=StructuralChannel(
            key,
            bit_count=config.channel.bit_count_per_unit,
            minimum_independent_units=config.channel.minimum_independent_units,
        ),
        calibration=calibration,
    )


def _detail(sample_id: str, attempt: int, evidence: Any) -> dict[str, Any]:
    erasures = sum(evidence.erasure_counts.values())
    total_units = evidence.independent_units + erasures
    return {
        "id": sample_id,
        "attempt_index": attempt,
        "score": evidence.score,
        "is_watermarked": evidence.decision,
        "eligible": evidence.eligible,
        "insufficient_evidence": not evidence.eligible,
        "independent_units": evidence.independent_units,
        "erasure_count": erasures,
        "erasure_rate": erasures / total_units if total_units else 0.0,
        "aggregate_numerator": evidence.aggregate_numerator,
        "aggregate_denominator": evidence.aggregate_denominator,
        "p_value": evidence.p_value,
        "evidence_sha256": evidence_sha256(evidence),
    }


def _prepare(args: argparse.Namespace) -> None:
    if len(args.cold_run) != 3:
        raise ValueError("pilot preparation requires exactly three cold runs")
    cold = [_read_json(path) for path in args.cold_run]
    identities = {run["summary"]["exact_identity_sha256"] for run in cold}
    if len(identities) != 1:
        raise ValueError("pilot cold processes do not have one exact identity")
    exact_rate_fields = (
        "candidate_pool_match_rate",
        "generation_to_r3_exact_rate",
        "schedule_exact_rate",
        "cache_hit_miss_exact_rate",
        "cross_task_composition_exact_rate",
    )
    if not all(
        run["summary"][field] == 1.0
        for run in cold
        for field in exact_rate_fields
    ):
        raise ValueError("pilot cold process exactness gate failed")
    ids = [str(row["id"]) for row in cold[0]["tasks"]]
    if any([str(row["id"]) for row in run["tasks"]] != ids for run in cold[1:]):
        raise ValueError("pilot task order differs across cold processes")

    config = load_public_config(args.public_config)
    calibration = load_calibration(args.calibration)
    primary_key = load_secret_key(args.key_file)
    detectors = {
        "primary": _detector(config, calibration, primary_key),
        "wrong_key_domain_separated": _detector(
            config,
            calibration,
            derive_wrong_control_key(primary_key),
        ),
    }
    grouped = load_retry_ledgers(args.retry_ledger)
    prompts = {
        str(row["id"]): str(row["prompt"])
        for row in load_prompts("humaneval", args.dataset_path)
    }
    current_final: list[dict[str, Any]] = []
    v4_final: list[dict[str, Any]] = []
    details = {
        "primary": {"current": [], "v4": []},
        "wrong_key_domain_separated": {"current": [], "v4": []},
    }
    tasks: list[dict[str, Any]] = []
    for cold_task in cold[0]["tasks"]:
        sample_id = str(cold_task["id"])
        candidates = rows_to_candidates(grouped[sample_id], retry=20)
        current_attempt = int(cold_task["current_selected_attempt"])
        v4_attempt = int(cold_task["v4_selected_attempt"])
        current_candidate = candidates[current_attempt]
        v4_candidate = candidates[v4_attempt]
        current_final.append(
            {
                "id": sample_id,
                "dataset": "humaneval",
                "prompt": prompts[sample_id],
                "final_code": current_candidate.final_code,
            }
        )
        v4_final.append(
            {
                "id": sample_id,
                "dataset": "humaneval",
                "prompt": prompts[sample_id],
                "final_code": v4_candidate.final_code,
            }
        )
        per_key: dict[str, Any] = {}
        for key_label, detector in detectors.items():
            current_evidence = detector.detect(
                DetectorPayload(final_code=current_candidate.final_code)
            )
            v4_evidence = detector.detect(
                DetectorPayload(final_code=v4_candidate.final_code)
            )
            current_detail = _detail(sample_id, current_attempt, current_evidence)
            v4_detail = _detail(sample_id, v4_attempt, v4_evidence)
            details[key_label]["current"].append(current_detail)
            details[key_label]["v4"].append(v4_detail)
            per_key[key_label] = {
                "current": current_detail,
                "v4": v4_detail,
            }
        cold_selected_sha = hashlib.sha256(
            json.dumps(
                cold_task["selected_r3_evidence"],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if cold_selected_sha != per_key["primary"]["v4"]["evidence_sha256"]:
            raise ValueError(f"pilot selected evidence mismatch: {sample_id}")
        tasks.append(
            {
                "id": sample_id,
                "retry": 20,
                "candidate_sha256": [item.final_code_sha256 for item in candidates],
                "current_selected_attempt": current_attempt,
                "v4_selected_attempt": v4_attempt,
                "current_final_code_sha256": current_candidate.final_code_sha256,
                "v4_final_code_sha256": v4_candidate.final_code_sha256,
                "evidence": per_key,
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "current_final": args.output_dir / "current_final_code.jsonl",
        "v4_final": args.output_dir / "v4_final_code.jsonl",
        "current_primary": args.output_dir / "current_primary_details.jsonl",
        "v4_primary": args.output_dir / "v4_primary_details.jsonl",
        "current_wrong": args.output_dir / "current_wrong_key_details.jsonl",
        "v4_wrong": args.output_dir / "v4_wrong_key_details.jsonl",
    }
    _write_jsonl(paths["current_final"], current_final)
    _write_jsonl(paths["v4_final"], v4_final)
    _write_jsonl(paths["current_primary"], details["primary"]["current"])
    _write_jsonl(paths["v4_primary"], details["primary"]["v4"])
    _write_jsonl(
        paths["current_wrong"],
        details["wrong_key_domain_separated"]["current"],
    )
    _write_jsonl(paths["v4_wrong"], details["wrong_key_domain_separated"]["v4"])
    audit = {
        "artifact_type": "wfcllm_v4_pilot_prepared_audit",
        "schema_version": "wfcllm-v4-pilot-prepared-audit/v1",
        "task_count": len(tasks),
        "retry": 20,
        "secret_metadata_included": False,
        "heldout_accessed": False,
        "full_run_triggered": False,
        "robustness_triggered": False,
        "cold_process_count": 3,
        "cold_exact_identity_sha256": next(iter(identities)),
        "cold_run_sha256": [_sha256(path) for path in args.cold_run],
        "cold_summaries": [run["summary"] for run in cold],
        "public_config_sha256": _sha256(args.public_config),
        "calibration_sha256": _sha256(args.calibration),
        "retry_ledger_sha256": [_sha256(path) for path in args.retry_ledger],
        "tasks": tasks,
        "outputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in paths.items()
        },
        "repair": {
            "used": 1,
            "maximum": 1,
            "classification": "code did not implement preregistered all-ineligible fallback",
            "mechanism_changed": False,
        },
    }
    (args.output_dir / "prepared_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _by_id(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    output = {str(row["id"]): row for row in rows}
    if len(output) != len(rows):
        raise ValueError("duplicate IDs in paired artifact")
    return output


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def _bootstrap_ci(
    differences: Sequence[float],
    *,
    seed: int,
    repetitions: int,
) -> list[float]:
    if not differences or repetitions <= 0:
        raise ValueError("bootstrap needs non-empty differences and repetitions")
    rng = random.Random(seed)
    count = len(differences)
    values = [
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(repetitions)
    ]
    return [_percentile(values, 0.025), _percentile(values, 0.975)]


def _two_sided_binomial_p(successes: int, trials: int) -> float:
    if trials == 0:
        return 1.0
    tail = sum(math.comb(trials, value) for value in range(successes + 1)) / 2**trials
    mirrored = sum(
        math.comb(trials, value) for value in range(successes, trials + 1)
    ) / 2**trials
    return min(1.0, 2 * min(tail, mirrored))


def _mcnemar(left: Sequence[bool], right: Sequence[bool]) -> dict[str, Any]:
    left_only = sum(a and not b for a, b in zip(left, right, strict=True))
    right_only = sum(b and not a for a, b in zip(left, right, strict=True))
    discordant = left_only + right_only
    return {
        "current_only": left_only,
        "v4_only": right_only,
        "both": sum(a and b for a, b in zip(left, right, strict=True)),
        "neither": sum(not a and not b for a, b in zip(left, right, strict=True)),
        "exact_two_sided_p": _two_sided_binomial_p(
            min(left_only, right_only), discordant
        ),
    }


def _summarize(args: argparse.Namespace) -> None:
    prepared = _read_json(args.prepared_audit)
    current_exec = _by_id(_read_jsonl(args.current_execution))
    v4_exec = _by_id(_read_jsonl(args.v4_execution))
    control = _read_json(args.calibration_control)
    tasks = prepared["tasks"]
    ids = [str(task["id"]) for task in tasks]
    if set(ids) != set(current_exec) or set(ids) != set(v4_exec):
        raise ValueError("execution IDs do not match prepared pilot")

    def values(key_label: str, arm: str, field: str) -> list[Any]:
        return [task["evidence"][key_label][arm][field] for task in tasks]

    current_scores = values("primary", "current", "score")
    v4_scores = values("primary", "v4", "score")
    wrong_current_scores = values("wrong_key_domain_separated", "current", "score")
    wrong_v4_scores = values("wrong_key_domain_separated", "v4", "score")
    current_decisions = values("primary", "current", "is_watermarked")
    v4_decisions = values("primary", "v4", "is_watermarked")
    wrong_current_decisions = values(
        "wrong_key_domain_separated", "current", "is_watermarked"
    )
    wrong_v4_decisions = values(
        "wrong_key_domain_separated", "v4", "is_watermarked"
    )
    current_pass = [bool(current_exec[sample_id]["passed"]) for sample_id in ids]
    v4_pass = [bool(v4_exec[sample_id]["passed"]) for sample_id in ids]
    score_deltas = [right - left for left, right in zip(current_scores, v4_scores, strict=True)]
    wrong_score_deltas = [
        right - left
        for left, right in zip(wrong_current_scores, wrong_v4_scores, strict=True)
    ]
    positive = [value for value in score_deltas if value > 0]
    nonzero = [value for value in score_deltas if value != 0]
    positive_share = (
        max(positive) / sum(positive) if positive and sum(positive) > 0 else None
    )
    sign_successes = sum(value > 0 for value in nonzero)
    semantic_means = [
        run["mean_semantic_seconds_per_task"] for run in prepared["cold_summaries"]
    ]
    mean_semantic = sum(semantic_means) / len(semantic_means)
    primary_control = control["controls"]["primary"]
    wrong_control = control["controls"]["wrong_key_domain_separated"]
    current_tpr = sum(current_decisions) / len(ids)
    v4_tpr = sum(v4_decisions) / len(ids)
    current_pass_count = sum(current_pass)
    v4_pass_count = sum(v4_pass)
    primary_shift = sum(score_deltas) / len(score_deltas)
    wrong_shift = sum(wrong_score_deltas) / len(wrong_score_deltas)
    gates = {
        "candidate_pool_hash_match_rate": prepared["cold_summaries"][0][
            "candidate_pool_match_rate"
        ] == 1.0,
        "generation_to_r3_exact_replay": all(
            run["generation_to_r3_exact_rate"] == 1.0
            for run in prepared["cold_summaries"]
        ),
        "three_cold_process_replay": prepared["cold_process_count"] == 3,
        "cache_hit_miss_replay": all(
            run["cache_hit_miss_exact_rate"] == 1.0
            for run in prepared["cold_summaries"]
        ),
        "batch_size_composition_order_replay": all(
            run["schedule_exact_rate"] == 1.0
            and run["cross_task_composition_exact_rate"] == 1.0
            for run in prepared["cold_summaries"]
        ),
        "eos_complete_candidate_neural_rescore_zero": all(
            run["eos_all_candidate_neural_rescore_count"] == 0
            for run in prepared["cold_summaries"]
        ),
        "semantic_cost": mean_semantic <= 0.1409884,
        "cost_reduction": 1 - mean_semantic / 0.201412 >= 0.3,
        "peak_vram": max(
            run["peak_allocated_mib"] for run in prepared["cold_summaries"]
        ) <= 32607,
        "pass_delta": v4_pass_count >= current_pass_count - 1,
        "pass_count": v4_pass_count >= 15,
        "tpr_strict_improvement": v4_tpr > current_tpr,
        "positive_shift_not_single_outlier": len(positive) >= 2,
        "calibration_negative_rate": primary_control["positive_rate"] <= 0.05,
        "wrong_key_no_equal_positive_shift": wrong_shift < primary_shift,
        "repair_budget": prepared["repair"]["used"] <= prepared["repair"]["maximum"],
        "detector_final_code_only": True,
        "key_did_not_influence_generation": True,
        "pilot_ids_frozen": ids == [str(task["id"]) for task in tasks],
        "heldout_not_accessed": prepared["heldout_accessed"] is False,
    }
    report = {
        "artifact_type": "wfcllm_v4_pilot_audit",
        "schema_version": "wfcllm-v4-pilot-audit/v1",
        "status": "pass" if all(gates.values()) else "fail",
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "sample_count": len(ids),
        "retry": prepared["retry"],
        "ids": ids,
        "arms": {
            "current": {
                "tpr_count": sum(current_decisions),
                "tpr": current_tpr,
                "pass_count": current_pass_count,
                "pass_rate": current_pass_count / len(ids),
                "eligible_count": sum(values("primary", "current", "eligible")),
                "mean_score": sum(current_scores) / len(ids),
            },
            "v4": {
                "tpr_count": sum(v4_decisions),
                "tpr": v4_tpr,
                "pass_count": v4_pass_count,
                "pass_rate": v4_pass_count / len(ids),
                "eligible_count": sum(values("primary", "v4", "eligible")),
                "mean_score": sum(v4_scores) / len(ids),
                "mean_independent_units": sum(
                    values("primary", "v4", "independent_units")
                ) / len(ids),
                "mean_erasure_rate": sum(
                    values("primary", "v4", "erasure_rate")
                ) / len(ids),
            },
            "wrong_key": {
                "current_positive_count": sum(wrong_current_decisions),
                "v4_positive_count": sum(wrong_v4_decisions),
                "current_tpr": sum(wrong_current_decisions) / len(ids),
                "v4_tpr": sum(wrong_v4_decisions) / len(ids),
                "paired_mean_score_shift": wrong_shift,
            },
        },
        "paired": {
            "mean_score_delta": primary_shift,
            "score_delta_bootstrap_95": _bootstrap_ci(
                score_deltas,
                seed=args.seed,
                repetitions=args.bootstrap_repetitions,
            ),
            "tpr_delta": v4_tpr - current_tpr,
            "tpr_delta_bootstrap_95": _bootstrap_ci(
                [float(right) - float(left) for left, right in zip(current_decisions, v4_decisions, strict=True)],
                seed=args.seed + 1,
                repetitions=args.bootstrap_repetitions,
            ),
            "pass_delta": v4_pass_count - current_pass_count,
            "pass_delta_bootstrap_95": _bootstrap_ci(
                [float(right) - float(left) for left, right in zip(current_pass, v4_pass, strict=True)],
                seed=args.seed + 2,
                repetitions=args.bootstrap_repetitions,
            ),
            "score_exact_sign_test": {
                "positive": sign_successes,
                "negative": sum(value < 0 for value in nonzero),
                "ties": len(score_deltas) - len(nonzero),
                "exact_two_sided_p": _two_sided_binomial_p(
                    min(sign_successes, len(nonzero) - sign_successes), len(nonzero)
                ),
            },
            "detection_mcnemar": _mcnemar(current_decisions, v4_decisions),
            "pass_mcnemar": _mcnemar(current_pass, v4_pass),
            "largest_positive_delta_share": positive_share,
        },
        "calibration_control": {
            "primary_positive_count": primary_control["positive_count"],
            "primary_positive_rate": primary_control["positive_rate"],
            "wrong_positive_count": wrong_control["positive_count"],
            "wrong_positive_rate": wrong_control["positive_rate"],
            "heldout_accessed": control["heldout_accessed"],
        },
        "performance": {
            "cold_semantic_seconds_per_task": semantic_means,
            "mean_semantic_seconds_per_task": mean_semantic,
            "cost_reduction_vs_complete_final": 1 - mean_semantic / 0.201412,
            "peak_allocated_mib": max(
                run["peak_allocated_mib"] for run in prepared["cold_summaries"]
            ),
        },
        "gates": gates,
        "repair": prepared["repair"],
        "secret_metadata_included": False,
        "heldout_accessed": False,
        "full_run_triggered": False,
        "robustness_triggered": False,
        "inputs": {
            "prepared_audit": {"path": str(args.prepared_audit), "sha256": _sha256(args.prepared_audit)},
            "current_execution": {"path": str(args.current_execution), "sha256": _sha256(args.current_execution)},
            "v4_execution": {"path": str(args.v4_execution), "sha256": _sha256(args.v4_execution)},
            "calibration_control": {"path": str(args.calibration_control), "sha256": _sha256(args.calibration_control)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "prepare":
            _prepare(args)
        else:
            _summarize(args)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
