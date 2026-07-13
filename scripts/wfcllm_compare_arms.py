#!/usr/bin/env python
"""Compare frozen Current-R20 and V2-R20 results with paired statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.metrics import build_detection_report  # noqa: E402

COMPARISON_SCHEMA_VERSION = "wfcllm-arm-comparison/v1"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare paired Current-R20 and V2-R20 experiment rows.",
    )
    parser.add_argument("--current-positive-details", required=True)
    parser.add_argument("--v2-positive-details", required=True)
    parser.add_argument("--current-negative-details", required=True)
    parser.add_argument("--v2-negative-details", required=True)
    parser.add_argument("--current-execution", required=True)
    parser.add_argument("--v2-execution", required=True)
    parser.add_argument("--current-recovery", default=None)
    parser.add_argument("--v2-replay", default=None)
    parser.add_argument("--v2-retry-ledger", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    return parser


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {exc.lineno}") from exc
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return value


def _by_id(
    rows: list[dict[str, Any]],
    *,
    label: str,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{label}[{index}] must contain an id")
        if sample_id in result:
            raise ValueError(f"{label} contains duplicate id: {sample_id}")
        result[sample_id] = row
    return result


def _require_same_ids(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    label: str,
) -> None:
    if set(left) != set(right):
        raise ValueError(
            f"{label} must match; missing={sorted(set(left) - set(right))}, "
            f"unexpected={sorted(set(right) - set(left))}"
        )


def _bool(row: dict[str, Any], field: str, *, label: str) -> bool:
    value = row.get(field)
    if not isinstance(value, bool):
        raise ValueError(f"{label} field {field} must be boolean")
    return value


def _enrich_with_passed(
    details: list[dict[str, Any]],
    execution_by_id: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **row,
            "passed": _bool(
                execution_by_id[str(row["id"])],
                "passed",
                label=f"execution[{row['id']}]",
            ),
        }
        for row in details
    ]


def _rate(values: list[bool]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mcnemar(left: list[bool], right: list[bool]) -> dict[str, int]:
    if len(left) != len(right):
        raise ValueError("paired values must have equal lengths")
    return {
        "current_only": sum(a and not b for a, b in zip(left, right)),
        "v2_only": sum(b and not a for a, b in zip(left, right)),
        "both": sum(a and b for a, b in zip(left, right)),
        "neither": sum(not a and not b for a, b in zip(left, right)),
    }


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _paired_bootstrap_ci(
    left: list[bool],
    right: list[bool],
    *,
    seed: int,
    repetitions: int,
) -> list[float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equal non-empty inputs")
    if repetitions <= 0:
        raise ValueError("bootstrap_repetitions must be positive")
    differences = [float(b) - float(a) for a, b in zip(left, right)]
    rng = random.Random(seed)
    sample_count = len(differences)
    replicates = [
        sum(differences[rng.randrange(sample_count)] for _ in range(sample_count))
        / sample_count
        for _ in range(repetitions)
    ]
    return [_percentile(replicates, 0.025), _percentile(replicates, 0.975)]


def _failure_taxonomy(rows: list[dict[str, Any]]) -> dict[str, int]:
    insufficient = [
        _bool(row, "insufficient_evidence", label=f"details[{row.get('id')}]")
        for row in rows
    ]
    decisions = [
        _bool(row, "is_watermarked", label=f"details[{row.get('id')}]")
        for row in rows
    ]
    return {
        "insufficient_evidence": sum(insufficient),
        "sufficient_but_low_score": sum(
            not insufficient_value and not decision
            for insufficient_value, decision in zip(insufficient, decisions)
        ),
        "detected": sum(decisions),
    }


def _sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _optional_evidence(args: argparse.Namespace) -> dict[str, Any]:
    evidence: dict[str, Any] = {}
    if args.current_recovery is not None:
        recovery = _load_json(args.current_recovery)
        evidence["current"] = {
            key: recovery.get(key)
            for key in (
                "accepted_count",
                "accepted_unique_count",
                "duplicate_accepted_count",
                "recovered_accepted_count",
                "recovered_accepted_unique_count",
                "accepted_to_recovered_ratio",
                "accepted_unique_to_recovered_ratio",
            )
        }
    if args.v2_replay is not None:
        replay = _load_json(args.v2_replay)
        evidence["v2_replay"] = {
            key: replay.get(key)
            for key in (
                "final_code_replay_rate",
                "ledger_selected_count",
                "ledger_final_code_match_rate",
                "ledger_score_replay_rate",
                "generation_to_final_score_pearson",
            )
        }
    if args.v2_retry_ledger is not None:
        selected = [
            row for row in _load_jsonl(args.v2_retry_ledger)
            if row.get("selected") is True
        ]
        selected_by_id = _by_id(selected, label="selected_retry_ledger")
        no_embedding = 0
        accepted_units = 0
        for sample_id, row in selected_by_id.items():
            quality = row.get("quality")
            if not isinstance(quality, dict) or not isinstance(
                quality.get("eligible"), bool
            ):
                raise ValueError(f"selected retry quality invalid for {sample_id}")
            no_embedding += not quality["eligible"]
            units = row.get("unit_count")
            if isinstance(units, bool) or not isinstance(units, int):
                raise ValueError(f"selected retry unit_count invalid for {sample_id}")
            accepted_units += units
        evidence["v2_selection"] = {
            "selected_count": len(selected),
            "no_embedding_count": no_embedding,
            "accepted_canonical_units": accepted_units,
        }
    return evidence


def _arm_summary(
    report: dict[str, Any],
    passed_values: list[bool],
) -> dict[str, Any]:
    primary = report["primary"]
    return {
        "passed": sum(passed_values),
        "pass_at_1": _rate(passed_values),
        "tpr": primary["tpr_at_target_fpr"],
        "observed_fpr": primary["observed_fpr"],
        "auroc": primary["auroc"],
        "positive_insufficient": primary["positive_insufficient_samples"],
        "positive_sufficient": primary["positive_sufficient_samples"],
        "score_distributions": report["score_distributions"],
        "detection_report": report,
    }


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    current_positive = _load_jsonl(args.current_positive_details)
    v2_positive = _load_jsonl(args.v2_positive_details)
    current_negative = _load_jsonl(args.current_negative_details)
    v2_negative = _load_jsonl(args.v2_negative_details)
    current_execution = _load_jsonl(args.current_execution)
    v2_execution = _load_jsonl(args.v2_execution)

    current_positive_by_id = _by_id(current_positive, label="current_positive")
    v2_positive_by_id = _by_id(v2_positive, label="v2_positive")
    _require_same_ids(
        current_positive_by_id,
        v2_positive_by_id,
        label="paired positive ids",
    )
    current_negative_by_id = _by_id(current_negative, label="current_negative")
    v2_negative_by_id = _by_id(v2_negative, label="v2_negative")
    _require_same_ids(
        current_negative_by_id,
        v2_negative_by_id,
        label="negative panel ids",
    )
    current_execution_by_id = _by_id(current_execution, label="current_execution")
    v2_execution_by_id = _by_id(v2_execution, label="v2_execution")
    _require_same_ids(
        current_positive_by_id,
        current_execution_by_id,
        label="current execution ids",
    )
    _require_same_ids(
        v2_positive_by_id,
        v2_execution_by_id,
        label="v2 execution ids",
    )

    ids = list(current_positive_by_id)
    current_passed = [
        _bool(current_execution_by_id[sample_id], "passed", label=sample_id)
        for sample_id in ids
    ]
    v2_passed = [
        _bool(v2_execution_by_id[sample_id], "passed", label=sample_id)
        for sample_id in ids
    ]
    current_decisions = [
        _bool(current_positive_by_id[sample_id], "is_watermarked", label=sample_id)
        for sample_id in ids
    ]
    v2_decisions = [
        _bool(v2_positive_by_id[sample_id], "is_watermarked", label=sample_id)
        for sample_id in ids
    ]
    current_detection_report = build_detection_report(
        _enrich_with_passed(current_positive, current_execution_by_id),
        current_negative,
    )
    v2_detection_report = build_detection_report(
        _enrich_with_passed(v2_positive, v2_execution_by_id),
        v2_negative,
    )
    current_summary = _arm_summary(current_detection_report, current_passed)
    v2_summary = _arm_summary(v2_detection_report, v2_passed)

    input_paths = {
        "current_positive_details": args.current_positive_details,
        "v2_positive_details": args.v2_positive_details,
        "current_negative_details": args.current_negative_details,
        "v2_negative_details": args.v2_negative_details,
        "current_execution": args.current_execution,
        "v2_execution": args.v2_execution,
    }
    return {
        "artifact_type": "wfcllm_arm_comparison",
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "single_seed": args.seed,
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "sample_count": len(ids),
        "negative_panel_count": len(current_negative),
        "current": current_summary,
        "v2": v2_summary,
        "deltas": {
            "pass_count": v2_summary["passed"] - current_summary["passed"],
            "pass_at_1": v2_summary["pass_at_1"] - current_summary["pass_at_1"],
            "tpr": v2_summary["tpr"] - current_summary["tpr"],
            "observed_fpr": (
                v2_summary["observed_fpr"] - current_summary["observed_fpr"]
            ),
            "auroc": v2_summary["auroc"] - current_summary["auroc"],
        },
        "paired": {
            "pass_mcnemar": _mcnemar(current_passed, v2_passed),
            "detection_mcnemar": _mcnemar(current_decisions, v2_decisions),
            "pass_delta_bootstrap_95": _paired_bootstrap_ci(
                current_passed,
                v2_passed,
                seed=args.seed,
                repetitions=args.bootstrap_repetitions,
            ),
            "tpr_delta_bootstrap_95": _paired_bootstrap_ci(
                current_decisions,
                v2_decisions,
                seed=args.seed + 1,
                repetitions=args.bootstrap_repetitions,
            ),
        },
        "failure_taxonomy": {
            "current": _failure_taxonomy(current_positive),
            "v2": _failure_taxonomy(v2_positive),
        },
        "evidence": _optional_evidence(args),
        "inputs": {
            name: {"path": str(Path(path)), "sha256": _sha256(path)}
            for name, path in input_paths.items()
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = _build_report(args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] paired arm comparison saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
