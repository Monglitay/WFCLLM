"""Unified Metric Contract extraction for WFCLLM run directories (ADR 0003).

Posthoc reporting utility: given a run directory, emit one standardized,
self-identifying record with dataset, language, model, calibration budget,
TPR, pass rate, AUROC, key identifier, and result eligibility. The extractor
is read-only, never writes back any detector-visible artifact, and is not on
the detection path.

This module folds in the aggregation semantics of the server-side JavaScript
evaluation tools that previously lived only inside the
``fast-fullgate-js-full164/evaluation-tools`` run directory
(``build_js_goal_summary.py``, ``evaluate_humanevalpack_js.py``,
``sanitize_humanevalpack_js.py``, ``regenerate_failed_humanevalpack_js.py``):

- detected-count based TPR over the full detector-input sample set,
- the experimental-bypass declaration, now read from run artifacts
  (gate manifest markers, ``gate_validation_skipped`` flags, calibration
  ``target_fpr``) instead of being hardcoded,
- post-hoc regeneration provenance, now derived from run-state
  ``details_path`` evidence instead of a hardcoded model name.

The goal-threshold gating at the end of ``build_js_goal_summary.py``
("tpr > 0.22 and pass_rate > 0.56, otherwise refuse to write") is
deliberately not inherited: extraction output is unconditional, and target
attainment is only ever recorded as data (ADR 0003).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "wfcllm-metric-contract/v1"
AUROC_DEFINITION = "pooled-hit-rate-rank/v1"
CONVENTIONAL_TARGET_FPR = 0.05

_STATEMENT_WINDOW_MARKER = "-statement-window/"
_PYTHON_DATASETS = {"humaneval", "mbpp"}
_HUMANEVALPACK_ID_PREFIXES = {"js", "javascript", "java", "cpp", "c++", "go", "rust"}
_PASS_MISSING_CAVEAT = "pass evaluation artifacts not present in run dir"


def _load_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload


def _load_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _first_existing_json(candidates: list[Path]) -> dict[str, Any] | None:
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        payload = _load_json_object(candidate)
        if payload is not None:
            return payload
    return None


def _run_identifier(run_dir: Path) -> str:
    """Directory name; a generic trailing ``run`` falls back to its parent."""
    resolved = run_dir.resolve()
    if resolved.name == "run" and resolved.parent.name:
        return resolved.parent.name
    return resolved.name


def _resolve_report_root(run_dir: Path) -> Path:
    if (run_dir / "reports" / "reference_report.json").is_file():
        return run_dir
    if (run_dir / "run" / "reports" / "reference_report.json").is_file():
        return run_dir / "run"
    return run_dir


def _empty_record(run_id: str) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset": None,
        "language": None,
        "generation_model": None,
        "sample_count": None,
        "target_fpr": None,
        "minimum_reliable_windows": None,
        "tpr": None,
        "tpr_is_conditional": False,
        "conditional_rule": None,
        "detected_count": None,
        "pass_rate": None,
        "pass_metric": None,
        "auroc": None,
        "auroc_definition": AUROC_DEFINITION,
        "key_identifier_sha256": None,
        "semantic_encoder_sha256": None,
        "eligibility": {
            "formal_eligible": None,
            "diagnostic_only": None,
            "experimental_only": None,
        },
        "experimental_bypass": None,
        "gate_validation_skipped": None,
        "unvalidated_gate_candidate": None,
        "caveats": [],
    }


def _language_from_contract(contract: Any) -> str | None:
    if isinstance(contract, str) and _STATEMENT_WINDOW_MARKER in contract:
        prefix = contract.split(_STATEMENT_WINDOW_MARKER, 1)[0]
        return prefix or None
    return None


def _dataset_from_sample_ids(sample_ids: list[Any]) -> str | None:
    prefixes = {
        str(sample_id).split("/", 1)[0].lower()
        for sample_id in sample_ids
        if "/" in str(sample_id)
    }
    if not prefixes:
        return None
    if prefixes <= {"humaneval"}:
        return "humaneval"
    if prefixes <= {"mbpp"}:
        return "mbpp"
    if prefixes <= _HUMANEVALPACK_ID_PREFIXES:
        return "humanevalpack"
    return None


def _string_values(payload: Any) -> list[str]:
    values: list[str] = []
    if isinstance(payload, str):
        values.append(payload)
    elif isinstance(payload, dict):
        for item in payload.values():
            values.extend(_string_values(item))
    elif isinstance(payload, list):
        for item in payload:
            values.extend(_string_values(item))
    return values


def _dataset_from_run_state_paths(run_state: dict[str, Any] | None) -> str | None:
    if run_state is None:
        return None
    joined = "\n".join(_string_values(run_state)).lower()
    if "humanevalpack" in joined:
        return "humanevalpack"
    if "humaneval" in joined:
        return "humaneval"
    if "mbpp" in joined:
        return "mbpp"
    return None


def _mann_whitney_auroc(
    positive_scores: list[float], negative_scores: list[float]
) -> float:
    """Mann-Whitney rank AUROC; ties count 0.5. Inputs are small in practice."""
    favourable = 0.0
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                favourable += 1.0
            elif positive == negative:
                favourable += 0.5
    return favourable / (len(positive_scores) * len(negative_scores))


def _pooled_negative_scores(calibration: dict[str, Any]) -> list[float]:
    buckets = calibration.get("reliable_window_count_buckets")
    if not isinstance(buckets, dict):
        return []
    pooled: list[float] = []
    for values in buckets.values():
        if isinstance(values, list):
            pooled.extend(float(value) for value in values)
    return pooled


def _format_rule(rule: Any) -> str | None:
    if not isinstance(rule, dict):
        return None
    field = rule.get("field")
    operator = rule.get("operator")
    bound = rule.get("value", rule.get("threshold"))
    if field is None or operator is None or bound is None:
        return None
    return f"{field}{operator}{bound}"


def _pass_fields(
    payload: dict[str, Any] | None,
) -> tuple[float | None, str | None, list[str]]:
    if payload is None:
        return None, None, [_PASS_MISSING_CAVEAT]
    caveats: list[str] = []
    if payload.get("pass_at_1") is not None:
        if payload.get("pass_at_10") is not None:
            caveats.append(
                f"pass@10={float(payload['pass_at_10'])} also recorded; "
                "the contract row reports pass@1"
            )
        return float(payload["pass_at_1"]), "pass@1", caveats
    if payload.get("pass_rate") is not None:
        metric = payload.get("pass_metric")
        return (
            float(payload["pass_rate"]),
            str(metric) if metric is not None else None,
            caveats,
        )
    return None, None, [_PASS_MISSING_CAVEAT]


def _run_state_flag(run_state: dict[str, Any] | None, flag: str) -> bool | None:
    if run_state is None:
        return None
    for section in run_state.values():
        if isinstance(section, dict) and flag in section:
            value = section[flag]
            if isinstance(value, bool):
                return value
    return None


def _regen_caveat(
    run_state: dict[str, Any] | None, sample_count: int | None
) -> str | None:
    if run_state is None:
        return None
    detect_section = run_state.get("detect")
    if not isinstance(detect_section, dict):
        return None
    details_path = detect_section.get("details_path")
    if not isinstance(details_path, str):
        return None
    details_name = Path(details_path).name
    if "regen" not in details_name.lower():
        return None
    caveat = (
        f"detect details_path '{details_name}' indicates samples were "
        "regenerated post-hoc with a separate generation pass"
    )
    match = re.search(r"regen[_-]*failed[_-]*(\d+)", details_name.lower())
    if match is not None:
        total = f"/{sample_count}" if sample_count is not None else ""
        caveat += f" ({match.group(1)}{total} samples affected)"
    return caveat


def _experimental_bypass(
    gate_validation_skipped: bool | None,
    unvalidated_gate_candidate: bool | None,
    experimental_only: bool | None,
) -> bool | None:
    evidence = (gate_validation_skipped, unvalidated_gate_candidate, experimental_only)
    if any(value is True for value in evidence):
        return True
    if all(value is None for value in evidence):
        return None
    return False


def _apply_identity(
    record: dict[str, Any],
    *,
    report: dict[str, Any] | None,
    calibration: dict[str, Any],
    gate_manifest: dict[str, Any] | None,
    run_state: dict[str, Any] | None,
    sample_ids: list[Any],
) -> None:
    caveats: list[str] = record["caveats"]
    report = report or {}

    language = report.get("language")
    if language is None:
        language = _language_from_contract(calibration.get("window_contract_version"))
    if language is None and gate_manifest is not None:
        language = _language_from_contract(gate_manifest.get("parser_contract"))
    dataset = report.get("dataset")
    if dataset is None:
        dataset = _dataset_from_sample_ids(sample_ids)
    if dataset is None:
        dataset = _dataset_from_run_state_paths(run_state)
    if language is None and dataset in _PYTHON_DATASETS:
        language = "python"

    record["language"] = language
    record["dataset"] = dataset
    if language is None:
        caveats.append("language could not be derived from run artifacts")
    if dataset is None:
        caveats.append("dataset could not be derived from run artifacts")

    model = report.get("generation_model")
    record["generation_model"] = model
    if model is None:
        caveats.append(
            "generation model not recorded in run artifacts; historical runs "
            "kept it only in launch scripts"
        )


def _extract_basic(record: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    record["dataset"] = report.get("dataset")
    record["language"] = report.get("language")
    sample_count = report.get("sample_count")
    record["sample_count"] = int(sample_count) if sample_count is not None else None
    record["caveats"].append(
        "basic report: no-watermark wiring run without detection, so TPR, "
        "AUROC, and calibration fields are unavailable"
    )
    record["caveats"].append(_PASS_MISSING_CAVEAT)
    record["caveats"].append(
        "generation model not recorded in run artifacts; historical runs "
        "kept it only in launch scripts"
    )
    return record


def _extract_conditional_layout(
    record: dict[str, Any],
    run_dir: Path,
    conditional_summary: dict[str, Any] | None,
    execution_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    caveats: list[str] = record["caveats"]
    final_code_path = run_dir / "final_code.jsonl"
    final_rows = _load_jsonl_rows(final_code_path) if final_code_path.is_file() else []

    dataset = None
    if execution_summary is not None:
        dataset = execution_summary.get("dataset")
    if dataset is None:
        datasets = {row.get("dataset") for row in final_rows} - {None}
        dataset = sorted(datasets)[0] if len(datasets) == 1 else None
    if dataset is None:
        dataset = _dataset_from_sample_ids([row.get("id") for row in final_rows])
    record["dataset"] = dataset
    record["language"] = "python" if dataset in _PYTHON_DATASETS else None
    if record["language"] is None:
        caveats.append("language could not be derived from run artifacts")
    if dataset is None:
        caveats.append("dataset could not be derived from run artifacts")

    positive = (
        conditional_summary.get("positive", {})
        if conditional_summary is not None
        else {}
    )
    sample_count = positive.get("total")
    if sample_count is None and execution_summary is not None:
        sample_count = execution_summary.get("total")
    if sample_count is None and final_rows:
        sample_count = len(final_rows)
    record["sample_count"] = int(sample_count) if sample_count is not None else None

    if conditional_summary is not None and positive.get("conditional_tpr") is not None:
        record["tpr"] = float(positive["conditional_tpr"])
        record["tpr_is_conditional"] = True
        eligibility_rule = _format_rule(conditional_summary.get("eligibility_rule"))
        detection_rule = _format_rule(conditional_summary.get("detection_rule"))
        record["conditional_rule"] = ";".join(
            part for part in (eligibility_rule, detection_rule) if part is not None
        ) or None
        detected = positive.get("detected")
        record["detected_count"] = int(detected) if detected is not None else None
        caveats.append(
            "TPR is conditional: computed on "
            f"{positive.get('eligible')} eligible of {positive.get('total')} "
            f"samples ({positive.get('excluded')} excluded by the eligibility rule)"
        )
    else:
        caveats.append(
            "conditional detection summary not present; TPR unavailable"
        )

    # Reference-report fields may still appear inside the two summary files.
    for field in (
        "target_fpr",
        "minimum_reliable_windows",
        "key_identifier_sha256",
        "semantic_encoder_sha256",
    ):
        for payload in (conditional_summary, execution_summary):
            if payload is not None and payload.get(field) is not None:
                record[field] = payload[field]
                break
    if record["target_fpr"] is None or record["key_identifier_sha256"] is None:
        caveats.append(
            "reference report and calibration artifacts not present; "
            "target_fpr, minimum_reliable_windows, and detector identity "
            "hashes are unavailable"
        )
    caveats.append(
        "AUROC source artifacts (detection_curve and pooled calibration "
        "negatives) not present; auroc is null"
    )

    pass_rate, pass_metric, pass_caveats = _pass_fields(execution_summary)
    record["pass_rate"] = pass_rate
    record["pass_metric"] = pass_metric
    caveats.extend(pass_caveats)

    caveats.append(
        "generation model not recorded in run artifacts; historical runs "
        "kept it only in launch scripts"
    )
    caveats.append(
        "gate-data manifest not present; eligibility markers unavailable"
    )
    return record


def _extract_standard(
    record: dict[str, Any],
    run_dir: Path,
    report_root: Path,
    report: dict[str, Any],
    run_state: dict[str, Any] | None,
) -> dict[str, Any]:
    caveats: list[str] = record["caveats"]

    calibration = report.get("calibration")
    if not isinstance(calibration, dict):
        calibration = (
            _load_json_object(report_root / "calibration" / "reference_calibration.json")
            or {}
        )
    record["target_fpr"] = calibration.get("target_fpr")
    record["minimum_reliable_windows"] = calibration.get("minimum_reliable_windows")
    record["key_identifier_sha256"] = calibration.get("key_identifier_sha256")
    record["semantic_encoder_sha256"] = calibration.get("semantic_encoder_sha256")

    gate_manifest = _first_existing_json(
        [
            report_root / "gate-data" / "manifest.json",
            run_dir / "gate-data" / "manifest.json",
            run_dir.parent / "gate-data" / "manifest.json",
        ]
    )

    curve = report.get("detection_curve")
    sample_ids: list[Any] = []
    if isinstance(curve, list) and curve:
        sample_ids = [row.get("id") for row in curve if isinstance(row, dict)]
        record["sample_count"] = len(curve)
        decisions = [
            row.get("decision") for row in curve if isinstance(row, dict)
        ]
        detected = sum(decision == "watermarked" for decision in decisions)
        record["detected_count"] = detected
        record["tpr"] = detected / len(curve)
        positive_scores = [
            float(row["hit_rate"])
            for row in curve
            if isinstance(row, dict) and row.get("hit_rate") is not None
        ]
    else:
        sample_count = report.get("sample_count")
        record["sample_count"] = (
            int(sample_count) if sample_count is not None else None
        )
        positive_scores = []
        caveats.append(
            "detection_curve missing: per-sample decisions cannot be "
            "recovered from aggregate hit/miss counts, so TPR is null"
        )

    negative_scores = _pooled_negative_scores(calibration)
    if positive_scores and negative_scores:
        record["auroc"] = _mann_whitney_auroc(positive_scores, negative_scores)
    else:
        caveats.append(
            "AUROC source incomplete (needs detection_curve hit rates and "
            "pooled calibration negative hit rates); auroc is null"
        )

    _apply_identity(
        record,
        report=report,
        calibration=calibration,
        gate_manifest=gate_manifest,
        run_state=run_state,
        sample_ids=sample_ids,
    )

    if gate_manifest is not None:
        record["eligibility"] = {
            "formal_eligible": gate_manifest.get("formal_eligible"),
            "diagnostic_only": gate_manifest.get("diagnostic_only"),
            "experimental_only": gate_manifest.get("experimental_only"),
        }
    else:
        caveats.append("gate-data manifest not present; eligibility markers unavailable")

    for flag in ("gate_validation_skipped", "unvalidated_gate_candidate"):
        value = report.get(flag)
        if not isinstance(value, bool):
            value = _run_state_flag(run_state, flag)
        record[flag] = value
    record["experimental_bypass"] = _experimental_bypass(
        record["gate_validation_skipped"],
        record["unvalidated_gate_candidate"],
        record["eligibility"]["experimental_only"],
    )

    target_fpr = record["target_fpr"]
    if isinstance(target_fpr, (int, float)) and target_fpr > CONVENTIONAL_TARGET_FPR:
        caveats.append(
            f"target_fpr={target_fpr} exceeds the conventional "
            f"{CONVENTIONAL_TARGET_FPR} budget: detection thresholds were relaxed"
        )

    regen = _regen_caveat(run_state, record["sample_count"])
    if regen is not None:
        caveats.append(regen)

    posthoc = _load_json_object(report_root / "reports" / "pass_report_posthoc.json")
    if posthoc is None:
        posthoc = _load_json_object(run_dir / "humaneval_summary.json")
    pass_rate, pass_metric, pass_caveats = _pass_fields(posthoc)
    record["pass_rate"] = pass_rate
    record["pass_metric"] = pass_metric
    caveats.extend(pass_caveats)
    return record


def extract_metric_contract(run_dir: Path) -> dict[str, Any]:
    """Extract one Metric Contract record from a run directory.

    Read-only and unconditional: missing metrics become nulls plus caveats,
    never a refusal to emit (ADR 0003).
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise ValueError(f"run directory does not exist: {run_dir}")

    record = _empty_record(_run_identifier(run_dir))
    report_root = _resolve_report_root(run_dir)
    report = _load_json_object(report_root / "reports" / "reference_report.json")
    run_state = _first_existing_json(
        [
            report_root / "run_state.json",
            run_dir / "run_state.json",
            run_dir.parent / "run_state.json",
        ]
    )

    if report is not None and report.get("basic_experiment") is True:
        return _extract_basic(record, report)
    if report is not None:
        return _extract_standard(record, run_dir, report_root, report, run_state)

    conditional_summary = _load_json_object(run_dir / "n_ge_3_tpr_summary.json")
    execution_summary = _load_json_object(run_dir / "humaneval_summary.json")
    if (
        conditional_summary is not None
        or execution_summary is not None
        or (run_dir / "final_code.jsonl").is_file()
    ):
        return _extract_conditional_layout(
            record, run_dir, conditional_summary, execution_summary
        )

    record["caveats"].append(
        "no recognized run artifacts found in run directory; all metrics null"
    )
    return record
