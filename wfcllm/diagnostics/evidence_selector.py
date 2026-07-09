from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wfcllm.method.contracts import require_diagnostic_marker

RANKING_MODES = {
    "evidence_first",
    "detector_first",
}


@dataclass(frozen=True)
class EvidenceCandidateFeatures:
    sample_id: str
    candidate_index: int
    detector_score: float
    scoreable_contexts: int = 0
    proxy_windows: int = 0
    insufficient_evidence: bool = False
    score_delta_vs_baseline: float | None = None
    proxy_delta_vs_baseline: int | None = None


def mark_evidence_selector_diagnostic(payload: dict[str, Any]) -> dict[str, Any]:
    marked = {
        **payload,
        "diagnostic_only": True,
        "not_official_method": True,
    }
    require_diagnostic_marker(marked)
    return marked


def select_candidate_rows(
    candidate_paths: list[Path],
    detail_paths: list[Path],
    *,
    min_detector_score: float | None = None,
    max_score_drop_vs_baseline: float | None = None,
    min_proxy_windows: int = 0,
    min_scoreable_contexts: int = 0,
    require_not_insufficient: bool = False,
    require_proxy_ge_baseline: bool = False,
    ranking_mode: str = "evidence_first",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not candidate_paths:
        raise ValueError("at least one --candidate-jsonl is required")
    if detail_paths and len(detail_paths) != len(candidate_paths):
        raise ValueError("--candidate-details count must match --candidate-jsonl count")
    if ranking_mode not in RANKING_MODES:
        raise ValueError(
            f"ranking_mode must be one of {sorted(RANKING_MODES)}, got {ranking_mode!r}"
        )

    candidate_rows = [_load_jsonl(path) for path in candidate_paths]
    detail_features = [_load_detail_features(path) for path in detail_paths]
    while len(detail_features) < len(candidate_rows):
        detail_features.append({})

    sample_ids = sorted({str(row.get("id") or "") for rows in candidate_rows for row in rows})
    selected_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []

    for sample_id in sample_ids:
        features: list[EvidenceCandidateFeatures] = []
        row_by_candidate: dict[int, dict[str, Any]] = {}
        for candidate_index, rows in enumerate(candidate_rows):
            row = _first_row_for_id(rows, sample_id)
            if row is None:
                continue
            detail = detail_features[candidate_index].get(sample_id, {})
            baseline_detail = detail_features[0].get(sample_id, {}) if detail_features else {}
            feature = _evidence_features(
                sample_id,
                detail,
                candidate_index=candidate_index,
                baseline_detail=baseline_detail if baseline_detail else None,
            )
            row_by_candidate[candidate_index] = row
            features.append(feature)

        eligible = _eligible_features(
            features,
            min_detector_score=min_detector_score,
            max_score_drop_vs_baseline=max_score_drop_vs_baseline,
            min_proxy_windows=min_proxy_windows,
            min_scoreable_contexts=min_scoreable_contexts,
            require_not_insufficient=require_not_insufficient,
            require_proxy_ge_baseline=require_proxy_ge_baseline,
        )
        selected = max(
            eligible,
            key=lambda item: _ranking_key(item, ranking_mode=ranking_mode),
        )
        selected_row = row_by_candidate[selected.candidate_index]
        selected_rows.append(_sanitized_output_row(selected_row))
        per_sample.append(
            {
                "id": sample_id,
                "selected_candidate_index": selected.candidate_index,
                "selection_reason": ranking_mode,
                "candidates": [feature.__dict__ for feature in features],
            }
        )

    analysis = {
        "artifact_type": "evidence_only_selector_diagnostic",
        "candidate_paths": [str(path) for path in candidate_paths],
        "detail_paths": [str(path) for path in detail_paths],
        "sample_count": len(selected_rows),
        "selected_by_candidate_index": _count_selected(per_sample),
        "evidence_counts": _evidence_counts(per_sample),
        "policy": {
            "min_detector_score": min_detector_score,
            "max_score_drop_vs_baseline": max_score_drop_vs_baseline,
            "min_proxy_windows": min_proxy_windows,
            "min_scoreable_contexts": min_scoreable_contexts,
            "require_not_insufficient": require_not_insufficient,
            "require_proxy_ge_baseline": require_proxy_ge_baseline,
            "ranking_mode": ranking_mode,
        },
        "per_sample": per_sample,
    }
    return selected_rows, mark_evidence_selector_diagnostic(analysis)


def selection_analysis_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Evidence-Only Candidate Selection",
        "",
        f"Selected rows: {analysis['sample_count']}",
        "",
        "## Selected by Candidate Index",
        "",
    ]
    for key, count in sorted(analysis["selected_by_candidate_index"].items()):
        lines.append(f"- candidate {key}: {count}")
    lines.extend(["", "## Evidence Counts", ""])
    for key, count in sorted(analysis["evidence_counts"].items()):
        lines.append(f"- {key}: {count}")
    return "\n".join(lines) + "\n"


def _evidence_features(
    sample_id: str,
    detail: dict[str, Any],
    *,
    candidate_index: int,
    baseline_detail: dict[str, Any] | None,
) -> EvidenceCandidateFeatures:
    detector_score = _float_feature(detail.get("score"), default=0.0)
    proxy_windows = _int_feature(detail.get("proxy_windows"))
    baseline_score = (
        _float_feature(baseline_detail.get("score"), default=0.0)
        if baseline_detail
        else None
    )
    baseline_proxy_windows = (
        _int_feature(baseline_detail.get("proxy_windows"))
        if baseline_detail
        else None
    )
    return EvidenceCandidateFeatures(
        sample_id=sample_id,
        candidate_index=candidate_index,
        detector_score=detector_score,
        scoreable_contexts=_int_feature(detail.get("scoreable_contexts")),
        proxy_windows=proxy_windows,
        insufficient_evidence=bool(detail.get("insufficient_evidence", False)),
        score_delta_vs_baseline=(
            detector_score - baseline_score if baseline_score is not None else None
        ),
        proxy_delta_vs_baseline=(
            proxy_windows - baseline_proxy_windows
            if baseline_proxy_windows is not None
            else None
        ),
    )


def _ranking_key(
    item: EvidenceCandidateFeatures,
    *,
    ranking_mode: str,
) -> tuple[object, ...]:
    if ranking_mode == "detector_first":
        return (
            item.detector_score,
            not item.insufficient_evidence,
            item.proxy_windows,
            item.scoreable_contexts,
            -item.candidate_index,
        )
    return (
        not item.insufficient_evidence,
        item.detector_score,
        item.proxy_windows,
        item.scoreable_contexts,
        -item.candidate_index,
    )


def _sanitized_output_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _eligible_features(
    features: list[EvidenceCandidateFeatures],
    *,
    min_detector_score: float | None,
    max_score_drop_vs_baseline: float | None,
    min_proxy_windows: int,
    min_scoreable_contexts: int,
    require_not_insufficient: bool,
    require_proxy_ge_baseline: bool,
) -> list[EvidenceCandidateFeatures]:
    if not features:
        return []
    baseline = min(features, key=lambda feature: feature.candidate_index)
    eligible = [baseline]
    for feature in features:
        if feature.candidate_index == baseline.candidate_index:
            continue
        if min_detector_score is not None and feature.detector_score < min_detector_score:
            continue
        if (
            max_score_drop_vs_baseline is not None
            and feature.score_delta_vs_baseline is not None
            and feature.score_delta_vs_baseline < -max_score_drop_vs_baseline
        ):
            continue
        if feature.proxy_windows < min_proxy_windows:
            continue
        if feature.scoreable_contexts < min_scoreable_contexts:
            continue
        if require_not_insufficient and feature.insufficient_evidence:
            continue
        if (
            require_proxy_ge_baseline
            and feature.proxy_delta_vs_baseline is not None
            and feature.proxy_delta_vs_baseline < 0
        ):
            continue
        eligible.append(feature)
    return eligible


def _load_detail_features(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("id") or ""): row
        for row in _load_jsonl(path)
    }


def _float_feature(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int_feature(value: Any) -> int:
    if isinstance(value, (list, tuple)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _first_row_for_id(rows: list[dict[str, Any]], sample_id: str) -> dict[str, Any] | None:
    for row in rows:
        if str(row.get("id") or "") == sample_id:
            return row
    return None


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _count_selected(per_sample: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_sample:
        key = str(row["selected_candidate_index"])
        counts[key] = counts.get(key, 0) + 1
    return counts


def _evidence_counts(per_sample: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in per_sample:
        selected_index = row["selected_candidate_index"]
        selected = next(
            candidate
            for candidate in row["candidates"]
            if candidate["candidate_index"] == selected_index
        )
        if not selected["insufficient_evidence"]:
            counts["sufficient_evidence"] = counts.get("sufficient_evidence", 0) + 1
        if selected["proxy_windows"] > 0:
            counts["has_proxy_windows"] = counts.get("has_proxy_windows", 0) + 1
        if selected["scoreable_contexts"] > 0:
            counts["has_scoreable_contexts"] = (
                counts.get("has_scoreable_contexts", 0) + 1
            )
    return counts
