"""Offline evaluation helpers for dual-channel code-watermark experiments."""

from __future__ import annotations

import ast
import json
import math
import re
from pathlib import Path
from typing import Any


def load_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records from disk."""
    artifact_path = Path(path)
    records: list[dict[str, Any]] = []
    for raw_line in artifact_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {artifact_path}")
        records.append(payload)
    return records


def write_jsonl_records(path: str | Path, records: list[dict[str, Any]]) -> None:
    """Write JSONL records to disk using UTF-8."""
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(record, ensure_ascii=False) for record in records)
    artifact_path.write_text(f"{payload}\n" if payload else "", encoding="utf-8")


def compute_pass_at_k(records: list[dict[str, Any]], k: int) -> float:
    """Compute HumanEval/MBPP-style pass@k from per-sample correctness rows."""
    if k <= 0:
        raise ValueError("k must be > 0")

    grouped: dict[str, list[bool]] = {}
    for record in records:
        task_id = str(record.get("task_id") or record.get("id") or "")
        if not task_id:
            raise ValueError("each record must provide task_id or id")
        grouped.setdefault(task_id, []).append(bool(record.get("passed", False)))

    if not grouped:
        return 0.0

    estimates = [_estimate_pass_at_k(len(outcomes), sum(outcomes), k) for outcomes in grouped.values()]
    return sum(estimates) / len(estimates)


def compute_retry_rate(records: list[dict[str, Any]]) -> float:
    """Compute average retry attempts per simple block."""
    attempts_total = 0
    total_blocks = 0
    for record in records:
        retry_summary = record.get("retry_summary") or {}
        if not isinstance(retry_summary, dict):
            continue
        attempts_total += int(retry_summary.get("attempts_total", 0) or 0)
        total_blocks += int(record.get("total_blocks", 0) or 0)
    return attempts_total / total_blocks if total_blocks else 0.0


def compute_average_latency(total_seconds: float, sample_count: int) -> float:
    """Compute average per-sample latency from a measured phase runtime."""
    return total_seconds / sample_count if sample_count > 0 else 0.0


def compute_roc_auc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Compute ROC AUC without external metric dependencies."""
    if not positive_scores or not negative_scores:
        return 0.0

    wins = 0.0
    total_pairs = len(positive_scores) * len(negative_scores)
    for positive in positive_scores:
        for negative in negative_scores:
            if positive > negative:
                wins += 1.0
            elif positive == negative:
                wins += 0.5
    return wins / total_pairs


def compute_tpr_at_fpr(
    positive_scores: list[float],
    negative_scores: list[float],
    target_fpr: float,
) -> float:
    """Compute the best achievable TPR at or below a target FPR."""
    if not 0.0 <= target_fpr <= 1.0:
        raise ValueError("target_fpr must be between 0 and 1")
    if not positive_scores or not negative_scores:
        return 0.0

    thresholds = sorted({*positive_scores, *negative_scores}, reverse=True)
    thresholds.append(max(thresholds) + 1.0)

    best_tpr = 0.0
    for threshold in thresholds:
        tpr = sum(score >= threshold for score in positive_scores) / len(positive_scores)
        fpr = sum(score >= threshold for score in negative_scores) / len(negative_scores)
        if fpr <= target_fpr:
            best_tpr = max(best_tpr, tpr)
    return best_tpr


def mode_score_field(mode: str) -> str:
    """Return the score field associated with an evaluation mode."""
    mapping = {
        "semantic-only": "z_score",
        "lexical-only": "lexical_z_score",
        "dual-channel": "joint_score",
    }
    try:
        return mapping[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported mode: {mode}") from exc


def extract_scores(records: list[dict[str, Any]], mode: str) -> list[float]:
    """Extract the comparison score for one detection mode."""
    field_name = mode_score_field(mode)
    scores: list[float] = []
    for record in records:
        if field_name in record:
            scores.append(float(record[field_name]))
    return scores


def mean_or_zero(values: list[float]) -> float:
    """Return the arithmetic mean or zero for empty inputs."""
    return sum(values) / len(values) if values else 0.0


def relative_delta(value: float, baseline: float) -> float:
    """Return relative delta against a baseline, guarding the zero case."""
    if baseline == 0.0:
        return 0.0 if value == 0.0 else math.inf
    return (value - baseline) / baseline


def build_perturbation_corpus(
    records: list[dict[str, Any]],
    output_path: str | Path,
    perturbation: str,
) -> dict[str, Any]:
    """Create a derived JSONL with a lightweight semantic-preserving perturbation."""
    transformed_records: list[dict[str, Any]] = []
    changed_samples = 0
    for record in records:
        updated = dict(record)
        original_code = str(record.get("generated_code", ""))
        transformed_code = apply_perturbation(original_code, perturbation)
        if transformed_code != original_code:
            changed_samples += 1
        updated["generated_code"] = transformed_code
        updated["perturbation"] = perturbation
        transformed_records.append(updated)

    write_jsonl_records(output_path, transformed_records)
    return {
        "path": str(Path(output_path)),
        "changed_samples": changed_samples,
        "total_samples": len(transformed_records),
    }


def apply_perturbation(code: str, perturbation: str) -> str:
    """Apply one offline perturbation used by the evaluation harness."""
    if perturbation == "formatting":
        return _format_code(code)
    if perturbation == "comments":
        return _inject_comment(code)
    if perturbation == "rename":
        return _rename_local_variable(code)
    if perturbation == "light-rewrite":
        return _light_rewrite(code)
    raise ValueError(f"unsupported perturbation: {perturbation}")


def _estimate_pass_at_k(n: int, c: int, k: int) -> float:
    if c <= 0:
        return 0.0
    k = min(k, n)
    if n - c < k:
        return 1.0
    miss_probability = 1.0
    for index in range(k):
        miss_probability *= (n - c - index) / (n - index)
    return 1.0 - miss_probability


def _format_code(code: str) -> str:
    try:
        return ast.unparse(ast.parse(code)).rstrip() + "\n"
    except SyntaxError:
        return code


def _inject_comment(code: str) -> str:
    if not code.strip():
        return code
    comment = "# offline-eval-comment"
    if code.startswith(comment):
        return code
    return f"{comment}\n{code}"


def _rename_local_variable(code: str) -> str:
    match = re.search(r"^(?P<indent>\s*)(?P<name>[A-Za-z_]\w*)\s*=", code, flags=re.MULTILINE)
    if match is None:
        return code
    variable_name = match.group("name")
    if variable_name in {"self", "cls"}:
        return code
    renamed = f"{variable_name}_renamed"
    return re.sub(rf"\b{re.escape(variable_name)}\b", renamed, code)


def _light_rewrite(code: str) -> str:
    match = re.search(r"^(?P<indent>\s*)return\s+(?P<expr>.+)$", code, flags=re.MULTILINE)
    if match is None:
        return code
    indent = match.group("indent")
    expr = match.group("expr")
    replacement = f"{indent}__wfcllm_value = {expr}\n{indent}return __wfcllm_value"
    return code[: match.start()] + replacement + code[match.end() :]
