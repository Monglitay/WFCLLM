from __future__ import annotations

from typing import Any

FORBIDDEN_QUALITY_GATE_FIELDS = {
    "hidden_tests",
    "public_doctests",
    "pass",
    "passed",
    "correctness_result",
    "ast_parse_success",
    "syntax_valid",
    "target_function_presence",
    "signature_compatible",
    "suspicious_tail",
    "truncation_heuristic",
    "duplicate_function_heuristic",
    "import_safety",
    "return_yield_presence",
    "prompt_contract_heuristic",
    "hardcoded_example_heuristic",
    "static_correctness_score",
    "complexity_score",
    "generated_vs_canonical_length_gap",
    "reference_correctness_proxy",
    "canonical_correctness_proxy",
    "logits_quality_proxy",
    "checkpoint_correctness_proxy",
}


def reject_quality_proxy_fields(value: Any) -> None:
    forbidden_path = _find_forbidden_quality_proxy(value, path="")
    if forbidden_path is not None:
        raise ValueError(f"quality proxy field is forbidden: {forbidden_path}")


def require_diagnostic_marker(payload: dict[str, Any]) -> None:
    if payload.get("diagnostic_only") is not True:
        raise ValueError("diagnostic_only must be true for diagnostic artifacts")
    if payload.get("not_official_method") is not True:
        raise ValueError("not_official_method must be true for diagnostic artifacts")


def _find_forbidden_quality_proxy(value: Any, *, path: str) -> str | None:
    if isinstance(value, dict):
        for key, nested in value.items():
            key_name = str(key)
            child_path = f"{path}.{key_name}" if path else key_name
            if key_name in FORBIDDEN_QUALITY_GATE_FIELDS:
                return child_path
            found = _find_forbidden_quality_proxy(nested, path=child_path)
            if found is not None:
                return found
    elif isinstance(value, list):
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            found = _find_forbidden_quality_proxy(item, path=child_path)
            if found is not None:
                return found
    return None
