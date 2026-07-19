from __future__ import annotations

import re
import unicodedata


FORBIDDEN_SECRET_FIELDS = frozenset(
    {
        "access_token",
        "api_key",
        "deployment_key",
        "key_material",
        "password",
        "private_key",
        "raw_secret",
        "raw_secret_key",
        "raw_training_key",
        "secret_key",
        "target_lsh_region",
    }
)

FORBIDDEN_FORMAL_QUALITY_PROXY_FIELDS = frozenset(
    {
        "ast_parse_success",
        "benchmark_outcome",
        "canonical_correctness_proxy",
        "checkpoint_correctness_proxy",
        "complexity_score",
        "correctness_result",
        "correctness_score",
        "duplicate_function_heuristic",
        "generated_vs_canonical_length_gap",
        "hardcoded_example_heuristic",
        "hidden_tests",
        "import_safety",
        "logits_quality_proxy",
        "pass",
        "passed",
        "prompt_contract_heuristic",
        "public_doctests",
        "quality_proxy",
        "quality_rank",
        "reference_correctness_proxy",
        "return_yield_presence",
        "signature_compatible",
        "static_correctness_score",
        "suspicious_tail",
        "syntax_valid",
        "target_function_presence",
        "test_result",
        "truncation_heuristic",
    }
)

UNSAFE_CHECKPOINT_STATE_FIELDS = frozenset(
    {
        "model_state",
        "model_state_dict",
        "optimizer_state",
        "optimizer_state_dict",
        "pickle",
        "rng_state",
        "scheduler_state",
        "scheduler_state_dict",
        "state_dict",
        "storage",
        "tensor",
    }
)

_CAMEL_BOUNDARY_1 = re.compile(r"(.)([A-Z][a-z]+)")
_CAMEL_BOUNDARY_2 = re.compile(r"([a-z0-9])([A-Z])")
_NON_IDENTIFIER = re.compile(r"[^a-z0-9]+")
_KNOWN_PLURAL_TOKENS = {
    "buffers": "buffer",
    "coefficients": "coefficient",
    "credentials": "credential",
    "correctnesses": "correctness",
    "doctests": "doctest",
    "gaps": "gap",
    "heuristics": "heuristic",
    "keys": "key",
    "labels": "label",
    "materials": "material",
    "models": "model",
    "moments": "moment",
    "optimizers": "optimizer",
    "outcomes": "outcome",
    "oracles": "oracle",
    "parameters": "parameter",
    "passes": "pass",
    "payloads": "payload",
    "proxies": "proxy",
    "qualities": "quality",
    "ranks": "rank",
    "rewards": "reward",
    "regions": "region",
    "results": "result",
    "scores": "score",
    "safeties": "safety",
    "secrets": "secret",
    "signatures": "signature",
    "states": "state",
    "successes": "success",
    "schedulers": "scheduler",
    "tensors": "tensor",
    "tests": "test",
    "values": "value",
    "valids": "valid",
    "weights": "weight",
}
_KNOWN_SINGULAR_TOKENS = {
    singular: plural for plural, singular in _KNOWN_PLURAL_TOKENS.items()
}
_IRREVERSIBLE_OR_PROVENANCE_SUFFIXES = (
    "_bank_id",
    "_count",
    "_hash",
    "_present",
    "_sha256",
    "_source",
    "_source_type",
)


def canonical_artifact_field_name(field: str) -> str:
    """Normalize public JSON field spellings without confusing ``parse`` with pass."""

    if not isinstance(field, str):
        raise ValueError("artifact field name must be a string")
    value = unicodedata.normalize("NFKC", field)
    if not value or any(not 0x20 <= ord(character) <= 0x7E for character in value):
        raise ValueError(
            "artifact field name must normalize to non-empty printable ASCII"
        )
    value = _CAMEL_BOUNDARY_1.sub(r"\1_\2", value)
    value = _CAMEL_BOUNDARY_2.sub(r"\1_\2", value).casefold()
    separated = _NON_IDENTIFIER.sub("_", value).strip("_")
    return "_".join(
        _KNOWN_PLURAL_TOKENS.get(token, token)
        for token in separated.split("_")
        if token
    )


def artifact_field_tokens_and_compact(field: str) -> tuple[tuple[str, ...], str]:
    canonical = canonical_artifact_field_name(field)
    tokens = tuple(token for token in canonical.split("_") if token)
    return tokens, "".join(tokens)


def _matches_named_field(field: str, expected_compacts: frozenset[str]) -> bool:
    _, compact = artifact_field_tokens_and_compact(field)
    return compact in expected_compacts


def _named_field_compact_variants(field: str) -> frozenset[str]:
    tokens = artifact_field_tokens_and_compact(field)[0]
    variants = {""}
    for token in tokens:
        options = {token}
        plural = _KNOWN_SINGULAR_TOKENS.get(token)
        if plural is not None:
            options.add(plural)
        variants = {prefix + option for prefix in variants for option in options}
    return frozenset(variants)


_FORBIDDEN_SECRET_COMPACTS = frozenset(
    compact
    for field in FORBIDDEN_SECRET_FIELDS
    for compact in _named_field_compact_variants(field)
)
_FORBIDDEN_QUALITY_COMPACTS = frozenset(
    compact
    for field in FORBIDDEN_FORMAL_QUALITY_PROXY_FIELDS
    for compact in _named_field_compact_variants(field)
)
_UNSAFE_CHECKPOINT_COMPACTS = frozenset(
    compact
    for field in UNSAFE_CHECKPOINT_STATE_FIELDS
    for compact in _named_field_compact_variants(field)
)


def is_forbidden_secret_field(field: str) -> bool:
    canonical = canonical_artifact_field_name(field)
    if _matches_named_field(field, _FORBIDDEN_SECRET_COMPACTS):
        return True
    if canonical.endswith(_IRREVERSIBLE_OR_PROVENANCE_SUFFIXES):
        return False
    tokens = set(artifact_field_tokens_and_compact(field)[0])
    return bool(
        "secret" in tokens
        or "credential" in tokens
        or {"raw", "training", "key"} <= tokens
        or {"deployment", "key"} <= tokens
        or {"key", "material"} <= tokens
        or {"target", "lsh", "region"} <= tokens
    )


def is_forbidden_formal_quality_proxy_field(field: str) -> bool:
    if canonical_artifact_field_name(field) == "semantic_preservation_passed":
        return False
    if _matches_named_field(field, _FORBIDDEN_QUALITY_COMPACTS):
        return True
    tokens = set(artifact_field_tokens_and_compact(field)[0])
    if tokens & {"correctness", "oracle", "reward"}:
        return True
    if tokens & {"pass", "passed", "passing"}:
        return True
    if "quality" in tokens:
        return True
    if {"syntax", "valid"} <= tokens or {"syntax", "validity"} <= tokens:
        return True
    outcome_tokens = {"fail", "failed", "outcome", "pass", "passed", "result"}
    if "test" in tokens and bool(tokens & outcome_tokens):
        return True
    if "benchmark" in tokens and bool(
        tokens & (outcome_tokens | {"score", "label"})
    ):
        return True
    return False


def is_unsafe_checkpoint_state_field(field: str) -> bool:
    return _matches_named_field(field, _UNSAFE_CHECKPOINT_COMPACTS)

POSTHOC_PASS_REQUIRED_FLAGS = {
    "posthoc_only",
    "not_used_for_generation",
    "not_used_for_retry",
    "not_used_for_selection",
    "not_used_for_calibration",
    "not_used_for_detection",
}
