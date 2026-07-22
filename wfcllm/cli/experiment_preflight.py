"""Fail-fast validation for the public multi-language experiment matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path


SUPPORTED_EXPERIMENT_PAIRS = frozenset(
    {
        ("python", "humaneval"),
        ("python", "mbpp"),
        ("cpp", "humanevalpack"),
        ("java", "humanevalpack"),
    }
)
SUPPORTED_EXPERIMENT_PROFILES = frozenset({"full", "fast"})


def validate_experiment_config(
    config: Mapping[str, object],
    language: str,
    dataset: str,
    profile: str,
) -> None:
    """Validate matrix identity and the no-carrier method invariant."""
    if (language, dataset) not in SUPPORTED_EXPERIMENT_PAIRS:
        raise ValueError(
            "unsupported language/dataset pair: "
            f"language={language!r}, dataset={dataset!r}"
        )
    if profile not in SUPPORTED_EXPERIMENT_PROFILES:
        raise ValueError(f"unsupported experiment profile: {profile!r}")

    generation = _mapping(config, "generation")
    experiment = _mapping(config, "experiment")
    semantic_lsh = _mapping(config, "semantic_lsh")
    method = _mapping(config, "method")
    rewrite = _mapping(method, "rewrite", prefix="method")

    _require_equal(generation, "language", language, "generation")
    _require_equal(generation, "dataset", dataset, "generation")
    _require_equal(experiment, "profile", profile, "experiment")
    if semantic_lsh.get("rule_name") != "semantic_lsh":
        raise ValueError(
            "experiment configs must use semantic_lsh; carrier-style "
            "keyed_text_region is forbidden"
        )
    if rewrite.get("strategy") == "keyed_text_region":
        raise ValueError("carrier rewrite strategy is forbidden")

    require_validated = _mapping(method, "gate", prefix="method").get(
        "require_validated"
    )
    expected_validated = profile == "full"
    if require_validated is not expected_validated:
        raise ValueError(
            "method.gate.require_validated must be "
            f"{expected_validated!r} for profile={profile!r}"
        )


def load_and_validate_experiment_config(
    path: Path,
    language: str,
    dataset: str,
    profile: str,
) -> dict[str, object]:
    """Load one explicit experiment config and validate its public identity."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid experiment config JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("experiment config must contain a JSON object")
    validate_experiment_config(value, language, dataset, profile)
    return value


def validate_runtime_capabilities(config: Mapping[str, object]) -> None:
    """Reject language paths that would otherwise fall into Python runtime code."""
    generation = _mapping(config, "generation")
    method = _mapping(config, "method")
    rewrite = _mapping(method, "rewrite", prefix="method")
    language = generation.get("language")
    strategy = rewrite.get("strategy")
    if language == "python":
        if strategy != "python_ast_equivalent":
            raise ValueError(
                "Python experiment requires the certified python_ast_equivalent "
                "rewrite strategy"
            )
        return
    if language in {"cpp", "java"}:
        if strategy != "model_semantic_window":
            raise ValueError(
                f"{language} experiments require model_semantic_window; "
                "Python AST and carrier rewrites are forbidden"
            )
        from wfcllm.windowing import (
            get_statement_unit_extractor,
            window_contract_for_language,
        )

        get_statement_unit_extractor(str(language))
        window_contract_for_language(str(language))
        return
    raise ValueError(f"unsupported runtime language: {language!r}")


def _mapping(
    parent: Mapping[str, object],
    key: str,
    *,
    prefix: str = "",
) -> Mapping[str, object]:
    value = parent.get(key)
    label = f"{prefix}.{key}" if prefix else key
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_equal(
    section: Mapping[str, object],
    key: str,
    expected: str,
    prefix: str,
) -> None:
    if section.get(key) != expected:
        raise ValueError(
            f"{prefix}.{key} must equal {expected!r}, got {section.get(key)!r}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="validate an experiment config")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--language", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--check-runtime-capabilities", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_and_validate_experiment_config(
            args.config,
            args.language,
            args.dataset,
            args.profile,
        )
        if args.check_runtime_capabilities:
            validate_runtime_capabilities(config)
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
