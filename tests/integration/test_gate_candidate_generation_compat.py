from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from wfcllm.cli.runners import _candidate_config_hash_matches
from wfcllm.gate.production import experiment_contract_hash


def _current_config() -> dict:
    return json.loads(
        Path("configs/wfcllm/gated_semantic_window_v1.json").read_text(
            encoding="utf-8"
        )
    )


def _legacy_generation_hash(config: dict) -> str:
    legacy = deepcopy(config)
    legacy["generation"]["max_new_tokens"] = 256
    legacy["generation"]["temperature"] = 0.25
    legacy["generation"].pop("program_finalizer")
    return experiment_contract_hash(legacy)


def test_candidate_accepts_exact_current_contract_hash() -> None:
    config = _current_config()

    assert _candidate_config_hash_matches(
        experiment_contract_hash(config), config
    )


def test_candidate_rejects_legacy_generation_hash() -> None:
    """The frozen generation-only upgrade shim is retired with gate-validate."""
    config = _current_config()

    assert not _candidate_config_hash_matches(
        _legacy_generation_hash(config), config
    )


def test_candidate_rejects_hash_when_gate_contract_changes() -> None:
    config = _current_config()
    trained_hash = experiment_contract_hash(config)
    changed = deepcopy(config)
    changed["method"]["gate"]["max_input_tokens"] += 1

    assert not _candidate_config_hash_matches(trained_hash, changed)


def test_fast_profile_accepts_any_recorded_candidate_hash() -> None:
    config = _current_config()
    config["experiment"] = {"profile": "fast"}

    assert _candidate_config_hash_matches("0" * 64, config)
    assert not _candidate_config_hash_matches(None, config)
