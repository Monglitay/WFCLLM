from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

from wfcllm.cli.runners import _unvalidated_candidate_config_hash_matches
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

    assert _unvalidated_candidate_config_hash_matches(
        experiment_contract_hash(config), config
    )


def test_candidate_accepts_only_frozen_generation_upgrade_from_legacy_hash() -> None:
    config = _current_config()

    assert _unvalidated_candidate_config_hash_matches(
        _legacy_generation_hash(config), config
    )


def test_candidate_rejects_legacy_hash_when_gate_contract_changes() -> None:
    config = _current_config()
    legacy_hash = _legacy_generation_hash(config)
    changed = deepcopy(config)
    changed["method"]["gate"]["max_input_tokens"] += 1

    assert not _unvalidated_candidate_config_hash_matches(legacy_hash, changed)
