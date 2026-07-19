from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.cli.runners import _resolve_program_finalizer
from wfcllm.generation.completion_finalizer import finalize_humaneval_program


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "wfcllm" / "gated_semantic_window_v1.json"


def test_resolves_humaneval_finalizer() -> None:
    finalizer, name = _resolve_program_finalizer(
        {"program_finalizer": "humaneval_target_function_v1"}
    )

    assert finalizer is finalize_humaneval_program
    assert name == "humaneval_target_function_v1"


def test_resolves_explicit_no_finalizer() -> None:
    finalizer, name = _resolve_program_finalizer({"program_finalizer": "none"})

    assert finalizer is None
    assert name == "none"


def test_rejects_unknown_finalizer() -> None:
    with pytest.raises(ValueError, match="unsupported generation program_finalizer"):
        _resolve_program_finalizer({"program_finalizer": "hidden_test_selector"})


def test_gated_config_uses_approved_single_generation_contract() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    generation = config["generation"]

    assert generation["max_new_tokens"] == 512
    assert generation["temperature"] == 0.0
    assert generation["program_finalizer"] == "humaneval_target_function_v1"
    assert generation["seed"] == 7
    assert generation["load_in_4bit"] is True
