"""Tests for generate-negative runner logic."""
import argparse
import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


@pytest.fixture
def neg_cfg_file(tmp_path):
    import json as _json
    cfg_data = {"generate_negative": {"source_mode": "llm", "lm_model_path": "", "dataset": "humaneval", "dataset_path": "data/datasets", "output_path": "data/neg.jsonl", "max_new_tokens": 512, "temperature": 0.8, "top_p": 0.95, "top_k": 50, "device": "cuda", "limit": None}}
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(_json.dumps(cfg_data))
    return cfg_path


def test_run_generate_negative_missing_lm_model_path(tmp_path, neg_cfg_file):
    """run_generate_negative returns 1 when lm_model_path is missing."""
    from wfcllm.cli.runners import run_generate_negative
    from wfcllm.orchestration.state import RunStateManager as RunState

    state = RunState(tmp_path / "state.json")

    args = argparse.Namespace(
        lm_model_path=None,
        dataset=None,
        dataset_path=None,
        negative_output=None,
        negative_limit=None,
        config=neg_cfg_file,
    )

    rc = run_generate_negative(args, state)
    assert rc == 1


def test_run_generate_negative_reference_mode_allows_missing_lm_model_path(
    tmp_path,
):
    from unittest.mock import patch, MagicMock
    from wfcllm.cli.runners import run_generate_negative
    from wfcllm.orchestration.state import RunStateManager as RunState

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(
        json.dumps(
            {
                "generate_negative": {
                    "source_mode": "reference",
                    "lm_model_path": "",
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "output_path": str(tmp_path / "neg.jsonl"),
                    "limit": None,
                }
            }
        )
    )
    state = RunState(tmp_path / "state.json")
    args = argparse.Namespace(
        lm_model_path=None,
        dataset=None,
        dataset_path=None,
        negative_output=None,
        negative_limit=None,
        config=cfg_path,
    )
    mock_gen = MagicMock()
    mock_gen.run.return_value = str(tmp_path / "neg.jsonl")

    with patch("wfcllm.extract.calibration.negative_corpus.NegativeCorpusGenerator", return_value=mock_gen):
        rc = run_generate_negative(args, state)

    assert rc == 0
    mock_gen.run.assert_called_once()


def test_run_generate_negative_calls_generator(tmp_path, neg_cfg_file):
    """run_generate_negative calls NegativeCorpusGenerator.run() and marks done."""
    from unittest.mock import patch, MagicMock
    from wfcllm.cli.runners import run_generate_negative
    from wfcllm.orchestration.state import RunStateManager as RunState

    state = RunState(tmp_path / "state.json")
    out_jsonl = str(tmp_path / "neg.jsonl")

    args = argparse.Namespace(
        lm_model_path="data/models/my-model",
        dataset="humaneval",
        dataset_path="data/datasets",
        negative_output=out_jsonl,
        negative_limit=None,
        config=neg_cfg_file,
    )

    mock_gen = MagicMock()
    mock_gen.run.return_value = out_jsonl

    with patch("wfcllm.extract.calibration.negative_corpus.NegativeCorpusGenerator", return_value=mock_gen):
        rc = run_generate_negative(args, state)

    assert rc == 0
    mock_gen.run.assert_called_once()
    assert state.is_done("generate-negative")


def test_base_config_uses_new_mainline_without_generate_negative_defaults():
    cfg = json.loads((CONFIGS_DIR / "base_config.json").read_text(encoding="utf-8"))
    assert cfg["method"]["name"] == "evidence_retry_seed7x3"
    assert cfg["runtime"]["default_phases"] == [
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
    ]
    assert "generate_negative" not in cfg
    assert "token_channel_train" not in cfg
