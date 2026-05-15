"""Watermark-phase integration tests migrated from test_run.py and test_run_config.py."""
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.conftest import PROJECT_ROOT, RUN_PY, RUNNERS_PY

CONFIGS_DIR = PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# Helpers (from TestRunWatermarkConfigNoFallback)
# ---------------------------------------------------------------------------

def _find_keyword_call(call_name: str, keyword_name: str) -> bool:
    tree = ast.parse(RUNNERS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == call_name:
            if any(keyword.arg == keyword_name for keyword in node.keywords):
                return True
    return False


# ---------------------------------------------------------------------------
# From TestRunWatermarkConfigNoFallback (test_run.py)
# ---------------------------------------------------------------------------

def test_run_watermark_no_enable_fallback():
    """run.py 构建 WatermarkConfig 不传 enable_fallback（已废弃）。"""
    source = RUN_PY.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "enable_fallback":
            raise AssertionError("run.py 仍传递了已废弃的 enable_fallback 参数")


def test_run_watermark_has_enable_cascade():
    """runners.py 构建 WatermarkConfig 传递 enable_cascade。"""
    source = RUNNERS_PY.read_text()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "enable_cascade":
            return
    raise AssertionError("runners.py 应传递 enable_cascade 参数给 WatermarkConfig")


def test_run_watermark_pipeline_config_receives_resume():
    assert _find_keyword_call("WatermarkPipelineConfig", "resume")


def test_run_watermark_pipeline_config_receives_sample_limit():
    assert _find_keyword_call("WatermarkPipelineConfig", "sample_limit")


# ---------------------------------------------------------------------------
# From TestBaseConfigFallbackCascade (test_run_config.py)
# ---------------------------------------------------------------------------

def test_no_enable_fallback_in_watermark_config():
    """base_config.json watermark 节不应有 enable_fallback 字段（已废弃）。"""
    cfg = json.loads((CONFIGS_DIR / "base_config.json").read_text())
    assert "enable_fallback" not in cfg.get("watermark", {}), (
        "base_config.json 的 watermark 节不应再有 enable_fallback"
    )


def test_enable_cascade_true_in_watermark_config():
    """base_config.json watermark 节的 enable_cascade 应为 true。"""
    cfg = json.loads((CONFIGS_DIR / "base_config.json").read_text())
    assert cfg.get("watermark", {}).get("enable_cascade") is True


def test_base_config_exposes_semantic_only_token_channel_defaults():
    cfg = json.loads((CONFIGS_DIR / "base_config.json").read_text(encoding="utf-8"))
    token_channel = cfg.get("watermark", {}).get("token_channel")

    assert token_channel == {
        "enabled": False,
        "channel_mode": "semantic-only",
    }


# ---------------------------------------------------------------------------
# Standalone tests from test_run_config.py
# ---------------------------------------------------------------------------

def test_base_config_b_restores_humaneval_best_known_region():
    cfg_path = CONFIGS_DIR / "base_config_B.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    anchors = cfg["watermark"]["adaptive_gamma"]["anchors"]

    assert cfg["watermark"]["lsh_d"] == 4
    assert cfg["extract"]["lsh_d"] == 4
    assert cfg["watermark"]["lsh_gamma"] == 0.75
    assert cfg["extract"]["lsh_gamma"] == 0.75
    assert anchors == {
        "p10": 0.75,
        "p50": 0.75,
        "p75": 0.50,
        "p90": 0.50,
        "p95": 0.25,
    }


def test_run_watermark_parses_token_channel_from_config(monkeypatch, tmp_path):
    import wfcllm.cli.runners as run_module

    captured: dict = {}

    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "watermark": {
                    "secret_key": "k",
                    "lm_model_path": "local-model",
                    "token_channel": {
                        "enabled": True,
                        "channel_mode": "lexical-only",
                        "model_path": "data/models/token-channel-demo",
                        "context_width": 64,
                        "delta": 1.5,
                        "lexical_min_block_tokens": 12,
                    },
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeState:
        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "encoder"

        @staticmethod
        def get(phase: str, key: str):
            return None

        @staticmethod
        def mark_done(*args, **kwargs):
            return None

    class FakeEncoder:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state):
            return None

        def to(self, device):
            return self

    class FakeGenerator:
        def __init__(self, lm_model, lm_tokenizer, encoder, encoder_tokenizer, config):
            captured["token_channel"] = config.token_channel

    class FakePipeline:
        def __init__(self, generator, config):
            return None

        def run(self) -> str:
            return str(tmp_path / "watermarked.jsonl")

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: object())
    monkeypatch.setattr("transformers.AutoModelForCausalLM.from_pretrained", lambda *args, **kwargs: object())
    monkeypatch.setattr("transformers.BitsAndBytesConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr("wfcllm.encoder.model.SemanticEncoder", FakeEncoder)
    monkeypatch.setattr("wfcllm.watermark.orchestrator.WatermarkGenerator", FakeGenerator)
    monkeypatch.setattr("wfcllm.watermark.pipeline.WatermarkPipeline", FakePipeline)
    monkeypatch.setattr("wfcllm.watermark.pipeline.WatermarkPipelineConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    args = SimpleNamespace(
        dataset=None,
        dataset_path=None,
        output_dir=None,
        sample_limit=None,
        embed_dim=None,
        secret_key=None,
        lm_model_path=None,
        resume=None,
        config=config_file,
        gamma_strategy=None,
        entropy_profile=None,
        profile_id=None,
    )

    rc = run_module.run_watermark(args, FakeState())
    assert rc == 0
    assert captured["token_channel"].enabled is True
    assert captured["token_channel"].mode == "lexical-only"
    assert captured["token_channel"].model_path == "data/models/token-channel-demo"
    assert captured["token_channel"].context_width == 64
    assert captured["token_channel"].delta == 1.5
    assert captured["token_channel"].lexical_min_block_tokens == 12


def test_run_watermark_rejects_non_object_token_channel_config(tmp_path, capsys):
    import wfcllm.cli.runners as run_module

    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "watermark": {
                    "secret_key": "k",
                    "lm_model_path": "local-model",
                    "token_channel": True,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeState:
        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "encoder"

        @staticmethod
        def get(phase: str, key: str):
            return None

    args = SimpleNamespace(
        dataset=None,
        dataset_path=None,
        output_dir=None,
        sample_limit=None,
        embed_dim=None,
        secret_key=None,
        lm_model_path=None,
        resume=None,
        config=config_file,
        gamma_strategy=None,
        entropy_profile=None,
        profile_id=None,
    )

    rc = run_module.run_watermark(args, FakeState())

    assert rc == 1
    assert "token_channel" in capsys.readouterr().err


def test_base_config_includes_adaptive_sections():
    cfg = json.loads((CONFIGS_DIR / "base_config.json").read_text(encoding="utf-8"))

    assert cfg["watermark"]["adaptive_gamma"] == {
        "enabled": True,
        "strategy": "piecewise_quantile",
        "profile_path": "data/calibration/humaneval_entropy_profile.json",
        "profile_id": "humaneval_entropy_profile",
        "gamma_min": 0.25,
        "gamma_max": 0.95,
        "anchors": {
            "p10": 0.95,
            "p50": 0.75,
            "p75": 0.55,
            "p90": 0.35,
            "p95": 0.25,
        },
    }
    assert cfg["extract"]["adaptive_detection"] == {
        "mode": "prefer-adaptive",
        "require_block_contract_check": True,
        "fail_on_structure_mismatch": True,
        "warn_on_numeric_mismatch": True,
        "exclude_invalid_samples": True,
    }
    assert cfg["extract"]["input_file"] is None
    assert cfg["extract"]["fpr"] == 0.05


def test_humaneval_subset_config_exposes_adaptive_experiment_defaults():
    cfg = json.loads(
        (CONFIGS_DIR / "humaneval_10_config.json").read_text(encoding="utf-8")
    )

    adaptive_gamma = cfg["watermark"]["adaptive_gamma"]
    assert cfg["watermark"]["sample_limit"] == 10
    assert adaptive_gamma["enabled"] is True
    assert adaptive_gamma["strategy"] == "piecewise_quantile"
    assert adaptive_gamma["profile_path"].startswith("data/calibration/")
    assert adaptive_gamma["profile_path"].endswith("_entropy_profile.json")
    assert adaptive_gamma["profile_id"] == Path(adaptive_gamma["profile_path"]).stem
    assert cfg["extract"]["adaptive_detection"]["mode"] == "prefer-adaptive"
    assert cfg["generate_negative"]["source_mode"] == "reference"
