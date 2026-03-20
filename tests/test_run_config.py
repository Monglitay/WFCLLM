"""Tests for run.py config loading."""
import json
import sys
from pathlib import Path

import pytest

# run.py 在项目根目录，需加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import load_config


def test_load_config_returns_dict(tmp_path):
    cfg = {"encoder": {"lr": 0.001}, "watermark": {}, "extract": {}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result["encoder"]["lr"] == 0.001


def test_load_config_missing_phase_ok(tmp_path):
    """缺少某阶段的 key，返回空 dict 而不报错。"""
    cfg = {"encoder": {"lr": 0.001}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result.get("watermark", {}) == {}


def test_load_config_file_not_found(tmp_path):
    with pytest.raises(SystemExit):
        load_config(tmp_path / "nonexistent.json")


def test_load_config_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{ not valid json }")
    with pytest.raises(SystemExit):
        load_config(f)


from run import build_parser


def test_parser_default_config():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/base_config.json")


def test_parser_custom_config():
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/my.json"])
    assert args.config == Path("configs/my.json")


def test_parser_accepts_adaptive_watermark_and_extract_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--gamma-strategy",
            "piecewise_quantile",
            "--entropy-profile",
            "configs/demo_profile.json",
            "--profile-id",
            "python__demo__v1",
            "--adaptive-detection-mode",
            "prefer-adaptive",
            "--strict-contract",
        ]
    )

    assert args.gamma_strategy == "piecewise_quantile"
    assert args.entropy_profile == "configs/demo_profile.json"
    assert args.profile_id == "python__demo__v1"
    assert args.adaptive_detection_mode == "prefer-adaptive"
    assert args.strict_contract is True


class TestBaseConfigFallbackCascade:
    def test_no_enable_fallback_in_watermark_config(self):
        """base_config.json watermark 节不应有 enable_fallback 字段（已废弃）。"""
        import json
        from pathlib import Path
        cfg = json.loads(Path("configs/base_config.json").read_text())
        assert "enable_fallback" not in cfg.get("watermark", {}), (
            "base_config.json 的 watermark 节不应再有 enable_fallback"
        )

    def test_enable_cascade_true_in_watermark_config(self):
        """base_config.json watermark 节的 enable_cascade 应为 true。"""
        import json
        from pathlib import Path
        cfg = json.loads(Path("configs/base_config.json").read_text())
        assert cfg.get("watermark", {}).get("enable_cascade") is True


def test_base_config_includes_adaptive_sections():
    cfg = json.loads(Path("configs/base_config.json").read_text(encoding="utf-8"))

    assert cfg["watermark"]["adaptive_gamma"] == {
        "enabled": False,
        "strategy": "piecewise_quantile",
        "profile_path": None,
        "profile_id": None,
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
        "mode": "fixed",
        "require_block_contract_check": True,
        "fail_on_structure_mismatch": True,
        "warn_on_numeric_mismatch": True,
        "exclude_invalid_samples": True,
    }


def test_humaneval_subset_config_exposes_adaptive_experiment_defaults():
    cfg = json.loads(
        Path("configs/humaneval_10_config.json").read_text(encoding="utf-8")
    )

    assert cfg["watermark"]["sample_limit"] == 10
    assert cfg["watermark"]["adaptive_gamma"]["enabled"] is True
    assert cfg["watermark"]["adaptive_gamma"]["strategy"] == "piecewise_quantile"
    assert cfg["watermark"]["adaptive_gamma"]["profile_path"] == (
        "data/calibration/humaneval_10_entropy_profile.json"
    )
    assert cfg["extract"]["adaptive_detection"]["mode"] == "prefer-adaptive"
