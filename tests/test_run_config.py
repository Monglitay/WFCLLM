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
