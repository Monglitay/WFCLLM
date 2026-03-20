"""Tests for run.py config loading."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

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


def test_base_config_extract_and_watermark_sections_have_matching_lsh_defaults():
    cfg = json.loads(Path("configs/base_config.json").read_text(encoding="utf-8"))
    assert cfg["extract"]["lsh_d"] == 4
    assert cfg["extract"]["lsh_gamma"] == 0.75
    assert cfg["extract"]["lsh_d"] == cfg["watermark"]["lsh_d"]
    assert cfg["extract"]["lsh_gamma"] == cfg["watermark"]["lsh_gamma"]


def test_run_extract_uses_resolved_gamma_for_calibration(monkeypatch, tmp_path):
    import run as run_module

    captured: dict = {}

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "generated_code": "def f():\n    return 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text(
        json.dumps({"generated_code": "def g():\n    return 2\n"}) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "extract": {
                    "secret_key": "k",
                    "input_file": str(input_file),
                    "calibration_corpus": str(calibration_file),
                    "lsh_d": 3,
                    "lsh_gamma": 0.5,
                    "fpr": 0.01,
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

    class FakeCalibrator:
        def __init__(self, scorer, gamma):
            captured["gamma"] = gamma

        def calibrate(self, corpus, fpr):
            return {"fpr_threshold": 1.2, "n_samples": len(corpus)}

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device):
            captured["resolved_threshold"] = config.fpr_threshold

    class FakePipeline:
        def __init__(self, detector, config):
            self._details = tmp_path / "sample_details.jsonl"
            self._summary = tmp_path / "sample_summary.json"

        @staticmethod
        def summary_path_for_details(details_path: Path) -> Path:
            return Path(details_path).parent / "sample_summary.json"

        def run(self) -> str:
            self._details.write_text("{}", encoding="utf-8")
            self._summary.write_text(
                json.dumps(
                    {
                        "meta": {"total_samples": 1},
                        "summary": {
                            "watermark_rate": 1.0,
                            "watermark_rate_ci_95": [1.0, 1.0],
                            "mean_z_score": 1.0,
                            "std_z_score": 0.0,
                            "mean_p_value": 0.1,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(self._details)

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: object())
    monkeypatch.setattr("wfcllm.encoder.model.SemanticEncoder", FakeEncoder)
    monkeypatch.setattr("wfcllm.extract.calibrator.ThresholdCalibrator", FakeCalibrator)
    monkeypatch.setattr("wfcllm.extract.scorer.BlockScorer", lambda *args, **kwargs: object())
    monkeypatch.setattr("wfcllm.watermark.keying.WatermarkKeying", lambda *args, **kwargs: object())
    monkeypatch.setattr("wfcllm.watermark.lsh_space.LSHSpace", lambda *args, **kwargs: object())
    monkeypatch.setattr("wfcllm.watermark.verifier.ProjectionVerifier", lambda *args, **kwargs: object())
    monkeypatch.setattr("wfcllm.extract.detector.WatermarkDetector", FakeDetector)
    monkeypatch.setattr("wfcllm.extract.pipeline.ExtractPipeline", FakePipeline)
    monkeypatch.setattr("wfcllm.extract.pipeline.ExtractPipelineConfig", lambda **kwargs: SimpleNamespace(**kwargs))

    args = SimpleNamespace(
        secret_key=None,
        input_file=str(input_file),
        extract_output_dir=None,
        embed_dim=None,
        fpr_threshold=None,
        resume=None,
        calibration_corpus=None,
        fpr=None,
        config=config_file,
    )

    rc = run_module.run_extract(args, FakeState())
    assert rc == 0
    assert captured["gamma"] == 0.75
    resolved_threshold = captured["resolved_threshold"]
    assert 0.5 < resolved_threshold < 2.5


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
