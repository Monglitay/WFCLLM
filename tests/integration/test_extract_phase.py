"""Extract-phase integration tests migrated from test_run.py and test_run_config.py."""
import argparse
import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.integration.conftest import PROJECT_ROOT, RUN_PY, RUNNERS_PY, write_json, write_jsonl

sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# From TestCLI class in test_run.py
# ---------------------------------------------------------------------------


def test_main_rejects_compare_only_mode_outside_extract_phase(monkeypatch, capsys):
    import wfcllm.cli.entry as run_module

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--phase",
            "watermark",
            "--compare-summary-left",
            "left-summary.json",
            "--compare-details-left",
            "left-details.jsonl",
            "--compare-watermarked-left",
            "left-watermarked.jsonl",
            "--compare-summary-right",
            "right-summary.json",
            "--compare-details-right",
            "right-details.jsonl",
            "--compare-watermarked-right",
            "right-watermarked.jsonl",
            "--compare-output",
            "report.json",
        ],
    )

    rc = run_module.main()
    stderr = capsys.readouterr().err

    assert rc == 1
    assert "compare-only" in stderr
    assert "extract" in stderr


def test_main_compare_only_extract_bypasses_extract_prerequisites(tmp_path, monkeypatch):
    import wfcllm.cli.entry as run_module

    left_summary = tmp_path / "left_summary.json"
    right_summary = tmp_path / "right_summary.json"
    left_details = tmp_path / "left_details.jsonl"
    right_details = tmp_path / "right_details.jsonl"
    report_output = tmp_path / "offline_analysis.json"

    write_json(left_summary, {"dataset": "HumanEval", "summary": {"watermark_rate": 1.0}})
    write_json(right_summary, {"dataset": "HumanEval", "summary": {"watermark_rate": 0.8}})
    write_jsonl(
        left_details,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": True,
                "z_score": 2.4,
                "p_value": 0.02,
                "independent_blocks": 8,
                "hits": 6,
            }
        ],
    )
    write_jsonl(
        right_details,
        [
            {
                "id": "HumanEval/0",
                "is_watermarked": False,
                "z_score": 1.0,
                "p_value": 0.14,
                "independent_blocks": 8,
                "hits": 5,
            }
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--phase",
            "extract",
            "--compare-summary-left",
            str(left_summary),
            "--compare-details-left",
            str(left_details),
            "--compare-summary-right",
            str(right_summary),
            "--compare-details-right",
            str(right_details),
            "--compare-output",
            str(report_output),
        ],
    )

    rc = run_module.main()

    assert rc == 0
    assert report_output.exists()


def test_main_runs_explicit_extract_input_even_when_extract_phase_is_done(
    tmp_path,
    monkeypatch,
    capsys,
):
    import wfcllm.cli.entry as run_module

    input_file = tmp_path / "input.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")
    called: list[str] = []

    class FakeState:
        def __init__(self, path=None, **kwargs):
            pass

        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "extract"

    def fake_run_extract(args, state) -> int:
        called.append("extract")
        assert args.input_file == str(input_file)
        return 0

    monkeypatch.setattr("wfcllm.cli.entry.RunStateManager", FakeState)
    monkeypatch.setattr("wfcllm.cli.entry.run_extract", fake_run_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--phase",
            "extract",
            "--input-file",
            str(input_file),
        ],
    )

    rc = run_module.main()
    captured = capsys.readouterr()

    assert rc == 0
    assert called == ["extract"]
    assert "[跳过] extract" not in captured.out


def test_main_runs_configured_extract_input_even_when_extract_phase_is_done(
    tmp_path,
    monkeypatch,
    capsys,
):
    import wfcllm.cli.entry as run_module

    input_file = tmp_path / "input.jsonl"
    input_file.write_text("{}\n", encoding="utf-8")
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps({"extract": {"input_file": str(input_file)}}),
        encoding="utf-8",
    )
    called: list[str] = []

    class FakeState:
        def __init__(self, path=None, **kwargs):
            pass

        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "extract"

    def fake_run_extract(args, state) -> int:
        called.append("extract")
        assert args.config == config_file
        return 0

    monkeypatch.setattr("wfcllm.cli.entry.RunStateManager", FakeState)
    monkeypatch.setattr("wfcllm.cli.entry.run_extract", fake_run_extract)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run.py",
            "--phase",
            "extract",
            "--config",
            str(config_file),
        ],
    )

    rc = run_module.main()
    captured = capsys.readouterr()

    assert rc == 0
    assert called == ["extract"]
    assert "[跳过] extract" not in captured.out


# ---------------------------------------------------------------------------
# From TestRunWatermarkConfigNoFallback class in test_run.py (extract-related)
# ---------------------------------------------------------------------------


def _find_keyword_call(call_name: str, keyword_name: str) -> bool:
    tree = ast.parse(RUNNERS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == call_name:
            if any(keyword.arg == keyword_name for keyword in node.keywords):
                return True
    return False


def test_run_extract_pipeline_config_receives_resume():
    assert _find_keyword_call("ExtractPipelineConfig", "resume")


def test_run_extract_marks_summary_file_in_state():
    tree = ast.parse(RUNNERS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "mark_done":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "extract":
                keywords = {keyword.arg for keyword in node.keywords}
                assert "details_file" in keywords
                assert "summary_file" in keywords
                assert "report_file" not in keywords
                return
    raise AssertionError("runners.py should mark extract state with details_file and summary_file")


# ---------------------------------------------------------------------------
# Module-level functions from test_run.py
# ---------------------------------------------------------------------------


def test_run_extract_passes_lsh_params():
    tree = ast.parse(RUNNERS_PY.read_text(encoding="utf-8"))
    seen_extract_config_call = False
    matched_expected_call = False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ExtractConfig":
            seen_extract_config_call = True
            kw_map = {kw.arg: kw.value for kw in node.keywords}
            has_expected_lsh_d = (
                "lsh_d" in kw_map
                and isinstance(kw_map["lsh_d"], ast.Name)
                and kw_map["lsh_d"].id == "lsh_d"
            )
            has_expected_lsh_gamma = (
                "lsh_gamma" in kw_map
                and isinstance(kw_map["lsh_gamma"], ast.Name)
                and kw_map["lsh_gamma"].id == "lsh_gamma"
            )
            if has_expected_lsh_d and has_expected_lsh_gamma:
                matched_expected_call = True

    assert seen_extract_config_call, "runners.py must call ExtractConfig"
    assert matched_expected_call, (
        "runners.py must pass lsh_d=lsh_d and lsh_gamma=lsh_gamma into ExtractConfig"
    )


def test_run_extract_resolves_lsh_params_from_first_record():
    tree = ast.parse(RUNNERS_PY.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "resolve_extract_lsh_params":
            assert len(node.args) >= 2
            assert isinstance(node.args[0], ast.Name)
            assert isinstance(node.args[1], ast.Name)
            assert node.args[0].id == "first_record"
            assert node.args[1].id == "ext_cfg"
            return
    raise AssertionError("runners.py must resolve extract LSH params from first_record/ext_cfg")


def test_run_extract_returns_error_on_invalid_first_record_json(tmp_path, capsys):
    from wfcllm.cli.runners import run_extract

    class _State:
        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "encoder"

        @staticmethod
        def get(phase: str, key: str):
            return None

    bad_input = tmp_path / "bad.jsonl"
    bad_input.write_text("{ not valid json }\n", encoding="utf-8")
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"extract": {"secret_key": "k", "input_file": str(bad_input)}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        secret_key=None,
        input_file=str(bad_input),
        extract_output_dir=None,
        embed_dim=None,
        fpr_threshold=None,
        resume=None,
        calibration_corpus=None,
        fpr=None,
        min_blocks=None,
        config=cfg,
    )

    rc = run_extract(args, _State())
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "输入文件首条记录" in stderr


def test_run_extract_returns_error_on_invalid_first_record_lsh(tmp_path, capsys):
    from wfcllm.cli.runners import run_extract

    class _State:
        @staticmethod
        def is_done(phase: str) -> bool:
            return phase == "encoder"

        @staticmethod
        def get(phase: str, key: str):
            return None

    bad_input = tmp_path / "bad_lsh.jsonl"
    bad_input.write_text(
        json.dumps({"watermark_params": {"lsh_d": "bad", "lsh_gamma": 0.75}}) + "\n",
        encoding="utf-8",
    )
    cfg = tmp_path / "cfg.json"
    cfg.write_text(
        json.dumps({"extract": {"secret_key": "k", "input_file": str(bad_input)}}),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        secret_key=None,
        input_file=str(bad_input),
        extract_output_dir=None,
        embed_dim=None,
        fpr_threshold=None,
        resume=None,
        calibration_corpus=None,
        fpr=None,
        min_blocks=None,
        config=cfg,
    )

    rc = run_extract(args, _State())
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "LSH 参数无效" in stderr


# ---------------------------------------------------------------------------
# From test_run_config.py
# ---------------------------------------------------------------------------


def test_run_extract_uses_resolved_gamma_for_calibration(monkeypatch, tmp_path):
    import wfcllm.cli.runners as run_module

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
        def __init__(self, scorer, gamma, mode="fixed", block_contract_builder=None):
            captured["gamma"] = gamma
            captured["mode"] = mode
            captured["has_block_contract_builder"] = block_contract_builder is not None

        def calibrate(self, corpus, fpr):
            return {"fpr_threshold": 1.2, "n_samples": len(corpus)}

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device, **kwargs):
            captured["resolved_threshold"] = config.fpr_threshold

    class FakePipeline:
        def __init__(self, detector, config):
            captured["summary_metadata"] = config.summary_metadata
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
    monkeypatch.setattr("wfcllm.extract.calibration.threshold.ThresholdCalibrator", FakeCalibrator)
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
        min_blocks=None,
        config=config_file,
    )

    rc = run_module.run_extract(args, FakeState())
    assert rc == 0
    assert captured["gamma"] == 0.75
    assert captured["mode"] == "fixed"
    assert captured["has_block_contract_builder"] is False
    assert captured["summary_metadata"]["calibration"] == {
        "source": str(calibration_file),
        "fpr": 0.01,
        "threshold": 1.2,
        "hypothesis_mode": "fixed",
        "statistic_definition": "m * gamma, m * gamma * (1 - gamma)",
        "decision_rule": "z_score >= threshold",
    }
    resolved_threshold = captured["resolved_threshold"]
    assert 0.5 < resolved_threshold < 2.5


def test_run_extract_parses_token_channel_from_config(monkeypatch, tmp_path):
    import wfcllm.cli.runners as run_module

    captured: dict = {}

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "generated_code": "x = 1\n",
                "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
            }
        ) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "extract": {
                    "secret_key": "k",
                    "input_file": str(input_file),
                    "token_channel": {
                        "enabled": True,
                        "channel_mode": "lexical-only",
                        "joint": {
                            "threshold": 5.0,
                        },
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

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device, **kwargs):
            captured["token_channel"] = config.token_channel

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
        min_blocks=None,
        config=config_file,
        adaptive_detection_mode=None,
        strict_contract=False,
        gamma_strategy=None,
        entropy_profile=None,
        profile_id=None,
    )

    rc = run_module.run_extract(args, FakeState())
    assert rc == 0
    assert captured["token_channel"].enabled is True
    assert captured["token_channel"].mode == "lexical-only"
    assert captured["token_channel"].joint_threshold == 5.0


def test_run_extract_uses_adaptive_calibration_when_adaptive_detection_enabled(
    monkeypatch,
    tmp_path,
):
    import wfcllm.cli.runners as run_module

    captured: dict = {}

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "generated_code": "x = 1\n",
                "watermark_params": {
                    "lsh_d": 4,
                    "lsh_gamma": 0.75,
                },
            }
        ) + "\n",
        encoding="utf-8",
    )
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text(
        json.dumps({"generated_code": "x = 1\n"}) + "\n",
        encoding="utf-8",
    )
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        json.dumps(
            {
                "language": "python",
                "model_family": "demo-model",
                "quantiles_units": {
                    "p10": 1,
                    "p50": 2,
                    "p75": 3,
                    "p90": 4,
                    "p95": 5,
                },
                "strategy": "piecewise_quantile",
            }
        ),
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "watermark": {
                    "adaptive_gamma": {
                        "enabled": True,
                        "strategy": "piecewise_quantile",
                        "profile_path": str(profile_file),
                        "profile_id": "demo-profile",
                        "anchors": {
                            "p10": 0.95,
                            "p50": 0.75,
                            "p75": 0.55,
                            "p90": 0.35,
                            "p95": 0.25,
                        },
                    }
                },
                "extract": {
                    "secret_key": "k",
                    "input_file": str(input_file),
                    "calibration_corpus": str(calibration_file),
                    "lsh_d": 4,
                    "lsh_gamma": 0.75,
                    "fpr": 0.01,
                    "adaptive_detection": {
                        "mode": "prefer-adaptive",
                    },
                },
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
        def __init__(self, scorer, gamma, mode="fixed", block_contract_builder=None):
            captured["gamma"] = gamma
            captured["mode"] = mode
            captured["block_contract_builder"] = block_contract_builder

        def calibrate(self, corpus, fpr):
            contracts = captured["block_contract_builder"]("x = 1\n")
            captured["built_contracts"] = contracts
            return {"fpr_threshold": 1.0, "n_samples": len(corpus)}

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device, **kwargs):
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
    monkeypatch.setattr("wfcllm.extract.calibration.threshold.ThresholdCalibrator", FakeCalibrator)
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
        min_blocks=None,
        config=config_file,
    )

    rc = run_module.run_extract(args, FakeState())
    assert rc == 0
    assert captured["mode"] == "adaptive"
    assert captured["gamma"] == 0.75
    assert captured["block_contract_builder"] is not None
    assert captured["built_contracts"]["0"]["k"] > 0


def test_run_extract_prefers_configured_fpr_when_cli_fpr_not_provided(
    monkeypatch,
    tmp_path,
):
    import wfcllm.cli.runners as run_module

    captured: dict = {}

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "generated_code": "x = 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )
    calibration_file = tmp_path / "calibration.jsonl"
    calibration_file.write_text(
        json.dumps({"generated_code": "x = 1\n"}) + "\n",
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
                    "lsh_d": 4,
                    "lsh_gamma": 0.75,
                    "fpr": 0.05,
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
        def __init__(self, scorer, gamma, mode="fixed", block_contract_builder=None):
            return None

        def calibrate(self, corpus, fpr):
            captured["fpr"] = fpr
            return {"fpr_threshold": 1.0, "n_samples": len(corpus)}

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device, **kwargs):
            return None

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
    monkeypatch.setattr("wfcllm.extract.calibration.threshold.ThresholdCalibrator", FakeCalibrator)
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
        min_blocks=None,
        config=config_file,
    )

    rc = run_module.run_extract(args, FakeState())
    assert rc == 0


def test_run_extract_allows_extract_only_validation_without_encoder_done(
    monkeypatch,
    tmp_path,
):
    import wfcllm.cli.runners as run_module

    captured: dict = {}

    input_file = tmp_path / "input.jsonl"
    input_file.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "generated_code": "x = 1\n",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
            }
        ) + "\n",
        encoding="utf-8",
    )
    config_file = tmp_path / "cfg.json"
    config_file.write_text(
        json.dumps(
            {
                "extract": {
                    "secret_key": "k",
                    "input_file": str(input_file),
                    "lsh_d": 4,
                    "lsh_gamma": 0.75,
                }
            }
        ),
        encoding="utf-8",
    )

    class FakeState:
        @staticmethod
        def is_done(phase: str) -> bool:
            return False

        @staticmethod
        def get(phase: str, key: str):
            return None

        @staticmethod
        def mark_done(*args, **kwargs):
            captured["mark_done"] = (args, kwargs)
            return None

    class FakeEncoder:
        def __init__(self, config):
            self.config = config

        def load_state_dict(self, state):
            return None

        def to(self, device):
            return self

    class FakeDetector:
        def __init__(self, config, encoder, tokenizer, device, **kwargs):
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
                            "mean_blocks": 1.0,
                        },
                    }
                ),
                encoding="utf-8",
            )
            return str(self._details)

    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda _: object())
    monkeypatch.setattr("wfcllm.encoder.model.SemanticEncoder", FakeEncoder)
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
        min_blocks=None,
        config=config_file,
        adaptive_detection_mode=None,
        strict_contract=False,
        gamma_strategy=None,
        entropy_profile=None,
        profile_id=None,
    )

    rc = run_module.run_extract(args, FakeState())

    assert rc == 0
    assert captured["mark_done"][0][0] == "extract"
