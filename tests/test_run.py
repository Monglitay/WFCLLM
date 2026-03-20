import argparse
import ast
import json
import sys
from pathlib import Path

import pytest


# ── 将项目根目录加入 sys.path（如果需要）
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import RunState, PHASES, ALL_PHASES


class TestRunState:
    def test_phases_order(self):
        assert PHASES == ["encoder", "watermark", "extract"]
        assert ALL_PHASES == ["encoder", "watermark", "extract", "generate-negative"]

    def test_initial_state_all_pending(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        for phase in PHASES:
            assert state.is_done(phase) is False

    def test_mark_done_persists(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder", checkpoint="data/checkpoints/encoder/encoder_epoch5.pt")

        # 重新加载
        state2 = RunState(state_file)
        assert state2.is_done("encoder") is True
        assert state2.get("encoder", "checkpoint") == "data/checkpoints/encoder/encoder_epoch5.pt"

    def test_reset_clears_all(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder")
        state.reset()
        assert state.is_done("encoder") is False

    def test_status_dict(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder", checkpoint="x.pt")
        status = state.status()
        assert status["encoder"]["done"] is True
        assert status["watermark"]["done"] is False
        assert status["extract"]["done"] is False


import subprocess


class TestRunGenerateNegative:
    @pytest.fixture
    def neg_cfg_file(self, tmp_path):
        import json as _json
        cfg_data = {"generate_negative": {"lm_model_path": "", "dataset": "humaneval", "dataset_path": "data/datasets", "output_path": "data/neg.jsonl", "max_new_tokens": 512, "temperature": 0.8, "top_p": 0.95, "top_k": 50, "device": "cuda", "limit": None}}
        cfg_path = tmp_path / "cfg.json"
        cfg_path.write_text(_json.dumps(cfg_data))
        return cfg_path

    def test_run_generate_negative_missing_lm_model_path(self, tmp_path, neg_cfg_file):
        """run_generate_negative returns 1 when lm_model_path is missing."""
        from run import run_generate_negative, RunState

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

    def test_run_generate_negative_calls_generator(self, tmp_path, neg_cfg_file):
        """run_generate_negative calls NegativeCorpusGenerator.run() and marks done."""
        from unittest.mock import patch, MagicMock
        from run import run_generate_negative, RunState

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

        with patch("wfcllm.extract.negative_corpus.NegativeCorpusGenerator", return_value=mock_gen):
            rc = run_generate_negative(args, state)

        assert rc == 0
        mock_gen.run.assert_called_once()
        assert state.is_done("generate-negative")


class TestCLI:
    def test_build_parser_parses_resume_argument(self):
        from run import build_parser

        args = build_parser().parse_args(["--phase", "extract", "--resume", "latest"])
        assert args.resume == "latest"

    def test_status_exits_zero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", "run.py", "--status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "encoder" in result.stdout

    def test_reset_exits_zero(self, tmp_path, monkeypatch):
        # 使用临时状态文件以免影响真实 data/
        monkeypatch.chdir(tmp_path)
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python",
             str(Path(__file__).parent.parent / "run.py"), "--reset"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "重置" in result.stdout or "reset" in result.stdout.lower()

    def test_unknown_phase_exits_nonzero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", "run.py", "--phase", "invalid"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0


class TestRunWatermarkConfigNoFallback:
    def _find_keyword_call(self, source_path: str, call_name: str, keyword_name: str) -> bool:
        tree = ast.parse(Path(source_path).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == call_name:
                if any(keyword.arg == keyword_name for keyword in node.keywords):
                    return True
        return False

    def test_run_watermark_no_enable_fallback(self):
        """run.py 构建 WatermarkConfig 不传 enable_fallback（已废弃）。"""
        source = Path("run.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_fallback":
                raise AssertionError("run.py 仍传递了已废弃的 enable_fallback 参数")

    def test_run_watermark_has_enable_cascade(self):
        """run.py 构建 WatermarkConfig 传递 enable_cascade。"""
        source = Path("run.py").read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_cascade":
                return
        raise AssertionError("run.py 应传递 enable_cascade 参数给 WatermarkConfig")

    def test_run_watermark_pipeline_config_receives_resume(self):
        assert self._find_keyword_call("run.py", "WatermarkPipelineConfig", "resume")

    def test_run_extract_pipeline_config_receives_resume(self):
        assert self._find_keyword_call("run.py", "ExtractPipelineConfig", "resume")

    def test_run_extract_marks_summary_file_in_state(self):
        tree = ast.parse(Path("run.py").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "mark_done":
                if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "extract":
                    keywords = {keyword.arg for keyword in node.keywords}
                    assert "details_file" in keywords
                    assert "summary_file" in keywords
                    assert "report_file" not in keywords
                    return
        raise AssertionError("run.py should mark extract state with details_file and summary_file")


def test_run_extract_passes_lsh_params():
    tree = ast.parse(Path("run.py").read_text(encoding="utf-8"))
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

    assert seen_extract_config_call, "run.py must call ExtractConfig"
    assert matched_expected_call, (
        "run.py must pass lsh_d=lsh_d and lsh_gamma=lsh_gamma into ExtractConfig"
    )


def test_run_extract_resolves_lsh_params_from_first_record():
    tree = ast.parse(Path("run.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "resolve_extract_lsh_params":
            assert len(node.args) >= 2
            assert isinstance(node.args[0], ast.Name)
            assert isinstance(node.args[1], ast.Name)
            assert node.args[0].id == "first_record"
            assert node.args[1].id == "ext_cfg"
            return
    raise AssertionError("run.py must resolve extract LSH params from first_record/ext_cfg")


def test_run_extract_returns_error_on_invalid_first_record_json(tmp_path, capsys):
    from run import run_extract

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
        config=cfg,
    )

    rc = run_extract(args, _State())
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "输入文件首条记录" in stderr


def test_run_extract_returns_error_on_invalid_first_record_lsh(tmp_path, capsys):
    from run import run_extract

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
        config=cfg,
    )

    rc = run_extract(args, _State())
    stderr = capsys.readouterr().err
    assert rc == 1
    assert "LSH 参数无效" in stderr
