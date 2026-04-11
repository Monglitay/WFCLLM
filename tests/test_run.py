import argparse
import ast
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RUN_PY = PROJECT_ROOT / "run.py"
README_MD = PROJECT_ROOT / "README.md"

# ── 将项目根目录加入 sys.path（如果需要）
sys.path.insert(0, str(PROJECT_ROOT))

from run import ALL_PHASES, PHASES, RunState


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


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
        cfg_data = {"generate_negative": {"source_mode": "llm", "lm_model_path": "", "dataset": "humaneval", "dataset_path": "data/datasets", "output_path": "data/neg.jsonl", "max_new_tokens": 512, "temperature": 0.8, "top_p": 0.95, "top_k": 50, "device": "cuda", "limit": None}}
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

    def test_run_generate_negative_reference_mode_allows_missing_lm_model_path(
        self,
        tmp_path,
    ):
        from unittest.mock import patch, MagicMock
        from run import run_generate_negative, RunState

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

        with patch("wfcllm.extract.negative_corpus.NegativeCorpusGenerator", return_value=mock_gen):
            rc = run_generate_negative(args, state)

        assert rc == 0
        mock_gen.run.assert_called_once()

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
    def test_cli_subprocess_invocations_do_not_use_bare_run_py_script_name(self):
        tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (isinstance(node.func, ast.Attribute) and node.func.attr == "run"):
                continue
            if not node.args or not isinstance(node.args[0], ast.List):
                continue
            constants = [
                elt.value
                for elt in node.args[0].elts
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str)
            ]
            assert "run.py" not in constants

    def test_build_parser_parses_resume_argument(self):
        from run import build_parser

        args = build_parser().parse_args(["--phase", "extract", "--resume", "latest"])
        assert args.resume == "latest"

    def test_build_parser_accepts_token_channel_flags(self):
        from run import build_parser

        args = build_parser().parse_args(
            [
                "--phase",
                "watermark",
                "--token-channel-enabled",
                "true",
                "--token-channel-mode",
                "dual-channel",
                "--token-channel-model-path",
                "data/models/token-channel-demo",
                "--token-channel-delta",
                "1.5",
                "--token-channel-joint-threshold",
                "5.0",
            ]
        )

        assert args.token_channel_enabled is True
        assert args.token_channel_mode == "dual-channel"
        assert args.token_channel_model_path == "data/models/token-channel-demo"
        assert args.token_channel_delta == pytest.approx(1.5)
        assert args.token_channel_joint_threshold == pytest.approx(5.0)

    def test_resolve_token_channel_config_applies_cli_overrides(self):
        from run import resolve_token_channel_config

        args = argparse.Namespace(
            token_channel_enabled=True,
            token_channel_mode="lexical-only",
            token_channel_model_path="data/models/token-channel-demo",
            token_channel_context_width=64,
            token_channel_switch_threshold=0.25,
            token_channel_delta=1.5,
            token_channel_ignore_repeated_ngrams=True,
            token_channel_ignore_repeated_prefixes=True,
            token_channel_debug_mode=True,
            token_channel_lexical_min_block_tokens=12,
            token_channel_lexical_retry_decay_start=3,
            token_channel_lexical_retry_disable_after=5,
            token_channel_lexical_gate_probe_tokens=20,
            token_channel_lexical_gate_min_fraction=0.2,
            token_channel_joint_semantic_weight=1.25,
            token_channel_joint_lexical_weight=0.6,
            token_channel_lexical_full_weight_min_positions=48,
            token_channel_joint_threshold=5.0,
        )

        resolved = resolve_token_channel_config(
            {
                "enabled": False,
                "channel_mode": "semantic-only",
                "delta": 2.0,
            },
            args,
        )

        assert resolved.enabled is True
        assert resolved.mode == "lexical-only"
        assert resolved.model_path == "data/models/token-channel-demo"
        assert resolved.context_width == 64
        assert resolved.switch_threshold == pytest.approx(0.25)
        assert resolved.delta == pytest.approx(1.5)
        assert resolved.ignore_repeated_ngrams is True
        assert resolved.ignore_repeated_prefixes is True
        assert resolved.debug_mode is True
        assert resolved.lexical_min_block_tokens == 12
        assert resolved.lexical_retry_decay_start == 3
        assert resolved.lexical_retry_disable_after == 5
        assert resolved.lexical_gate_probe_tokens == 20
        assert resolved.lexical_gate_min_fraction == pytest.approx(0.2)
        assert resolved.joint_semantic_weight == pytest.approx(1.25)
        assert resolved.joint_lexical_weight == pytest.approx(0.6)
        assert resolved.lexical_full_weight_min_positions == 48
        assert resolved.joint_threshold == pytest.approx(5.0)

    def test_resolve_token_channel_config_preserves_value_error_for_invalid_joint_config(self):
        from run import resolve_token_channel_config

        args = argparse.Namespace(
            token_channel_enabled=True,
            token_channel_mode=None,
            token_channel_model_path=None,
            token_channel_context_width=None,
            token_channel_switch_threshold=None,
            token_channel_delta=None,
            token_channel_ignore_repeated_ngrams=None,
            token_channel_ignore_repeated_prefixes=None,
            token_channel_debug_mode=None,
            token_channel_lexical_min_block_tokens=None,
            token_channel_lexical_retry_decay_start=None,
            token_channel_lexical_retry_disable_after=None,
            token_channel_lexical_gate_probe_tokens=None,
            token_channel_lexical_gate_min_fraction=None,
            token_channel_joint_semantic_weight=None,
            token_channel_joint_lexical_weight=None,
            token_channel_lexical_full_weight_min_positions=None,
            token_channel_joint_threshold=None,
        )

        with pytest.raises(ValueError, match="joint must be a JSON object"):
            resolve_token_channel_config({"joint": 1}, args)

    def test_help_lists_token_channel_flags(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "--token-channel-enabled" in result.stdout
        assert "--token-channel-model-path" in result.stdout
        assert "--token-channel-joint-threshold" in result.stdout

    def test_token_channel_train_help_matches_documented_loader_state(self):
        readme_text = README_MD.read_text(encoding="utf-8")

        assert "当前入口仅用于加载离线训练缓存 / teacher cache 并校验参数" in readme_text

        result = subprocess.run(
            [
                "conda",
                "run",
                "-n",
                "WFCLLM",
                "python",
                "-m",
                "wfcllm.watermark.token_channel.train",
                "--help",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "Offline token-channel training loader" in result.stdout
        assert "--corpus-cache" in result.stdout
        assert "--teacher-cache" in result.stdout

    def test_status_exits_zero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "encoder" in result.stdout

    def test_reset_exits_zero(self, tmp_path, monkeypatch):
        # 使用临时状态文件以免影响真实 data/
        monkeypatch.chdir(tmp_path)
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--reset"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "重置" in result.stdout or "reset" in result.stdout.lower()

    def test_unknown_phase_exits_nonzero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", str(RUN_PY), "--phase", "invalid"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_run_offline_analysis_writes_json_report(self, tmp_path):
        from run import run_offline_analysis

        left_summary = tmp_path / "left_summary.json"
        right_summary = tmp_path / "right_summary.json"
        left_details = tmp_path / "left_details.jsonl"
        right_details = tmp_path / "right_details.jsonl"
        left_watermarked = tmp_path / "left_watermarked.jsonl"
        right_watermarked = tmp_path / "right_watermarked.jsonl"
        report_output = tmp_path / "offline_analysis.json"

        _write_json(
            left_summary,
            {
                "dataset": "HumanEval",
                "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
                "summary": {"watermark_rate": 1.0},
            },
        )
        _write_json(
            right_summary,
            {
                "dataset": "HumanEval",
                "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
                "summary": {"watermark_rate": 0.0},
            },
        )
        _write_jsonl(
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
        _write_jsonl(
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
        _write_jsonl(
            left_watermarked,
            [
                {
                    "id": "HumanEval/0",
                    "watermark_params": {"lsh_d": 3, "lsh_gamma": 0.5},
                    "total_blocks": 8,
                    "embedded_blocks": 6,
                    "failed_blocks": 0,
                    "fallback_blocks": 0,
                    "embed_rate": 0.75,
                }
            ],
        )
        _write_jsonl(
            right_watermarked,
            [
                {
                    "id": "HumanEval/0",
                    "watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75},
                    "total_blocks": 8,
                    "embedded_blocks": 5,
                    "failed_blocks": 1,
                    "fallback_blocks": 0,
                    "embed_rate": 0.625,
                }
            ],
        )

        args = argparse.Namespace(
            compare_summary_left=str(left_summary),
            compare_details_left=str(left_details),
            compare_watermarked_left=str(left_watermarked),
            compare_summary_right=str(right_summary),
            compare_details_right=str(right_details),
            compare_watermarked_right=str(right_watermarked),
            compare_output=str(report_output),
        )

        rc = run_offline_analysis(args)

        assert rc == 0
        assert report_output.exists()
        report = json.loads(report_output.read_text(encoding="utf-8"))
        assert set(report) == {
            "compatibility",
            "parameter_diff",
            "detail_delta",
            "embedding_delta",
            "anomalies",
            "regression_classification",
        }

    def test_main_rejects_compare_only_mode_outside_extract_phase(self, monkeypatch, capsys):
        import run as run_module

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

    def test_main_compare_only_extract_bypasses_extract_prerequisites(self, tmp_path, monkeypatch):
        import run as run_module

        left_summary = tmp_path / "left_summary.json"
        right_summary = tmp_path / "right_summary.json"
        left_details = tmp_path / "left_details.jsonl"
        right_details = tmp_path / "right_details.jsonl"
        report_output = tmp_path / "offline_analysis.json"

        _write_json(left_summary, {"dataset": "HumanEval", "summary": {"watermark_rate": 1.0}})
        _write_json(right_summary, {"dataset": "HumanEval", "summary": {"watermark_rate": 0.8}})
        _write_jsonl(
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
        _write_jsonl(
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
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        import run as run_module

        input_file = tmp_path / "input.jsonl"
        input_file.write_text("{}\n", encoding="utf-8")
        called: list[str] = []

        class FakeState:
            @staticmethod
            def is_done(phase: str) -> bool:
                return phase == "extract"

        def fake_run_phase(phase: str, args, state) -> int:
            called.append(phase)
            assert phase == "extract"
            assert args.input_file == str(input_file)
            return 0

        monkeypatch.setattr(run_module, "RunState", FakeState)
        monkeypatch.setattr(run_module, "run_phase", fake_run_phase)
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
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        import run as run_module

        input_file = tmp_path / "input.jsonl"
        input_file.write_text("{}\n", encoding="utf-8")
        config_file = tmp_path / "cfg.json"
        config_file.write_text(
            json.dumps({"extract": {"input_file": str(input_file)}}),
            encoding="utf-8",
        )
        called: list[str] = []

        class FakeState:
            @staticmethod
            def is_done(phase: str) -> bool:
                return phase == "extract"

        def fake_run_phase(phase: str, args, state) -> int:
            called.append(phase)
            assert phase == "extract"
            assert args.config == config_file
            return 0

        monkeypatch.setattr(run_module, "RunState", FakeState)
        monkeypatch.setattr(run_module, "run_phase", fake_run_phase)
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


class TestRunWatermarkConfigNoFallback:
    def _find_keyword_call(self, call_name: str, keyword_name: str) -> bool:
        tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == call_name:
                if any(keyword.arg == keyword_name for keyword in node.keywords):
                    return True
        return False

    def test_run_watermark_no_enable_fallback(self):
        """run.py 构建 WatermarkConfig 不传 enable_fallback（已废弃）。"""
        source = RUN_PY.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_fallback":
                raise AssertionError("run.py 仍传递了已废弃的 enable_fallback 参数")

    def test_run_watermark_has_enable_cascade(self):
        """run.py 构建 WatermarkConfig 传递 enable_cascade。"""
        source = RUN_PY.read_text()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.keyword) and node.arg == "enable_cascade":
                return
        raise AssertionError("run.py 应传递 enable_cascade 参数给 WatermarkConfig")

    def test_run_watermark_pipeline_config_receives_resume(self):
        assert self._find_keyword_call("WatermarkPipelineConfig", "resume")

    def test_run_watermark_pipeline_config_receives_sample_limit(self):
        assert self._find_keyword_call("WatermarkPipelineConfig", "sample_limit")

    def test_run_extract_pipeline_config_receives_resume(self):
        assert self._find_keyword_call("ExtractPipelineConfig", "resume")

    def test_run_extract_marks_summary_file_in_state(self):
        tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
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
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
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
    tree = ast.parse(RUN_PY.read_text(encoding="utf-8"))
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
