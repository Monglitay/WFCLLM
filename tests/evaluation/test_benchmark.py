"""Tests for the benchmark evaluation module."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from wfcllm.datasets.loaders.local import TestCase, load_test_cases


def test_load_test_cases_humaneval() -> None:
    cases = load_test_cases("humaneval", "data/datasets")
    assert len(cases) == 164
    case = cases["HumanEval/0"]
    assert case.task_id == "HumanEval/0"
    assert case.entry_point is not None
    assert "check" in case.test_code or "assert" in case.test_code


def test_load_test_cases_mbpp() -> None:
    cases = load_test_cases("mbpp", "data/datasets")
    assert len(cases) > 0
    first_key = next(iter(cases))
    case = cases[first_key]
    assert case.task_id.startswith("mbpp/")
    assert case.entry_point is None
    assert "assert" in case.test_code


def test_load_test_cases_invalid_dataset() -> None:
    with pytest.raises(ValueError, match="must be one of"):
        load_test_cases("invalid", "data/datasets")


from wfcllm.evaluation.benchmark import TestExecutor


def test_executor_passing_code() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def add(a, b):\n    return a + b\n"
    test_code = "assert add(1, 2) == 3\nassert add(0, 0) == 0\n"
    assert executor.run_test(code, test_code) is True


def test_executor_failing_code() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def add(a, b):\n    return a - b\n"
    test_code = "assert add(1, 2) == 3\n"
    assert executor.run_test(code, test_code) is False


def test_executor_timeout() -> None:
    executor = TestExecutor(timeout=1.0)
    code = "import time\ndef slow():\n    time.sleep(10)\n"
    test_code = "slow()\n"
    assert executor.run_test(code, test_code) is False


def test_executor_syntax_error() -> None:
    executor = TestExecutor(timeout=5.0)
    code = "def broken(\n"
    test_code = "assert True\n"
    assert executor.run_test(code, test_code) is False


from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner


def test_benchmark_runner_pass_at_k() -> None:
    """BenchmarkRunner computes pass@k from execution results."""
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        num_candidates=2,
        output_dir="/tmp/test_benchmark",
    )
    # Mock: 2 tasks, 2 candidates each. Task A: both pass. Task B: 1 pass, 1 fail.
    mock_records = [
        {"id": "HumanEval/0", "generated_code": "def f(): return 1", "candidate_index": 0},
        {"id": "HumanEval/0", "generated_code": "def f(): return 1", "candidate_index": 1},
        {"id": "HumanEval/1", "generated_code": "def g(): return 2", "candidate_index": 0},
        {"id": "HumanEval/1", "generated_code": "def g(): return 2", "candidate_index": 1},
    ]
    correctness = [True, True, True, False]

    runner = BenchmarkRunner(config)
    result = runner._compute_pass_at_k(mock_records, correctness)
    assert result["pass_at_1"] == pytest.approx(0.75)
    assert result["pass_at_2"] == pytest.approx(1.0)


def test_benchmark_runner_compute_auroc() -> None:
    """BenchmarkRunner computes AUROC from positive/negative details."""
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        output_dir="/tmp/test_benchmark",
    )
    positive_details = [
        {"id": "HumanEval/0", "z_score": 3.0, "lexical_z_score": 2.5, "joint_score": 4.0},
        {"id": "HumanEval/1", "z_score": 2.5, "lexical_z_score": 2.0, "joint_score": 3.5},
    ]
    negative_details = [
        {"id": "HumanEval/0", "z_score": 0.5, "lexical_z_score": 0.3, "joint_score": 0.6},
        {"id": "HumanEval/1", "z_score": 0.2, "lexical_z_score": 0.1, "joint_score": 0.3},
    ]
    runner = BenchmarkRunner(config)
    result = runner._compute_detection_metrics(positive_details, negative_details)

    assert result["semantic"]["auroc"] == pytest.approx(1.0)
    assert result["lexical"]["auroc"] == pytest.approx(1.0)
    assert result["joint"]["auroc"] == pytest.approx(1.0)
    assert 0.0 <= result["semantic"]["tpr_at_1pct_fpr"] <= 1.0


def test_benchmark_runner_run_with_existing_results(tmp_path: Path) -> None:
    """BenchmarkRunner.run() produces a report from pre-existing artifacts."""
    # Create mock watermarked JSONL (2 candidates for 2 tasks)
    watermarked_dir = tmp_path / "watermarked"
    watermarked_dir.mkdir()
    records = [
        {"id": "task_0", "generated_code": "def f():\n    return 1\n"},
        {"id": "task_0", "generated_code": "def f():\n    return 1\n"},
        {"id": "task_1", "generated_code": "def g():\n    return 2\n"},
        {"id": "task_1", "generated_code": "def g():\n    return 2\n"},
    ]
    wm_file = watermarked_dir / "candidates.jsonl"
    wm_file.write_text(
        "\n".join(json.dumps(r) for r in records), encoding="utf-8"
    )

    # Create mock positive details
    pos_details = tmp_path / "positive_details.jsonl"
    pos_records = [
        {"id": "task_0", "z_score": 2.5, "lexical_z_score": 1.8, "joint_score": 3.0},
        {"id": "task_1", "z_score": 3.0, "lexical_z_score": 2.0, "joint_score": 3.5},
    ]
    pos_details.write_text(
        "\n".join(json.dumps(r) for r in pos_records), encoding="utf-8"
    )

    # Create mock negative details
    neg_details = tmp_path / "negative_details.jsonl"
    neg_records = [
        {"id": "task_0", "z_score": 0.3, "lexical_z_score": 0.1, "joint_score": 0.4},
        {"id": "task_1", "z_score": 0.5, "lexical_z_score": 0.2, "joint_score": 0.5},
    ]
    neg_details.write_text(
        "\n".join(json.dumps(r) for r in neg_records), encoding="utf-8"
    )

    # Create mock test cases
    mock_test_cases = {
        "task_0": TestCase(task_id="task_0", entry_point=None, test_code="assert f() == 1"),
        "task_1": TestCase(task_id="task_1", entry_point=None, test_code="assert g() == 2"),
    }

    output_dir = tmp_path / "output"
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        num_candidates=2,
        watermarked_dirs=[str(watermarked_dir)],
        positive_details=str(pos_details),
        negative_details=str(neg_details),
        output_dir=str(output_dir),
    )
    runner = BenchmarkRunner(config)
    with patch.object(runner, "_load_test_cases", return_value=mock_test_cases):
        report = runner.run()

    assert "pass_at_1" in report["metrics"]
    assert "detection" in report["metrics"]
    assert "semantic" in report["metrics"]["detection"]
    assert report["metrics"]["detection"]["semantic"]["auroc"] == pytest.approx(1.0)


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.evaluate import _build_parser


def test_bench_cli_parser_existing_results() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "bench",
        "--dataset", "humaneval",
        "--watermarked-dirs", "dir1", "dir2",
        "--negative-details", "neg.jsonl",
        "--positive-details", "pos.jsonl",
        "--output-dir", "/tmp/out",
    ])
    assert args.subcommand == "bench"
    assert args.dataset == "humaneval"
    assert args.watermarked_dirs == ["dir1", "dir2"]
    assert args.negative_details == "neg.jsonl"
    assert args.positive_details == "pos.jsonl"


def test_bench_cli_parser_auto_generate() -> None:
    parser = _build_parser()
    args = parser.parse_args([
        "bench",
        "--dataset", "mbpp",
        "--config", "configs/base_config.json",
        "--auto-generate",
        "--num-candidates", "10",
    ])
    assert args.subcommand == "bench"
    assert args.dataset == "mbpp"
    assert args.auto_generate is True
    assert args.num_candidates == 10


def test_benchmark_runner_auto_generate_invokes_phases(tmp_path: Path) -> None:
    """In auto-generate mode, BenchmarkRunner calls watermark + extract phases."""
    output_dir = tmp_path / "output"
    config = BenchmarkConfig(
        dataset="humaneval",
        config_path="configs/base_config.json",
        dataset_path="data/datasets",
        num_candidates=2,
        auto_generate=True,
        negative_corpus="data/negative_corpus.jsonl",
        output_dir=str(output_dir),
    )
    runner = BenchmarkRunner(config)

    commands_called: list[list[str]] = []

    def mock_run_command(cmd, env=None):
        commands_called.append(cmd)
        # Create a fake JSONL file when watermark phase is called
        if "--phase" in cmd and "watermark" in cmd:
            out_idx = cmd.index("--output-dir") + 1
            out_dir = Path(cmd[out_idx])
            out_dir.mkdir(parents=True, exist_ok=True)
            fake_jsonl = out_dir / "fake_watermarked.jsonl"
            fake_jsonl.write_text(
                '{"id": "HumanEval/0", "generated_code": "def f(): return 1"}\n',
                encoding="utf-8",
            )
        return None

    with patch.object(runner, "_run_command", mock_run_command):
        with patch.object(runner, "_load_test_cases", return_value={}):
            with patch.object(runner, "_evaluate_correctness", return_value=[]):
                with patch.object(runner, "_load_details", return_value=[]):
                    runner.run()

    watermark_calls = [c for c in commands_called if "--phase" in c and "watermark" in c]
    extract_calls = [c for c in commands_called if "--phase" in c and "extract" in c]
    assert len(watermark_calls) == 2
    assert len(extract_calls) >= 1
