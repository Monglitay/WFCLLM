# Extraction Benchmark Metrics Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pass@1, Pass@10, and AUROC metrics to the extraction evaluation via an offline benchmark script, starting with HumanEval and MBPP datasets.

**Architecture:** New `wfcllm/evaluation/benchmark.py` module provides TestExecutor (subprocess-based test execution) and BenchmarkRunner (orchestration). CLI entry via new `bench` subcommand in `scripts/evaluate.py`. Dataset loader extended with `load_test_cases`.

**Tech Stack:** Python subprocess for code execution, existing `compute_pass_at_k` and `compute_roc_auc` from `wfcllm/evaluation/code_execution.py`, HuggingFace datasets for loading test cases.

## File Structure

| File | Responsibility |
|------|---------------|
| `wfcllm/datasets/loaders/local.py` | Extend: add `TestCase` dataclass + `load_test_cases()` |
| `wfcllm/evaluation/benchmark.py` | New: `TestExecutor`, `BenchmarkConfig`, `BenchmarkRunner` |
| `scripts/evaluate.py` | Extend: add `bench` subcommand |
| `tests/evaluation/__init__.py` | New: empty package init |
| `tests/evaluation/test_benchmark.py` | New: all unit tests |

---

### Task 1: Extend dataset loader with `load_test_cases`

**Files:**
- Modify: `wfcllm/datasets/loaders/local.py`
- Test: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Create test file with failing test for load_test_cases**

```python
# tests/evaluation/test_benchmark.py
"""Tests for the benchmark evaluation module."""
from __future__ import annotations

import pytest

from wfcllm.datasets.loaders.local import load_test_cases


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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_load_test_cases_humaneval -v`
Expected: FAIL with `ImportError` (load_test_cases not yet defined)

- [ ] **Step 3: Implement load_test_cases in local.py**

Add to `wfcllm/datasets/loaders/local.py`:

```python
from dataclasses import dataclass


@dataclass
class TestCase:
    task_id: str
    entry_point: str | None
    test_code: str


def load_test_cases(dataset: str, dataset_path: str) -> dict[str, TestCase]:
    """Load test cases for HumanEval or MBPP from local dataset cache."""
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            f"dataset must be one of {SUPPORTED_DATASETS}, got '{dataset}'"
        )

    path = str(Path(dataset_path) / dataset)

    if dataset == "humaneval":
        ds = load_dataset(
            "openai/openai_humaneval",
            cache_dir=path,
            download_mode="reuse_cache_if_exists",
        )
        cases: dict[str, TestCase] = {}
        for split in ds:
            for item in ds[split]:
                cases[item["task_id"]] = TestCase(
                    task_id=item["task_id"],
                    entry_point=item["entry_point"],
                    test_code=item["test"],
                )
        return cases

    # mbpp
    ds = load_dataset(
        "google-research-datasets/mbpp",
        "full",
        cache_dir=path,
        download_mode="reuse_cache_if_exists",
    )
    cases = {}
    for split in ds:
        for item in ds[split]:
            task_id = f"mbpp/{item['task_id']}"
            cases[task_id] = TestCase(
                task_id=task_id,
                entry_point=None,
                test_code="\n".join(item["test_list"]),
            )
    return cases
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/datasets/loaders/local.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add load_test_cases for HumanEval/MBPP test execution"
```

---

### Task 2: Implement TestExecutor

**Files:**
- Create: `wfcllm/evaluation/benchmark.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing tests for TestExecutor**

Append to `tests/evaluation/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_executor_passing_code -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement TestExecutor**

Create `wfcllm/evaluation/benchmark.py`:

```python
"""Offline benchmark evaluation: Pass@1, Pass@10, AUROC for code watermarking."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestExecutor:
    """Execute code + test cases via subprocess, return pass/fail."""

    timeout: float = 5.0

    def run_test(self, code: str, test_code: str) -> bool:
        """Run generated code against test assertions. Returns True if all pass."""
        full_source = f"{code}\n{test_code}\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_source)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def execute_humaneval(
        self, generated_code: str, test_code: str, entry_point: str
    ) -> bool:
        """HumanEval-style execution: code + test + check(entry_point)."""
        full_test = f"{test_code}\ncheck({entry_point})\n"
        return self.run_test(generated_code, full_test)

    def execute_mbpp(self, generated_code: str, test_code: str) -> bool:
        """MBPP-style execution: code + assert statements."""
        return self.run_test(generated_code, test_code)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_executor_passing_code tests/evaluation/test_benchmark.py::test_executor_failing_code tests/evaluation/test_benchmark.py::test_executor_timeout tests/evaluation/test_benchmark.py::test_executor_syntax_error -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add TestExecutor for subprocess-based code test execution"
```

---

### Task 3: Implement BenchmarkRunner core (Pass@k computation)

**Files:**
- Modify: `wfcllm/evaluation/benchmark.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing test for BenchmarkRunner pass@k**

Append to `tests/evaluation/test_benchmark.py`:

```python
from unittest.mock import patch
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_pass_at_k -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Implement BenchmarkConfig and BenchmarkRunner._compute_pass_at_k**

Add to `wfcllm/evaluation/benchmark.py`:

```python
import json
from typing import Any

from wfcllm.evaluation.code_execution import compute_pass_at_k


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark evaluation."""

    dataset: str
    config_path: str
    dataset_path: str = "data/datasets"
    num_candidates: int = 10
    timeout_per_test: float = 5.0
    watermarked_dirs: list[str] | None = None
    positive_details: str | None = None
    negative_details: str | None = None
    auto_generate: bool = False
    negative_corpus: str | None = None
    output_dir: str = "data/eval/benchmark"


class BenchmarkRunner:
    """Orchestrate benchmark evaluation: Pass@k + AUROC."""

    def __init__(self, config: BenchmarkConfig):
        self._config = config

    def _compute_pass_at_k(
        self,
        records: list[dict[str, Any]],
        correctness: list[bool],
    ) -> dict[str, float]:
        """Compute pass@1 and pass@10 from records + correctness annotations."""
        annotated = []
        for record, is_correct in zip(records, correctness):
            annotated.append({**record, "is_correct": is_correct})

        result: dict[str, float] = {}
        result["pass_at_1"] = compute_pass_at_k(annotated, k=1)
        k = min(10, self._config.num_candidates)
        result[f"pass_at_{k}"] = compute_pass_at_k(annotated, k=k)
        if k == 10:
            result["pass_at_10"] = result[f"pass_at_{k}"]
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_pass_at_k -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add BenchmarkConfig and pass@k computation to BenchmarkRunner"
```

---

### Task 4: Implement BenchmarkRunner AUROC computation

**Files:**
- Modify: `wfcllm/evaluation/benchmark.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing test for AUROC computation**

Append to `tests/evaluation/test_benchmark.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_compute_auroc -v`
Expected: FAIL with `AttributeError`

- [ ] **Step 3: Implement _compute_detection_metrics**

Add to `BenchmarkRunner` in `wfcllm/evaluation/benchmark.py`:

```python
    def _compute_detection_metrics(
        self,
        positive_details: list[dict[str, Any]],
        negative_details: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Compute AUROC and TPR@1%FPR for each available score field."""
        from wfcllm.evaluation.code_execution import compute_roc_auc, compute_tpr_at_fpr

        score_fields = {
            "semantic": "z_score",
            "lexical": "lexical_z_score",
            "joint": "joint_score",
        }
        result: dict[str, dict[str, float]] = {}
        for channel, field in score_fields.items():
            pos_scores = [r[field] for r in positive_details if field in r]
            neg_scores = [r[field] for r in negative_details if field in r]
            if not pos_scores or not neg_scores:
                continue
            result[channel] = {
                "auroc": compute_roc_auc(pos_scores, neg_scores),
                "tpr_at_1pct_fpr": compute_tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01),
            }
        return result
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_compute_auroc -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add AUROC/TPR@FPR detection metrics to BenchmarkRunner"
```

---

### Task 5: Implement BenchmarkRunner.run() end-to-end orchestration

**Files:**
- Modify: `wfcllm/evaluation/benchmark.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing test for BenchmarkRunner.run() with pre-existing results**

Append to `tests/evaluation/test_benchmark.py`:

```python
import json
import tempfile
from pathlib import Path


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
```

Also add this import at the top of the test file:

```python
from unittest.mock import patch
from wfcllm.datasets.loaders.local import TestCase
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_run_with_existing_results -v`
Expected: FAIL with `AttributeError` (run method not defined)

- [ ] **Step 3: Implement BenchmarkRunner.run()**

Add to `BenchmarkRunner` in `wfcllm/evaluation/benchmark.py`:

```python
    def run(self) -> dict[str, Any]:
        """Execute the full benchmark evaluation and return the report."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Load test cases
        test_cases = self._load_test_cases()

        # Load or generate watermarked candidates
        watermarked_records = self._load_watermarked_records()

        # Execute tests for pass@k
        executor = TestExecutor(timeout=self._config.timeout_per_test)
        correctness = self._evaluate_correctness(watermarked_records, test_cases, executor)
        pass_metrics = self._compute_pass_at_k(watermarked_records, correctness)

        # Save execution results
        exec_results_path = output_dir / f"{self._config.dataset}_exec_results.jsonl"
        self._save_execution_results(exec_results_path, watermarked_records, correctness)

        # Load detection details for AUROC
        positive_details = self._load_details(self._config.positive_details)
        negative_details = self._load_details(self._config.negative_details)
        detection_metrics = self._compute_detection_metrics(positive_details, negative_details)

        # Build report
        report: dict[str, Any] = {
            "dataset": self._config.dataset,
            "num_candidates": self._config.num_candidates,
            "num_tasks": len(test_cases),
            "metrics": {
                **pass_metrics,
                "detection": detection_metrics,
            },
            "details": {
                "execution_results_path": str(exec_results_path),
                "positive_details_path": self._config.positive_details,
                "negative_details_path": self._config.negative_details,
            },
        }

        # Write report
        report_path = output_dir / f"{self._config.dataset}_benchmark_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    def _load_test_cases(self) -> dict[str, Any]:
        from wfcllm.datasets.loaders.local import load_test_cases
        return load_test_cases(self._config.dataset, self._config.dataset_path)

    def _load_watermarked_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self._config.watermarked_dirs:
            for dir_path in self._config.watermarked_dirs:
                for jsonl_file in Path(dir_path).glob("*.jsonl"):
                    records.extend(self._read_jsonl(jsonl_file))
        return records

    def _evaluate_correctness(
        self,
        records: list[dict[str, Any]],
        test_cases: dict[str, Any],
        executor: TestExecutor,
    ) -> list[bool]:
        results: list[bool] = []
        for record in records:
            task_id = str(record.get("id", ""))
            code = str(record.get("generated_code", ""))
            tc = test_cases.get(task_id)
            if tc is None:
                results.append(False)
                continue
            if tc.entry_point is not None:
                passed = executor.execute_humaneval(code, tc.test_code, tc.entry_point)
            else:
                passed = executor.execute_mbpp(code, tc.test_code)
            results.append(passed)
        return results

    def _save_execution_results(
        self,
        path: Path,
        records: list[dict[str, Any]],
        correctness: list[bool],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for record, is_correct in zip(records, correctness):
            lines.append(json.dumps(
                {"id": record.get("id"), "is_correct": is_correct},
                ensure_ascii=False,
            ))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_details(self, path: str | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        return self._read_jsonl(Path(path))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_run_with_existing_results -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): implement BenchmarkRunner.run() end-to-end orchestration"
```

---

### Task 6: Add `bench` subcommand to scripts/evaluate.py

**Files:**
- Modify: `scripts/evaluate.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing test for CLI argument parsing**

Append to `tests/evaluation/test_benchmark.py`:

```python
import sys
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_bench_cli_parser_existing_results -v`
Expected: FAIL (bench subcommand not defined)

- [ ] **Step 3: Add bench subcommand to scripts/evaluate.py**

Add to `_build_parser()` in `scripts/evaluate.py`, after the `dual_parser` block:

```python
    bench_parser = subparsers.add_parser(
        "bench",
        help="compute Pass@1, Pass@10, AUROC from watermarked candidates + negative corpus",
    )
    bench_parser.add_argument(
        "--dataset", required=True, choices=["humaneval", "mbpp"],
    )
    bench_parser.add_argument("--config", default="configs/base_config.json")
    bench_parser.add_argument("--dataset-path", default="data/datasets")
    bench_parser.add_argument(
        "--watermarked-dirs", nargs="+", default=None,
        help="directories containing watermarked candidate JSONL files",
    )
    bench_parser.add_argument("--positive-details", default=None)
    bench_parser.add_argument("--negative-details", default=None)
    bench_parser.add_argument("--negative-corpus", default=None)
    bench_parser.add_argument("--auto-generate", action="store_true")
    bench_parser.add_argument("--num-candidates", type=int, default=10)
    bench_parser.add_argument("--timeout", type=float, default=5.0)
    bench_parser.add_argument("--output-dir", default="data/eval/benchmark")
    bench_parser.set_defaults(func=_cmd_bench)
```

Add the `_cmd_bench` function:

```python
def _cmd_bench(args: argparse.Namespace) -> int:
    from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner

    config = BenchmarkConfig(
        dataset=args.dataset,
        config_path=args.config,
        dataset_path=args.dataset_path,
        num_candidates=args.num_candidates,
        timeout_per_test=args.timeout,
        watermarked_dirs=args.watermarked_dirs,
        positive_details=args.positive_details,
        negative_details=args.negative_details,
        auto_generate=args.auto_generate,
        negative_corpus=args.negative_corpus,
        output_dir=args.output_dir,
    )
    runner = BenchmarkRunner(config)
    report = runner.run()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_bench_cli_parser_existing_results tests/evaluation/test_benchmark.py::test_bench_cli_parser_auto_generate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/evaluate.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add bench subcommand to scripts/evaluate.py for Pass@k/AUROC"
```

---

### Task 7: Add auto-generate mode to BenchmarkRunner

**Files:**
- Modify: `wfcllm/evaluation/benchmark.py`
- Modify: `tests/evaluation/test_benchmark.py`

- [ ] **Step 1: Write failing test for auto-generate mode**

Append to `tests/evaluation/test_benchmark.py`:

```python
from wfcllm.evaluation.benchmark import BenchmarkRunner


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

    def mock_run_command(cmd: list[str], env=None):
        commands_called.append(cmd)
        from wfcllm.evaluation.dual_channel import CommandRunResult
        return CommandRunResult(exit_code=0, stdout="", stderr="")

    with patch.object(runner, "_run_command", mock_run_command):
        with patch.object(runner, "_load_test_cases", return_value={}):
            with patch.object(runner, "_load_watermarked_records", return_value=[]):
                with patch.object(runner, "_load_details", return_value=[]):
                    runner.run()

    # Should have called watermark phase num_candidates times + extract phases
    watermark_calls = [c for c in commands_called if "--phase" in c and "watermark" in c]
    extract_calls = [c for c in commands_called if "--phase" in c and "extract" in c]
    assert len(watermark_calls) == 2
    assert len(extract_calls) >= 1  # at least negative extract
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_auto_generate_invokes_phases -v`
Expected: FAIL

- [ ] **Step 3: Implement auto-generate logic in BenchmarkRunner**

Add to `BenchmarkRunner` in `wfcllm/evaluation/benchmark.py`:

```python
    def _auto_generate(self) -> tuple[list[dict[str, Any]], str, str]:
        """Run watermark + extract phases automatically, return (records, pos_details_path, neg_details_path)."""
        import os
        import time

        output_dir = Path(self._config.output_dir)
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")

        all_records: list[dict[str, Any]] = []
        watermarked_paths: list[Path] = []

        for i in range(self._config.num_candidates):
            candidate_dir = output_dir / f"watermarked_candidate_{i + 1}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "run.py",
                "--config", self._config.config_path,
                "--phase", "watermark",
                "--dataset", self._config.dataset,
                "--output-dir", str(candidate_dir),
            ]
            self._run_command(cmd, env)
            jsonl_files = list(candidate_dir.glob("*.jsonl"))
            if jsonl_files:
                path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                watermarked_paths.append(path)
                records = self._read_jsonl(path)
                for r in records:
                    r["candidate_index"] = i
                all_records.extend(records)

        # Run extract on positive (first candidate as representative)
        pos_extract_dir = output_dir / "positive_extract"
        if watermarked_paths:
            cmd = [
                "python", "run.py",
                "--config", self._config.config_path,
                "--phase", "extract",
                "--input-file", str(watermarked_paths[0]),
                "--extract-output-dir", str(pos_extract_dir),
            ]
            self._run_command(cmd, env)
        pos_details_path = str(pos_extract_dir / f"{watermarked_paths[0].stem}_details.jsonl") if watermarked_paths else ""

        # Run extract on negative corpus
        neg_extract_dir = output_dir / "negative_extract"
        neg_corpus = self._config.negative_corpus or "data/negative_corpus.jsonl"
        neg_corpus_path = Path(neg_corpus)
        cmd = [
            "python", "run.py",
            "--config", self._config.config_path,
            "--phase", "extract",
            "--input-file", str(neg_corpus_path),
            "--extract-output-dir", str(neg_extract_dir),
        ]
        self._run_command(cmd, env)
        neg_details_path = str(neg_extract_dir / f"{neg_corpus_path.stem}_details.jsonl")

        return all_records, pos_details_path, neg_details_path

    def _run_command(self, cmd: list[str], env: dict[str, str] | None = None) -> Any:
        """Run a subprocess command, raise on failure."""
        import subprocess as _sp
        result = _sp.run(cmd, capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "command failed")
        return result
```

Update `BenchmarkRunner.run()` to use auto-generate when configured:

```python
    def run(self) -> dict[str, Any]:
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        test_cases = self._load_test_cases()

        if self._config.auto_generate:
            watermarked_records, pos_path, neg_path = self._auto_generate()
            if not self._config.positive_details:
                self._config.positive_details = pos_path
            if not self._config.negative_details:
                self._config.negative_details = neg_path
        else:
            watermarked_records = self._load_watermarked_records()

        # ... rest unchanged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark.py::test_benchmark_runner_auto_generate_invokes_phases -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark.py
git commit -m "feat(evaluation): add auto-generate mode to BenchmarkRunner"
```

---

### Task 8: Create tests/__init__.py and run full test suite

**Files:**
- Create: `tests/evaluation/__init__.py`

- [ ] **Step 1: Create package init**

```python
# tests/evaluation/__init__.py
```

- [ ] **Step 2: Run the full test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/ -v`
Expected: All tests PASS

- [ ] **Step 3: Run existing tests to verify no regressions**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --ignore=tests/evaluation/ -x -q 2>&1 | tail -20`
Expected: No new failures

- [ ] **Step 4: Commit**

```bash
git add tests/evaluation/__init__.py
git commit -m "chore: add tests/evaluation package init"
```

