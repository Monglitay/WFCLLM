# Fix pass@1 = 0: Round 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the critical evaluation bug causing pass@1 = 0, then re-run the benchmark to get a baseline score.

**Architecture:** The watermarked output stores `prompt` (function signature + docstring) and `generated_code` (function body) separately. The evaluation must concatenate them before executing tests. Additionally, tune generation temperature to reduce placeholder outputs.

**Tech Stack:** Python, pytest, subprocess-based code execution

---

### Task 1: Fix `_evaluate_correctness` to prepend prompt to generated code

**Files:**
- Modify: `wfcllm/evaluation/benchmark.py:251-270`
- Test: `tests/evaluation/test_benchmark.py` (create if not exists)

- [ ] **Step 1: Write the failing test**

Create `tests/evaluation/test_benchmark_correctness.py`:

```python
"""Test that _evaluate_correctness prepends prompt to generated_code."""
import pytest
from unittest.mock import patch, MagicMock
from wfcllm.evaluation.benchmark import BenchmarkConfig, BenchmarkRunner, TestExecutor
from wfcllm.datasets.loaders.local import TestCase


def test_evaluate_correctness_prepends_prompt():
    """The full function (prompt + body) must be passed to the executor."""
    config = BenchmarkConfig(dataset="humaneval", config_path="configs/base_config.json")
    runner = BenchmarkRunner(config)

    records = [{
        "id": "HumanEval/0",
        "prompt": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
        "generated_code": "    return a + b\n",
    }]
    test_cases = {
        "HumanEval/0": TestCase(
            task_id="HumanEval/0",
            entry_point="add",
            test_code="def check(candidate):\n    assert candidate(1, 2) == 3\n",
        ),
    }
    executor = TestExecutor(timeout=5.0)
    results = runner._evaluate_correctness(records, test_cases, executor)
    assert results == [True], "prompt + generated_code should form a valid function"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark_correctness.py -v`
Expected: FAIL — currently only `generated_code` (the body) is passed, so `add` is never defined.

- [ ] **Step 3: Fix `_evaluate_correctness` in benchmark.py**

In `wfcllm/evaluation/benchmark.py`, change line 260 from:

```python
            code = str(record.get("generated_code", ""))
```

to:

```python
            prompt = str(record.get("prompt", ""))
            body = str(record.get("generated_code", ""))
            code = prompt + body if prompt else body
```

- [ ] **Step 4: Run test to verify it passes**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/test_benchmark_correctness.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/evaluation/benchmark.py tests/evaluation/test_benchmark_correctness.py
git commit -m "fix(evaluation): prepend prompt to generated_code in benchmark correctness check"
```

---

### Task 2: Run benchmark on existing watermarked data to get baseline pass@1

**Files:**
- None modified — this is a measurement step

- [ ] **Step 1: Run the benchmark evaluation**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py bench \
  --dataset humaneval \
  --config configs/base_config.json \
  --watermarked-dirs data/watermarked \
  --negative-corpus data/negative_corpus.jsonl \
  --positive-details data/results/humaneval_20260506_154557_details.jsonl \
  --negative-details data/results/negative/negative_corpus_details.jsonl \
  --output-dir data/eval/benchmark_round1
```

- [ ] **Step 2: Record the baseline pass@1 result**

Read the output JSON and note the pass@1 value. This is the baseline after fixing the evaluation bug.

---

### Task 3: Tune generation parameters to reduce placeholder outputs

**Files:**
- Modify: `configs/base_config.json`

- [ ] **Step 1: Adjust temperature and top_p for better code generation**

In `configs/base_config.json`, change the watermark section:

```json
"temperature": 0.8,
"top_p": 0.95,
"max_new_tokens": 768,
"repetition_penalty": 1.1
```

Rationale: temperature 0.5 is too low for a 7B model to generate diverse, correct code. Raising to 0.8 gives the model more freedom to find correct solutions while still being constrained. Increasing max_new_tokens from 512 to 768 gives more room for longer solutions.

- [ ] **Step 2: Commit config change**

```bash
git add configs/base_config.json
git commit -m "feat(config): tune generation params for better code quality"
```

---

### Task 4: Re-generate watermarked code and evaluate

**Files:**
- Output: `data/watermarked/` (new JSONL file)
- Output: `data/eval/benchmark_round1/` (evaluation results)

- [ ] **Step 1: Run watermark generation**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py \
  --config configs/base_config.json \
  --phase watermark
```

- [ ] **Step 2: Run benchmark evaluation on new output**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate.py bench \
  --dataset humaneval \
  --config configs/base_config.json \
  --watermarked-dirs data/watermarked \
  --negative-corpus data/negative_corpus.jsonl \
  --output-dir data/eval/benchmark_round1_tuned
```

- [ ] **Step 3: Compare pass@1 before and after tuning**

Check the benchmark report JSON for the new pass@1 value and compare with the baseline from Task 2.
