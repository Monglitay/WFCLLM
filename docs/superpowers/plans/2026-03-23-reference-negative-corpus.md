# Reference Negative Corpus Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `generate-negative` 增加 `reference | llm` 双来源模式，并默认使用数据集原生参考解生成负样本语料。

**Architecture:** 先在共享 dataset loader 中补统一的 reference-solution 读取接口，再在 `NegativeCorpusGenerator` 中按 `source_mode` 分流；最后把 `run.py` 和配置/测试接上，保证 reference 模式不依赖 LLM。

**Tech Stack:** Python 3.11, pytest, dataclasses, HuggingFace datasets cache, JSONL

---

### Task 1: Add shared reference-solution loader

**Files:**
- Modify: `wfcllm/common/dataset_loader.py`
- Modify: `tests/common/test_dataset_loader.py`

- [ ] **Step 1: Write failing tests for HumanEval reference loading**

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest -q tests/common/test_dataset_loader.py -k reference
```

- [ ] **Step 3: Implement `load_reference_solutions()`**

- [ ] **Step 4: Run tests to verify pass**

### Task 2: Add reference/llm switch to negative corpus generator

**Files:**
- Modify: `wfcllm/extract/negative_corpus.py`
- Modify: `tests/extract/test_negative_corpus.py`

- [ ] **Step 1: Write failing tests for `source_mode=\"reference\"`**

- [ ] **Step 2: Verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest -q tests/extract/test_negative_corpus.py -k reference
```

- [ ] **Step 3: Add `NegativeCorpusConfig.source_mode` and branch generator behavior**

- [ ] **Step 4: Run focused generator tests**

### Task 3: Wire run.py and config behavior

**Files:**
- Modify: `run.py`
- Modify: `tests/test_run.py`
- Modify: `tests/test_run_config.py`
- Modify: `configs/base_config.json`

- [ ] **Step 1: Write failing tests for `run_generate_negative()` in reference mode**

- [ ] **Step 2: Verify failure**

- [ ] **Step 3: Implement CLI/config plumbing**

- [ ] **Step 4: Update base config default to `reference`**

- [ ] **Step 5: Run focused run/config tests**

### Task 4: Regression verification

**Files:**
- Verify only

- [ ] **Step 1: Run full focused suite**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest -q \
  tests/common/test_dataset_loader.py \
  tests/extract/test_negative_corpus.py \
  tests/test_run.py \
  tests/test_run_config.py
```

- [ ] **Step 2: Summarize resulting usage**

- [ ] **Step 3: Optionally dry-run `generate-negative` in reference mode**
