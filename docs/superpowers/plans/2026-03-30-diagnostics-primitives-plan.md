# Diagnostics Primitives Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the diagnostics data model, hashing helper, and per-sample summary helper so downstream artifacts can persist compact route-one evidence without touching runtime blocks yet.

**Architecture:** Introduce `wfcllm/watermark/diagnostics.py` to host `FailureReason`, `BlockLifecycleRecord`, and the summary helpers, keeping the API JSON-safe (dataclasses + `to_dict`). Drive it with a new `tests/watermark/test_diagnostics.py` module that validates failure reason counting, hash helper behavior, serialization, and summary aggregations.
**Tech Stack:** Python dataclasses/enum, `hashlib` for SHA-256, `typing` for structured responses, `pytest` for unit testing.

---

### Task 1: Diagnostics primitives and summary helpers

**Files:**
- Create: `wfcllm/watermark/diagnostics.py`
- Create: `tests/watermark/test_diagnostics.py`

- [ ] **Step 1: Write the failing test**

```python
def test_block_lifecycle_summary_counts_examples():
    records = [BlockLifecycleRecord(...)]
    summary = summarize_sample_diagnostics(records)
    assert summary["retry_summary"]["blocks_with_retry"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/watermark/test_diagnostics.py -k summary` at the root 
Expected: FAIL with `NameError` or assertion because helpers and dataclasses don’t exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement `FailureReason` enum, `BlockLifecycleRecord` dataclass (with `to_dict`), SHA-256 helper `hash_block_text`, and `summarize_sample_diagnostics` that aggregates all required counters per spec (retry counts, cascade triggers, failure reason totals, rescued/unrescued counts, JSON-safe output with `diagnostics_version`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/watermark/test_diagnostics.py -k summary` -- expect PASS.

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/diagnostics.py tests/watermark/test_diagnostics.py
git commit -m "feat: add diagnostics primitives"
```
