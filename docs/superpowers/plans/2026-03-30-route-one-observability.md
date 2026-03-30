# Route-One Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a route-one observability layer that explains how each simple block moves through initial verify, retry, cascade, and final embedding outcome without changing the current detector threshold or watermark decision rule.

**Architecture:** Introduce a dedicated watermark diagnostics module that owns normalized failure reasons, block-lifecycle records, and per-sample summary aggregation. Thread structured verification details out of `verifier.py`, `retry_loop.py`, `generator.py`, and `cascade.py`, then persist compact summaries in the existing watermarked artifact and full block ledgers in a new diagnostics JSONL file. Finally, extend offline analysis to read the new optional fields and surface route-one anomaly flags without breaking older artifacts.

**Tech Stack:** Python 3.13, dataclasses, hashlib, JSON/JSONL, pytest, existing watermark pipeline and offline-analysis helpers

**Spec source of truth:** `/root/autodl-tmp/WFCLLM/docs/superpowers/specs/2026-03-30-route-one-observability-design.md`

**Working directory assumption:** Run commands from `/root/autodl-tmp/WFCLLM`

---

## File Structure

### New files

- `wfcllm/watermark/diagnostics.py`
  - Own the route-one diagnostics data model, failure-reason normalization, block text hashing, per-sample summary aggregation, and JSON-serializable helpers.
- `tests/watermark/test_diagnostics.py`
  - Cover failure reason normalization, block-ledger serialization, and per-sample summary rollups.

### Existing files to modify

- `wfcllm/watermark/verifier.py`
  - Surface structured verification details needed to explain `signature` vs `margin` failures.
- `wfcllm/watermark/retry_loop.py`
  - Record attempt-level route-one diagnostics with normalized failure reasons and no-block outcomes.
- `wfcllm/watermark/cascade.py`
  - Return structured cascade metadata instead of leaving rollback information implicit in stack state.
- `wfcllm/watermark/generator.py`
  - Build logical block lifecycles, attach retry/cascade evidence, and expose per-sample diagnostic summaries alongside existing stats.
- `wfcllm/watermark/pipeline.py`
  - Persist compact summary fields to the existing watermarked JSONL and write a sibling block-ledger JSONL under `data/diagnostics/`.
- `wfcllm/extract/offline_analysis.py`
  - Load optional route-one summaries and optional ledger artifacts, then compute new anomaly flags and comparison outputs.

### Existing tests to modify

- `tests/watermark/test_retry_loop.py`
- `tests/watermark/test_cascade.py`
- `tests/watermark/test_generator.py`
- `tests/watermark/test_pipeline.py`
- `tests/extract/test_offline_analysis.py`

---

### Task 1: Add the diagnostics data model and summary helpers

**Files:**
- Create: `wfcllm/watermark/diagnostics.py`
- Create: `tests/watermark/test_diagnostics.py`

- [ ] **Step 1: Write the failing diagnostics unit tests**

```python
from wfcllm.watermark.diagnostics import (
    BlockLifecycleRecord,
    FailureReason,
    summarize_sample_diagnostics,
)


def test_summarize_sample_diagnostics_counts_retry_and_cascade_outcomes():
    record = BlockLifecycleRecord(
        sample_id="HumanEval/38",
        block_ordinal=0,
        node_type="expression_statement",
        parent_node_type="module",
        block_text_hash="abc",
        initial_verify={"passed": False, "failure_reason": FailureReason.SIGNATURE_MISS.value},
        retry_attempts=[{"attempt_index": 1, "failure_reason": FailureReason.NO_BLOCK_GENERATED.value}],
        cascade_events=[{"triggered": True, "compound_node_type": "if_statement"}],
        final_outcome={"embedded": False, "rescued_by_retry": False, "rescued_by_cascade": False},
    )

    summary = summarize_sample_diagnostics([record])

    assert summary["retry_summary"]["blocks_with_retry"] == 1
    assert summary["retry_summary"]["attempts_no_block"] == 1
    assert summary["cascade_summary"]["cascade_triggers"] == 1
    assert summary["failure_reason_counts"]["signature_miss"] == 1
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run:

```bash
python -m pytest tests/watermark/test_diagnostics.py -v
```

Expected: import failure for `wfcllm.watermark.diagnostics`.

- [ ] **Step 3: Implement the diagnostics primitives**

```python
from dataclasses import asdict, dataclass, field
from enum import Enum
import hashlib


class FailureReason(str, Enum):
    SIGNATURE_MISS = "signature_miss"
    MARGIN_MISS = "margin_miss"
    SIGNATURE_AND_MARGIN_MISS = "signature_and_margin_miss"
    NO_BLOCK_GENERATED = "no_block_generated"
    CASCADE_REPLACED = "cascade_replaced"
    UNKNOWN = "unknown"


def hash_block_text(block_text: str) -> str:
    return hashlib.sha256(block_text.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Expand the tests to cover JSON-safe serialization and rescued block rollups**

```python
def test_block_lifecycle_record_to_dict_is_json_serializable():
    record = BlockLifecycleRecord(
        sample_id="HumanEval/0",
        block_ordinal=1,
        node_type="return_statement",
        parent_node_type="module",
        block_text_hash="deadbeef",
    )

    payload = record.to_dict()

    assert payload["sample_id"] == "HumanEval/0"
    assert payload["block_ordinal"] == 1
```

- [ ] **Step 5: Run the focused diagnostics tests until they pass**

Run:

```bash
python -m pytest tests/watermark/test_diagnostics.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the diagnostics primitives**

```bash
git add wfcllm/watermark/diagnostics.py tests/watermark/test_diagnostics.py
git commit -m "feat: add route-one diagnostics data model"
```

### Task 2: Enrich verifier and retry-loop attempt diagnostics

**Files:**
- Modify: `wfcllm/watermark/verifier.py`
- Modify: `wfcllm/watermark/retry_loop.py`
- Modify: `tests/watermark/test_retry_loop.py`

- [ ] **Step 1: Add failing retry-loop tests for classified failure reasons**

```python
from wfcllm.watermark.retry_loop import RetryLoop
from wfcllm.watermark.verifier import VerifyResult


def test_retry_records_signature_miss(mock_ctx, config, mock_verifier, mock_keying, mock_entropy):
    mock_verifier.verify.return_value = VerifyResult(
        passed=False,
        min_margin=0.5,
        lsh_signature=(1, 0, 1, 0),
        in_valid_set=False,
    )

    result = RetryLoop(...).run(checkpoint, original_event)

    assert result.diagnostics.per_attempt[0].failure_reason == "signature_miss"


def test_retry_records_no_block_generated_when_budget_ends(mock_ctx, config, mock_verifier, mock_keying, mock_entropy):
    mock_ctx.forward_and_sample.return_value = mock_ctx.eos_id

    result = RetryLoop(...).run(checkpoint, original_event)

    assert result.diagnostics.per_attempt[0].no_block is True
    assert result.diagnostics.per_attempt[0].failure_reason == "no_block_generated"
```

- [ ] **Step 2: Run the retry-loop tests to verify failure**

Run:

```bash
python -m pytest tests/watermark/test_retry_loop.py -k "failure_reason or no_block_generated" -v
```

Expected: `VerifyResult` missing fields and/or diagnostics missing `failure_reason`.

- [ ] **Step 3: Extend `VerifyResult` and retry attempt recording with structured failure metadata**

```python
@dataclass
class VerifyResult:
    passed: bool
    min_margin: float
    lsh_signature: tuple[int, ...] = ()
    in_valid_set: bool = False
```

```python
info = AttemptInfo(
    sig=result.lsh_signature or None,
    min_margin=result.min_margin,
    passed=result.passed,
    text=event.block_text[:80],
    failure_reason=failure_reason.value,
    in_valid_set=result.in_valid_set,
    block_text_hash=hash_block_text(event.block_text),
)
```

- [ ] **Step 4: Add a focused test for `margin_miss` vs `signature_and_margin_miss`**

```python
def test_retry_distinguishes_margin_only_failure(...):
    mock_verifier.verify.return_value = VerifyResult(
        passed=False,
        min_margin=0.0001,
        lsh_signature=(1, 1, 1, 1),
        in_valid_set=True,
    )

    result = RetryLoop(...).run(checkpoint, original_event)

    assert result.diagnostics.per_attempt[0].failure_reason == "margin_miss"
```

- [ ] **Step 5: Run the retry-loop file and make it pass**

Run:

```bash
python -m pytest tests/watermark/test_retry_loop.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the retry-loop observability changes**

```bash
git add wfcllm/watermark/verifier.py wfcllm/watermark/retry_loop.py tests/watermark/test_retry_loop.py
git commit -m "feat: classify retry failure reasons"
```

### Task 3: Capture block lifecycles in generator and cascade

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `wfcllm/watermark/cascade.py`
- Modify: `tests/watermark/test_cascade.py`
- Modify: `tests/watermark/test_generator.py`

- [ ] **Step 1: Write failing generator tests for per-sample summaries and lifecycle records**

```python
def test_generate_attaches_route_one_diagnostic_summary(...):
    result = gen.generate("prompt")

    assert result.diagnostic_summary["diagnostics_version"] == 1
    assert "retry_summary" in result.diagnostic_summary
    assert "cascade_summary" in result.diagnostic_summary


def test_generate_marks_block_rescued_by_retry(...):
    result = gen.generate("prompt")

    assert result.block_ledgers[0]["final_outcome"]["rescued_by_retry"] is True
```

- [ ] **Step 2: Run the focused generator/cascade tests to verify failure**

Run:

```bash
python -m pytest tests/watermark/test_generator.py tests/watermark/test_cascade.py -k "diagnostic or rescued or cascade" -v
```

Expected: `GenerateResult` missing diagnostic fields and cascade metadata not serializable.

- [ ] **Step 3: Add lifecycle collection to `generator.py` and structured cascade metadata to `cascade.py`**

```python
@dataclass
class GenerateResult:
    ...
    diagnostic_summary: dict[str, object] = field(default_factory=dict)
    block_ledgers: list[dict[str, object]] = field(default_factory=list)
```

```python
return {
    "triggered": True,
    "compound_node_type": cascade_cp.compound_event.node_type,
    "failed_simple_count_before_cascade": len(cascade_cp.failed_simple_blocks),
    "restored_total_blocks": stats.total_blocks,
    "restored_embedded_blocks": stats.embedded_blocks,
    "restored_failed_blocks": stats.failed_blocks,
}
```

- [ ] **Step 4: Add a failing test for cascade replacement bookkeeping**

```python
def test_generate_marks_pending_failed_block_as_cascade_replaced(...):
    result = gen.generate("prompt")

    assert result.block_ledgers[0]["cascade_events"][0]["triggered"] is True
    assert result.block_ledgers[0]["final_outcome"]["rescued_by_cascade"] is True
```

- [ ] **Step 5: Run the generator/cascade tests until they pass**

Run:

```bash
python -m pytest tests/watermark/test_generator.py tests/watermark/test_cascade.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the generator/cascade lifecycle capture**

```bash
git add wfcllm/watermark/generator.py wfcllm/watermark/cascade.py tests/watermark/test_generator.py tests/watermark/test_cascade.py
git commit -m "feat: capture route-one block lifecycles"
```

### Task 4: Persist summaries in the watermarked artifact and write the block-ledger JSONL

**Files:**
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `tests/watermark/test_pipeline.py`

- [ ] **Step 1: Write failing pipeline tests for diagnostic summary fields and ledger output**

```python
def test_pipeline_writes_route_one_summary_fields(tmp_path):
    ...
    row = json.loads(Path(output_path).read_text(encoding="utf-8").splitlines()[0])

    assert row["diagnostics_version"] == 1
    assert row["retry_summary"]["blocks_with_retry"] == 1


def test_pipeline_writes_block_ledger_jsonl(tmp_path):
    ...
    ledger_path = Path(tmp_path) / "diagnostics" / "humaneval_20260330_120000_block_ledger.jsonl"

    assert ledger_path.exists()
```

- [ ] **Step 2: Run the pipeline tests to verify failure**

Run:

```bash
python -m pytest tests/watermark/test_pipeline.py -k "diagnostic or ledger" -v
```

Expected: missing summary fields and no diagnostics JSONL emitted.

- [ ] **Step 3: Extend the pipeline to write backward-compatible summaries and a sibling ledger file**

```python
record.update(result.diagnostic_summary)
```

```python
diagnostics_dir = out_dir.parent / "diagnostics"
diagnostics_dir.mkdir(parents=True, exist_ok=True)
```

```python
for ledger_row in result.block_ledgers:
    ledger_file.write(json.dumps(ledger_row, ensure_ascii=False) + "\n")
```

- [ ] **Step 4: Add a regression test for cascade visibility**

```python
def test_pipeline_exposes_cascade_recovery_without_overloading_fallback_blocks(tmp_path):
    ...
    assert row["cascade_summary"]["cascade_triggers"] == 1
    assert row["fallback_blocks"] == 0
```

- [ ] **Step 5: Run the pipeline tests until they pass**

Run:

```bash
python -m pytest tests/watermark/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the persistence layer changes**

```bash
git add wfcllm/watermark/pipeline.py tests/watermark/test_pipeline.py
git commit -m "feat: persist route-one diagnostics artifacts"
```

### Task 5: Extend offline analysis to consume optional route-one evidence

**Files:**
- Modify: `wfcllm/extract/offline_analysis.py`
- Modify: `tests/extract/test_offline_analysis.py`

- [ ] **Step 1: Add failing offline-analysis tests for optional diagnostic summaries**

```python
def test_load_watermarked_artifact_preserves_optional_route_one_summary(tmp_path):
    _write_jsonl(
        tmp_path / "watermarked.jsonl",
        [
            {
                "id": "HumanEval/0",
                "total_blocks": 8,
                "embedded_blocks": 6,
                "failed_blocks": 2,
                "fallback_blocks": 0,
                "embed_rate": 0.75,
                "diagnostics_version": 1,
                "retry_summary": {"blocks_with_retry": 2, "retry_exhausted_blocks": 1},
                "cascade_summary": {"cascade_triggers": 1, "cascade_rescued_blocks": 0},
                "failure_reason_counts": {"signature_miss": 3},
            }
        ],
    )

    artifact = load_watermarked_artifact(tmp_path / "watermarked.jsonl")

    assert artifact.records["HumanEval/0"]["retry_summary"]["blocks_with_retry"] == 2
```

- [ ] **Step 2: Run the offline-analysis tests to verify failure**

Run:

```bash
python -m pytest tests/extract/test_offline_analysis.py -k "route_one or diagnostic" -v
```

Expected: missing optional diagnostic parsing and missing anomaly flags.

- [ ] **Step 3: Extend offline-analysis models and anomaly generation**

```python
if record.get("retry_summary", {}).get("retry_exhausted_blocks", 0) > 0 and delta.flip_direction == "true_to_false":
    anomaly_flags.append("near_miss_with_exhausted_retry")
```

```python
if record.get("cascade_summary", {}).get("cascade_triggers", 0) > 0 and record.get("cascade_summary", {}).get("cascade_rescued_blocks", 0) == 0:
    anomaly_flags.append("cascade_no_recovery")
```

- [ ] **Step 4: Add a test for backward compatibility with older artifacts**

```python
def test_load_watermarked_artifact_keeps_older_rows_compatible(tmp_path):
    ...
    artifact = load_watermarked_artifact(tmp_path / "legacy.jsonl")

    assert artifact.records["HumanEval/0"]["embed_rate"] == 0.75
```

- [ ] **Step 5: Run the offline-analysis tests until they pass**

Run:

```bash
python -m pytest tests/extract/test_offline_analysis.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit the offline-analysis extensions**

```bash
git add wfcllm/extract/offline_analysis.py tests/extract/test_offline_analysis.py
git commit -m "feat: analyze route-one diagnostics offline"
```

### Task 6: Run focused verification and capture the artifact contract

**Files:**
- Modify: `tests/watermark/test_generator.py`
- Modify: `tests/watermark/test_pipeline.py`
- Modify: `tests/extract/test_offline_analysis.py`

- [ ] **Step 1: Run the complete focused verification set**

Run:

```bash
python -m pytest \
  tests/watermark/test_diagnostics.py \
  tests/watermark/test_retry_loop.py \
  tests/watermark/test_cascade.py \
  tests/watermark/test_generator.py \
  tests/watermark/test_pipeline.py \
  tests/extract/test_offline_analysis.py -v
```

Expected: PASS.

- [ ] **Step 2: Add one regression test that checks the ledger and summary stay in sync**

```python
def test_pipeline_summary_matches_ledger_rollup(tmp_path):
    ...
    assert row["rescued_blocks"] == 1
    assert sum(1 for item in ledger_rows if item["final_outcome"]["rescued_by_retry"]) == 1
```

- [ ] **Step 3: Run the focused verification set again**

Run:

```bash
python -m pytest \
  tests/watermark/test_diagnostics.py \
  tests/watermark/test_retry_loop.py \
  tests/watermark/test_cascade.py \
  tests/watermark/test_generator.py \
  tests/watermark/test_pipeline.py \
  tests/extract/test_offline_analysis.py -v
```

Expected: PASS with no schema-regression failures.

- [ ] **Step 4: Smoke-test one tiny watermark pipeline run for artifact creation**

Run:

```bash
python run.py --phase watermark --dataset humaneval --sample-limit 1 --secret-key test-key
```

Expected: one new `data/watermarked/humaneval_*.jsonl` row and one new `data/diagnostics/humaneval_*_block_ledger.jsonl` file.

- [ ] **Step 5: Commit the verification-driven contract lock-in**

```bash
git add tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/extract/test_offline_analysis.py
git commit -m "test: lock route-one diagnostics artifact contract"
```
