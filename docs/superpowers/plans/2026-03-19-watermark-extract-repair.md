# Watermark Extract Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复“生成端显示大量嵌入成功、提取端却大面积 miss”的主链路问题，使提取阶段严格复用生成阶段的真实 LSH 参数与校准口径，并修正 cascade/diagnostic 统计失真。

**Architecture:** 修复分三层进行。第一层修 `run_extract()` 的参数传播，让 detector 与 calibrator 使用和 watermark 阶段完全一致的 `lsh_d/lsh_gamma`；第二层把嵌入参数写入水印 JSONL 元数据，并在提取时优先从输入文件恢复，避免配置漂移；第三层收敛 diagnostic/cascade 口径，使对齐报告不再把中间态 compound 事件误当成最终可提取信号。

**Tech Stack:** Python 3.11, pytest, dataclasses, JSONL pipeline, tree-sitter-python, HuggingFace encoder inference

---

## Root Cause Summary

- `run.py:435` 在 watermark 阶段把 `lsh_d=4`、`lsh_gamma=0.75` 正确传入 `WatermarkConfig`。
- `run.py:585` 在 extract 阶段构造 `ExtractConfig` 时漏传 `lsh_d`、`lsh_gamma`，导致 `wfcllm/extract/config.py` 默认回退到 `3/0.5`。
- `run.py:562` 的自动校准虽然读取了 `extract` 配置中的 `lsh_d/lsh_gamma`，但这些值来自 `configs/base_config.json` 的 `extract` 节，当前与 watermark 节不一致：`configs/base_config.json:49` 是 `4/0.75`，`configs/base_config.json:56` 是 `3/0.5`。
- 因此提取端同时发生两种错位：
  - LSH 空间错位：超平面与 valid set 变了。
  - 统计口径错位：`HypothesisTester` 和 `ThresholdCalibrator` 的 `gamma` 变了，阈值也跟着错。
- 第二梯队问题是 diagnostic/cascade 口径漂移：
  - `experiment/embed_extract_alignment/diagnostic_generator.py` 仍在 `_diag_try_cascade()` 中验证中间态 compound 事件，与 `wfcllm/watermark/generator.py:199` 的主实现不一致。
  - `wfcllm/watermark/generator.py` 的 `stats` 在发生 rollback/cascade 后不会回滚已累计的 simple success 计数，导致日志和最终代码可提取信号可能不一致。

---

## File Structure

### Core production files

- Modify: `run.py`
- Modify: `configs/base_config.json`
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `wfcllm/extract/config.py`
- Modify: `wfcllm/extract/pipeline.py`

### Diagnostic / experiment files

- Modify: `experiment/embed_extract_alignment/diagnostic_generator.py`
- Modify: `experiment/embed_extract_alignment/models.py`
- Modify: `experiment/embed_extract_alignment/aligner.py`
- Modify: `experiment/embed_extract_alignment/report.py`
- Create: `tools/debug_extract_alignment.py`

### Tests

- Modify: `tests/test_run.py`
- Modify: `tests/test_run_config.py`
- Modify: `tests/watermark/test_pipeline.py`
- Modify: `tests/extract/test_pipeline.py`
- Modify: `tests/experiment/embed_extract_alignment/test_aligner.py`
- Create: `tests/extract/test_extract_param_resolution.py`
- Create: `tests/watermark/test_embed_metadata_roundtrip.py`

---

## Task 1: Lock the primary regression with failing tests

**Files:**
- Modify: `tests/test_run.py`
- Modify: `tests/test_run_config.py`
- Create: `tests/extract/test_extract_param_resolution.py`

- [ ] **Step 1: Add an AST-level test proving `run_extract()` passes `lsh_d` and `lsh_gamma` into `ExtractConfig`**

```python
def test_run_extract_passes_lsh_params():
    tree = ast.parse(Path("run.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "ExtractConfig":
            kws = {kw.arg for kw in node.keywords}
            assert "lsh_d" in kws
            assert "lsh_gamma" in kws
            return
    raise AssertionError("run.py must pass lsh_d/lsh_gamma into ExtractConfig")
```

- [ ] **Step 2: Add a config test requiring extract defaults to match watermark defaults unless explicitly overridden**

```python
def test_extract_lsh_defaults_match_watermark():
    cfg = json.loads(Path("configs/base_config.json").read_text(encoding="utf-8"))
    assert cfg["extract"]["lsh_d"] == cfg["watermark"]["lsh_d"]
    assert cfg["extract"]["lsh_gamma"] == cfg["watermark"]["lsh_gamma"]
```

- [ ] **Step 3: Add a pure unit test for parameter resolution precedence**

```python
def test_extract_prefers_embedded_metadata_over_stale_config():
    record = {"watermark_params": {"lsh_d": 4, "lsh_gamma": 0.75}}
    ext_cfg = {"lsh_d": 3, "lsh_gamma": 0.5}
    resolved = resolve_extract_lsh_params(record, ext_cfg)
    assert resolved == (4, 0.75)
```

- [ ] **Step 4: Run the narrow failing test set**

Run:

```bash
pytest -q tests/test_run.py tests/test_run_config.py tests/extract/test_extract_param_resolution.py
```

Expected:

- `test_run_extract_passes_lsh_params` FAILS before implementation.
- `test_extract_lsh_defaults_match_watermark` FAILS with current `configs/base_config.json`.
- `test_extract_prefers_embedded_metadata_over_stale_config` FAILS because helper does not exist yet.

- [ ] **Step 5: Commit**

```bash
git add tests/test_run.py tests/test_run_config.py tests/extract/test_extract_param_resolution.py
git commit -m "test: lock extract lsh parameter propagation"
```

---

## Task 2: Persist embed-side parameters into watermark output

**Files:**
- Modify: `wfcllm/watermark/pipeline.py`
- Create: `tests/watermark/test_embed_metadata_roundtrip.py`
- Modify: `tests/watermark/test_pipeline.py`

- [ ] **Step 1: Add a failing test asserting watermark JSONL records carry non-secret embed metadata**

```python
def test_pipeline_writes_watermark_params(tmp_path, generator, pipeline_config):
    output_path = pipeline.run()
    row = json.loads(Path(output_path).read_text(encoding="utf-8").splitlines()[0])
    assert row["watermark_params"]["lsh_d"] == 4
    assert row["watermark_params"]["lsh_gamma"] == 0.75
```

- [ ] **Step 2: Implement minimal metadata serialization in `wfcllm/watermark/pipeline.py`**

```python
record["watermark_params"] = {
    "lsh_d": generator._config.lsh_d,
    "lsh_gamma": generator._config.lsh_gamma,
    "margin_base": generator._config.margin_base,
    "margin_alpha": generator._config.margin_alpha,
}
```

- [ ] **Step 3: Keep metadata intentionally non-secret**

```python
assert "secret_key" not in record["watermark_params"]
```

- [ ] **Step 4: Run focused tests**

Run:

```bash
pytest -q tests/watermark/test_pipeline.py tests/watermark/test_embed_metadata_roundtrip.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/pipeline.py tests/watermark/test_pipeline.py tests/watermark/test_embed_metadata_roundtrip.py
git commit -m "feat: persist watermark lsh metadata in output records"
```

---

## Task 3: Resolve extract-side LSH parameters from input metadata and pass them everywhere

**Files:**
- Modify: `run.py`
- Modify: `wfcllm/extract/config.py`
- Modify: `tests/test_run.py`
- Modify: `tests/extract/test_extract_param_resolution.py`

- [ ] **Step 1: Introduce a small helper in `run.py` to resolve extract parameters**

```python
def resolve_extract_lsh_params(first_record: dict, ext_cfg: dict) -> tuple[int, float]:
    params = first_record.get("watermark_params") or {}
    lsh_d = params.get("lsh_d", ext_cfg.get("lsh_d", 3))
    lsh_gamma = params.get("lsh_gamma", ext_cfg.get("lsh_gamma", 0.5))
    return int(lsh_d), float(lsh_gamma)
```

- [ ] **Step 2: Document and test the homogeneity assumption for watermark output**

```python
# WatermarkPipeline writes one JSONL per watermark run/config, so the file is
# assumed homogeneous. The extractor resolves LSH params from the first record
# and warns if later records disagree.
```

- [ ] **Step 3: Load the first JSONL record before calibration**

```python
with open(input_file, encoding="utf-8") as f:
    first_line = next((line.strip() for line in f if line.strip()), "")
first_record = json.loads(first_line) if first_line else {}
lsh_d, lsh_gamma = resolve_extract_lsh_params(first_record, ext_cfg)
```

- [ ] **Step 4: Use the resolved values in both calibrator and detector**

```python
lsh_space = LSHSpace(secret_key, embed_dim, lsh_d)
keying = WatermarkKeying(secret_key, lsh_d, lsh_gamma)

extract_config = ExtractConfig(
    secret_key=secret_key,
    embed_dim=embed_dim,
    fpr_threshold=fpr_threshold,
    lsh_d=lsh_d,
    lsh_gamma=lsh_gamma,
)
```

- [ ] **Step 5: Emit an explicit warning when metadata and config disagree**

```python
if "watermark_params" in first_record:
    cfg_pair = (ext_cfg.get("lsh_d"), ext_cfg.get("lsh_gamma"))
    meta_pair = (lsh_d, lsh_gamma)
    if all(v is not None for v in cfg_pair) and cfg_pair != meta_pair:
        print(
            f"[警告] extract 配置 LSH 参数 {cfg_pair} 与输入文件元数据 {meta_pair} 不一致；"
            f"优先使用输入文件元数据",
            file=sys.stderr,
        )
```

- [ ] **Step 6: Add explicit config fields to `ExtractConfig` construction test coverage**

```python
assert ExtractConfig(secret_key="k", lsh_d=4, lsh_gamma=0.75).lsh_d == 4
assert ExtractConfig(secret_key="k", lsh_d=4, lsh_gamma=0.75).lsh_gamma == 0.75
```

- [ ] **Step 7: Run the focused test set**

Run:

```bash
pytest -q tests/test_run.py tests/extract/test_extract_param_resolution.py
```

Expected: PASS

- [ ] **Step 8: Commit (optional if your environment should remain commit-free)**

```bash
git add run.py wfcllm/extract/config.py tests/test_run.py tests/extract/test_extract_param_resolution.py
git commit -m "fix: propagate embed lsh params into extract stage"
```

---

## Task 4: Align config defaults and recalibration behavior

**Files:**
- Modify: `configs/base_config.json`
- Modify: `tests/test_run_config.py`
- Modify: `tests/extract/test_pipeline.py`

- [ ] **Step 1: Update `configs/base_config.json` so the `extract` section matches the `watermark` section**

```json
"extract": {
  "lsh_d": 4,
  "lsh_gamma": 0.75
}
```

- [ ] **Step 2: Add a regression test for calibrated threshold path**

```python
def test_run_extract_uses_resolved_gamma_for_calibration(monkeypatch):
    captured = {}
    class FakeCalibrator:
        def __init__(self, scorer, gamma):
            captured["gamma"] = gamma
    ...
    assert captured["gamma"] == 0.75
```

- [ ] **Step 3: Add a smoke test documenting the expected threshold behavior**

```python
assert 0.5 < resolved_threshold < 2.5
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
pytest -q tests/test_run_config.py tests/extract/test_pipeline.py
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add configs/base_config.json tests/test_run_config.py tests/extract/test_pipeline.py
git commit -m "fix: align extract defaults and calibration gamma"
```

---

## Task 5: Fix cascade/diagnostic accounting drift

**Files:**
- Modify: `experiment/embed_extract_alignment/diagnostic_generator.py`
- Modify: `experiment/embed_extract_alignment/models.py`
- Modify: `experiment/embed_extract_alignment/aligner.py`
- Modify: `experiment/embed_extract_alignment/report.py`
- Modify: `tests/experiment/embed_extract_alignment/test_aligner.py`
- Modify: `tests/watermark/test_generator_integration.py`

- [ ] **Step 1: Write a failing test proving cascade diagnostics remain compound-aligned, but are clearly separated from simple-score agreement**

```python
def test_cascade_diagnostic_preserves_compound_alignment():
    report = ...
    assert report.compound_aligned_count >= 1
    assert report.score_disagree_count == 0  # compound pairs excluded from score disagreement
```

- [ ] **Step 2: Change diagnostic recording to mirror production semantics without deleting compound alignment**

```python
# stop treating the cascade helper as an independent watermark-verification path
# but still preserve compound EmbedEvent alignment against all_blocks with
# extract_score=None, matching the spec's compound-pair model
```

- [ ] **Step 3: Add explicit metadata that distinguishes production simple events from diagnostic compound probes**

```python
@dataclass
class EmbedEvent:
    ...
    is_diagnostic_compound_probe: bool = False
```

- [ ] **Step 4: Update report summaries to split simple-vs-compound mismatch counters without removing compound pairs**

```python
text_mismatch_simple_only
text_mismatch_compound_only
```

- [ ] **Step 5: Run the diagnostic test slice**

Run:

```bash
pytest -q tests/experiment/embed_extract_alignment/test_aligner.py tests/watermark/test_generator_integration.py
```

Expected: PASS

- [ ] **Step 6: Commit (optional if your environment should remain commit-free)**

```bash
git add experiment/embed_extract_alignment/diagnostic_generator.py experiment/embed_extract_alignment/aligner.py experiment/embed_extract_alignment/report.py tests/experiment/embed_extract_alignment/test_aligner.py tests/watermark/test_generator_integration.py
git commit -m "fix: align cascade diagnostics with production semantics"
```

---

## Task 6: Add a one-shot local debug tool for prompt-level forensics

**Files:**
- Create: `tools/debug_extract_alignment.py`
- Modify: `tests/extract/test_pipeline.py`

- [ ] **Step 1: Create a CLI that accepts `--prompt-id`, `--input-file`, `--use-embedded-params`**

```python
parser.add_argument("--prompt-id", required=True)
parser.add_argument("--use-embedded-params", action="store_true")
```

- [ ] **Step 2: Print the exact mismatch dimensions separately**

```python
print({
    "text_mismatch_count": report.text_mismatch_count,
    "parent_mismatch_count": report.parent_mismatch_count,
    "score_disagree_count": report.score_disagree_count,
    "resolved_lsh_d": lsh_d,
    "resolved_lsh_gamma": lsh_gamma,
})
```

- [ ] **Step 3: Emit side-by-side samples for the first few mismatches**

```python
for pair in report.aligned_pairs[:5]:
    ...
```

- [ ] **Step 4: Add a smoke test that the script can resolve metadata without crashing**

Run:

```bash
pytest -q tests/extract/test_pipeline.py
```

Expected: PASS

- [ ] **Step 5: Commit**
- [ ] **Step 5: Commit (optional if your environment should remain commit-free)**

```bash
git add tools/debug_extract_alignment.py tests/extract/test_pipeline.py
git commit -m "feat: add prompt-level extract alignment debugger"
```

---

## Task 7: Verify the real-world fix on the archived dataset

**Files:**
- No code changes required unless verification reveals new regression.

- [ ] **Step 1: Re-run the narrow unit suite**

Run:

```bash
pytest -q tests/test_run.py tests/test_run_config.py tests/watermark/test_pipeline.py tests/extract/test_pipeline.py tests/extract/test_extract_param_resolution.py tests/watermark/test_embed_metadata_roundtrip.py tests/experiment/embed_extract_alignment/test_aligner.py tests/watermark/test_generator_integration.py
```

Expected: PASS

- [ ] **Step 2: Re-run extract on the archived watermark file**

Run:

```bash
/root/miniconda3/envs/WFCLLM/bin/python run.py --phase extract --config configs/base_config.json --input-file data/watermarked/humaneval_20260318_204554.jsonl --secret-key 1010
```

Expected:

- detector no longer uses `3/0.5`;
- calibration log reflects `gamma=0.75`;
- overall mean hit rate increases materially vs the current archived run.

- [ ] **Step 3: Spot-check the known bad prompts**

Run:

```bash
/root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/128 --input-file data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params
/root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/23 --input-file data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params
/root/miniconda3/envs/WFCLLM/bin/python tools/debug_extract_alignment.py --prompt-id HumanEval/45 --input-file data/watermarked/humaneval_20260318_204554.jsonl --use-embedded-params
```

Expected:

- parameter resolution shows `lsh_d=4`, `lsh_gamma=0.75`;
- text/parent mismatches are either zero or confined to non-scored diagnostic paths;
- score disagreement shrinks materially on these prompts.

- [ ] **Step 4: Record post-fix evidence in a short markdown note**

Create:

```text
docs/superpowers/specs/2026-03-19-extract-repair-verification.md
```

Content:

- before/after watermark rate
- before/after average hit rate
- 3 prompt spot-check outcomes

- [ ] **Step 5: Commit (optional if your environment should remain commit-free)**

```bash
git add docs/superpowers/specs/2026-03-19-extract-repair-verification.md
git commit -m "docs: record extract repair verification results"
```

---

## Implementation Notes

- Do **not** start by changing the diagnostic experiment. The primary production failure is parameter drift in `run_extract()`.
- Keep the first production fix minimal:
  - restore parameter propagation;
  - align config defaults;
  - recalibrate with the same `gamma` used in embedding.
- Only after the production path is fixed should you adjust `DiagnosticGenerator`; otherwise you risk “fixing the microscope before fixing the patient.”
- Treat each watermark JSONL as homogeneous output from one watermark run/config. The extractor may resolve parameters from the first record, but must warn if later records disagree.
- Do **not** persist `secret_key` into JSONL metadata.
- Prefer input-file metadata over `extract` config when both are present; emit a warning if they disagree.

---

## Evidence This Plan Must Preserve

- Archived dataset evidence:
  - `data/watermarked/humaneval_20260318_204554.jsonl`
  - `data/results/humaneval_20260318_204554_details.jsonl`
  - `data/results/humaneval_20260318_204554_summary.json`
- Logs:
  - `logs/0318_204544_watermark.log`
  - `logs/0319_152031_extract.log`
- Known reproductions:
  - `HumanEval/128`: current extract params produce ~2/10 hits; watermark-side params recover near-embed hit count.
  - `HumanEval/23`, `HumanEval/45`, `HumanEval/139`: same pattern.

---

## Expected Outcome

After Task 4, production extraction should stop using the wrong LSH space and wrong null hypothesis. After Task 5, alignment diagnostics should stop over-reporting text mismatch from cascade-only intermediate states. After Task 7, we should have a reproducible before/after artifact set that proves whether the remaining gap is statistical thresholding, true text drift, or residual cascade accounting.
