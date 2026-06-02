# Dual-Channel Strong ACW Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strong ACW-style token channel to `WFCLLM` while keeping the existing block-level semantic watermark as the primary acceptance and rollback path.

**Architecture:** Keep the current semantic channel intact, add a separate token-channel subsystem under `wfcllm/watermark/token_channel/`, hook it into generation as a token-level bias layer, replay it in extraction as a lexical detector, and fuse semantic and lexical scores in a transparent joint result. Build the feature in vertical slices so each milestone is testable offline and can be disabled independently.

**Tech Stack:** Python 3.11, PyTorch, existing HuggingFace model/tokenizer interfaces, existing `wfcllm` watermark/extract pipelines, pytest, offline local model assets.

---

## File Map

### Create

- `wfcllm/watermark/token_channel/__init__.py`
- `wfcllm/watermark/token_channel/config.py`
- `wfcllm/watermark/token_channel/protocol.py`
- `wfcllm/watermark/token_channel/model.py`
- `wfcllm/watermark/token_channel/runtime.py`
- `wfcllm/watermark/token_channel/train_corpus.py`
- `wfcllm/watermark/token_channel/teacher.py`
- `wfcllm/watermark/token_channel/train.py`
- `wfcllm/watermark/token_channel/features.py`
- `wfcllm/common/offline_code_eval.py`
- `wfcllm/extract/token_channel.py`
- `scripts/evaluate_dual_channel.py`
- `tests/watermark/token_channel/test_config.py`
- `tests/watermark/token_channel/test_protocol.py`
- `tests/watermark/token_channel/test_model.py`
- `tests/watermark/token_channel/test_runtime.py`
- `tests/watermark/token_channel/test_train_corpus.py`
- `tests/watermark/token_channel/test_features.py`
- `tests/extract/test_token_channel.py`
- `tests/extract/test_joint_detection.py`

### Modify

- `wfcllm/watermark/config.py`
- `wfcllm/watermark/generator.py`
- `wfcllm/watermark/pipeline.py`
- `wfcllm/extract/config.py`
- `wfcllm/extract/detector.py`
- `wfcllm/extract/pipeline.py`
- `wfcllm/extract/hypothesis.py`
- `run.py`
- `configs/base_config.json`
- `tests/watermark/test_config.py`
- `tests/watermark/test_generator.py`
- `tests/watermark/test_pipeline.py`
- `tests/extract/test_config.py`
- `tests/extract/test_detector.py`
- `tests/extract/test_pipeline.py`
- `tests/test_run.py`
- `README.md`

### Existing Files To Read Before Editing

- `docs/superpowers/specs/2026-04-10-dual-channel-strong-acw-design.md`
- `wfcllm/watermark/config.py`
- `wfcllm/watermark/generator.py`
- `wfcllm/watermark/pipeline.py`
- `wfcllm/extract/config.py`
- `wfcllm/extract/detector.py`
- `wfcllm/extract/pipeline.py`
- `run.py`

## Task 1: Add Token-Channel Configuration Surface

**Files:**
- Create: `wfcllm/watermark/token_channel/config.py`
- Modify: `wfcllm/watermark/config.py`
- Modify: `wfcllm/extract/config.py`
- Modify: `tests/watermark/test_config.py`
- Test: `tests/watermark/token_channel/test_config.py`
- Test: `tests/extract/test_config.py`

- [ ] **Step 1: Write the failing config tests**

```python
from wfcllm.watermark.config import WatermarkConfig


def test_token_channel_defaults_disabled():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.token_channel.enabled is False
    assert cfg.token_channel.channel_mode == "semantic-only"
    assert cfg.token_channel.joint_threshold == 4.0
```

- [ ] **Step 2: Run the config tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py tests/extract/test_config.py tests/watermark/token_channel/test_config.py -v`
Expected: FAIL with missing `token_channel` config objects/imports.

- [ ] **Step 3: Add minimal token-channel dataclasses**

```python
@dataclass
class TokenChannelConfig:
    enabled: bool = False
    channel_mode: str = "semantic-only"
    model_path: str | None = None
    context_width: int = 4
    switch_threshold: float = 0.7
    delta: float = 2.0
    ignore_repeated_ngrams: bool = False
    ignore_repeated_prefixes: bool = False
    joint_semantic_weight: float = 1.0
    joint_lexical_weight: float = 0.5
    lexical_full_weight_min_positions: int = 64
    joint_threshold: float = 4.0
    lexical_min_block_tokens: int = 8
    lexical_retry_decay_start: int = 2
    lexical_retry_disable_after: int = 4
    lexical_gate_probe_tokens: int = 16
    lexical_gate_min_fraction: float = 0.10
    debug_mode: bool = False
```

- [ ] **Step 4: Wire config into watermark and extract configs**

```python
@dataclass
class WatermarkConfig:
    ...
    token_channel: TokenChannelConfig = field(default_factory=TokenChannelConfig)
```

- [ ] **Step 5: Re-run the config tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_config.py tests/extract/test_config.py tests/watermark/token_channel/test_config.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/token_channel/config.py wfcllm/watermark/config.py wfcllm/extract/config.py tests/watermark/test_config.py tests/extract/test_config.py tests/watermark/token_channel/test_config.py
git commit -m "feat: add token channel configuration"
```

## Task 2: Implement the Token-Channel Partition Protocol

**Files:**
- Create: `wfcllm/watermark/token_channel/protocol.py`
- Test: `tests/watermark/token_channel/test_protocol.py`

- [ ] **Step 1: Write the failing protocol tests**

```python
def test_partition_uses_full_vocab_pairing():
    logits = torch.tensor([0.9, 0.8, 0.2, 0.1])
    green, red = build_partition(logits=logits, prefix_ids=(1, 2), secret_key="k")
    assert len(green) + len(red) == 4
    assert set(green).isdisjoint(red)
```

- [ ] **Step 2: Run the protocol tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_protocol.py -v`
Expected: FAIL with missing protocol helpers.

- [ ] **Step 3: Implement deterministic full-vocab pairing protocol**

```python
def build_partition(logits: torch.Tensor, prefix_ids: tuple[int, ...], secret_key: str) -> PartitionResult:
    sorted_ids = torch.argsort(logits, descending=True).tolist()
    rng = seeded_rng(prefix_ids, secret_key)
    ...
```

- [ ] **Step 4: Add repeat-filter helpers needed by detection**

```python
def make_prefix_key(prefix_ids: tuple[int, ...]) -> tuple[int, ...]:
    return prefix_ids
```

- [ ] **Step 5: Re-run the protocol tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_protocol.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add wfcllm/watermark/token_channel/protocol.py tests/watermark/token_channel/test_protocol.py
git commit -m "feat: add token channel partition protocol"
```

## Task 3: Add Token-Channel Model Loading and Runtime API

**Files:**
- Create: `wfcllm/watermark/token_channel/features.py`
- Create: `wfcllm/watermark/token_channel/model.py`
- Create: `wfcllm/watermark/token_channel/runtime.py`
- Create: `tests/watermark/token_channel/test_features.py`
- Create: `tests/watermark/token_channel/test_model.py`
- Create: `tests/watermark/token_channel/test_runtime.py`

- [ ] **Step 1: Write failing model/runtime tests**

```python
def test_runtime_returns_gate_and_partition_logits(fake_token_channel_model):
    runtime = TokenChannelRuntime(fake_token_channel_model, config)
    decision = runtime.score_prefix(prefix_ids=[1, 2, 3], features=feature_vector)
    assert hasattr(decision, "switch_logit")
    assert decision.preference_logits.shape[-1] == 8
```

- [ ] **Step 2: Run the runtime tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_features.py tests/watermark/token_channel/test_model.py tests/watermark/token_channel/test_runtime.py -v`
Expected: FAIL with missing runtime/model classes.

- [ ] **Step 3: Add structural feature extraction and structure-mask helpers**

```python
@dataclass(frozen=True)
class TokenChannelFeatures:
    node_type: str
    parent_node_type: str
    block_relative_offset: int
    in_code_body: bool
    structure_mask: bool
```

- [ ] **Step 4: Add a minimal model artifact contract**

```python
@dataclass(frozen=True)
class TokenChannelArtifact:
    model_path: str
    context_width: int
    tokenizer_name: str
    metadata_path: str
```

- [ ] **Step 5: Add runtime wrapper around the model, features, and protocol**

```python
class TokenChannelRuntime:
    def score_prefix(self, prefix_ids: list[int], features: TokenChannelFeatures) -> TokenChannelDecision:
        output = self._model(torch.tensor(prefix_ids[-self._context_width:]), features)
        return TokenChannelDecision(...)
```

- [ ] **Step 6: Add artifact metadata persistence and compatibility checks**

```python
def save_token_channel_artifact_metadata(path: Path, metadata: dict[str, object]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
```

- [ ] **Step 7: Define the metadata schema and enforce it at load time**

```python
required_keys = {
    "schema_version",
    "tokenizer_name",
    "tokenizer_vocab_size",
    "context_width",
    "feature_version",
    "training_config",
}
```

- [ ] **Step 8: Re-run the runtime tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_features.py tests/watermark/token_channel/test_model.py tests/watermark/token_channel/test_runtime.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add wfcllm/watermark/token_channel/features.py wfcllm/watermark/token_channel/model.py wfcllm/watermark/token_channel/runtime.py tests/watermark/token_channel/test_features.py tests/watermark/token_channel/test_model.py tests/watermark/token_channel/test_runtime.py
git commit -m "feat: add token channel runtime model wrapper"
```

## Task 4: Build the Offline Training Corpus and Teacher Extraction Path

**Files:**
- Create: `wfcllm/watermark/token_channel/train_corpus.py`
- Create: `wfcllm/watermark/token_channel/teacher.py`
- Create: `wfcllm/watermark/token_channel/train.py`
- Create: `tests/watermark/token_channel/test_train_corpus.py`
- Modify: `wfcllm/common/transform/engine.py`
- Modify: `wfcllm/watermark/token_channel/features.py`

- [ ] **Step 0: Add explicit structure-mask tests for excluded regions**

```python
def test_structure_mask_excludes_imports_signatures_and_decorators():
    masks = build_structure_masks(source_code)
    assert masks["import_statement"] is False
    assert masks["decorator"] is False
    assert masks["function_signature"] is False
    assert masks["class_header"] is False
```

- [ ] **Step 1: Write the failing corpus-builder tests**

```python
def test_build_training_rows_collects_prefix_entropy_and_next_token(tmp_path):
    rows = build_training_rows(samples=[sample], context_width=4)
    assert rows[0]["prefix_tokens"]
    assert "entropy" in rows[0]
    assert "continuation_diversity" in rows[0]
    assert "node_type" in rows[0]
    assert "structure_mask" in rows[0]
```

- [ ] **Step 2: Run the corpus tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_corpus.py -v`
Expected: FAIL with missing corpus/teacher helpers.

- [ ] **Step 3: Implement corpus building from base samples plus positive transforms**

```python
def build_augmented_variants(source_code: str) -> list[str]:
    return transform_engine.apply_positive_variants(source_code)
```

- [ ] **Step 4: Attach structural features and code-body masks to each training row**

```python
row["node_type"] = contract.node_type
row["parent_node_type"] = contract.parent_node_type or "module"
row["block_relative_offset"] = relative_offset
row["structure_mask"] = is_structure_safe_position(...)
```

- [ ] **Step 4a: Implement deterministic AST-to-token mask reconstruction for the formal exclusions**

```python
excluded_regions = collect_excluded_token_spans(
    source_code,
    excluded_node_types={
        "import_statement",
        "import_from_statement",
        "decorator",
        "function_signature",
        "class_header",
    },
)
```

- [ ] **Step 5: Compute explicit switch-target labels from the approved rule**

```python
row["switch_target"] = int(
    row["structure_mask"]
    and row["entropy"] >= entropy_threshold
    and row["continuation_diversity"] >= diversity_threshold
)
```

- [ ] **Step 6: Implement persistent corpus-cache and teacher-cache formats**

```python
def save_training_cache(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("wb") as fh:
        pickle.dump(rows, fh)
```

- [ ] **Step 7: Implement teacher extraction helpers**

```python
def extract_teacher_rows(tokenizer, model, text: str, context_width: int) -> list[dict[str, object]]:
    ...
```

- [ ] **Step 8: Implement a train-entry skeleton with dataset loading only**

```python
def main() -> None:
    parser = build_parser()
    ...
```

- [ ] **Step 9: Re-run the corpus tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_corpus.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add wfcllm/watermark/token_channel/train_corpus.py wfcllm/watermark/token_channel/teacher.py wfcllm/watermark/token_channel/train.py wfcllm/common/transform/engine.py tests/watermark/token_channel/test_train_corpus.py
git commit -m "feat: add token channel training corpus builder"
```

## Task 5: Train the Token-Channel Model with TDD Loss Coverage

**Files:**
- Modify: `wfcllm/watermark/token_channel/__init__.py`
- Modify: `wfcllm/watermark/token_channel/model.py`
- Modify: `wfcllm/watermark/token_channel/train.py`
- Modify: `tests/watermark/token_channel/test_model.py`

- [ ] **Step 1: Write failing loss tests for switch and token heads**

```python
def test_compute_loss_returns_distill_ce_and_switch_terms():
    loss_dict = model.compute_loss(batch, output)
    assert set(loss_dict) >= {"total_loss", "distillation_loss", "ce_loss", "switch_loss"}
```

- [ ] **Step 2: Add a failing test for switch-target consumption**

```python
def test_switch_loss_uses_precomputed_switch_target():
    batch = {"switch_target": torch.tensor([1.0])}
    loss_dict = model.compute_loss(batch, output)
    assert loss_dict["switch_loss"] > 0
```

- [ ] **Step 3: Run the model tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_model.py -v`
Expected: FAIL with missing `compute_loss` or missing terms.

- [ ] **Step 4: Implement the minimal dual-head model and losses**

```python
total_loss = alpha_distill * distillation_loss + alpha_ce * ce_loss + alpha_switch * switch_loss
```

- [ ] **Step 5: Make `compute_loss` consume persisted `switch_target` labels**

```python
switch_target = batch["switch_target"].to(output["switch"].dtype)
```

- [ ] **Step 6: Add train-loop smoke coverage**

```python
def train_one_epoch(...):
    ...
```

- [ ] **Step 7: Add checkpoint export and metadata save steps to the training path**

```python
torch.save(model.state_dict(), checkpoint_dir / "pytorch_model.bin")
save_token_channel_artifact_metadata(checkpoint_dir / "metadata.json", metadata)
```

- [ ] **Step 8: Record training-validation evidence required by the spec**

```text
Write out switch-target positive/negative counts, train loss, validation loss, and per-epoch switch loss so Section 13.1 can be checked directly.
```

- [ ] **Step 9: Re-run the model tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_model.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add wfcllm/watermark/token_channel/__init__.py wfcllm/watermark/token_channel/model.py wfcllm/watermark/token_channel/train.py tests/watermark/token_channel/test_model.py
git commit -m "feat: add token channel training losses"
```

## Task 6: Integrate Token-Channel Bias into Watermark Generation

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `run.py`
- Modify: `configs/base_config.json`
- Modify: `tests/watermark/test_generator.py`
- Modify: `tests/watermark/test_pipeline.py`
- Modify: `tests/test_run.py`

- [ ] **Step 1: Write failing generation integration tests**

```python
def test_dual_channel_mode_uses_token_bias_before_block_verification(...):
    result = generator.generate(prompt)
    assert result.diagnostic_summary["token_channel_enabled"] is True
```

- [ ] **Step 2: Run the generation tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/test_run.py -v`
Expected: FAIL with missing token-channel wiring/config resolution.

- [ ] **Step 3: Load token-channel runtime in generator construction path**

```python
if config.token_channel.enabled:
    self._token_channel = TokenChannelRuntime.from_artifact(...)
```

- [ ] **Step 3a: Enforce tokenizer and feature-version compatibility at generator load time**

```python
runtime = TokenChannelRuntime.from_artifact(...)
runtime.validate_compatibility(tokenizer=self._tokenizer, feature_version=FEATURE_VERSION)
```

- [ ] **Step 4: Compute runtime structural features before each token decision**

```python
features = build_runtime_features(ctx, current_block_contract, token_index)
```

- [ ] **Step 5: Enforce `structure_mask` before any runtime token bias**

```python
if not features.structure_mask:
    skip_token_channel = True
```

- [ ] **Step 6: Apply token-channel bias in the token loop before simple-block acceptance**

```python
decision = self._token_channel.score_prefix(prefix_ids, features=features)
scores = apply_green_bias(scores, decision, config.token_channel)
```

- [ ] **Step 7: Add runtime protection rules from the spec**

```python
if block_token_count < config.token_channel.lexical_min_block_tokens:
    disable_token_channel = True
```

- [ ] **Step 7a: Add retry-decay behavior starting at failure count 2**

```python
if retry_count >= config.token_channel.lexical_retry_decay_start:
    current_delta = config.token_channel.delta * 0.5
```

- [ ] **Step 7b: Add retry-disable behavior at failure count 4**

```python
if retry_count >= config.token_channel.lexical_retry_disable_after:
    disable_token_channel = True
```

- [ ] **Step 7c: Add low-gate-fraction shutdown for the first 16 scorable tokens**

```python
if gate_open_fraction < config.token_channel.lexical_gate_min_fraction:
    disable_token_channel = True
```

- [ ] **Step 7d: Disable token-altering post-processing on token-channel outputs**

```python
if config.token_channel.enabled:
    postprocess_generation = False
```

- [ ] **Step 8: Add explicit `lexical-only` generation semantics**

```python
if config.token_channel.channel_mode == "lexical-only":
    skip_semantic_verification = True
    skip_semantic_retry = True
```

- [ ] **Step 9: Expose mode/config parsing in `run.py`, `configs/base_config.json`, and `tests/test_run_config.py`**

```json
"token_channel": {
  "enabled": false,
  "channel_mode": "semantic-only"
}
```

- [ ] **Step 10: Re-run the generation tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/test_run_config.py -v`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add wfcllm/watermark/generator.py wfcllm/watermark/pipeline.py run.py configs/base_config.json tests/watermark/test_generator.py tests/watermark/test_pipeline.py tests/test_run_config.py
git commit -m "feat: integrate token channel into generation"
```

## Task 7: Add Token-Channel Replay Detection and Joint Scoring

**Files:**
- Create: `wfcllm/extract/token_channel.py`
- Modify: `wfcllm/extract/detector.py`
- Modify: `wfcllm/extract/hypothesis.py`
- Modify: `wfcllm/extract/pipeline.py`
- Create: `tests/extract/test_token_channel.py`
- Create: `tests/extract/test_joint_detection.py`
- Modify: `tests/extract/test_detector.py`
- Modify: `tests/extract/test_pipeline.py`

- [ ] **Step 1: Write failing lexical detection tests**

```python
def test_token_channel_detector_replays_green_hits():
    result = detector.detect(code)
    assert result.lexical_result.num_positions_scored == 3
    assert result.lexical_result.z_score > 0
```

- [ ] **Step 2: Run the extraction tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py tests/extract/test_pipeline.py tests/extract/test_token_channel.py tests/extract/test_joint_detection.py -v`
Expected: FAIL with missing lexical result fields and detector logic.

- [ ] **Step 3: Implement lexical replay detector**

```python
for idx in range(prefix_len, len(token_ids)):
    features = rebuild_runtime_features_from_final_code(...)
    if not features.structure_mask:
        continue
    decision = runtime.score_prefix(token_ids[:idx], features=features)
    if decision.gate_open:
        ...
```

- [ ] **Step 3a: Enforce tokenizer and feature-version compatibility at extraction load time**

```python
runtime.validate_compatibility(tokenizer=self._tokenizer, feature_version=FEATURE_VERSION)
```

- [ ] **Step 3b: Ensure extraction replays the exact final tokenizer-visible code**

```python
code_for_detection = load_generated_code_without_token_altering_postprocess(...)
```

- [ ] **Step 4: Add repeat-filter handling for n-grams and prefixes**

```python
if config.ignore_repeated_prefixes and prefix_key in seen_prefixes:
    continue
if config.ignore_repeated_ngrams and ngram_key in seen_ngrams:
    continue
```

- [ ] **Step 5: Extend extraction result objects to carry semantic, lexical, and joint results**

```python
@dataclass
class LexicalDetectionResult:
    num_positions_scored: int
    num_green_hits: int
    green_fraction: float
    lexical_z_score: float
    lexical_p_value: float

@dataclass
class JointDetectionResult:
    semantic_z: float
    lexical_z: float
    joint_score: float
    p_joint: float
    prediction: bool
    confidence: float
    rationale: str
```

- [ ] **Step 6: Add explicit `lexical-only` extraction semantics**

```python
if token_channel_config.channel_mode == "lexical-only":
    semantic_result = None
    joint_result = lexical_result.to_joint_equivalent()
```

- [ ] **Step 7: Implement weighted fusion, p-value, confidence, and rationale labels**

```python
joint_score = semantic_weight * semantic_z + lexical_weight * lexical_support_factor * lexical_z
rationale = describe_joint_result(semantic_z=semantic_z, lexical_z=lexical_z)
```

- [ ] **Step 8: Extend details/summary pipeline output**

```python
row["lexical_z_score"] = result.lexical_result.z_score
row["joint_score"] = result.joint_result.joint_score
```

- [ ] **Step 9: Re-run the extraction tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py tests/extract/test_pipeline.py tests/extract/test_token_channel.py tests/extract/test_joint_detection.py -v`
Expected: PASS.

- [ ] **Step 10: Commit**

```bash
git add wfcllm/extract/token_channel.py wfcllm/extract/detector.py wfcllm/extract/hypothesis.py wfcllm/extract/pipeline.py tests/extract/test_detector.py tests/extract/test_pipeline.py tests/extract/test_token_channel.py tests/extract/test_joint_detection.py
git commit -m "feat: add token channel detection and fusion"
```

## Task 8: Add CLI, Documentation, and End-to-End Verification

**Files:**
- Modify: `README.md`
- Modify: `run.py`
- Modify: `configs/base_config.json`
- Modify: `tests/test_run.py`
- Modify: `docs/superpowers/specs/2026-04-10-dual-channel-strong-acw-design.md` only if implementation reveals required clarifications

- [ ] **Step 1: Write failing CLI/docs coverage tests**

```python
def test_run_parser_accepts_token_channel_flags():
    parser = build_parser()
    args = parser.parse_args(["--phase", "watermark", "--token-channel-enabled", "true"])
    assert args.token_channel_enabled is True
```

- [ ] **Step 2: Run the CLI tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -v`
Expected: FAIL with unrecognized token-channel arguments or missing config resolution.

- [ ] **Step 3: Add parser/config resolution for token-channel options**

```python
parser.add_argument("--token-channel-enabled", ...)
parser.add_argument("--token-channel-mode", ...)
```

- [ ] **Step 4: Update README with training, generation, and detection commands**

```markdown
HF_HUB_OFFLINE=1 conda run -n WFCLLM python -m wfcllm.watermark.token_channel.train ...
```

- [ ] **Step 5: Run focused smoke checks**

Run: `conda run -n WFCLLM python -m compileall wfcllm run.py scripts`
Expected: PASS.

- [ ] **Step 6: Run the targeted full test slice**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ tests/extract/ tests/test_run.py -v`
Expected: PASS.

- [ ] **Step 7: Run the project smoke command**

Run: `conda run -n WFCLLM python run.py --status`
Expected: PASS and print phase status.

- [ ] **Step 8: Commit**

```bash
git add README.md run.py configs/base_config.json tests/test_run.py
git commit -m "docs: add token channel usage and verification"
```

## Task 9: Build an Offline Evaluation Harness for Spec Thresholds

**Files:**
- Create: `wfcllm/common/offline_code_eval.py`
- Create: `scripts/evaluate_dual_channel.py`
- Create: `tests/extract/test_joint_detection.py` if not already created in Task 7
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-04-10-dual-channel-strong-acw-design.md` only if implementation reveals unavoidable threshold clarification needs

- [ ] **Step 1: Write a failing harness smoke test or dry-run expectation**

```python
def test_evaluate_dual_channel_builds_all_three_modes(tmp_path):
    result = run_evaluation(...)
    assert set(result) >= {"semantic-only", "lexical-only", "dual-channel"}
```

- [ ] **Step 2: Run the harness smoke check to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_joint_detection.py -k harness -v`
Expected: FAIL with missing evaluation harness.

- [ ] **Step 3: Implement `scripts/evaluate_dual_channel.py`**

```python
def run_evaluation(dataset: str, config_path: str, output_dir: str) -> dict[str, object]:
    ...
```

- [ ] **Step 4: Implement local correctness evaluation helpers for HumanEval and MBPP**

```python
def compute_pass_at_k(records: list[dict[str, object]], k: int) -> float:
    ...
```

- [ ] **Step 5: Make the harness compute the required artifacts**

```text
For semantic-only, lexical-only, and dual-channel: compute pass@1/pass@10 deltas, retry delta, latency delta, ROC AUC, TPR@1% FPR, and perturbation results.
```

- [ ] **Step 6: Make the harness call the real generation and extraction entrypoints**

```text
For each mode, run `python run.py --phase watermark ...` to produce JSONL, then run `python run.py --phase extract ...` to produce details and summary artifacts, then aggregate metrics from those artifacts.
```

- [ ] **Step 6a: Define the negative-score path for ROC and TPR/FPR metrics**

```text
Use `run.py --phase generate-negative` or an existing negative corpus JSONL to produce non-watermarked samples, then run extraction on that corpus in each mode so the harness has both positive and negative score distributions.
```

- [ ] **Step 7: Add offline evaluation helper commands to README draft notes**

```markdown
python scripts/evaluate_dual_channel.py --dataset humaneval --config configs/base_config.json
```

- [ ] **Step 8: Run semantic-only evaluation slice**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py tests/extract/test_joint_detection.py -k semantic -v`
Expected: PASS and produce baseline detector metrics fixtures.

- [ ] **Step 9: Run lexical-only evaluation slice**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py tests/extract/test_joint_detection.py -k lexical -v`
Expected: PASS and confirm lexical-only metric assertions.

- [ ] **Step 10: Run dual-channel evaluation slice**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_detector.py tests/extract/test_joint_detection.py -k joint -v`
Expected: PASS and confirm joint metric assertions.

- [ ] **Step 11: Run the real evaluation harness on the offline benchmark split**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM python scripts/evaluate_dual_channel.py --dataset humaneval --config configs/base_config.json --output-dir data/eval/dual_channel`
Expected: PASS and write per-mode metric artifacts for generation quality, retry/latency, ROC AUC, TPR@1% FPR, and perturbation checks.

- [ ] **Step 12: Run perturbation checks**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_joint_detection.py -k perturb -v`
Expected: PASS for formatting, comments, renaming, and light rewrite checks.

- [ ] **Step 13: Record whether spec thresholds are met**

```text
Capture pass@1/pass@10 deltas, retry delta, latency delta, lexical ROC AUC, lexical TPR@1% FPR, and joint uplift.
```

- [ ] **Step 14: Commit README or fixture updates if needed**

```bash
git add wfcllm/common/offline_code_eval.py scripts/evaluate_dual_channel.py README.md tests/extract/test_joint_detection.py
git commit -m "test: add dual-channel evaluation coverage"
```

## Task 10: Final Regression and Branch Finish

**Files:**
- Modify: only files touched by fixes from verification

- [ ] **Step 1: Run full offline test suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
Expected: PASS.

- [ ] **Step 2: Run compile smoke again**

Run: `conda run -n WFCLLM python -m compileall wfcllm run.py scripts`
Expected: PASS.

- [ ] **Step 3: Inspect git diff before handoff**

Run: `git status --short`
Expected: only intended token-channel implementation changes remain.

- [ ] **Step 4: Create final implementation commit(s) if fixes were needed**

```bash
git add <touched-files>
git commit -m "fix: resolve token channel regression issues"
```

- [ ] **Step 5: Prepare execution handoff note**

```text
Summarize semantic-only, lexical-only, and dual-channel verification results.
```
