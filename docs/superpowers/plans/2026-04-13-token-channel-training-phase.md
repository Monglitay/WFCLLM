# Token-Channel Training Phase Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an official offline `run.py --phase token-channel-train` workflow that builds the training cache, trains the token-channel model, exports the artifact, validates compatibility, and prints a usable summary.

**Architecture:** Keep `run.py` thin by adding an optional phase, a dedicated `token_channel_train` config section, and a workflow entrypoint under `wfcllm/watermark/token_channel/`. Reuse the existing token-channel helpers in `train_corpus.py`, `train.py`, and `model.py`, and cover the feature with offline-friendly routing, workflow, and failure-path tests.

**Tech Stack:** Python 3.11, existing `wfcllm` CLI/config pattern, PyTorch, local HuggingFace tokenizer/model loading, pytest, offline local datasets and model assets.

---

## File Map

### Create

- `wfcllm/watermark/token_channel/train_workflow.py`
- `tests/watermark/token_channel/test_train_workflow.py`
- `docs/superpowers/plans/2026-04-13-token-channel-training-phase.md`

### Modify

- `run.py`
- `configs/base_config.json`
- `README.md`
- `tests/test_run.py`

### Existing Files To Read Before Editing

- `docs/superpowers/specs/2026-04-13-token-channel-training-phase-design.md`
- `run.py`
- `configs/base_config.json`
- `wfcllm/common/dataset_loader.py`
- `wfcllm/watermark/token_channel/train.py`
- `wfcllm/watermark/token_channel/train_corpus.py`
- `wfcllm/watermark/token_channel/model.py`
- `tests/test_run.py`
- `tests/watermark/token_channel/test_train_corpus.py`
- `README.md`

## Task 1: Add `run.py` Phase Routing and Config Surface

**Files:**
- Modify: `run.py`
- Modify: `configs/base_config.json`
- Test: `tests/test_run.py`

- [ ] **Step 1: Write the failing `run.py` routing tests**

```python
def test_run_accepts_token_channel_train_phase(monkeypatch, tmp_path):
    from run import main

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "token_channel_train": {
                    "dataset": "humaneval",
                    "dataset_path": "data/datasets",
                    "lm_model_path": "data/models/deepseek-coder-7b-base",
                    "model_path": "data/models/token-channel",
                    "cache_path": "data/token_channel/train_corpus.json",
                    "context_width": 128,
                    "hidden_size": 64,
                    "batch_size": 128,
                    "epochs": 3,
                    "lr": 0.001,
                    "entropy_threshold": 1.0,
                    "diversity_threshold": 2,
                    "split_ratio": 0.9,
                    "seed": 0,
                }
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    called = {}

    def fake_runner(args, state):
        called["phase"] = args.phase
        called["dataset"] = args.dataset
        return 0

    monkeypatch.setattr("run.run_token_channel_train", fake_runner)

    rc = main([
        "--config", str(config_path),
        "--phase", "token-channel-train",
        "--dataset", "humaneval",
        "--lm-model-path", "data/models/deepseek-coder-7b-base",
    ])

    assert rc == 0
    assert called == {"phase": "token-channel-train", "dataset": "humaneval"}
```

- [ ] **Step 2: Run the routing tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -k token_channel_train -v`
Expected: FAIL because `token-channel-train` is not yet a supported phase/runner.

- [ ] **Step 3: Add the new optional phase and runner slot in `run.py`**

```python
PHASES = ["encoder", "watermark", "extract"]
OPTIONAL_PHASES = ["generate-negative", "token-channel-train"]


def run_phase(phase: str, args: argparse.Namespace, state: RunState) -> int:
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
        "generate-negative": run_generate_negative,
        "token-channel-train": run_token_channel_train,
    }
    return runners[phase](args, state)
```

- [ ] **Step 4: Add CLI arguments and config loading for the new phase**

```python
parser.add_argument("--token-channel-cache-path", type=str)
parser.add_argument("--token-channel-model-path", type=str)
parser.add_argument("--token-channel-context-width", type=int)
parser.add_argument("--token-channel-hidden-size", type=int)
parser.add_argument("--token-channel-batch-size", type=int)
parser.add_argument("--token-channel-epochs", type=int)
parser.add_argument("--token-channel-lr", type=float)
parser.add_argument("--token-channel-entropy-threshold", type=float)
parser.add_argument("--token-channel-diversity-threshold", type=int)
parser.add_argument("--token-channel-split-ratio", type=float)
parser.add_argument("--token-channel-seed", type=int)
```

- [ ] **Step 5: Add a dedicated `token_channel_train` section to `configs/base_config.json`**

```json
"token_channel_train": {
  "dataset": "humaneval",
  "dataset_path": "data/datasets",
  "lm_model_path": "data/models/deepseek-coder-7b-base",
  "model_path": "data/models/token-channel",
  "cache_path": "data/token_channel/train_corpus.json",
  "context_width": 128,
  "hidden_size": 64,
  "batch_size": 128,
  "epochs": 3,
  "lr": 0.001,
  "entropy_threshold": 1.0,
  "diversity_threshold": 2,
  "split_ratio": 0.9,
  "seed": 0
}
```

- [ ] **Step 6: Re-run the routing tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -k token_channel_train -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add run.py configs/base_config.json tests/test_run.py
git commit -m "feat: add token channel train phase routing"
```

## Task 2: Add the Workflow Config and Summary API

**Files:**
- Create: `wfcllm/watermark/token_channel/train_workflow.py`
- Test: `tests/watermark/token_channel/test_train_workflow.py`

- [ ] **Step 1: Write the failing workflow config tests**

```python
def test_workflow_config_rejects_invalid_split_ratio():
    with pytest.raises(ValueError, match="split_ratio"):
        TokenChannelTrainWorkflowConfig(
            dataset="humaneval",
            dataset_path="data/datasets",
            lm_model_path="data/models/deepseek-coder-7b-base",
            model_path="data/models/token-channel",
            cache_path="data/token_channel/train_corpus.json",
            context_width=128,
            hidden_size=64,
            batch_size=128,
            epochs=3,
            lr=0.001,
            entropy_threshold=1.0,
            diversity_threshold=2,
            split_ratio=1.0,
            seed=0,
        )
```

- [ ] **Step 2: Run the workflow config tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py -k config -v`
Expected: FAIL because the workflow module and config dataclass do not exist yet.

- [ ] **Step 3: Add the workflow config and summary dataclasses**

```python
@dataclass(frozen=True)
class TokenChannelTrainWorkflowConfig:
    dataset: str
    dataset_path: str
    lm_model_path: str
    model_path: str
    cache_path: str
    context_width: int
    hidden_size: int
    batch_size: int
    epochs: int
    lr: float
    entropy_threshold: float
    diversity_threshold: int
    split_ratio: float = 0.9
    seed: int = 0

    def __post_init__(self) -> None:
        if self.dataset not in {"humaneval", "mbpp"}:
            raise ValueError("dataset must be one of: humaneval, mbpp")
        if not Path(self.lm_model_path).exists():
            raise ValueError("lm_model_path must exist")
        if self.context_width <= 0:
            raise ValueError("context_width must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be >= 0")
        if self.diversity_threshold < 1:
            raise ValueError("diversity_threshold must be >= 1")
        if not 0 < self.split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1")
```

- [ ] **Step 4: Add helpers that normalize config inputs and summarize results**

```python
@dataclass(frozen=True)
class TokenChannelTrainWorkflowSummary:
    dataset: str
    training_rows: int
    train_rows: int
    validation_rows: int
    artifact_dir: str
    cache_path: str
    compatibility_ok: bool
    epochs: tuple[TokenChannelEpochMetrics, ...]
    switch_target_positive_count: int
    switch_target_negative_count: int
```

- [ ] **Step 5: Add unit tests for config validation and summary formatting helpers**

```python
def test_summary_to_lines_includes_epoch_metrics():
    summary = TokenChannelTrainWorkflowSummary(...)
    lines = format_token_channel_training_summary(summary)
    assert any("epoch=1" in line for line in lines)
    assert any("compatibility: ok" in line for line in lines)


@pytest.mark.parametrize(
    ("field_name", "value", "match"),
    [
        ("context_width", 0, "context_width"),
        ("hidden_size", 0, "hidden_size"),
        ("batch_size", 0, "batch_size"),
        ("epochs", 0, "epochs"),
        ("lr", 0.0, "lr"),
        ("entropy_threshold", -1.0, "entropy_threshold"),
        ("diversity_threshold", 0, "diversity_threshold"),
    ],
)
def test_workflow_config_rejects_invalid_values(field_name, value, match):
    kwargs = valid_workflow_config_kwargs(tmp_path)
    kwargs[field_name] = value
    with pytest.raises(ValueError, match=match):
        TokenChannelTrainWorkflowConfig(**kwargs)
```

- [ ] **Step 6: Re-run the workflow config tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py -k "config or summary" -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add wfcllm/watermark/token_channel/train_workflow.py tests/watermark/token_channel/test_train_workflow.py
git commit -m "feat: add token channel training workflow config"
```

## Task 3: Implement Corpus Normalization, Training, Export, and Validation Workflow

**Files:**
- Modify: `wfcllm/watermark/token_channel/train_workflow.py`
- Test: `tests/watermark/token_channel/test_train_workflow.py`

- [ ] **Step 1: Write the failing end-to-end workflow orchestration tests**

```python
def test_workflow_builds_cache_trains_exports_and_validates(monkeypatch, tmp_path):
    config = TokenChannelTrainWorkflowConfig(...)

    monkeypatch.setattr(
        "wfcllm.watermark.token_channel.train_workflow.load_reference_solutions",
        lambda dataset, dataset_path: [{"generated_code": "def f():\n    return 1\n"}, {"generated_code": "def g():\n    return 2\n"}],
    )
    monkeypatch.setattr(
        "wfcllm.watermark.token_channel.train_workflow.build_training_rows",
        lambda **kwargs: [row_fixture_one, row_fixture_two],
    )

    summary = run_token_channel_training_workflow(config)

    assert summary.training_rows == 2
    assert summary.compatibility_ok is True
```

- [ ] **Step 2: Run the workflow orchestration tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py -k workflow -v`
Expected: FAIL because the workflow entrypoint does not yet orchestrate the full flow.

- [ ] **Step 3: Implement dataset-row normalization into token-channel samples**

```python
def _normalize_reference_solution_rows(rows: list[dict[str, object]]) -> list[dict[str, str]]:
    samples: list[dict[str, str]] = []
    for row in rows:
        source_code = row.get("generated_code")
        if not isinstance(source_code, str) or not source_code:
            raise ValueError("reference solution row must contain non-empty generated_code")
        samples.append({"source_code": source_code})
    return samples
```

- [ ] **Step 4: Implement deterministic split logic and tiny-dataset validation**

```python
def _split_training_rows(rows: list[dict[str, object]], split_ratio: float, seed: int) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(rows) < 2:
        raise ValueError("token-channel training requires at least 2 rows")
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_index = min(len(shuffled) - 1, max(1, int(len(shuffled) * split_ratio)))
    return shuffled[:split_index], shuffled[split_index:]
```

- [ ] **Step 5: Implement the full workflow body using existing helpers**

```python
def run_token_channel_training_workflow(config: TokenChannelTrainWorkflowConfig) -> TokenChannelTrainWorkflowSummary:
    dataset_rows = load_reference_solutions(config.dataset, config.dataset_path)
    samples = _normalize_reference_solution_rows(dataset_rows)
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_path)
    teacher_model = _load_teacher_model(config.lm_model_path)
    training_rows = build_training_rows(...)
    Path(config.cache_path).parent.mkdir(parents=True, exist_ok=True)
    save_training_cache(config.cache_path, training_rows)
    train_rows, validation_rows = _split_training_rows(training_rows, config.split_ratio, config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TokenChannelModel(
        vocab_size=len(tokenizer),
        context_width=config.context_width,
        hidden_size=config.hidden_size,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    def iter_batches(rows_subset: list[dict[str, object]]):
        for index in range(0, len(rows_subset), config.batch_size):
            batch_rows = rows_subset[index:index + config.batch_size]
            if batch_rows:
                yield build_token_channel_batch(
                    batch_rows,
                    context_width=config.context_width,
                    device=device,
                )

    epoch_metrics: list[TokenChannelEpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        random.Random(config.seed + epoch).shuffle(train_rows)
        epoch_metrics.append(
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_batches=iter_batches(train_rows),
                validation_batches=iter_batches(validation_rows),
                epoch=epoch,
            )
        )

    evidence = build_training_evidence(rows=training_rows, epochs=epoch_metrics)
    metadata = {
        "schema_version": "token-channel/v1",
        "tokenizer_name": tokenizer.name_or_path,
        "tokenizer_vocab_size": len(tokenizer),
        "context_width": config.context_width,
        "feature_version": FEATURE_VERSION,
        "training_config": {
            "dataset": config.dataset,
            "dataset_path": config.dataset_path,
            "cache_path": config.cache_path,
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "entropy_threshold": config.entropy_threshold,
            "diversity_threshold": config.diversity_threshold,
            "split_ratio": config.split_ratio,
            "seed": config.seed,
        },
    }

    Path(config.model_path).mkdir(parents=True, exist_ok=True)
    paths = save_token_channel_training_artifacts(...)
    artifact = load_token_channel_artifact(config.model_path)
    require_token_channel_compatibility(...)
    return TokenChannelTrainWorkflowSummary(...)
```

- [ ] **Step 6: Add a focused metadata test before artifact reload validation**

```python
def test_build_artifact_metadata_contains_required_compatibility_fields(tmp_path):
    config = TokenChannelTrainWorkflowConfig(**valid_workflow_config_kwargs(tmp_path))
    metadata = build_token_channel_training_metadata(config=config, tokenizer_name="local/model", tokenizer_vocab_size=32000)
    assert metadata["schema_version"] == "token-channel/v1"
    assert metadata["tokenizer_name"] == "local/model"
    assert metadata["tokenizer_vocab_size"] == 32000
    assert metadata["context_width"] == config.context_width
    assert metadata["feature_version"] == FEATURE_VERSION
    assert metadata["training_config"]["model_path"] == config.model_path
```

- [ ] **Step 7: Add failure-path tests for empty datasets, malformed rows, and compatibility failure**

```python
def test_workflow_rejects_reference_row_without_generated_code():
    with pytest.raises(ValueError, match="generated_code"):
        _normalize_reference_solution_rows([{}])


def test_workflow_rejects_single_training_row(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "wfcllm.watermark.token_channel.train_workflow.build_training_rows",
        lambda **kwargs: [row_fixture_one],
    )
    config = TokenChannelTrainWorkflowConfig(**valid_workflow_config_kwargs(tmp_path))
    with pytest.raises(ValueError, match="at least 2 rows"):
        run_token_channel_training_workflow(config)
```

- [ ] **Step 8: Re-run the workflow tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/test_train_workflow.py -v`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add wfcllm/watermark/token_channel/train_workflow.py tests/watermark/token_channel/test_train_workflow.py
git commit -m "feat: implement token channel training workflow"
```

## Task 4: Wire the Workflow into `run.py` Status, State, and CLI Output

**Files:**
- Modify: `run.py`
- Modify: `tests/test_run.py`

- [ ] **Step 1: Write the failing `run_state` and summary output tests**

```python
def test_token_channel_train_marks_run_state_with_paths(monkeypatch, tmp_path):
    state = RunState(tmp_path / "run_state.json")

    monkeypatch.setattr(
        "run.run_token_channel_training_workflow",
        lambda config: TokenChannelTrainWorkflowSummary(
            dataset="humaneval",
            training_rows=20,
            train_rows=18,
            validation_rows=2,
            artifact_dir="data/models/token-channel",
            cache_path="data/token_channel/train_corpus.json",
            compatibility_ok=True,
            epochs=(TokenChannelEpochMetrics(epoch=1, train_loss=1.0, validation_loss=0.9, switch_loss=0.2),),
            switch_target_positive_count=11,
            switch_target_negative_count=9,
        ),
    )

    args = build_args_for_token_channel_train(...)
    rc = run_token_channel_train(args, state)

    assert rc == 0
    assert state.get("token-channel-train", "artifact_dir") == "data/models/token-channel"
```

- [ ] **Step 2: Run the `run.py` integration tests to verify failure**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -k "token_channel_train and (state or summary or status)" -v`
Expected: FAIL because the runner does not yet build config, print the summary, or record state metadata.

- [ ] **Step 3: Add a `run_token_channel_train()` runner that builds workflow config from config + CLI overrides**

```python
def run_token_channel_train(args: argparse.Namespace, state: RunState) -> int:
    cfg = load_config(args.config)
    section = cfg.get("token_channel_train", {})
    workflow_config = resolve_token_channel_train_config(section, args)
    summary = run_token_channel_training_workflow(workflow_config)
    for line in format_token_channel_training_summary(summary):
        print(line)
    state.mark_done(
        "token-channel-train",
        dataset=summary.dataset,
        cache_path=summary.cache_path,
        artifact_dir=summary.artifact_dir,
    )
    return 0
```

- [ ] **Step 4: Add tests for config precedence, prefixed flag handling, and model-path overrides**

```python
def test_token_channel_train_cli_overrides_config_values():
    config = resolve_token_channel_train_config(
        {"epochs": 3, "batch_size": 128, "model_path": "data/models/token-channel"},
        argparse.Namespace(
            token_channel_epochs=5,
            token_channel_batch_size=64,
            token_channel_model_path="data/models/token-channel-alt",
            ...,
        ),
    )
    assert config.epochs == 5
    assert config.batch_size == 64
    assert config.model_path == "data/models/token-channel-alt"
```

- [ ] **Step 5: Re-run the `run.py` token-channel tests**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py -k token_channel_train -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add run.py tests/test_run.py
git commit -m "feat: wire token channel training into run cli"
```

## Task 5: Document the New Workflow and Run Focused Validation

**Files:**
- Modify: `README.md`
- Modify: `run.py`
- Modify: `tests/test_run.py`
- Modify: `tests/watermark/token_channel/test_train_workflow.py`

- [ ] **Step 1: Write the failing documentation-adjacent test if needed, otherwise add focused assertions to existing CLI tests**

```python
def test_token_channel_train_summary_mentions_artifact_and_cache_paths(...):
    ...
    assert "data/models/token-channel" in captured.out
    assert "data/token_channel/train_corpus.json" in captured.out


def test_token_channel_train_reports_overwrite_targets(tmp_path, monkeypatch, capsys):
    cache_path = tmp_path / "train_corpus.json"
    model_dir = tmp_path / "token-channel"
    cache_path.write_text("{}", encoding="utf-8")
    model_dir.mkdir()
    monkeypatch.setattr("run.run_token_channel_training_workflow", lambda config: summary_fixture(cache_path, model_dir))
    ...
    assert str(cache_path) in captured.out
    assert str(model_dir) in captured.out
```

- [ ] **Step 2: Update `run.py` to print overwrite notices before workflow execution when cache or artifact paths already exist**

```python
if Path(workflow_config.cache_path).exists():
    print(f"[提示] 将覆盖训练缓存：{workflow_config.cache_path}")
if Path(workflow_config.model_path).exists():
    print(f"[提示] 将覆盖 token-channel 产物目录：{workflow_config.model_path}")
```

- [ ] **Step 3: Update `README.md` to document the official workflow**

```markdown
### Token-Channel Training Command

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py \
    --phase token-channel-train \
    --config configs/base_config.json \
    --dataset humaneval \
    --lm-model-path data/models/deepseek-coder-7b-base
```

This phase rebuilds `data/token_channel/train_corpus.json`, trains the token-channel model, writes `model.pt`, `metadata.json`, and `training_evidence.json`, then reloads the artifact and validates compatibility.
```

- [ ] **Step 4: Run the focused validation suite**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/watermark/token_channel/test_train_workflow.py -v`
Expected: PASS.

- [ ] **Step 5: Run the syntax smoke check**

Run: `conda run -n WFCLLM python -m compileall wfcllm run.py`
Expected: PASS with no syntax errors.

- [ ] **Step 6: Run the broader offline regression slice for touched areas**

Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/ tests/test_run.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add README.md run.py tests/test_run.py tests/watermark/token_channel/test_train_workflow.py
git commit -m "docs: document token channel training workflow"
```

## Final Verification

- [ ] Run: `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run.py tests/watermark/token_channel/test_train_workflow.py tests/watermark/token_channel/ -v`
Expected: PASS.

- [ ] Run: `conda run -n WFCLLM python -m compileall wfcllm run.py`
Expected: PASS.

- [ ] Confirm the documented command shape still matches the implemented flags in `run.py` and the defaults in `configs/base_config.json`.
