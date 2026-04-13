# Token-Channel Training Phase Design

**Date:** 2026-04-13  
**Author:** OpenCode  
**Status:** Design approved, pending implementation

## 1. Overview

This design adds a first-class `token-channel-train` phase to `run.py` so the repository can produce a compatible `data/models/token-channel` artifact through an officially supported offline workflow.

`token-channel-train` is an explicit opt-in phase. It is not part of the default no-argument `python run.py` pipeline.

The repository already contains the core building blocks for token-channel training:

- corpus construction in `wfcllm/watermark/token_channel/train_corpus.py`
- model training helpers in `wfcllm/watermark/token_channel/train.py`
- artifact export and compatibility validation in `wfcllm/watermark/token_channel/model.py`

What is missing today is the workflow layer that connects these pieces into a single supported command. `README.md` explicitly notes that the current token-channel CLI only loads cached assets and does not run the complete training/export flow.

This design fills that gap without changing the underlying token-channel algorithm.

## 2. Goals and Non-Goals

### 2.1 Goals

- Add a new `run.py` phase for official token-channel training.
- Support a full offline workflow in one command:
  - load reference solutions from a supported local dataset
  - build the training corpus cache
  - train the token-channel model
  - export the artifact bundle
  - reload and validate the exported artifact
- Keep `run.py` as the user-facing entrypoint.
- Reuse existing token-channel training helpers instead of re-implementing them.
- Expose the common training hyperparameters through CLI arguments.
- Print a concise training summary that confirms whether the artifact is ready to use.

### 2.2 Non-Goals

- V1 does not redesign the token-channel model architecture.
- V1 does not add support for arbitrary JSONL or custom free-form corpora.
- V1 does not add network-backed dataset or model fetching.
- V1 does not introduce experiment tracking, checkpoint resumption, or multi-run registry management.
- V1 does not replace the existing module-level training helpers as internal primitives.

## 3. Confirmed Product Decisions

- The official entrypoint is `run.py`, not `python -m wfcllm.watermark.token_channel.train`.
- The new phase name is `token-channel-train`.
- The new phase is optional, like `generate-negative`, rather than a new default main pipeline stage.
- V1 supports repository-standard offline datasets only, starting with the same reference-solution loading path already used elsewhere.
- V1 covers the full workflow, not only the final training loop.
- V1 exposes common tuning knobs on the CLI, including `context_width`, `hidden_size`, `batch_size`, `epochs`, `lr`, `entropy_threshold`, and `diversity_threshold`.
- Successful completion means three things happen in one run:
  - artifact files are written
  - artifact reload and compatibility validation succeed
  - a readable summary is printed

## 4. Options Considered

### 4.1 Recommended: thin `run.py` phase plus workflow orchestrator

Add a new phase to `run.py`, then delegate the actual workflow orchestration to a dedicated token-channel workflow module under `wfcllm/watermark/token_channel/`.

**Pros**

- Matches the repository's unified CLI direction.
- Keeps orchestration separate from training primitives.
- Minimizes code duplication.
- Keeps future token-channel features localized.

**Cons**

- Requires one more integration layer.

### 4.2 Alternative: place the whole workflow directly in `run.py`

Implement dataset loading, cache construction, model training, export, and validation inline in `run.py`.

**Pros**

- Very direct entrypoint.

**Cons**

- Bloats `run.py`.
- Mixes command routing with training details.
- Makes testing and later extension harder.

### 4.3 Alternative: make the module CLI primary and let `run.py` proxy it

Turn `wfcllm.watermark.token_channel.train` into the true complete workflow CLI and let `run.py` wrap it.

**Pros**

- Keeps token-channel logic self-contained.

**Cons**

- Conflicts with the chosen single-entrypoint user experience.
- Creates two mental models for the same operation.

### 4.4 Decision

Adopt option 4.1. `run.py` remains the official interface, while orchestration stays in the token-channel package.

## 5. Architecture

### 5.1 Top-Level Structure

The implementation is divided into three layers:

1. `run.py` phase routing and CLI/config normalization
2. token-channel workflow orchestration
3. existing token-channel training and artifact helpers

This keeps user input handling, business workflow, and model-specific helpers separate.

### 5.2 Responsibilities by Module

**`run.py`**

- Registers `token-channel-train` as an optional phase that is available through `--phase`, visible in status output, and tracked in `run_state`.
- Adds `token-channel-train` to the supported phase list.
- Parses and validates the user-facing arguments for this phase.
- Converts raw CLI values into a dedicated workflow config object.
- Calls the workflow entrypoint and reports final success/failure.

On success, `run_state` should record enough metadata to make `--status` useful, at minimum the dataset name, cache path, and artifact directory path.

The phase follows the repository's existing `--config` pattern. The implementation should load `configs/base_config.json` by default, read a dedicated top-level `token_channel_train` config section, and then apply explicit CLI overrides on top.

**`wfcllm/watermark/token_channel/train_workflow.py`**

- Owns the full end-to-end orchestration.
- Loads reference solutions from the supported offline dataset.
- Adapts dataset rows into the sample contract expected by token-channel corpus builders.
- Instantiates the tokenizer and teacher model from `lm-model-path`.
- Builds and saves the training corpus cache.
- Trains the token-channel model using existing helper functions.
- Exports the model artifact and evidence files.
- Reloads the artifact and runs compatibility validation.
- Returns a structured summary object for CLI printing.

**Existing helper modules**

- `train_corpus.py` remains responsible for building training rows.
- `train.py` remains responsible for batching, epochs, and artifact persistence helpers.
- `model.py` remains responsible for metadata validation, export, reload, and compatibility checks.

## 6. CLI and Configuration Surface

### 6.1 Command Shape

The supported command shape is:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM python run.py \
    --config configs/base_config.json \
    --phase token-channel-train \
    --dataset humaneval \
    --dataset-path data/datasets \
    --lm-model-path data/models/deepseek-coder-7b-base \
    --token-channel-model-path data/models/token-channel
```

The phase must participate in the same config-loading flow already used by the existing phases.

### 6.2 Required Inputs

- `--phase token-channel-train`
- `--dataset` unless provided in config
- `--lm-model-path` unless provided in config

The dataset must be one of the locally supported datasets already handled by repository dataset loaders.

The new config surface is a dedicated top-level section:

```json
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
    "seed": 0
  }
}
```

CLI values override config values using the same precedence model already present elsewhere in `run.py`.

### 6.3 Optional Inputs

- The token-channel training phase uses prefixed flags such as `--token-channel-lr`, `--token-channel-batch-size`, and `--token-channel-epochs` to avoid conflicting with the repository's existing encoder-oriented unprefixed training flags.
- `--dataset-path` default: `data/datasets`
- `--token-channel-model-path` default: `data/models/token-channel`
- `--token-channel-cache-path` default: `data/token_channel/train_corpus.json`
- `--token-channel-context-width`
- `--token-channel-hidden-size`
- `--token-channel-batch-size`
- `--token-channel-epochs`
- `--token-channel-lr`
- `--token-channel-entropy-threshold`
- `--token-channel-diversity-threshold`
- `--token-channel-split-ratio`
- `--token-channel-seed`

### 6.4 Default Behavior

- The workflow always rebuilds the training cache before training.
- Existing cache and artifact files may be overwritten deliberately.
- The CLI must state when it is overwriting a previously existing cache or artifact path.
- Validation runs automatically after export.
- A successful run ends with both written artifacts and a printed summary.

The deliberate cache rebuild is important because the artifact must match the tokenizer actually used for generation. Silent reuse of an old cache would make that easier to get wrong.

## 7. Workflow Data Flow

### 7.1 Step 1: load reference solutions

The workflow loads reference solutions from the chosen local dataset using `wfcllm.common.dataset_loader.load_reference_solutions()`. V1 does not add a new corpus format.

Because the existing dataset loader returns rows with fields such as `generated_code`, while token-channel corpus construction expects samples containing `source_code`, the workflow must include a dedicated normalization step before corpus building.

### 7.2 Step 2: build training rows

The workflow constructs the teacher tokenizer and teacher model from `lm-model-path`, then feeds normalized samples into `build_training_rows()`.

The normalization contract is explicit in V1:

- input dataset row: existing reference-solution row from the dataset loader
- workflow sample passed into corpus construction: `{"source_code": row["generated_code"]}`

If a loaded row does not contain a usable `generated_code` string, the workflow fails instead of silently skipping it.

The resulting rows are written to `train_corpus.json` through the existing cache persistence helpers.

### 7.3 Step 3: train model

The workflow reads the cached rows, shuffles them deterministically, performs a train/validation split, builds batches using the existing helper, and trains `TokenChannelModel` for the configured number of epochs.

The split contract is fixed in V1:

- default `split_ratio` is `0.9`, meaning 90% train and 10% validation
- default seed is `0`
- shuffling uses that seed before splitting
- at least 2 rows are required; otherwise the workflow fails
- splits must remain disjoint
- train rows are `rows[:split_index]`
- validation rows are `rows[split_index:]`
- `split_index` is `min(len(rows) - 1, max(1, int(len(rows) * split_ratio)))`

This gives planning and test work a concrete deterministic contract without allowing overlapping train and validation samples.

Per-epoch metrics are collected and transformed into training evidence.

### 7.4 Step 4: export artifact bundle

The workflow writes the following files into the artifact directory:

- `model.pt`
- `metadata.json`
- `training_evidence.json`

Metadata must include tokenizer identity, tokenizer vocabulary size, context width, feature version, and a training-config block with the relevant training parameters.

### 7.5 Step 5: reload and validate artifact

After export, the workflow immediately reloads the artifact through the existing artifact loader and validates compatibility against the tokenizer and workflow settings used during training.

If this reload or compatibility validation fails, the phase fails.

## 8. Outputs and Summary

### 8.1 Files

On success, the phase writes:

- the training corpus cache at the configured cache path
- the model checkpoint at `<artifact_dir>/model.pt`
- the metadata file at `<artifact_dir>/metadata.json`
- the evidence file at `<artifact_dir>/training_evidence.json`

### 8.2 CLI Summary

The final CLI summary should include:

- dataset name
- total number of training rows
- train/validation split sizes
- switch target positive and negative counts across all persisted training rows
- per-epoch train loss, validation loss, and switch loss
- artifact directory path
- compatibility validation result

The summary is not meant to be a full experiment report. Its job is to confirm that the workflow produced a usable artifact and to surface obviously bad training runs.

V1 should print metrics for every epoch, because the evidence file is already stored per epoch and the configured epoch count is expected to remain small.

## 9. Error Handling

### 9.1 Configuration Validation

The workflow fails early for invalid configuration, including:

- unsupported dataset name
- missing or unreadable `lm-model-path`
- non-positive `context_width`
- non-positive `hidden_size`
- non-positive `batch_size`
- non-positive `epochs`
- non-positive `lr`
- `split_ratio` outside `(0, 1)`
- negative `entropy_threshold`
- `diversity_threshold < 1`

### 9.2 Data Validation

The workflow fails fast when:

- the dataset loads no reference solutions
- corpus construction produces no training rows
- the validation split is empty
- required row fields are malformed

### 9.3 Artifact Validation

The workflow fails if:

- artifact export does not write the required files
- artifact reload fails
- metadata validation fails
- tokenizer name, tokenizer vocab size, context width, or feature version do not match

The CLI must not report success if the artifact exists but is unusable.

## 10. Testing Strategy

### 10.1 `run.py` integration tests

- verify that `token-channel-train` is accepted as a valid phase
- verify that CLI arguments are routed into the workflow config correctly
- verify that the phase prints a success summary when the workflow returns success

### 10.2 Workflow unit tests

- mock dataset loading, teacher model creation, tokenizer creation, and artifact save/load boundaries
- verify the workflow executes the expected steps in order
- verify metadata contains the expected compatibility fields
- verify the summary object includes the expected metrics and paths

### 10.3 Failure-path tests

- empty dataset
- empty training rows
- empty validation split
- artifact reload failure
- compatibility mismatch after export

### 10.4 Validation philosophy

Tests should remain offline and lightweight. V1 should not add expensive end-to-end model-training tests with large real models. Small fixtures and mocks are sufficient for workflow coverage.

## 11. Implementation Notes

- Prefer a dedicated workflow config dataclass rather than passing raw CLI dictionaries through the stack.
- Keep `run.py` thin; do not move training math or corpus logic into it.
- Reuse existing token-channel helper functions wherever possible.
- Preserve current metadata schema and compatibility rules.
- Keep all path reads and writes UTF-8 where JSON is involved.

## 12. Acceptance Criteria

This design is considered implemented when all of the following are true:

- `run.py` supports `--phase token-channel-train`
- one command can build the cache, train the model, export the artifact, and validate the result
- the workflow operates fully offline against local datasets and local model assets
- the exported artifact is compatible with the tokenizer and settings used during training
- the CLI prints a concise success summary
- relevant tests cover both success and failure paths
