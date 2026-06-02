# AGENTS.md

Guidance for agentic coding agents working in `WFCLLM`.
Audited against `CLAUDE.md`, `README.md`, `requirements.txt`,
`configs/base_config.json`, current `wfcllm/`, `scripts/`, `tools/`, and
`tests/` code on 2026-06-02.

## Scope and Priority

- This file applies to the whole repository rooted at `WFCLLM/`.
- Follow the root `CLAUDE.md` first. This file adds a more current project map
  and practical coding/test guidance for agents.
- If a user instruction conflicts with this file, follow the user instruction.
- Treat `experiment/` as reference-only legacy research code.
- Never import `experiment.*` from production code under `wfcllm/`.
- Legacy tests may still use `experiment/` as a comparison fixture; do not copy
  that pattern into production modules.
- If `code-watermark-main/` is present, treat it as external/reference code.
  Do not import it from `wfcllm/` unless the user explicitly asks for a
  deliberate migration/refactor.

## Repository Snapshot

- Main package: `wfcllm/`
- Thin CLI entrypoint: `run.py`, forwarding to `wfcllm.cli.entry:main`
- Main config: `configs/base_config.json`
- Other experiment configs: `configs/exp*.json`, `configs/humaneval_10_config.json`
- Ablation config example: `configs/ablation/example_dual_channel.yaml`
- Tests: `tests/` with about 100 `test_*.py` files across module groups
- Main docs: `README.md`, `CLAUDE.md`, `docs/design/`, `docs/plans/`,
  `docs/superpowers/`
- Supporting scripts: `scripts/`
- Download/debug helper tools: `tools/`
- Offline model root: `data/models/`
- Offline dataset root: `data/datasets/`
- Run state: `data/run_state.json` (generated and gitignored)

## Current Architecture

- `wfcllm.cli`
  - `arguments.py`: argparse surface for `run.py`
  - `entry.py`: main runtime wiring
  - `config_resolver.py`: JSON config loading, CLI override merging, token-channel
    and adaptive-gamma resolution
  - `runners.py`: CLI-bound phase runner functions
- `wfcllm.orchestration`
  - `state.py`: phase names and `RunStateManager`
  - `phase_registry.py`: phase-to-runner registry
  - `pipeline.py`: skip/force/prereq orchestration
  - `prereq.py`: import-time prereq registration framework
- `wfcllm.encoder`
  - Semantic encoder config, dataset construction, CodeT5/LoRA model,
    train/evaluate/trainer code
- `wfcllm.pretrain`
  - Optional pretrain phase that dispatches `encoder` then lexical
    token-channel training in canonical order
- `wfcllm.watermark`
  - `orchestrator.py`: `WatermarkGenerator` and generation-time semantic/token
    channel logic
  - `semantic_channel.py`, `retry_loop.py`, `cascade.py`, `context.py`,
    `interceptor.py`, `verifier.py`, `lsh_space.py`, `keying.py`
  - `pipeline.py`: dataset-level watermark JSONL writer and diagnostics sidecar
  - `adaptive_gamma/`: entropy profile building and piecewise-quantile scheduling
  - `token_channel/`: lexical channel config, protocol, features, runtime,
    model artifact helpers, and training workflow
- `wfcllm.extract`
  - `detector.py`, `scorer.py`, `hypothesis.py`, `dp_selector.py`,
    `alignment.py`, `pipeline.py`
  - `calibration/`: negative corpus generation and FPR threshold calibration
  - `token_channel.py`: replay detector for lexical channel evidence
- `wfcllm.datasets`
  - Adapter/registry layer for benchmark datasets
  - Implemented adapters: `humaneval`, `mbpp`
  - `humanevalpack` is a skeleton interface for future multilingual support
- `wfcllm.lang`
  - Adapter/registry layer for languages
  - Python is the implemented adapter
  - `cpp`, `java`, and `js` packages are placeholders
  - Python parser and transform rules live under `wfcllm/lang/python/`
- `wfcllm.common`
  - Shared typed registry, checkpoint/resume helpers, and canonical block
    contract helpers
- `wfcllm.evaluation`
  - Offline code execution, benchmark reports, detection regression reports,
    and dual-channel evaluation
- `wfcllm.ablation`
  - YAML sweep parsing, Cartesian expansion, sequential runner, and metric
    extraction

## CLI and Phase Contract

- `run.py` is intentionally a thin shim. Do not move business logic back into it.
- Phase source of truth is `wfcllm/orchestration/state.py`:
  - Main phases run by default: `encoder`, `watermark`, `extract`
  - Optional phases: `pretrain`, `generate-negative`, `token-channel-train`,
    `build-entropy-profile`
- `run.py --phase <name>` accepts every phase in `ALL_PHASES`.
- Running without `--phase` executes only the main three phases.
- `--status` and `--reset` use `RunStateManager` and `data/run_state.json`.
- Completed phases are skipped unless `--force` is set.
- Extract with an explicit input file bypasses the usual completed-phase skip.
- Compare-only extract mode requires all left/right summary/details and output
  flags; it is only valid with `--phase extract`.
- Config loading returns `{}` when a config file is missing, but invalid JSON
  exits with an error.
- CLI overrides are merged in `wfcllm.cli.config_resolver`; do not duplicate
  override logic in phase code.
- Explicit boolean CLI options use true/false-style strings through
  `parse_optional_bool`.
- Extract LSH parameters should prefer the first watermarked JSONL row's
  `watermark_params` over extract config values when present.

## Environment

- Use the Conda environment `WFCLLM`.
- Do not install dependencies or run project code from `base`.
- Install dependencies from `requirements.txt`.
- PyTorch/CUDA and `bitsandbytes` are intentionally not fully pinned; adapt them
  to the target server.
- Tests and normal development should assume offline execution.
- Prefer local resources over HuggingFace Hub access.
- For model/dataset-sensitive commands, set offline environment variables:
  `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1`
  when appropriate.

## Setup Commands

```bash
conda create -n WFCLLM python=3.11 -y
conda run -n WFCLLM pip install torch
conda run -n WFCLLM pip install -r requirements.txt
```

## Smoke Commands

There is no formal build system. No repo-managed `pyproject.toml`, `Makefile`,
`tox.ini`, `setup.cfg`, `pytest.ini`, `ruff`, `black`, `isort`, `flake8`,
`mypy`, `pyright`, or `pylint` config was found.

Use these non-invasive checks:

```bash
conda run -n WFCLLM python run.py --status
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Useful phase examples:

```bash
conda run -n WFCLLM python run.py
conda run -n WFCLLM python run.py --phase encoder
conda run -n WFCLLM python run.py --phase watermark --config configs/base_config.json
conda run -n WFCLLM python run.py --phase extract --config configs/base_config.json
conda run -n WFCLLM python run.py --phase generate-negative --config configs/base_config.json
conda run -n WFCLLM python run.py --phase token-channel-train --config configs/base_config.json
conda run -n WFCLLM python run.py --phase pretrain --stages encoder lexical
conda run -n WFCLLM python run.py --phase build-entropy-profile \
  --build-profile-input-log data/logs/watermark_debug.log \
  --build-profile-output data/calibration/humaneval_entropy_profile.json \
  --build-profile-language python \
  --build-profile-model-family deepseek-coder-7b-base
```

## Test Commands

Always include `HF_HUB_OFFLINE=1` with pytest commands unless the user explicitly
asks for online behavior.

Whole suite:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Useful targeted commands:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/common/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/datasets/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/encoder/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/token_channel/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/evaluation/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ablation/ -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/ -v
```

Examples for narrow changes:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_cli_resolution.py -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_orchestration.py -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark/test_pipeline.py::TestWatermarkPipelineRun::test_run_creates_jsonl -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_offline_analysis.py -k route_one -v
```

Useful pytest variants:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -q
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v --tb=short
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -x -v
```

For tests that instantiate the semantic encoder, prefer the local model path:

```python
EncoderConfig(model_name="data/models/codet5-base")
```

## Script Entry Points

- Prefer `scripts/evaluate.py` for unified offline evaluation:
  - `exec`: pass@k from candidate JSONL files
  - `detection`: regression report from saved summary/details artifacts
  - `dual`: dual-channel end-to-end harness
  - `bench`: Pass@1/Pass@10/AUROC benchmark runner
- `scripts/evaluate_dual_channel.py` is a deprecated wrapper around
  `wfcllm.evaluation.dual_channel`.
- Prefer `scripts/build_entropy_profile.py` and
  `scripts/calibrate_threshold.py` over deprecated `scripts/calibrate.py`.
- `scripts/generate_negative_corpus.py` remains available for direct negative
  corpus generation, but `run.py --phase generate-negative` is the preferred
  workflow entry.
- `scripts/run_ablation.py` runs YAML sweeps through `wfcllm.ablation`.
- `scripts/build_whitening.py` and `scripts/validate_whitening.py` operate on
  calibration/whitening artifacts.
- `tools/download_*` and `tools/verify_*` are local resource helpers.
- `tools/debug_extract_alignment.py` relies on `run.py` re-exporting
  `resolve_extract_lsh_params`; do not remove that re-export without updating
  the tool.

## Artifact Contracts

Preserve stable JSON/JSONL field names unless the user explicitly asks for a
schema migration and tests are updated.

- Watermark output JSONL rows are produced by `WatermarkPipeline` and include
  stable fields such as:
  `id`, `dataset`, `prompt`, `generated_code`, `blocks`, `adaptive_mode`,
  `profile_id`, `alignment_summary`, `total_blocks`, `embedded_blocks`,
  `failed_blocks`, `fallback_blocks`, `embed_rate`, `diagnostics_ledger_rows`,
  `watermark_params`, and optional `token_channel`.
- Watermark diagnostics sidecars are written beside watermarked outputs as
  `<watermarked_stem>_block_ledger.jsonl` under a diagnostics directory.
- `watermark_params` is public metadata. It must not contain `secret_key`.
- Extract details JSONL rows include fields such as:
  `id`, `mode`, `is_watermarked`, `z_score`, `p_value`,
  `independent_blocks`, `hits`, and optional semantic/lexical/joint/alignment
  fields.
- Extract summaries are written as `<stem>_summary.json` and include `meta` and
  `summary` sections.
- Negative corpus JSONL should be compatible with extraction/calibration and
  include `id`, `dataset`, `prompt`, and `generated_code`.
- Token-channel training artifacts live under `data/models/token-channel/` by
  default and use `model.pt`, `metadata.json`, `training_state.pt`, and
  `training_evidence.json`.
- Token-channel metadata schema is `token-channel/v1`; update compatibility
  tests if changing it.
- Block contracts are the embed/extract alignment contract. Preserve
  `ordinal`, `block_id`, `node_type`, `parent_node_type`, `block_text_hash`,
  `start_line`, `end_line`, `entropy_units`, `gamma_target`, `k`, and
  `gamma_effective` unless doing a deliberate schema migration.

## Code Style

- Prefer minimal, direct changes.
- Match the current module boundaries instead of creating parallel systems.
- Use `from __future__ import annotations` in library modules.
- Group imports as standard library, third-party, then local imports.
- Separate import groups with one blank line.
- Prefer explicit imports over wildcard imports.
- Use `pathlib.Path` for path handling.
- Use UTF-8 explicitly for file reads and writes.
- Use modern type hints such as `list[str]`, `dict[str, Any]`, and
  `str | None`.
- Add type hints to public functions, methods, and dataclass fields.
- Prefer dataclasses for config objects and result/value records.
- Use `@dataclass(frozen=True)` for immutable records when mutation is
  unnecessary.
- Use `__post_init__` to enforce dataclass invariants.
- Use snake_case for functions, variables, and module names.
- Use PascalCase for classes.
- Use UPPER_CASE for module-level constants.
- Keep identifiers and most docstrings in English to match the codebase.
- Chinese user-facing CLI text is acceptable where existing commands already
  use it.
- Avoid clever abstractions when a small local helper or inline logic is
  clearer.

## Error Handling and Logging

- Validate config and external input early.
- This codebase commonly raises `ValueError` for invalid config, malformed data,
  and incompatible artifacts.
- Chain parsing/conversion failures with `raise ... from exc`.
- Do not use bare `except:`.
- Avoid swallowing malformed JSON or structure mismatches unless nearby resume
  logic already treats them as ignorable.
- Preserve fail-fast behavior in pipeline/config code.
- Use `logging.getLogger(__name__)` for module loggers.
- Add logging only where it improves observability or debugging.
- Keep logs factual and low-noise.
- Do not leak secret material such as `secret_key` into public JSON/JSONL
  artifacts.

## Module-Specific Guidance

- CLI changes usually need updates in `wfcllm/cli/arguments.py`,
  `wfcllm/cli/config_resolver.py`, `wfcllm/cli/runners.py`,
  `wfcllm/orchestration/state.py`, integration tests, and README/AGENTS docs.
- New phases must update `ALL_PHASES`, phase registration in
  `wfcllm.cli.entry`, runner tests, and status/reset expectations.
- New config sections should be represented by dataclasses or focused resolver
  helpers, not ad hoc dict access spread across pipelines.
- Watermark generator changes should preserve tests that monkeypatch
  `WatermarkGenerator` internals, `ProjectionVerifier`, retry behavior, and
  cascade behavior.
- Token-channel runtime and extraction must stay tokenizer-compatible. Artifact
  metadata, tokenizer name/vocab size, context width, and feature version are
  compatibility inputs.
- Adaptive gamma changes must preserve embed/extract roundtrip behavior and
  block-contract alignment checks.
- Extraction should continue to prefer embedded metadata for LSH/adaptive
  parameters when available.
- Dataset additions should go through `wfcllm.datasets.adapter.DatasetAdapter`
  and registry side-effect imports in `wfcllm/datasets/__init__.py`.
- Language additions should go through `wfcllm.lang.adapter.LanguageAdapter`,
  registry side-effect imports, parser/block extraction tests, and transform
  rule tests.
- Evaluation code may run subprocesses for correctness tests. Keep timeouts and
  temp-file cleanup explicit.
- Ablation sweeps should remain resumable via `results.jsonl` and per-run
  config copies.

## Test Style

- Use `pytest`.
- Use `tmp_path` for filesystem-isolated tests.
- Use `pytest.fixture` for reusable setup.
- Use `monkeypatch`, `unittest.mock.patch`, and `MagicMock` for expensive
  model/dataset/external boundaries.
- Use `pytest.raises(..., match=...)` for validation/error tests.
- Use `pytest.approx(...)` for floating-point assertions.
- Keep test names in `test_<behavior>` style.
- Prefer updating nearby existing test files over adding scattered new files.
- Every public function/class should have corresponding tests unless it is a
  trivial import shim.
- For docs-only changes, tests are normally not required; verify the document
  content and diff instead.

## Data and Git Hygiene

- Check `git status --short` before editing. This repository may contain
  unrelated user work and untracked experiment outputs.
- Never revert user changes unless explicitly requested.
- Do not add large local resources unless the user explicitly asks.
- `data/models/`, `data/datasets/`, `data/checkpoints/`, `data/results/`,
  `data/run_state.json`, and `data/token_channel/` are gitignored.
- Many `data/watermarked*`, `data/diagnostics`, and `data/eval` artifacts may
  exist locally. Treat them as experiment artifacts; only commit them when they
  are intentionally tracked fixtures or user-requested outputs.
- `__pycache__/` and `*.pyc` are ignored and should not be committed.
- Commit format is `<type>: <description>`.
- Allowed commit types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.

## Practical Agent Workflow

- Read `CLAUDE.md` before editing.
- Inspect the target module and its nearby tests first.
- Confirm whether a change touches persisted artifact schemas.
- Make the smallest coherent change inside existing module boundaries.
- Add or update nearby tests for behavioral changes.
- Run the smallest relevant pytest target after a meaningful change.
- For broad or cross-module changes, run:
  `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
- For syntax-only smoke validation, run:
  `conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools`
- If full validation was not feasible, state exactly what was and was not run.
