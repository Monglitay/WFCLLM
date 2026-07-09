# AGENTS.md

Guidance for agentic coding agents working in `WFCLLM`.
Audited against `CLAUDE.md`, `README.md`, `configs/base_config.json`,
`configs/wfcllm/`, current `wfcllm/`, `scripts/`, `tools/`, and `tests/`
on 2026-07-09.

## Scope And Priority

- This file applies to the whole repository rooted at `WFCLLM/`.
- Follow the root `CLAUDE.md` first. This file adds practical coding,
  testing, and artifact guidance for agents.
- If a user instruction conflicts with this file, follow the user instruction.
- Treat `experiment/`, `archive/`, `configs/legacy/`, `scripts/legacy/`, and
  `docs/archive/` as reference-only historical material.
- Never import `experiment.*` or archived modules from production code under
  `wfcllm/`.
- Legacy tests are archived and excluded from default pytest discovery by
  `pytest.ini`; do not copy their import patterns into active tests.

## Current Architecture

- `wfcllm.method`
  - Preset contracts, method config dataclasses, artifact path helpers, and
    official run layout helpers.
- `wfcllm.generation`
  - Structure-aware generation-time evidence collection, retry state machine,
    candidate sidecars, and final-code JSONL pipeline.
- `wfcllm.semantic`
  - Semantic LSH rules, feature extraction, and entropy helpers used by the
    official WFCLLM method.
- `wfcllm.detection`
  - Strict final-code-only detector input validation, calibration, metrics,
    and detection pipeline.
- `wfcllm.audit`
  - Detector-input integrity, no-quality-gate checks, and artifact marker
    validation.
- `wfcllm.diagnostics`
  - Non-official diagnostic selector and upper-bound analysis helpers.
- `wfcllm.cli`
  - `run.py` argparse surface, phase registry wiring, config loading, and
    phase runner functions.
- `wfcllm.orchestration`
  - Phase source of truth, run-state persistence, prereq handling, and
    skip/force orchestration.
- `wfcllm.datasets`, `wfcllm.lang`, `wfcllm.encoder`, `wfcllm.common`
  - Dataset adapters, language adapters, encoder support, and shared helpers.

## CLI And Phase Contract

- `run.py` is intentionally a thin shim that forwards to
  `wfcllm.cli.entry:main`.
- Phase source of truth is `wfcllm/orchestration/state.py`.
- Main phases are `generate`, `calibrate`, `detect`, `report`, and `audit`.
- Optional active phases are `encoder`, `posthoc-pass-report`, and
  `diagnostic-selector`.
- Historical phase implementations are available only through explicit
  `legacy-*` phase names plus `--legacy`; live code should not call archived
  modules directly.
- Current `run.py` mainline phase runners establish the official phase names
  and run-state contract. Artifact-producing low-level scripts remain under
  `scripts/` until those phase runners are wired to the full pipeline.
- Official detector input rows contain exactly `id`, `dataset`, `prompt`, and
  `final_code`.
- Diagnostic selector outputs must be marked `diagnostic_only=true` and
  `not_official_method=true`.
- Posthoc pass reports must carry the marker documented in
  `docs/NO_QUALITY_GATE_PROTOCOL.md`.

## Configs

- Use `configs/base_config.json` for the default mainline config.
- Use `configs/wfcllm/evidence_retry_seed7x3.json` for the official preset.
- Use `configs/wfcllm/repro_humaneval_full164.json` for the documented
  HumanEval reproduction contract.
- Historical experiment configs belong under `configs/legacy/`.
- Do not add old mainline config sections to active root configs.

## Artifact Contracts

- Official run artifacts use the `data/runs/<run_id>/` layout documented in
  `docs/ARTIFACT_SCHEMA.md`.
- `inputs/final_code.jsonl` is the only official positive detector input.
- Generation audit rows, candidate sidecars, raw attempt summaries, and
  posthoc reports are not detector inputs.
- Do not leak secrets into public JSON, JSONL, reports, or docs.
- Do not commit local models, datasets, checkpoints, large run outputs, logits
  dumps, generated logs, or papers.

## Environment

- Use the conda environment `WFCLLM`.
- Do not install dependencies or run project code from `base`.
- Install dependencies from `requirements.txt`.
- Tests and normal development should assume offline execution.
- Prefer local resources under `data/models/` and `data/datasets/`.
- For pytest and model/dataset-sensitive commands, set `HF_HUB_OFFLINE=1`;
  also set `TRANSFORMERS_OFFLINE=1` and `HF_DATASETS_OFFLINE=1` when
  appropriate.

## Smoke Commands

```bash
conda run -n WFCLLM python run.py --status
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest --collect-only -q
```

## Test Commands

Active tests are rooted at `tests/`; archived tests are excluded by
`pytest.ini`.

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/semantic -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

If full `tests/` fails because local models or datasets are absent, report the
first failing command, first error line, and affected files. Do not describe a
resource-bound failure as a successful full-suite run.

## Script Entry Points

- `scripts/wfcllm_generate.py`: low-level artifact-producing generation entry
  point while `run.py` phase runners remain a phase/state contract.
- `scripts/wfcllm_sanitize_final_code.py`: sanitize candidate JSONL into the
  strict final-code detector schema.
- `scripts/evaluate.py exec`: pass@k from candidate JSONL files.
- `scripts/evaluate.py detection`: regression report from saved summary/details
  artifacts.
- `scripts/evaluate.py bench`: benchmark report from existing candidate and
  detection artifacts. It does not auto-run archived phases.
- `scripts/evaluate.py dual`: archived harness guidance only.

## Code Style

- Prefer minimal, direct changes inside existing module boundaries.
- Use `from __future__ import annotations` in library modules.
- Use `pathlib.Path` for path handling.
- Use UTF-8 explicitly for file reads and writes.
- Use modern type hints such as `list[str]`, `dict[str, Any]`, and
  `str | None`.
- Prefer dataclasses for config objects and result records.
- Raise `ValueError` for invalid config, malformed data, and incompatible
  artifacts unless nearby code uses a narrower exception.
- Use `logging.getLogger(__name__)` for module loggers.
- Keep logs factual and low-noise.

## Git Hygiene

- Check `git status --short` before editing.
- Do not revert unrelated user changes.
- Keep commits scoped to code, tests, configs, scripts, docs, and archive files
  needed for the task.
- Do not add `data/`, `papers/`, `review-stage/`, local server output,
  checkpoints, large JSONL files, or generated logs to the branch.
- Commit format is `<type>: <description>`.
- Allowed commit types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
