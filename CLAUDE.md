# WFCLLM Agent Guide

This repository contains the live WFCLLM mainline: a structure-aware generation-time semantic watermarking method for code LLMs. The default official method preset is `evidence_retry_seed7x3`.

## Project Architecture

```text
wfcllm/
  method/
  generation/
  semantic/
  detection/
  audit/
  diagnostics/
  cli/
  orchestration/
  datasets/
  lang/
  encoder/
  common/
```

`run.py` is intentionally a thin shim that forwards to `wfcllm.cli.entry:main`. Keep business logic inside the package.

## Mainline Phases

The live phases are:

- `generate`
- `calibrate`
- `detect`
- `report`
- `audit`

Running without `--phase` executes the `PHASES` sequence from `wfcllm/orchestration/state.py`. Keep config `runtime.default_phases` aligned with that source of truth until orchestration explicitly supports config-driven phase sequencing. `--status` and run state handling belong to `wfcllm.orchestration`.

## Method Rules

- The official preset is `evidence_retry_seed7x3`.
- Official detector input rows contain exactly `id`, `dataset`, `prompt`, and `final_code`.
- Pass/test/correctness proxies are forbidden during generation, retry, final selection, calibration, and detection.
- Posthoc pass reports are allowed only as after-the-fact utility reports and must carry the full marker documented in `docs/NO_QUALITY_GATE_PROTOCOL.md`.
- Diagnostic selectors are not official methods. Their outputs must carry `diagnostic_only=true` and `not_official_method=true`.
- Do not leak secrets into public JSON, JSONL, reports, or docs.

## Repository Boundaries

- Live package code is under `wfcllm/`.
- Archived legacy code is reference-only under `archive/`.
- Historical docs are reference-only under `docs/archive/`.
- Never import `experiment.*` from production code under `wfcllm/`.
- Do not import archived code from live package modules.
- Treat large local resources under `data/models/`, `data/datasets/`, `data/checkpoints/`, and run outputs as local artifacts unless explicitly requested.

## Development Commands

Use the `WFCLLM` conda environment.

```bash
conda run -n WFCLLM python run.py --status
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Use `HF_HUB_OFFLINE=1` for pytest commands unless the user explicitly asks for online behavior. Prefer local model and dataset paths under `data/`.

## Coding Conventions

- Prefer small, direct changes inside existing module boundaries.
- Use `pathlib.Path` for paths.
- Use UTF-8 explicitly for file reads and writes.
- Use modern type hints such as `list[str]`, `dict[str, Any]`, and `str | None`.
- Use dataclasses for config and value records when appropriate.
- Keep library identifiers and most docstrings in English.
- Raise `ValueError` for invalid config, malformed data, and incompatible artifacts unless nearby code uses a narrower exception.
- Keep logs factual and low-noise.

## Testing Guidance

- Use `pytest`.
- Use `tmp_path` for filesystem-isolated tests.
- Use `monkeypatch`, `unittest.mock.patch`, and `MagicMock` for model, dataset, and external boundaries.
- Use `pytest.raises(..., match=...)` for validation paths.
- Use `pytest.approx(...)` for floating-point assertions.
- Update nearby tests for behavior changes.

For broad changes, run the targeted suites first, then the full suite if feasible:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

## Git Hygiene

- Check `git status --short` before editing.
- Do not revert unrelated user changes.
- Do not commit local models, datasets, checkpoints, large run artifacts, or generated logs.
- Commit format is `<type>: <description>`.
- Allowed commit types: `feat`, `fix`, `refactor`, `test`, `docs`, `chore`.
