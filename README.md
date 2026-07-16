# WFCLLM: Structure-Aware Generation-Time Semantic Watermarking for Code LLMs

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. The default and official baseline preset remains `evidence_retry_seed7x3`. The optional `gated_semantic_window_v1` preset is experimental; offline fake tests establish its implementation contract and wiring, not real gate effectiveness or detector improvement.

WFCLLM embeds semantic LSH evidence while code is generated, using structure-aware boundary detection, rollback-capable evidence-only retry, and strict final-code-only statistical detection. The official protocol forbids pass/test/correctness proxies during generation, retry, calibration, detection, and final candidate selection.

## Quick Start

Run commands from the repository root in the `WFCLLM` conda environment.

```bash
python run.py --status
python run.py --config configs/base_config.json
python run.py --phase generate
python run.py --phase detect --input data/runs/<run_id>/inputs/final_code.jsonl
python run.py --phase audit --run-dir data/runs/<run_id>
```

To select the experimental gated configuration, use
`--config configs/wfcllm/gated_semantic_window_v1.json`. Its default phase
sequence is `gate-data`, `gate-train`, `gate-validate`, `generate`, `calibrate`,
`detect`, `report`, `audit`. A separately validated external bundle may start
at `generate` and then uses the normal five main phases.

The default run root is `data/runs/`. Generated official detector inputs are written as `inputs/final_code.jsonl` with exactly `id`, `dataset`, `prompt`, and `final_code`.

The command block above is the official `run.py` workflow contract for the mainline phases. In this refactor branch, the phase runners currently establish phase names and run-state behavior; for direct artifact-producing generation while those runners are being wired to the full pipeline, use `scripts/wfcllm_generate.py` with the same preset values documented in [docs/REPRO_EVIDENCE_RETRY_SEED7X3.md](docs/REPRO_EVIDENCE_RETRY_SEED7X3.md).

## Official Method

The main method preset is documented in [docs/WFCLLM_METHOD.md](docs/WFCLLM_METHOD.md). Reproduction notes for the default preset are in [docs/REPRO_EVIDENCE_RETRY_SEED7X3.md](docs/REPRO_EVIDENCE_RETRY_SEED7X3.md).

Core protocol references:

- [No Quality Gate Protocol](docs/NO_QUALITY_GATE_PROTOCOL.md)
- [Strict Code-Only Detector](docs/STRICT_CODE_ONLY_DETECTOR.md)
- [Artifact Schema](docs/ARTIFACT_SCHEMA.md)
- [Diagnostic Upper Bound](docs/DIAGNOSTIC_UPPER_BOUND.md)
- [Legacy Archive](docs/LEGACY_ARCHIVE.md)

Known metrics for the current official preset are intentionally modest. Detector evidence remains the main limitation, and the preset is the best official single-configuration generation-time candidate rather than a solved detector result.

## Repository Layout

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
configs/
  base_config.json
  wfcllm/
scripts/
  wfcllm_generate.py
  wfcllm_sanitize_final_code.py
  evaluate.py
tests/
docs/
archive/
```

The live package is `wfcllm/`. Archived legacy code and historical documentation are kept under `archive/` and `docs/archive/` for traceability.

## Environment

Use the conda environment named `WFCLLM`.

```bash
conda create -n WFCLLM python=3.11 -y
conda run -n WFCLLM pip install torch
conda run -n WFCLLM pip install -r requirements.txt
```

Development and tests should assume offline execution unless a user explicitly requests online behavior.

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/detection -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/audit -v
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/diagnostics -v
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

For model-sensitive tests, prefer local resources under `data/models/` and `data/datasets/`.

The gated workflow requires all resources locally: a supported base generation
model, the semantic encoder, `data/models/codet5-small`, source datasets, a
private 32-key training bank, a disjoint private 8-key holdout bank, and a
deployment key supplied through the CLI's file/environment options. It never
downloads models or datasets automatically. Keys stay outside configs and
public artifacts; only identifiers and hashes may be published.

## Mainline Phases

`run.py` is a thin entrypoint that forwards to `wfcllm.cli.entry:main`.

Default mainline phases:

- `generate`
- `calibrate`
- `detect`
- `report`
- `audit`

The experimental gated preset prepends three phases:

- `gate-data` builds provenance-bound, no-quality-proxy supervision.
- `gate-train` trains the close/suitable gate with the fixed six-loss contract.
- `gate-validate` fits thresholds, validates float/int8 stability, and is the
  only formal gate-bundle publisher.

The default config is `configs/base_config.json`, which points to the official method preset and the `data/runs/` artifact root.

## Protocol Summary

Detector input is final code only. Audit rows, generation traces, retry decisions, candidate sidecars, and posthoc pass reports are not detector inputs.

Posthoc pass reports must carry the marker described in [docs/NO_QUALITY_GATE_PROTOCOL.md](docs/NO_QUALITY_GATE_PROTOCOL.md). They can be used to report utility after the run, but they cannot influence generation, retry, final selection, calibration, or detection.

Diagnostic selector outputs must be marked `diagnostic_only=true` and `not_official_method=true`; see [docs/DIAGNOSTIC_UPPER_BOUND.md](docs/DIAGNOSTIC_UPPER_BOUND.md).

Do not commit `data/runs/`, gate datasets, key banks, models, checkpoints,
bundles, generated JSONL, logs, or other experiment artifacts. The repository
tracks contracts, code, small configs, tests, and documentation only.
