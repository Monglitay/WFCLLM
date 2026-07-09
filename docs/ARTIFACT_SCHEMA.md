# Artifact Schema

WFCLLM writes official run artifacts under `data/runs/<run_id>/`.

## Run Layout

```text
data/runs/<run_id>/
  config/
    resolved_config.json
    method_preset.json
  inputs/
    final_code.jsonl
  generation/
    audit.jsonl
    candidate_sidecar.jsonl
    raw_attempt_summary.jsonl
  calibration/
    reference_calibration.json
  detection/
    positive_details.jsonl
    reference_negative_details.jsonl
    model_negative_details.jsonl
  reports/
    reference_report.json
    model_negative_report.json
    pass_report_posthoc.json
  audit/
    detector_input_integrity.json
    no_quality_gate_integrity.json
    artifact_integrity.json
  logs/
    run.log
```

## Detector Input Contract

`inputs/final_code.jsonl` is the only official positive detector input. Each row must contain exactly:

- `id`
- `dataset`
- `prompt`
- `final_code`

## Generation Sidecars

Files under `generation/` are audit or diagnostic artifacts. They are useful for reproducibility and debugging, but they are not detector inputs.

## Reports

`reports/pass_report_posthoc.json` is allowed only after official generation, calibration, detection, and selection decisions are complete. It must include the marker from `docs/NO_QUALITY_GATE_PROTOCOL.md`.

## Integrity Audits

The `audit/` directory records detector-input and no-quality-gate integrity checks. These artifacts support reproducibility and review; they do not add detector evidence.
