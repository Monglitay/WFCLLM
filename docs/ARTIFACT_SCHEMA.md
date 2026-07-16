# Artifact Schema

WFCLLM's official run artifact contract is rooted at `data/runs/<run_id>/`. Artifact-producing workflows must write the layout below when connected to the mainline pipeline.

## Run Layout

```text
data/runs/<run_id>/
  config/
    resolved_config.json
    method_preset.json
  inputs/
    final_code.jsonl
  gate-data/
    manifest.json
    source_manifest.json
    window_groups.jsonl
    candidate_attempts.jsonl
    labels.jsonl
    split_manifest.json
    training_key_bank_manifest.json
    feasibility_summary.json
  gate-train/
    resolved_training_config.json
    training_metrics.jsonl
    development_summary.json
    checkpoints/
    candidate_bundle/
    candidate_bundle_manifest.json
  gate-validate/
    validation_summary.json
    agreement_details.jsonl
    bundle/
    gate_bundle_manifest.json
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

The three `gate-*` directories belong only to the experimental
`gated_semantic_window_v1` method. The official `evidence_retry_seed7x3`
baseline retains its five-stage layout and does not require them.

## Detector Input Contract

`inputs/final_code.jsonl` is the only official positive detector input. Each row must contain exactly:

- `id`
- `dataset`
- `prompt`
- `final_code`

Calibration and detection may open this file, their negative final-code
corpora, the validated gate bundle, and the frozen calibration artifact. They
must never read `gate-data/`, training checkpoints, generation audit rows, or
candidate sidecars as detector evidence.

## Gated Producer/Consumer Contract

| Artifact | Required public fields | Producer | Allowed consumer |
| --- | --- | --- | --- |
| `gate-data/manifest.json` | schema/config/input hashes, source and split contract versions, `formal_eligible` | `gate-data` | `gate-train`, audit |
| `window_groups.jsonl` | group/source IDs, normalized structure, spans and parent descriptor | `gate-data` | `gate-train`, audit |
| `candidate_attempts.jsonl` | group/window/candidate IDs, rewrite budget, structural/LSH outcomes | `gate-data` | label builder, audit |
| `labels.jsonl` | group/window IDs, close/suitable labels, budget summaries | `gate-data` | `gate-train`, audit |
| `split_manifest.json` | source groups, split IDs and provenance hashes | `gate-data` | `gate-train`, `gate-validate`, audit |
| `training_key_bank_manifest.json` | key-bank identifier/count and hash only | `gate-data` | `gate-train`, audit |
| `training_metrics.jsonl` | six named losses and aggregate structural metrics | `gate-train` | report, audit |
| `candidate_bundle_manifest.json` | model/tokenizer/config hashes, candidate status | `gate-train` | `gate-validate`, audit |
| `validation_summary.json` | fitted thresholds, float/int8 agreement, false-positive and coverage metrics | `gate-validate` | bundle publisher, report, audit |
| `agreement_details.jsonl` | sample/group IDs and stability decisions; no raw keys | `gate-validate` | audit |
| `gate_bundle_manifest.json` | bundle tree hash, contract hashes and `validated=true` | `gate-validate` | generate, calibrate, detect, audit |
| `generation/audit.jsonl` | closed spans, gate probabilities, rewrite decisions, `audit_only=true` | generate | audit/report only |
| `generation/candidate_sidecar.jsonl` | candidate and selected-index trace, `not_detector_input=true` | generate | audit only |
| `calibration/reference_calibration.json` | gated detector mode, bundle/config/key identifiers, negative-corpus hash, reliable-window background | calibrate | detect, report, audit |
| `detection/*_details.jsonl` | final-code-derived spans, parent descriptor, gate probabilities, hit/miss/abstain and margin | detect | report, audit |

Key-bank and deployment key material is runtime-only. Public artifacts contain
only non-reversible identifiers or hashes. A formal gate bundle must come from
the real `gate-validate` publisher, have a matching tree hash, and carry
`validated=true` with `diagnostic_test_backend=false`. Fake integration
artifacts are always `diagnostic_only=true`, `not_official_method=true`,
`formal_eligible=false`, and cannot be relabelled as formal artifacts.

## Generation Sidecars

Files under `generation/` are audit or diagnostic artifacts. They are useful for reproducibility and debugging, but they are not detector inputs.

## Reports

`reports/pass_report_posthoc.json` is allowed only after official generation, calibration, detection, and selection decisions are complete. It must include the marker from `docs/NO_QUALITY_GATE_PROTOCOL.md`.

For gated runs, the formal report keeps the method and detector mode separate
from proxy-baseline numbers and reports gate coverage, hit/miss/abstain counts,
rewrite cost, calibration, and the detection curve. A pre-existing pass@1
artifact may only be merged with all posthoc markers intact.

## Integrity Audits

The `audit/` directory records detector-input and no-quality-gate integrity checks. These artifacts support reproducibility and review; they do not add detector evidence.
