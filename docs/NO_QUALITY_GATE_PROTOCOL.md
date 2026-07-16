# No Quality Gate Protocol

WFCLLM is evaluated as a generation-time semantic watermarking method. The official protocol forbids using pass/test/correctness proxies as control signals during generation, retry, calibration, detection, and final candidate selection.

## Allowed Signals

- Structure-aware code boundary events.
- Parser structure validity used only to establish formal window boundaries;
  this is not evidence that code is correct or high quality.
- Key-independent aggregates of semantic LSH margin/stability measured across
  the training key bank. Raw keys and a deployment-key target region are never
  public gate inputs.
- Gate close/suitable probabilities from the frozen validated bundle.
- Retry budget counters and rollback state.
- Candidate 0 (the original complete window), bounded key-blind rewrites, and
  deterministic rollback/restore state.
- Sampling metadata needed to continue generation.
- Final-code-only detector statistics derived from `id`, `dataset`, `prompt`, and `final_code`.
- Posthoc utility reports, after all official generation and detection decisions are complete.

## Forbidden Signals

- Unit-test pass or fail results during generation.
- Benchmark correctness labels during generation.
- Runtime execution outcomes during retry.
- Candidate selection based on pass/test/correctness proxies.
- Calibration thresholds selected from pass/test/correctness proxies.
- Detection scores that read generation traces, retry traces, audit logs, candidate sidecars, or posthoc utility reports.
- Any proxy named or structured as quality, correctness, pass, fail, test outcome, benchmark outcome, reward, or oracle feedback when used as a control signal.
- Syntax-valid or parser-valid flags treated as a quality rank, reward, or
  correctness signal. Structural validity only permits the parser/window
  contract to continue.
- Single-key/deployment-key labels, target LSH region, raw training/holdout
  keys, or any decision derived from knowing the deployment target.
- Gate-data, gate-training metrics, checkpoints, validation rows, generation
  audit rows, or candidate sidecars read by calibration/detection.

## Posthoc Pass Report Marker

Any posthoc pass report must carry this exact marker:

```json
{
  "posthoc_only": true,
  "not_used_for_generation": true,
  "not_used_for_retry": true,
  "not_used_for_selection": true,
  "not_used_for_calibration": true,
  "not_used_for_detection": true
}
```

The marker records that the report is an after-the-fact utility measurement. It does not make the report eligible for any official method decision.

Posthoc correctness can compare utility and enforce the configured reporting
non-inferiority boundary, but it must never flow back into generation, retry,
candidate selection, gate-data labels, gate training, gate validation,
calibration, or detection. The formal eight-stage workflow therefore ends all
generation and detector decisions before a posthoc pass report is produced or
merged.
