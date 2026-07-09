# No Quality Gate Protocol

WFCLLM is evaluated as a generation-time semantic watermarking method. The official protocol forbids using pass/test/correctness proxies as control signals during generation, retry, calibration, detection, and final candidate selection.

## Allowed Signals

- Structure-aware code boundary events.
- Semantic LSH evidence derived from candidate code structure.
- Retry budget counters and rollback state.
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
