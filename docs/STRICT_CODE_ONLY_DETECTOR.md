# Strict Code-Only Detector

The official WFCLLM detector consumes final code only. It must not read generation traces, retry traces, audit rows, candidate sidecars, sampling metadata, or posthoc utility reports.

## Official Input

Official input exactly:

```json
{"id": "HumanEval/0", "dataset": "humaneval", "prompt": "...", "final_code": "..."}
```

Each JSONL row must contain exactly these fields:

- `id`
- `dataset`
- `prompt`
- `final_code`

## Rejected Inputs

Detector input validation rejects rows containing extra method-control or trace fields, including:

- `generated_code`
- `blocks`
- `watermark_params`
- `audit`
- `audit_only`
- `audit_event_id`
- `candidate_sidecar`
- `raw_attempt_summary`
- `pass_report`
- keys beginning with `generation_`
- keys beginning with `audit_`

The detector may write details and reports, but those outputs are downstream artifacts. They are not valid detector inputs for the same run.
