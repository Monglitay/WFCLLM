# WFCLLM Method

WFCLLM is a structure-aware generation-time semantic watermarking method for code LLMs. It embeds semantic LSH evidence during generation through structure-aware boundary detection, rollback-capable evidence-only retry, and final-code-only statistical detection.

The default official method preset is `evidence_retry_seed7x3`.

The protocol forbids pass/test/correctness proxies during generation, retry, calibration, detection, and final candidate selection.

## Method Summary

WFCLLM operates on generated code structure rather than post-generation utility labels. During generation, structure-aware boundaries define evidence opportunities. Retry decisions use semantic evidence only, and official detection consumes only the final code artifact.

## Current Known Metrics

| Metric | Value |
| --- | --- |
| pass@1 | 104/164 = 63.41% |
| AUROC | 0.660080 |
| TP@<=8FP | 43/164 |
| FP | 8 |
| positive insufficient | 20 |
| model-negative AUROC | 0.510332 |

## Official Boundaries

- Official preset: `evidence_retry_seed7x3`
- Official run root: `data/runs/<run_id>/`
- Official detector input: `inputs/final_code.jsonl`
- Official detector input fields: `id`, `dataset`, `prompt`, `final_code`
- Utility pass reports: posthoc only, never control signals
- Diagnostic selectors: not official methods

Detector evidence remains the main limitation. The current preset is the best official single-configuration generation-time candidate, not a solved detector result.
