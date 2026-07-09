# Legacy Archive

Legacy WFCLLM materials are preserved for traceability under:

```text
archive/legacy_wfcllm_2026_07/
```

Historical documents are preserved under:

```text
docs/archive/
```

These archived materials are reference-only. They are not imported by the live `wfcllm/` package and are not part of the official mainline method.

## Archive Contents

The archive keeps historical code roots, selected tests, historical configs, legacy scripts, and experiment summaries needed to understand earlier SAWR-era development. The archive manifest is the source of truth for the expected preserved roots.

## Large File Policy

The archive should not track large local artifacts such as:

- model weights
- datasets
- checkpoints
- generated JSONL run outputs
- logits dumps
- execution logs
- temporary analysis caches

Large or reproducible artifacts should remain outside git under ignored local paths such as `data/models/`, `data/datasets/`, `data/checkpoints/`, `data/results/`, and `data/runs/`.

## Live Code Boundary

Live code may point users to the archive for historical guidance, but it must not import archived modules. If historical behavior is needed in the mainline, migrate the necessary behavior into `wfcllm/` with focused tests and current protocol checks.
