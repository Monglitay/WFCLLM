# Phase A Contract and Replay Report

## Decision

Phase A passes the contract/replay gate and permits the preregistered 30-task
pilot. This is not a performance-success decision. The two new V2 debug
samples both failed HumanEval execution, so the debug run supplies no positive
Pass@1 evidence; the frozen paired pilot remains necessary.

## Frozen runtime

- Server project: `/root/autodl-tmp/WFCLLM`
- Generation code commit: `5accc59234c13231e404b10c0f1e7cc17c33f456`
- Baseline commit: `25b3c6c001809f80c5314020e4048a7e68960963`
- Seed: `20260713`
- New-generation retry attempts: exactly `20`
- Generation model: local `deepseek-coder-1.3b-instruct`, BF16, no 4-bit
- Encoder: local CodeT5 base, LoRA, BF16, embedding dimension 128
- Encoder checkpoint SHA-256:
  `9da80dfdbb3f8e682298e588e822f15967e460c9dfd640a243aad8ed1ba51d25`
- V2 signature width: 16 bits
- GPU: NVIDIA GeForce RTX 5090

Generation and detection use the same `load_v2_signature_scorer` path, which
constructs the same CodeT5 encoder, checkpoint, precision, embedding width,
secret-conditioned LSH space, and target-signature implementation on both
sides.

## Existing-final-code replay

Eight pre-existing positive final-code rows and eight ordinary negative rows
were scored twice under the new V2 implementation. These samples were not V2
embedded with the new private key; their purpose was deterministic loader and
canonicalization validation.

- Exact score/metadata replay: 8/8 (`100%`)
- Positive correct-key score median: `0.4986979167`
- Positive derived-wrong-key score median: `0.5`
- Ordinary-negative score median: `0.534375`

As expected for non-V2-embedded historical code, this check does not show a
correct-key separation and is not used as effectiveness evidence.

## New V2-R20 debug replay

The first two preregistered pilot IDs (`HumanEval/124` and `HumanEval/111`)
were generated as a Phase A debug subset. Each sample used all 20 attempts;
the run therefore contains 40 raw attempts.

- Final-code rows: 2/2 complete
- Retry-ledger rows: 40/40 complete
- Exactly one selected attempt per sample: 2/2
- Repeated final-code score replay: 2/2 (`100%`)
- Selected ledger final-code SHA match: 2/2 (`100%`)
- Generation-score to final-score equality: 2/2 (`100%`)
- Generation-score/final-score Pearson correlation: approximately `1.0`
- Selected canonical units: 7 total (3 and 4)
- Accepted-to-recovered canonical-unit ratio: `1.0`
- Correct-key score median: `0.5807291667`
- Derived-wrong-key score median: `0.4765625`
- Ordinary-negative score median in the two-row debug panel: `0.5234375`
- Static-quality-eligible attempts: 20/40
- Selected attempts passing the static quality gate: 2/2
- HumanEval execution: 0/2

The correct-key shift is directionally positive in these two samples, but two
samples are far too few for a TPR claim. The 0/2 execution result is retained
as adverse evidence and is not reclassified as a generation failure.

## Contract checks

- The V2 detector input validator accepts exactly `id`, `dataset`, `prompt`,
  and `final_code`; scorer input is only `final_code`.
- Retry ledgers and candidate sidecars are explicitly audit-only and forbidden
  as detector input.
- Canonical units are re-extracted from saved final code; no generation-time
  unit list is loaded by the detector.
- Target signatures and the LSH space are secret-key conditioned.
- Correct-key and derived-wrong-key scorers use identical public/model config.
- V1 artifacts fail in the V2 loader; V2 artifacts fail in the V1 loader.
- The public calibration artifact contains only the key hash, never the raw
  key.
- Silent raw-secret scans passed for the replay report, retry ledger, and both
  calibration artifacts.

## Artifact hashes

- Existing-code replay JSON:
  `64cdd956d30f3e71547616cd9627092f0bcbeb7a5026dbee1c7610a35ea30b50`
- Debug final-code JSONL:
  `b926859abfc3d8e6ef535023383c5d510ca9dc07aed83686eab46e97e538de5b`
- Debug retry ledger JSONL:
  `18c9164779998bc0f8464f933e239727bc4537577736547a96991d09d389ffca`
- Debug replay JSON:
  `e2338ada84b36f1432037e4b298b9adc9f2276edfadf17386bb2757e9ddbc9ef`
- Debug execution summary JSON:
  `04b51e066453036440f78814e09348de342693af2e1034da1abf53a99cd8d6b8`
- Frozen Current v1 calibration JSON:
  `a0cbe1f5fe6cd114209e260c03ea199edcb2504882c4c557144cb2054ac2c475`
- Frozen V2 calibration JSON:
  `049a1a1617c92c83a5a6469e6968edf7cd6bba9a524a899e72aa42e0dbb45abb`

Raw generation artifacts remain untracked under
`data/experiments/watermark_v2_retry20_20260713/`. They are referenced by
hash, not committed. No secret material is included in this report.
