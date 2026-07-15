# Watermark Mechanism V4 Results

## Final status

`CanonicalStructuralReplay-V4-R20` is a complete **negative result**. It fixes the
V3 batch-dependent exact-replay defect and is far below the semantic-cost gate,
but it does not satisfy the joint detection and correctness claim.

## Root-cause conclusion

The Stage A verdict is A2. Batch-shape-dependent float32 neural inference is
supported, and CPU whitening contributes another batch-shaped numerical path.
The earliest hooked divergence was T5 block 0 self-attention; the persisted formal
chain differed by whitening pre-normalization. Fixed sequence length was not
sufficient, deterministic CUDA flags did not remove the effect, and fixed
physical shape explained why V3's fixed-batch experiment recovered 30/30.

## Candidate decision

- Candidate A: rejected; exact but 0.310876049 seconds/task.
- Candidate B: rejected; 18 erasure-mask mismatch rows and no certified bound.
- Candidate C: selected; deterministic canonical AST/role/bounded-dataflow
  representation, HMAC signature/target domains, final-code-only R3.
- Candidate D: rejected; no formal evidence or signal benefit over C.

## Debug and pilot

| Metric | Result | Gate |
|---|---:|---:|
| Debug generation→R3 exact | 8/8 × 3 cold processes | 100% |
| Pilot candidate hashes | 600/600 | 100% |
| Pilot generation→R3 exact | 30/30 × 3 cold processes | 100% |
| Batch/order/composition/cache exact | 100% | 100% |
| Semantic seconds/task | 0.022659895 | ≤0.1409884 |
| Cost reduction vs 0.201412 | 88.7495% | ≥30% |
| Peak CUDA allocation | 0 MiB | ≤32607 MiB |
| Current TPR | 0/30 | baseline |
| V4 TPR | 0/30 | strictly greater: FAIL |
| Current pass | 18/30 | baseline |
| V4 pass | 15/30 | ≥15: PASS; delta ≥-1: FAIL |
| Mean paired score delta | +0.060468750 | descriptive |

Calibration primary/wrong-key rates were 2/230 (0.8696%) and 1/230
(0.4348%). Wrong-key pilot V4 positives were 1/30, but its paired score shift was
-0.022751323 rather than the primary +0.060468750.

## Escalation status

- V4 held-out FPR: not measured; held-out detector path remained closed.
- HumanEval-164: not run.
- Robustness: not run.
- Post-pilot tuning: none.
- Targeted repairs used: 1/1, solely to restore preregistered fallback behavior.

## Comparison with preserved V3

V3's separate pilot had Current/V3 TPR 1/30 versus 7/30, pass 15/30 versus
15/30, semantic cost 0.101131791 seconds/task, and only 27/30 exact replay. V4's
separate pilot has perfect exact replay and lower cost, but Current/V4 TPR 0/30
versus 0/30 and pass 18/30 versus 15/30. These studies use different pilot IDs;
the comparison is descriptive, not paired.
