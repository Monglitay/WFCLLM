# Watermark Mechanism V3 Failure Postmortem

## Failure

The formal R3 contract required generation-time evidence and final-code-only detector evidence to agree exactly on every pilot task. After one permitted implementation repair, three of thirty selected programs still differed in one unit's quantized semantic embedding. The failed IDs were HumanEval/34, HumanEval/39, and HumanEval/71.

## Root cause

Canonical AST reconstruction was stable, and candidate final-code bytes matched. The remaining nondeterminism sits in neural representation inference: the same semantic context embedded in different dynamic batch compositions can move by a small float32 amount. Fixed sequence padding removed variable sequence-length effects, but dynamic batch size/composition still altered near-boundary encoder outputs. Integer quantization at scale 4096 then converted some small numerical changes into different discrete vectors, causing signature-bit and exact-evidence divergence.

This is a mechanism-level contract problem, not a cosmetic test discrepancy. A final-code detector cannot reproduce evidence that depends on the generation-time batch neighborhood.

## Repair history

The original dynamic-padding implementation reached 26/30 exact replay. The single permitted repair investigated fixed shape inference. Fixed batch 32 × sequence 256 reached 30/30 debug/pilot exactness but cost about 0.265 s/task and violated the 0.1409884 s/task gate. The frozen final repair retained sequence length 256 with dynamic batches, restoring mean cost to 0.101132 s/task, but exactness reached only 27/30. No further change was allowed.

## What worked

- Shared candidate-pool equality was exact (600/600 hashes).
- Incremental event batching eliminated all-candidate EOS semantic rescoring.
- Mean semantic overhead fell by roughly 49.79% relative to the measured complete-final semantic baseline.
- Aggregate HumanEval correctness stayed 15/30.
- Primary-key pilot scores shifted upward, while wrong-key controls did not show the same shift.

## Why the positive signal does not rescue the method

The detector contract defines the watermark. When generation and R3 do not reconstruct identical signed evidence, apparent TPR and score improvements combine intended selection pressure with representation-path inconsistency. Those outcomes remain useful diagnostics but cannot validate the mechanism.

## Constraints for any V4

- Do not reuse the unopened V3 held-out split after designing against this failure; create a new calibration/held-out split.
- Do not claim exactness from high cosine similarity; compare discrete evidence bit for bit.
- Do not let batch composition influence a signed representation. Candidate options include deterministic per-context inference, an integerized representation, a margin/dead-zone with explicit erasures, or a purely canonical structural signature.
- Treat fixed-batch performance and exactness as joint preregistered gates.
- Preserve shared raw pools and final-only R3 detection.
- Run robustness only after clean base exactness and held-out FPR gates.
