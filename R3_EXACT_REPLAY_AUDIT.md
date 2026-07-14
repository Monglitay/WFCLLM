# R3 Exact Replay Audit

## Contract

R3 receives only final code, the public frozen config, and a private key file. It may not consume prompt, task ID, raw candidate pool, generation trace, event ledger, token probabilities, or generation-time unit metadata. Exactness compares sorted unit IDs, canonical-context hashes, quantized embeddings, signature bits, target bits, match counts, aggregate numerator/denominator, and decision.

## Outcome

- Exact: 27/30 (90.0%)
- Required: 30/30 (100%)
- Mismatches: `HumanEval/34` unit 3 quantized embedding; `HumanEval/39` unit 6 quantized embedding; `HumanEval/71` unit 2 quantized embedding
- Aggregate gate: **FAIL**

All three failures are representation-level mismatches after canonical extraction, not differences in the final-code bytes. Fixed sequence padding removed most batch-shape sensitivity, but dynamic batch composition still changed float32 CodeT5 inference near quantization/projection boundaries. Because the channel signs a quantized neural representation, even one boundary flip violates the bit-exact generation/detection contract.

The observer path is read-only and streams saved final-code trajectories line by line. The final detector independently reparses only the selected final code. Replay of saved trajectories validates the semantic observer and selector against the same raw V2 pool, but it does not claim live token-runtime equivalence for speculative/rolled-back contexts that were absent from the saved trajectory.

The single allowed repair was exhausted. Further changes—deterministic per-context kernels, wider quantization dead zones, integerized encoder inference, or a non-neural canonical representation—would constitute a new preregistered mechanism, not a repair of this result.
