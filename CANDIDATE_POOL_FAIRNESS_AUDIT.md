# Candidate Pool Fairness Audit

The formal Current-R20 and Streaming-SharedPool-V3-R20 selectors consumed the same ordered raw retry trajectories.

- 30 pilot task IDs match the preregistration exactly.
- Every task contains attempt indices 0 through 19 exactly once.
- Every run-summary task contains 20 ordered candidate SHA-256 values.
- Recomputing SHA-256 from the saved UTF-8 final code matched 600/600 recorded hashes.
- Candidate-pool hash match rate: 100.0%.
- The V3 private key does not enter generation, sampling seeds, logits, stopping, syntax/static quality evaluation, repair, or raw candidate construction.
- V3 performs no end-of-sequence semantic rescore of all complete candidates; recorded count is zero. It incrementally consumes public closure events and performs exactly one replay for the selected final candidate.

The candidate ledger is the saved V2 retry-20 pilot pool. This is an exact shared-pool replay study rather than a fresh live generation run. That design isolates selector effects and guarantees pool equality; it does not measure end-to-end wall-clock generation latency.
