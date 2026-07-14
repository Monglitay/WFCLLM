# V4 Candidate-Pool Fairness Audit

## Result

Candidate-pool fairness is **PASS**. Current-R20 and V4-R20 consumed the same 20
ordered final-code candidates for each of the 30 preregistered tasks. All 600
per-attempt SHA-256 values were preserved, and every V4 runtime reported identical
input/output pool hashes and a 1.0 match rate.

The source is the preserved 164-task V2 retry-20 ledger generated before V4 was
designed. Each HumanEval ID appears exactly once per attempt index 0..19. The V4
pilot IDs were selected and committed before the pilot and have zero overlap with
V3 debug/pilot IDs.

## Influence checks

- Candidate generation did not load or access the V4 key.
- V4 did not modify code, logits, seeds, stopping, validity, or candidate order.
- Static validity and quality fields came directly from the frozen ledger.
- The only V4 action was selection among saved candidates and one selected-final
  detector replay.
- No candidate was deleted after observing correctness or detection outcomes.

This is a frozen shared-pool replay experiment. It isolates selection effects but
does not claim a new live model-generation run or end-to-end generation latency.
