# V4 Result-to-Claim Gate

- **claim_supported:** no
- **confidence:** high
- **integrity_status:** warn
- **pending_external_review:** true

## Intended claim

V4 was intended to satisfy, jointly: batch-invariant exact final-code-only replay,
the semantic cost and VRAM gates, correctness preservation, calibration and
held-out FPR at or below 5%, and strictly better detection than Current.

## What the results support

- A deterministic final-code-only structural representation can produce 30/30
  strict selected replay in three cold processes and remain invariant to tested
  batch/order/composition/cache schedules.
- Current and V4 can consume exactly the same 600 ordered candidates without key
  influence on generation.
- The tested CPU structural selection plus one final replay costs 0.022659895
  seconds/task and allocates zero CUDA memory.
- The frozen selector produces a positive paired structural-score shift
  (+0.06046875; bootstrap interval entirely positive) that is not reproduced by
  the wrong-key paired score shift.
- Calibration primary and wrong-key positive rates are below 5%.

## What the results do not support

- V4 is not detected more often than Current: both are 0/30.
- V4 does not preserve paired correctness within the frozen tolerance: 15/30
  versus 18/30.
- The score shift is not watermark success and does not establish TPR at 5% FPR.
- There is no held-out FPR, HumanEval-164, robustness, live-generation, multi-seed,
  cross-model, cross-language, or natural-code evidence.
- Candidate C should not be called neural semantic hashing; its formal evidence is
  canonical structural/data-flow evidence.

## Missing evidence

Held-out negative performance, full-benchmark detection/correctness, attacks,
additional seeds/models/languages, live generation latency, and an independent
final result auditor are missing. These experiments are intentionally absent
because the pilot failed and the escalation policy closed them.

## Suggested claim revision

The defensible claim is: “V4 demonstrates a batch-invariant, final-code-only,
strictly replayable and inexpensive structural evidence path, while its frozen
selector fails to yield calibrated detection gains and harms paired correctness
on the preregistered pilot.”

## Next experiments needed

Only a new preregistration may proceed. It should use new splits and IDs and test
a genuinely new evidence/selection mechanism, such as certified discrete neural
representations, efficient integer/reference inference, or a correctness-aware
selection contract frozen before outcomes. V4 threshold or Candidate C must not
be tuned on the current results.

## Routing

Verdict `no`: record the postmortem, preserve artifacts, stop this mechanism, and
do not run ablations. Final external result-to-claim review remains pending because
the launched independent reviewer did not return after the session interruption.
