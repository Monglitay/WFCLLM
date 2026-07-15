# Watermark Mechanism V4 Failure Postmortem

## Terminal condition

V4 closes as a complete preregistered negative result at the 30-task pilot. The
mechanism passed root-cause, exact replay, fairness, calibration-control, security,
cost, VRAM, and absolute pass-count checks, but failed two pilot hard gates:

- V4 TPR was 0/30 and Current TPR was 0/30, so strict improvement was false.
- V4 passed 15/30 and Current passed 18/30, so the paired pass delta was -3 rather
  than the required minimum of -1.

This terminal condition forbids held-out, full, robustness, and further tuning.

## Root cause addressed

Stage A classified the V3 hypothesis as A2: batch-shape-dependent neural inference
is supported, with an additional material whitening component. The earliest
instrumented neural divergence appeared in T5 block 0 self-attention; the saved
formal encoder/projection chain's first persistent generation/R3 difference was
visible at whitening pre-normalization. Physical batch shape was the dominant
factor. Composition and order did not add measurable differences once shape was
fixed. Fixed sequence length alone was insufficient, and deterministic CUDA
flags did not eliminate shape-dependent values.

This evidence explains V3's 27/30 quantized replay but does not establish a
certified universal perturbation bound or identify one CUDA kernel as the sole
cause.

## Why Candidate C was selected

Candidate A obtained exact fixed-shape neural replay but cost 0.310876049 seconds
per task, over twice the 0.1409884 gate. Candidate B's empirical margin probe had
18 erasure-mask mismatch rows and no certified bound. Candidate D added no formal
evidence beyond Candidate C. Candidate C was the only probed mechanism that was
batch invariant, final-code-only, fast, key-isolated, and implementable without a
new unverifiable numerical assumption.

Candidate C succeeded on engineering invariants but failed the scientific joint
claim. It increased the mean signed score by 0.06046875, yet the frozen
calibration tail contained twelve values at or above 0.1875. Under conservative
ties, a pilot score of 0.1875 has p=(1+12)/231=0.0563 and is not positive. The
highest V4 pilot statistic did not exceed this boundary, so V4 produced 0/30
primary decisions.

## Correctness tradeoff

V4 selected a different attempt from Current for 23/30 tasks. Within the same
static quality tier, the structural statistic changed which candidate won. The
paired correctness cells were Current-only 4, V4-only 1, both 14, neither 11.
Thus the selector traded three net correct programs for a positive but
non-detecting score shift. This is not an execution artifact: both arms were run
against the same local HumanEval tests and timeout.

## Targeted repair

The first cold pilot invocation found an implementation defect before any result
artifact was produced: a task whose 20 candidates were all static-ineligible
raised instead of applying the preregistered Current fallback. The one allowed
repair added `fallback_count` and restored Current's frozen fallback ranking. It
was covered by a failing-then-passing regression test and committed as `2ed8a0c`.
No mechanism or outcome-facing parameter changed. The repair budget is exhausted.

## What not to infer

- Exact replay does not imply a useful detector.
- A significant paired score shift is not TPR success.
- Calibration rate is not held-out FPR.
- The saved-pool replay is not a live generation latency study.
- Candidate C's discrete structure should not be described as neural semantic
  generalization.
- V3 and V4 pilot counts should not be pooled because their IDs are disjoint.

## Future work

Any follow-on must be a new mechanism and preregistration. Useful directions are
structural statistics with correctness-aware selection constraints fixed before
outcomes, genuinely certified discrete neural representations, integer/reference
encoders that meet the cost gate, or larger independent evidence families that
improve upper-tail separation. The current split, threshold, pilot IDs, and opened
calibration results must not be reused for confirmatory tuning.
