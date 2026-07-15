# V4 Experiment Integrity Audit

**Date:** 2026-07-15

**Auditor:** primary agent, non-independent

**Overall verdict:** WARN
**Integrity status:** warn (no data-integrity FAIL; final cross-agent review unavailable)

The final independent read-only auditor was launched but did not return a usable
report before the session interruption. This report is therefore explicitly a
non-independent primary-agent audit. The Stage A root-cause design did receive an
independent read-only review; that does not make this final result audit independent.

## A. Ground-truth provenance: PASS

HumanEval correctness used locally cached benchmark tests loaded by
`wfcllm.datasets.loaders.local.load_test_cases` and executed by
`scripts/wfcllm_humaneval_exec.py`. The 30 Current and V4 inputs contained the
same preregistered IDs. Ground truth was not generated from model outputs.

Classification: `real_gt` for correctness; `self_supervised_statistical` for
watermark detection and replay identity.

## B. Score normalization: PASS

The formal score is the raw sum of fixed per-unit numerators divided by the raw
sum of fixed 32-bit denominators. No value is divided by a model-output maximum,
mean, or pilot statistic. Empirical p-values use only the frozen 230-score
calibration artifact and conservative greater-than-or-equal ties. Raw score,
numerator, denominator, units, p-value, and decision are retained.

## C. Result existence and recomputation: PASS

The tracked `PILOT_AUDIT_V4.json` matches the ignored raw pilot audit SHA-256 at
creation. Recomputed primary figures are Current/V4 TPR 0/30 versus 0/30,
Current/V4 pass 18/30 versus 15/30, mean paired score delta +0.06046875,
semantic time 0.022659895 seconds/task, and two failed gates (`pass_delta`,
`tpr_strict_improvement`). The three cold-process exact identity hashes match.

## D. Code-path and dead-code review: WARN

The formal pilot executed `CandidateRuntime.select`, strict final-code-only
`V4Detector.detect`, one `replay_selected`, the HumanEval executor, and frozen
statistics code. The public cache was exercised by dedicated audit paths and tests
but is not integrated into `CandidateRuntime.select`; therefore cache invariance is
supported, while a claim of production runtime cache acceleration would not be.
The measured runtime did not rely on cache hits, so the cost result is conservative
with respect to that omission.

## E. Scope: WARN

The positive experiment is a single-seed, saved-pool, 30-task HumanEval pilot.
It is sufficient to reject the preregistered joint claim, but insufficient for
cross-seed, cross-model, cross-language, live-generation, natural-code, held-out
FPR, full-benchmark, or robustness claims. Reports consistently state these limits.

## F. Candidate-pool and retry integrity: PASS

Every preregistered ID had exactly 20 attempts indexed 0..19. Current and V4 used
the same 600 ordered final-code hashes. V4 did not modify logits, stopping, seeds,
validity, code, or order. The pool predates V4 and candidate generation never
loaded the V4 key.

## G. Threshold, split, and access isolation: PASS with limitation

The public calibration reference and target FPR were frozen before the pilot.
V4 calibration and held-out panels are disjoint from V3 and filtered for ID/code/
AST/HumanEval overlaps. All recorded application flags say held-out was not passed
to a detector, and no full or robustness artifact directory exists. No OS-level
syscall tracing was active, so this is an application/artifact proof rather than a
kernel-forensic proof.

## H. Replay and runtime integrity: PASS

Three independent cold processes produced one exact identity hash. Generation/R3,
batch/order/composition, cross-task composition, and cache hit/miss comparisons
were strict over all discrete fields, not cosine similarity. The formal V4 path is
CPU structural processing, uses no encoder, and allocated zero CUDA memory.

## I. Repair and post-hoc tuning: PASS

One of one allowed repairs restored the preregistered Current fallback for a pool
with no static-eligible candidate. A failing regression test preceded the fix.
No threshold, mechanism, representation, precision, split, ID, retry, or evidence
field was changed. No post-pilot tuning occurred.

## J. Secret/public boundary: PASS

Both V4 key files remain mode 0600 and ignored. Staged scans before every commit
found zero raw-key matches. Final exact-byte scans cover tracked and public raw
artifacts and emit no key hash or fingerprint. The user later relaxed key
publicity, but the stricter original boundary was retained.

## Claim impact

- Batch-invariant exact discrete replay: supported within the tested V4 path.
- Performance below the frozen gate: supported on this server and saved pool.
- Detection superiority and correctness preservation: unsupported.
- Held-out/full/robustness/generalization: no evidence; correctly not run.
- Overall watermark-success claim: rejected.
