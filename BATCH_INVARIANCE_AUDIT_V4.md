# V4 Batch-Invariance Audit

## Scope and result

Stage E exercised the three V3 regression IDs (`HumanEval/34`, `/39`, `/71`) and
the five preregistered new debug IDs (`/131`, `/116`, `/119`, `/133`, `/9`). Each
task consumed exactly 20 ordered candidates from the preserved pre-V4 retry pool.

The audit result is **PASS**:

- Candidate-pool equality: 160/160 per cold process, 100%.
- Generation evidence to final-code-only R3 evidence: 8/8 per process, 100%.
- Three cold processes: one unique exact identity hash, 100%.
- Per-task scheduling conditions: 8 tasks × 6 nominal batch sizes × 5 orders,
  all exact in every process.
- Cross-task adversarial composition: forward, reverse, deterministic permutation,
  short-first, and long-first, all exact in every process.
- Cache miss/hit reconstruction: 100% exact.
- EOS all-candidate neural semantic rescoring: zero.

## What was compared

The comparison did not use cosine similarity or a relaxed tolerance. It compared
all frozen discrete evidence fields: sorted unit IDs, canonical-context hashes,
representation bytes, quantized values, erasure masks, signature bits, target
bits, match counts, per-unit numerators and denominators, aggregate numerator and
denominator, independent-unit count, eligibility, empirical p-value, and final
decision.

The common cross-process exact identity SHA-256 is
`a6f89db32ad87d86e97258a5aa304e4bd19f38ff082295de6c47fd70670c560f`.
This hash covers task order, all candidate final-code hashes, Current and V4
selected attempts, every candidate evidence hash, and the full selected
generation/R3 evidence objects. It excludes timings and process labels.

## Interpretation

The formal V4 representation is a pure deterministic function of final code and
the frozen public canonicalization config. Nominal batch size, grouping,
composition, order, padding, CUDA kernel choice, TF32, warm-up, and other
candidates do not participate in this representation. The Stage E matrix is
therefore an implementation regression test of that contract, not a claim that
the V3 neural encoder became batch invariant.

The preserved V3 numerical diagnosis remains unchanged: V3 first diverged in the
encoder and its perturbations were amplified at whitening/quantization. V4 avoids
that unstable neural representation in its signed evidence rather than weakening
the replay comparator.

## Access boundary

Only frozen calibration negatives were used for calibration and key controls.
The V4 held-out detector path was not entered, and no full or robustness run was
triggered during this audit.
