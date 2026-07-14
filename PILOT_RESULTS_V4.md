# Watermark Mechanism V4 Pilot Results

## Decision

The frozen 30-task, retry-20 pilot is a **negative result**. Two hard gates failed:

1. V4 passed 15/30 tasks versus Current's 18/30, a paired delta of -3; the
   preregistered minimum was -1.
2. Current and V4 both detected 0/30; V4 therefore did not strictly improve TPR.

Per the preregistration, held-out negatives were not opened in the detector,
HumanEval-164 was not run, robustness was not run, and no result-driven repair or
parameter change was attempted.

## Primary outcomes

| Metric | Current-R20 | CanonicalStructuralReplay-V4-R20 | Paired delta |
|---|---:|---:|---:|
| TPR at frozen empirical decision | 0/30 (0%) | 0/30 (0%) | 0 |
| HumanEval pass | 18/30 (60%) | 15/30 (50%) | -3 |
| Eligible final codes | 25/30 | 28/30 | +3 |
| Mean signed structural score | 0.009938823 | 0.070407573 | +0.060468750 |

The score delta is positive and distributed across the sample rather than driven
by one outlier: 21 positive, 1 negative, and 8 tied task deltas; the largest
positive delta contributes 14.4822% of the positive sum. The paired score-delta
bootstrap 95% interval is [0.035838190, 0.087272983], and the exact two-sided sign
test gives p=0.0000109673. This supports a selection-score shift only. It does not
support successful watermark detection because neither arm produced a primary-key
positive under the frozen calibration rule.

## Correctness statistics

- Pass delta: -3 tasks.
- Paired bootstrap 95% interval for pass-rate delta: [-0.233333, 0.033333].
- McNemar cells: Current-only 4, V4-only 1, both 14, neither 11.
- Exact two-sided McNemar p-value: 0.375.

V4 met the absolute 15/30 pass floor but failed the paired `Current - 1` floor.

## Replay, fairness, and performance

- Ordered raw candidate hashes: 600/600 identical.
- Generation-to-R3 exact discrete replay: 30/30 in each of three cold processes.
- Cold-process exact identity hashes: one unique value,
  `52b9ac6e6cfcf819b9b5165f06cca1f3055cdddaeaf0fd91881e021ddd4bb63e`.
- Cache hit/miss, batch-size, composition, order, and cross-task adversarial
  schedules: 100% exact.
- EOS all-candidate neural semantic rescore count: zero.
- Mean semantic cost: 0.022659895 seconds/task.
- Reduction versus 0.201412 seconds/task: 88.7495%.
- Peak CUDA allocated memory: 0 MiB.

## Calibration and wrong-key controls

The calibration panel contains 230 newly frozen MBPP train/validation negatives.
The public calibration reference was fitted without the primary private key.

- Primary-key calibration positives: 2/230 = 0.8696%.
- Domain-separated wrong-key calibration positives: 1/230 = 0.4348%.
- Wrong-key pilot positives on Current: 0/30.
- Wrong-key pilot positives on V4: 1/30.
- Wrong-key paired mean score shift: -0.022751323, opposite the primary-key
  +0.060468750 shift.

Both calibration controls pass the 5% gate, and the wrong key does not reproduce
the primary paired score shift. The lone wrong-key V4 pilot positive is reported
without deletion or reinterpretation.

## Repair accounting

The first pilot cold-process attempt terminated before producing outcome artifacts
because a pool with no static-quality-eligible candidate raised an exception. The
frozen public config explicitly required the existing Current static-quality
fallback for this case. One targeted repair restored the preregistered ranking
`(quality_tier, -fallback_count, -attempt_index)` and added a regression test.
It did not change the representation, calibration, threshold, key, precision,
pilot IDs, retry count, or exact comparator. Commit: `2ed8a0c`.

This consumed the entire repair budget of one.

## Relationship to V3

V3's preserved, different 30-task pilot reported Current 1/30 versus V3 7/30 TPR,
with 15/30 pass in both arms and 27/30 exact replay. V4 uses a disjoint pilot, so
those counts are historical context and are not pooled with V4 statistics. V4
solves exact replay and performance, but its frozen structural statistic does not
cross the calibrated decision boundary on this pilot.
