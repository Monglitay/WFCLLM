# Watermark Mechanism V4 Final Audit

## Verdict

V4 is closed as a **complete negative result with disclosed audit warnings**. No
integrity check found fabricated ground truth, self-normalized scores, candidate
pool inequality, threshold tuning, held-out misuse, secret influence on generation,
or unreported full/robustness execution. The warning is procedural: the final
result audit is non-independent and lacks OS-level file-access tracing.

## Closure checklist

- Independent V4 branch/worktree from V3 commit `8693e08`: PASS.
- V3 worktree clean and HEAD unchanged at `8693e08`: PASS.
- Root-cause controlled matrix and independent design review: PASS.
- Literature review and runnable candidate probes: PASS.
- New V4 calibration/held-out split and freeze: PASS.
- Formal mechanism preregistered before pilot: PASS.
- TDD implementation and existing regression preservation: PASS.
- Debug exactness/performance gates: PASS.
- Pilot candidate pool and exact replay gates: PASS.
- Pilot detection superiority: FAIL (0/30 versus 0/30).
- Pilot paired correctness: FAIL (15/30 versus 18/30).
- Repair budget: PASS, 1/1.
- Held-out/full/robustness stopped after pilot failure: PASS.
- Calibration is not mislabeled as held-out FPR: PASS.
- Score shift is not mislabeled as watermark success: PASS.
- Secret/public artifact boundary: PASS.
- Final independent result audit: WARN, unavailable after interruption.

## Final claim

The formal V4 path achieves the narrow engineering claim of batch-invariant exact
semantic/structural evidence replay at low cost. The broader watermark mechanism
claim is unsupported because detection did not improve and correctness degraded
beyond the gate. The result is suitable as a preserved negative finding, not as a
positive watermark result.
