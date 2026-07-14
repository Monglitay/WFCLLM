# V4 False-Positive-Rate Audit

## Calibration controls

The calibration-only panel contains 230 V4-negative examples from a new MBPP
train/validation split after exact ID, code-content, AST, V3 test, and HumanEval
overlap filtering. The fitted empirical reference is public and key-independent.

- Primary key: 2/230 positives, 0.8696%.
- Domain-separated wrong key: 1/230 positives, 0.4348%.
- Target: at most 5%.

Both calibration controls pass. Conservative greater-than-or-equal ties and the
minimum-three-independent-unit rule were used.

## Held-out status

No held-out FPR exists for V4. The held-out file was created and hashed at split
freeze, but the detector path was not opened because the pilot failed two hard
gates. Calibration FPR is not described as held-out FPR anywhere in the V4
result. HumanEval pilot decisions are positive-arm TPR observations and are not
used as negative FPR evidence.
