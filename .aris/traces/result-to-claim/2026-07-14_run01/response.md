# Independent Reviewer Response

- `claim_supported: no`
- `confidence: high`

## What the results support

V2 meets the absolute quality floor exactly at 82/164. Its final-code-only
contract is strongly supported by 100% final-code, ledger-code, and
ledger-score replay and generation/final score correlation near 1.0. V2 raises
sufficient evidence from 135/164 to 163/164 and improves AUROC from 0.62262 to
0.73134. It also lowers held-out false positives from four to three relative
to Current.

## What the results do not support

The central claim is not supported. V2 TPR is 0.23171 versus Current's 0.25610,
and the paired TPR difference CI `[-0.12195, 0.07317]` does not support a
positive effect. V2 held-out FPR is 0.07317, above the 0.05 hard limit. Since
the operating point violates the held-out FPR constraint, the TPR must be
described as TPR at the frozen calibration threshold, not as achieved legal
TPR@5%FPR. Higher evidence density and AUROC cannot replace the failed FPR and
TPR gates. Pass has no margin, and one post-gate exploratory seed cannot
support confirmatory or stability claims.

## Missing evidence

An untouched evaluation with a newly frozen independent calibration split and
held-out negative split would be required for any future candidate. The
current held-out panel may not be used for threshold selection. Mechanism
diagnosis should explain the AUROC improvement alongside upper-tail FPR
failure, evidence-count scaling, effective sample size, and Current-only/V2-only
detection discordances.

## Suggested revision

State that V2 improved evidence recoverability, sufficient-evidence coverage,
and exploratory AUROC, but failed held-out FPR and TPR-superiority gates. Do
not claim legal TPR@5%FPR improvement.

## Routing

Retain V2 as experimental/diagnostic, keep Current/v1 as default, preserve the
negative artifacts, and do not continue tuning on the exposed panels.
