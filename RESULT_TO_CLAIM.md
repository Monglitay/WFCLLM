# Result-to-Claim Decision: Watermark Mechanism V2 Retry-20

## Decision

`claim_supported = no` with high confidence for the target claim that V2-R20
improves legal `TPR@5%FPR` over equal-budget Current-R20 while preserving
HumanEval quality. The full run was authorized only after the pilot gate
failed, so it is `post-gate exploratory` rather than confirmatory.

Two decisive full-data conditions reject the target claim:

1. V2 held-out FPR is 3/41 = 7.32%, above the 5% hard ceiling.
2. V2 TPR at the frozen calibration threshold is 38/164 = 23.17%, below
   Current's 42/164 = 25.61%.

Because actual held-out FPR exceeded 5%, the observed TPR must not be packaged
as a legally achieved held-out `TPR@5%FPR`.

## Claim matrix

| Claim | Evidence | Judgment |
|---|---|---|
| V2 is generation-time and key-conditioned | secret-conditioned canonical selection | supported |
| Detector is final-code-only | strict four-field input; 164/164 independent replay | supported |
| Generation objective equals detector score | ledger score replay 164/164; Pearson approximately 1.0 | supported |
| Canonical evidence survives final formatting | final-code and ledger code match 164/164 | supported |
| V2 increases evidence availability | sufficient 163/164 versus 135/164 | supported on this seed |
| V2 preserves the absolute Pass floor | 82/164 = 50.00% | supported exactly, no margin |
| V2 improves whole-range ranking | AUROC 0.7313 versus 0.6226 | supported exploratorily |
| V2 TPR exceeds Current-R20 | 38/164 versus 42/164 | not supported |
| V2 held-out FPR is at most 5% | 3/41 = 7.32% | not supported |
| Improvement has paired statistical support | TPR delta CI `[-0.12195, 0.07317]` | not supported |
| Full result is confirmatory | pilot gate failed before override | not supported |
| Cross-seed stability | one seed only | not evaluated |

## Supported findings

On one frozen seed with equal retry=20 budgets, V2 improved its engineering
contract: sufficient-evidence coverage increased by 28 tasks, final-code
reconstruction was exact, and AUROC increased by 0.1087. Pass@1 was 82/164,
exactly at the required floor. V2 also reduced held-out false positives from
four to three relative to Current.

These are diagnostic subclaims. They demonstrate that objective mismatch and
final-format evidence loss were fixed; they do not establish legal watermark
effectiveness improvement.

## Unsupported findings

- V2 did not increase frozen-threshold TPR; it decreased it by four tasks.
- V2 did not control held-out FPR at or below 5%.
- More sufficient evidence did not produce more detected positives.
- Higher AUROC did not yield a compliant frozen operating point.
- Pass met 50% with no margin and was two tasks below Current.
- Paired confidence intervals include zero and do not support superiority.
- A post-gate exploratory run cannot be relabeled confirmatory.
- One seed cannot support robust or cross-seed claims.

## Precise allowed narrative

> In a post-gate exploratory, single-seed, 164-task HumanEval experiment with
> retry=20, V2-R20 met the minimum Pass@1 floor (82/164), achieved 100%
> final-code replay, increased sufficient-evidence coverage from 135/164 to
> 163/164, and increased AUROC from 0.6226 to 0.7313. However, held-out FPR at
> the frozen threshold remained 7.32%, above 5%, and TPR decreased from 25.61%
> to 23.17%. V2 improved evidence recoverability and whole-range ranking but
> failed the legal FPR and TPR-improvement gates.

## Prohibited claims

- “V2 improves legal TPR@5%FPR over Current-R20.”
- “V2 controls held-out FPR below 5%.”
- “V2 is a successful final method.”
- “V2 is statistically superior.”
- “V2 is robust across seeds.”
- “More canonical evidence caused better final detection.”

## Missing evidence

The current held-out panel has been opened and must not be reused to select a
new threshold or detector. A future candidate requires a new preregistration,
independent calibration and held-out negative splits, and a frozen detector
before evaluation. Existing raw scores may be used only for diagnosis of
calibration-to-heldout tail shift, score scaling versus evidence count,
related-unit effective sample size, and the 35 Current-only versus 31 V2-only
detection discordances.

## Routing

Per the user's final decision, no more experiments or repairs are run. Keep V2
code and tests as explicitly experimental, retain the failed artifacts, and
keep Current/v1 as default. Final disposition: **experiment and audit complete;
method improvement failed the hard gates**.
