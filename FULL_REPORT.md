# HumanEval Full Retry-20 Report

## Final disposition

**Experiment complete; method improvement failed the hard gates.**

The preregistered 30-task pilot did not pass its entry gate. The user later
authorized one direct 164-task run to observe full-data behavior. Therefore,
all results in this report are explicitly classified as **post-gate
exploratory**, not confirmatory. V2 remains experimental and is not promoted
to the default method.

V2-R20 met the absolute quality floor exactly at 82/164, but failed both
watermark-effectiveness gates: held-out FPR was 3/41 = 7.32% (greater than
5%), and TPR at the frozen calibration threshold was 38/164 = 23.17%, below
Current-R20's 42/164 = 25.61%. Because observed held-out FPR exceeded 5%, the
reported TPR is not described as a legally achieved held-out `TPR@5%FPR`.

## Frozen protocol

- Dataset: HumanEval, all 164 tasks.
- Seed: `20260713`; exactly one seed.
- Retry: `evidence_retry_attempts=20` for every new Current and V2 generation.
- Same local generation model, encoder checkpoint, prompts, decoding
  parameters, maximum tokens, task ordering, and execution environment.
- Thresholds frozen from the independent 82-row calibration-negative split.
- Held-out negatives: 41 rows; opened once only after both 164-row arms were
  assembled and closed.
- No threshold, split, seed, task list, detector, or aggregation rule was
  changed after held-out access.
- Full-run classification: `post-gate exploratory`;
  `confirmatory_claim_allowed=false`.

## Primary results

| Metric | Current-R20 | V2-R20 | V2 - Current |
|---|---:|---:|---:|
| HumanEval Pass@1 | 84/164 (51.22%) | 82/164 (50.00%) | -2 tasks (-1.22 pp) |
| TPR at frozen threshold | 42/164 (25.61%) | 38/164 (23.17%) | -4 tasks (-2.44 pp) |
| Held-out FPR | 4/41 (9.76%) | 3/41 (7.32%) | -1 negative (-2.44 pp) |
| AUROC | 0.6226 | 0.7313 | +0.1087 |
| Sufficient evidence | 135/164 | 163/164 | +28 tasks |
| Insufficient evidence | 29/164 | 1/164 | -28 tasks |
| Sufficient but below threshold | 93/164 | 125/164 | +32 tasks |
| Detected | 42/164 | 38/164 | -4 tasks |
| No embedding | not directly comparable | 2/164 | -- |

V2 materially increased evidence coverage and improved rank ordering over the
whole score range, but those improvements did not produce a compliant frozen
operating point or higher TPR.

## Score distributions

| Arm / class | min | p25 | median | p75 | max |
|---|---:|---:|---:|---:|---:|
| Current positive | 0.0000 | 0.1738 | 0.3384 | 0.6021 | 1.3222 |
| Current held-out negative | 0.0000 | 0.0000 | 0.1824 | 0.4365 | 1.0212 |
| V2 positive | -1.3416 | 0.7071 | 1.3416 | 1.8009 | 3.1820 |
| V2 held-out negative | -1.5000 | -0.4472 | 0.0000 | 1.4142 | 2.4749 |

The V2 positive distribution moved upward, but the held-out negative upper
tail also remained high. AUROC increased by 0.1087, yet three of 41 held-out
negatives crossed the frozen threshold. AUROC improvement cannot replace the
preregistered FPR and TPR hard gates.

## Paired statistics

- Pass pairs: Current-only 13, V2-only 11, both pass 71, neither pass 69.
- Detection pairs: Current-only 35, V2-only 31, both detected 7, neither
  detected 91.
- Paired bootstrap 95% CI for V2 minus Current Pass@1:
  `[-0.07317, 0.04878]`.
- Paired bootstrap 95% CI for V2 minus Current TPR:
  `[-0.12195, 0.07317]`.

The TPR point estimate is negative and its confidence interval crosses zero.
There is no paired statistical evidence for V2 superiority. The Pass interval
also does not prove non-inferiority, although V2 satisfies the absolute 50%
floor on this one seed.

## Evidence recovery and replay

Current accepted 365 nominal units (303 unique). Final-code re-extraction
recovered 333 nominal units and 272 unique units, giving 91.23% nominal and
89.77% unique accepted-to-recovered ratios.

V2 selected one final row for every task, recorded 635 accepted canonical
units, and used quality-first no-embedding fallback on two tasks. Its replay
contract was exact:

- final-code score replay: 164/164 (100%);
- selected-ledger final-code match: 164/164 (100%);
- selected-ledger score replay: 164/164 (100%);
- generation-time objective versus final detector score Pearson correlation:
  approximately 1.0;
- detector input: final code plus public configuration and secret key only.

Candidate sidecars, retry ledgers, task correctness, labels, task IDs as hidden
state, generation scores, and selection manifests were not detector inputs.
They were used only for posthoc audit. This supports the V2 engineering
contract, but not the intended effectiveness claim.

## Pilot versus full direction

The repaired 30-task development iteration tied Current TPR (7/30 versus
7/30), exceeded its frozen negative-panel FPR target (3/30 = 10%), and had
higher Pass (16/30 versus 13/30). In the full exploratory run, V2 retained its
evidence/replay advantage and higher AUROC, but Pass changed to -2 tasks and
TPR to -4 tasks versus Current. Evidence density and replay were directionally
consistent; Pass and TPR superiority were not.

## Hard-gate audit

| Required condition | Evidence | Status |
|---|---|---|
| Generation-time key-conditioned watermark | V2 generation and key-conditioned canonical selection | pass |
| Final-code-only detector | strict 164-row inputs and independent replay | pass |
| All new generation retry=20 | frozen commands and ledgers | pass |
| Preregistered pilot passes | failed before user-authorized full run | **fail** |
| V2 Pass@1 at least 82/164 | 82/164 | pass, no margin |
| Held-out FPR at most 5% | 3/41 = 7.32% | **fail** |
| V2 TPR above Current | 38/164 versus 42/164 | **fail** |
| Paired evidence supports improvement | TPR delta CI crosses zero | **fail** |
| Final-code replay 100% | 164/164 | pass |
| No secret leakage | zero exact raw-secret matches | pass |

## Artifact closure

- Current assembled final-code SHA-256:
  `fd6f1d4d9ad2bf7dd394c070650a189403865d38323546d1329ed2b44ec7c373`.
- V2 assembled final-code SHA-256:
  `96911e4ebef4bb42caf545f5f0702ab342372403d814a30dc1c48afcf720cec1`.
- Paired comparison SHA-256:
  `01a28b025363e0358c9ad7febe87e55bfd1130c5aacac52e56466cb8d896fdf8`.
- Held-out split SHA-256:
  `8a0180e9339c1d7c2af7fadcc11ea149d8cce725b972f4c887f2de87f572b8ce`.
- Held-out opened at `2026-07-13T21:51:52+08:00`; scoring completed at
  `2026-07-13T21:52:07+08:00`.
- Current and V2 independent integrity audits each validated 164 strict input
  rows and found zero raw-secret matches across tracked files and the retained
  experiment tree.
- Final verification: 734 tests passed with 27 warnings; compileall passed.

Only one seed was run. No multi-seed mean, standard deviation, cross-seed
stability, or general robustness claim is made.

## Final conclusion

V2 fixed generation/detection objective mismatch, final-format evidence loss,
and much of the evidence-sparsity problem. It did not solve calibrated
upper-tail separation: ordinary negatives still crossed the frozen threshold,
and the extra sufficient evidence mostly became sufficient-but-low-score
positives. The legitimate conclusion is:

> Experiment and audit complete; V2 improved evidence recoverability and
> exploratory AUROC, but method improvement failed the pilot, held-out FPR,
> TPR-superiority, and paired-evidence hard gates.
