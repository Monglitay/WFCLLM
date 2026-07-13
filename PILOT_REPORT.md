# Watermark Mechanism V2 Retry-20 Pilot Report

## Decision

The frozen single-seed 30-task pilot is complete. The result is negative: the
original V2 and its single allowed development repair both fail the
preregistered watermark gates. The repaired candidate is therefore not the
final method, V2 remains experimental, and the 164-task confirmation is not
permitted by the preregistration.

After this pilot decision was frozen, the user explicitly authorized one
164-task run. That later run is a separate `post-gate exploratory` validation:
it does not change any pilot metric, cannot make this gate pass retroactively,
and cannot be described as preregistered confirmation.

The quality gate did work: Current-R20 passed 13/30 tasks, original V2-R20
passed 15/30, and repaired V2 passed 16/30. The watermark claim did not work.
Original V2 detected 4/30 positives, below Current's 7/30. The one repair
recovered 7/30 but only tied Current, while its frozen pilot-negative FPR was
3/30 (10%), above the 5% gate.

## Frozen setting

- Seed: `20260713`, one seed only.
- HumanEval IDs: the exact 30 IDs in
  `configs/wfcllm/v2_retry20_preregistration.json`, in frozen order.
- Model: local DeepSeek-Coder-1.3B-Instruct, BF16, no 4-bit loading.
- Encoder: local CodeT5-base, LoRA, BF16, embedding dimension 128.
- Encoder checkpoint SHA-256:
  `9da80dfdbb3f8e682298e588e822f15967e460c9dfd640a243aad8ed1ba51d25`.
- Decoding: max 256 new tokens, temperature 0.25, top-p 0.95, top-k 0,
  retry repetition penalty 4.0.
- Every generated arm used exactly 20 full-completion attempts. No new
  retry-3 or retry-5 run was used.
- Calibration negatives: frozen 82-row split. The threshold is a conservative
  strict upper order statistic followed by `nextafter(+infinity)`.
- Pilot FPR panel: frozen 30-row development panel. It is not the held-out
  41-row final panel.
- At the time of this pilot decision, the held-out 41-row negative panel had
  not been passed to a detector. Any later, once-only use belongs exclusively
  to the user-authorized post-gate exploratory full report.

## Primary results

| Arm | Pass@1 | TPR at frozen 5% threshold | Pilot-negative FPR | AUROC | No embedding | Insufficient | Sufficient, low score |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current-R20 | 13/30 (43.33%) | 7/30 (23.33%) | 4/30 (13.33%) | 0.7267 | 4 | 6 | 17 |
| Original V2-R20, trimmed mean | 15/30 (50.00%) | 4/30 (13.33%) | 3/30 (10.00%) | 0.6944 | 0 | 0 | 26 |
| V2 development repair, standardized bits | 16/30 (53.33%) | 7/30 (23.33%) | 3/30 (10.00%) | 0.7128 | 0 | 0 | 23 |
| Preregistered ablation: V2 detector + v1 selection | 13/30 (43.33%) | 2/30 (6.67%) | 3/30 (10.00%) | 0.3822 | n/a | 2 | 26 |

The Current-R20 pilot-panel FPR also exceeds 5%; that does not excuse the V2
gate. The candidate was required to have FPR at most 5% and TPR strictly above
Current under the frozen rules.

### Frozen thresholds

- Current v1: `0.477098868374439`.
- Original V2 16-bit trimmed mean: `0.6562500000000001`.
- Repaired V2 standardized bit sum: `1.9445436482630059`.

No threshold was selected from the 30-row pilot panel. The repair changed the
prespecified statistic and was recalibrated once on the same independent
82-row calibration split; it did not reuse the pilot or held-out panels for
threshold selection.

## Score distributions

Quantiles are min / p25 / median / p75 / max.

| Arm | Positive scores | Pilot-negative scores |
|---|---|---|
| Current-R20 | 0 / 0.1633 / 0.3482 / 0.6117 / 1.3222 | 0 / 0 / 0 / 0.1838 / 1.0212 |
| Original V2 | 0.4375 / 0.5508 / 0.5703 / 0.6250 / 0.7813 | 0.3750 / 0.5156 / 0.5313 / 0.5625 / 0.7500 |
| Repaired V2 | -0.7071 / 0.7539 / 1.0607 / 1.7166 / 3.1820 | -2.0000 / 0 / 0.5949 / 1.0317 / 2.8284 |

Original V2 exhibits a broad median shift, not merely one high outlier, but its
95th-percentile positive and negative scores are both approximately 0.7047.
The repair also shifts the median, yet its upper tails remain close: positive
p95 is 2.4004 and negative p95 is 2.3158. That is why a legal calibration tail
does not translate the central shift into higher TPR.

Wrong-key behavior was directionally correct. For the repaired selected code,
the derived-wrong-key score maximum was 1.5, below the 1.9445 threshold. This
does not rescue the ordinary-negative FPR or comparative TPR failures.

## Evidence recovery and objective alignment

| Arm | Accepted evidence | Recovered evidence | Recovery ratio | Generation/final score correlation |
|---|---:|---:|---:|---:|
| Current-R20 | 70 events, 50 unique | 67 events, 47 unique | 0.957 events, 0.940 unique | 0.188 accepted count; 0.324 unique count |
| Original V2 | 105 unique canonical units | 105 | 1.000 | 1.000 |
| Repaired V2 | 111 unique canonical units | 111 | 1.000 | 1.000 |

All 30 original V2 selected-ledger hashes and generation scores exactly match
an independent final-code detector process. For the repair, all 30 selection
manifest hashes and scores also exactly match final-code re-extraction. Repeated
final-code replay is 30/30 for both. The official detector reads only
`id`, `dataset`, `prompt`, and `final_code`, then scores only `final_code` plus
public configuration and the private key.

The replay evidence falsifies “embedding is lost by final formatting” as the
pilot bottleneck. It also shows that making generation and detection exactly
isomorphic is not sufficient for a legal TPR improvement.

## Quality, length, retry, and generation cost

The original V2 quality gate admitted 554/600 raw attempts. Every selected
original and repaired V2 candidate was syntax-valid and executable-bodied;
neither arm emitted `no_embedding`. Current-R20 emitted two syntax errors and
one timeout in the post-selection correctness harness.

For repaired V2:

- selected retry index: mean 6.43, median 5, range 0--19;
- score versus retry index Pearson correlation: 0.048;
- score versus final-code length Pearson correlation: -0.166;
- score versus Pass/fail point-biserial correlation: 0.018;
- mean score: 1.248 for passed code and 1.218 for failed code;
- mean length: 620 characters for passed code and 823 for failed code;
- selected candidates were 30/30 syntax-valid;
- all-attempt internal v1 candidate counter: mean 2.74;
- repaired selected-attempt internal counter: mean 2.80.

The near-zero score/Pass correlation indicates that the static gate prevented
obvious damage without using hidden tests, while the watermark objective itself
did not predict correctness. Longer failed completions remain a quality risk,
but the repair did not select by raw length.

The selected-attempt sidecars contain 1,098 candidate observations for
Current-R20 (36.6/task) and 859 for original V2 (28.6/task). The repaired arm
reuses saved attempts, so it has no additional generation cost. Current screen
wall time was about 46m41s (93.4s/task); original V2 screen wall time was at
most 49m14s (98.5s/task). Each arm used 600 full attempts. No GPU-heavy job ran
concurrently.

## Preregistered ablation

The sole ablation reselected the V2 attempt ledger with the old v1 accepted-hit
key and then used the V2 detector. Its final-code SHA-256 is
`110f545b23411f2db116b70308c1dc802b4dcc8b625218d497b07a9ea2ab866a`,
exactly the Current-R20 final-code hash. This independently confirms that the
two generated arms used the same raw attempts and that their material
difference was the final selection rule.

The ablation had Pass 13/30, TPR 2/30, and AUROC 0.3822. Original aligned V2
selection raised those to Pass 15/30, TPR 4/30, and AUROC 0.6944. Thus aligned
selection has a real mechanism effect relative to using the old selector with
the new detector. It nevertheless fails to outperform the complete Current-R20
system at the legal operating point.

## Paired statistics

Original V2 versus Current-R20:

- Pass difference: +2/30 (+0.0667), paired bootstrap 95% CI
  `[-0.1000, 0.2333]`.
- TPR difference: -3/30 (-0.1000), paired bootstrap 95% CI
  `[-0.3000, 0.1000]`.
- Pass discordance: Current-only 3, V2-only 5, both 10, neither 12.
- Detection discordance: Current-only 7, V2-only 4, both 0, neither 19.

Repaired V2 versus Current-R20:

- Pass difference: +3/30 (+0.1000), paired bootstrap 95% CI
  `[-0.0667, 0.2667]`.
- TPR difference: 0/30, paired bootstrap 95% CI
  `[-0.2333, 0.2333]`.
- Pass discordance: Current-only 2, V2-only 5, both 11, neither 12.
- Detection discordance: Current-only 6, V2-only 6, both 1, neither 17.

The intervals are wide, as expected for a screening pilot, and do not support
an effectiveness-improvement claim.

## Sample-level failure taxonomy

For repaired V2 versus Current:

- Current-only Pass: `HumanEval/54`, `HumanEval/92`.
- V2-only Pass: `HumanEval/87`, `HumanEval/154`, `HumanEval/39`,
  `HumanEval/123`, `HumanEval/147`.
- Both Pass: `HumanEval/56`, `HumanEval/122`, `HumanEval/55`,
  `HumanEval/13`, `HumanEval/34`, `HumanEval/48`, `HumanEval/52`,
  `HumanEval/50`, `HumanEval/46`, `HumanEval/22`, `HumanEval/152`.
- Current-only detected: `HumanEval/65`, `HumanEval/122`, `HumanEval/55`,
  `HumanEval/13`, `HumanEval/0`, `HumanEval/32`.
- V2-only detected: `HumanEval/87`, `HumanEval/145`, `HumanEval/34`,
  `HumanEval/100`, `HumanEval/50`, `HumanEval/22`.
- Both detected: `HumanEval/111`.

The different detected IDs and only one shared true positive show that the tie
in TPR is not a stable taskwise dominance hidden by aggregation.

## Gate audit

| Pilot requirement | Original V2 | Single repair |
|---|---|---|
| Pass at least 15/30 | pass, 15 | pass, 16 |
| No more than one Pass below Current | pass, +2 | pass, +3 |
| TPR strictly above Current 7/30 | fail, 4 | fail, 7 (tie) |
| Broad positive shift | partial pass | partial pass |
| Pilot-negative FPR at most 5% | fail, 10% | fail, 10% |
| Recovery not worse | pass, 100% | pass, 100% |
| Final-code replay 100% | pass | pass |
| No sidecar detector input | pass | pass |

Because the only repair still fails two hard gates, the preregistered stopping
decision was required rather than optional. The later user-authorized full run
is explicitly outside that confirmatory path and does not alter this audit.

## Hypothesis outcomes

- **H1, wrong selection objective:** partially supported. Alignment makes
  generation/final correlation exactly 1.0 and improves strongly over the v1
  selection ablation, but it does not beat Current-R20 TPR.
- **H2, correlated/repeated evidence:** canonical deduplication eliminates
  insufficient evidence and achieves perfect recovery, but the predicted legal
  TPR/AUROC improvement is falsified. The remaining ordinary-negative tail is
  too heavy.
- **H3, static quality gate:** supported on the pilot. Both V2 variants avoid
  syntax/truncation failures and meet the Pass screen. This is a pilot-only,
  single-seed result, not a full-dataset guarantee.

## Key artifact hashes

- Current final code: `110f545b23411f2db116b70308c1dc802b4dcc8b625218d497b07a9ea2ab866a`.
- Original V2 final code: `0411a7939da8c17e4e01db6013583477dfb16b1a36023569f435094b38ad2974`.
- Original V2 retry ledger: `ee0fde132edb83afbc6a131b60eac575799deda1f97cddcd9afacfd6d9217ebb`.
- Repaired final code: `6a307816c36bf9a34e7cdb4ee39c3ca6ebd67f6c8e696d5b9c39a9a224f4a2d6`.
- Repaired selection manifest: `80f4e35c8a7b5fe4371bd8b99bf64bdc3bbaa07f5db29e1602f69a1624074fbf`.
- Repaired calibration: `3af57a28a78cb3326c0b7cd243f95c357a6cf06deb246f46fe8654dabb3f69dd`.
- Repaired comparison: `a8d16fa22c047e15393de58b8b9260d682a860d0cdaf83a8a261f5d2be4a11d0`.

Raw run artifacts remain untracked under `data/experiments/`; reports record
their hashes. They are audit inputs, never detector inputs.
