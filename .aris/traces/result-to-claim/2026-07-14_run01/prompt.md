# Result-to-Claim Reviewer Prompt

Judge whether the following target claim is supported:

> V2-R20 improves legal TPR@5%FPR over equal-budget Current-R20 while
> preserving HumanEval quality.

Frozen experiment: one seed (`20260713`), HumanEval 164 tasks, retry=20, same
model and decoding. The 30-task pilot failed before the user authorized a
post-gate exploratory full run.

Results supplied to the independent reviewer:

- Current: Pass 84/164, TPR 42/164, held-out FPR 4/41, AUROC 0.62262.
- V2: Pass 82/164, TPR 38/164, held-out FPR 3/41, AUROC 0.73134.
- V2 minus Current: Pass -2 tasks, TPR -0.02439, FPR -0.02439, AUROC
  +0.10872.
- Paired 95% bootstrap CIs: Pass `[-0.07317, 0.04878]`; TPR
  `[-0.12195, 0.07317]`.
- V2 final-code, ledger-code, and ledger-score replay: 100%; generation/final
  Pearson approximately 1.0.

Return `claim_supported`, supported and unsupported findings, missing evidence,
revised claim, next steps, and confidence. Do not inflate a single-seed,
post-gate exploratory result.
