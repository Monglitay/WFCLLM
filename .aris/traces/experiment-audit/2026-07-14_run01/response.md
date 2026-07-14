# Independent Read-Only Audit Response

Snapshot SHA-256:
`761b7536086bd8bb793bf25c871d8194b1b2ecbc2b918fbb83a183e05b754ff7`.
Twenty-seven files were reviewed without modification.

## Verdict

- Before stale-report remediation: **FAIL**.
- After reports are replaced with actual full results: **WARN**.
- Target method-improvement claim: **unsupported**.
- Honest negative-result claim: **supported**.

The only explicit pre-remediation integrity failure was that
`FULL_REPORT.md:3-30`, `EXPERIMENT_INTEGRITY_AUDIT.md:47-50,183-191`, and
`RESULT_TO_CLAIM.md:23-24,58-64` said full evaluation had not run and held-out
negatives were unopened. This contradicted `postprocess_manifest.json:23-54`
and `paired_comparison.json:4-410`. Replacing those three reports before commit
converts result existence/numeric agreement to PASS.

## A. Ground-truth provenance: WARN

No fake-ground-truth pattern was found. Both execution JSONLs contain 164
unique HumanEval IDs and recompute to Current Pass 84 and V2 Pass 82.
`scripts/wfcllm_compare_arms.py:105-115,275-311` requires a boolean execution
result for every ID. However, the 27-file snapshot omitted the executor
implementation, official test-loader invocation, dataset test hash, and final
code, so official HumanEval provenance could not be proved from the snapshot
alone.

## B. Score normalization: PASS

No model-output max/min/mean normalization was found. V2 uses the predefined
Bernoulli(0.5) bit-null z score described in `V2_METHOD.md:54-68`; raw score,
p-value, threshold, matched bits, and total bits remain in retained details.
Calibration and detection are separate in
`scripts/wfcllm_v2_detect.py:118-135`. Paired reports consume raw decisions
without arm-wise rescaling (`scripts/wfcllm_compare_arms.py:322-329`).

## C. Result existence/number match: FAIL before remediation, PASS after

Independent JSONL recounts match all published values:

| Metric | Current | V2 |
|---|---:|---:|
| unique positive IDs | 164 | 164 |
| unique held-out IDs | 41 | 41 |
| unique execution IDs | 164 | 164 |
| Pass | 84 | 82 |
| detected positives | 42 | 38 |
| held-out false positives | 4 | 3 |
| TPR | 25.61% | 23.17% |
| FPR | 9.76% | 7.32% |
| AUROC | 0.6226204640 | 0.7313355146 |

All 41 negative IDs equal the frozen preregistration held-out list. All
manifest-listed snapshot hashes match. The standalone detection reports lack
execution labels and correctly expose missing Pass coverage; final Pass values
must come from execution/paired artifacts, as the remediated reports do.

## D. Dead code: PASS for included scripts

The comparison, V2 replay, integrity audit, and V1/V2 detector handlers are
called on paths with retained output artifacts. The archived `dual` evaluator
explicitly returns an error and is not represented as a completed metric. This
assessment covers the six scripts in the snapshot, not the entire repository.

## E. Scope: WARN

The actual scope is HumanEval, Current-R20 and V2-R20, one seed, retry=20, one
41-row held-out panel, and a full run authorized after pilot failure.
`v2_retry20_preregistration.json:6-7,148-157` and
`postprocess_manifest.json:23-26` enforce single-seed exploratory language.
No confirmatory, multi-seed, comprehensive, or robustness claim is supported.

## F. Evaluation type: PASS with classification

HumanEval Pass is intended real-GT, subject to the A provenance warning.
Watermark metrics are controlled treatment-label statistical evaluation:
watermarked generation is positive and frozen ordinary code is negative.
Replay is deterministic integrity evidence, not an effectiveness metric.

## Additional checks

- Held-out isolation: **WARN**. Manifest records one opening after assembly and
  41 frozen unique IDs, but the snapshot lacks an append-only access log.
- Final-code-only: **PASS for V2**. V2 detect accepts input and calibration only
  (`scripts/wfcllm_v2_detect.py:60-71,128-134`); replay scores final code before
  optional ledger comparison (`scripts/wfcllm_v2_replay.py:222-249`).
- Schema isolation: **WARN**. Versioned outputs are consistent, but calibration
  loaders and compatibility tests were outside the snapshot.
- Seed: **PASS**, consistently `20260713`.
- Retry: **WARN in snapshot**. Preregistration and postprocess say 20, but raw
  full attempt ledgers were not included.
- Secret scan: **PASS, artifact-backed**. The implementation scans tracked and
  experiment files and keeps the key out of reports
  (`scripts/wfcllm_integrity_audit.py:136-200`); Current scanned 779 files and
  V2 780, both with zero matches.

## Claim impact

Supported within the exploratory seed: generation-time/key-conditioned/final-
code-only V2 contract, 164/164 deterministic replay, objective alignment,
evidence coverage increase, and AUROC increase. Unsupported: held-out FPR at
or below 5%, V2 TPR superiority, paired statistical superiority, successful
final method, or cross-seed stability.

Recommended post-remediation fields:

```text
overall_verdict = WARN
integrity_status = warn
result_to_claim = unsupported
numeric_full_gate_passed = false
```

WARN reflects independent snapshot scope, not evidence of fabricated results.
Even if additional provenance/ledger/access evidence upgraded integrity to
PASS, method effectiveness would remain failed.
