# WFCLLM V2 Retry-20 Experiment Integrity Audit

**Experiment date:** 2026-07-13

**Finalization date:** 2026-07-14

**Project:** WFCLLM

**Classification:** post-gate exploratory
**Integrity status after report remediation:** **WARN**

## Overall verdict

The retained snapshot's code, manifests, execution outputs, detection details,
replay records, and independent secret scans are internally consistent. There
is no evidence of fabricated ground truth, self-normalized performance,
phantom result files, detector dependence on generation sidecars, threshold
retuning after held-out access, or raw-secret leakage.

The audit status is WARN rather than PASS for two reasons. First, the experiment
is one seed on one benchmark and the full run occurred only after the
preregistered pilot gate failed. Second, the independent 27-file snapshot did
not contain the HumanEval executor implementation, final-code bodies, complete
full retry ledger, calibration loader/tests, or an append-only held-out access
log. These limitations cap the independent evidence; they do not contradict
the reported negative result.

The read-only audit snapshot initially found a C-class documentation failure:
`FULL_REPORT.md`, `EXPERIMENT_INTEGRITY_AUDIT.md`, and `RESULT_TO_CLAIM.md`
still said that full evaluation had not run and held-out negatives were
unopened. Those files predated the user-authorized full run. They were replaced
before submission with the complete 164-task negative result. The stale claims
are not retained in the final commit.

## A. Ground-truth provenance: WARN

No evidence indicates that model outputs were used as ground truth. The
comparison tool reads retained per-task execution rows and requires a boolean
`passed` field for every frozen ID (`scripts/wfcllm_compare_arms.py:105-115`,
`275-311`). The final Pass counts independently recompute from 164 unique rows
per arm as Current 84 and V2 82.

However, the snapshot did not include the actual HumanEval executor, official
test-loader invocation, dataset test hash, or final-code bodies. Therefore the
reviewer could not prove official-dataset provenance from this snapshot alone.
HumanEval Pass is intended `real_gt` with this provenance warning; detection is
a controlled treatment-label statistical evaluation.

## B. Score normalization: PASS

No metric is normalized by the maximum, minimum, or mean of the model's own
outputs. The comparison tool consumes raw frozen detector decisions, observed
FPR, TPR, and AUROC, and computes paired deltas directly
(`scripts/wfcllm_compare_arms.py:254-262`, `330-372`). The V2 detector loads a
versioned calibration artifact and scores strict input records
(`scripts/wfcllm_v2_detect.py:128-144`). Raw score and p-value quantiles are
retained alongside threshold decisions.

## C. Result existence and numeric agreement: PASS after remediation

The postprocess manifest records both 164-row final-code hashes, positive and
held-out detail hashes, execution hashes, replay hash, paired-comparison hash,
and a `complete` status. Its metrics are Current Pass 84, TPR 0.25609756, FPR
0.09756098, AUROC 0.62262046; V2 Pass 82, TPR 0.23170732, FPR 0.07317073,
AUROC 0.73133551
(`data/experiments/watermark_v2_retry20_20260713/full_exploratory/full_results/postprocess_manifest.json:3-15`,
`40-52`). These numbers match `paired_comparison.json`, both detection reports,
and the final reports.

The initial snapshot documentation mismatch was a real FAIL and was treated as
a release blocker. Updating all three stale reports converts this check to PASS;
the audit trace preserves the pre-remediation finding.

## D. Dead-code and pipeline-use assessment: PASS with limited scope

The metric functions used for the final claim are on the executed postprocess
path: execution rows are loaded and ID-checked, arm summaries are built,
McNemar discordances and paired bootstrap intervals are computed, and the
result is written by the comparison script
(`scripts/wfcllm_compare_arms.py:275-372`). Replay code re-scores final code,
checks ledger code hashes and scores, and emits aggregate replay rates
(`scripts/wfcllm_v2_replay.py:212-277`). Their outputs exist and are hashed in
the complete manifest. No claim depends on an uncalled experimental metric.

## E. Scope assessment: WARN

The actual scope is HumanEval, two equal-budget arms, one frozen seed, one
pilot development iteration, and one post-gate exploratory full run. The
preregistration explicitly says `single_seed_only=true` and prohibits
cross-seed claims
(`configs/wfcllm/v2_retry20_preregistration.json:6-7`, `157`). Final documents
use that restricted language. Claims of robustness, generality, confirmatory
success, or multi-seed stability are unsupported and prohibited.

## F. Evaluation classification: PASS

- HumanEval Pass@1: `real_gt`.
- Watermark detection: key-conditioned statistical evaluation against frozen
  ordinary-code calibration and held-out negative panels.
- Replay: deterministic self-consistency/integrity check, not an effectiveness
  metric.
- No human-judge or model-generated-reference evaluation is represented as
  ground truth.

## Held-out isolation: WARN

The preregistration freezes target FPR 0.05, calibration/held-out task IDs, and
the rule that held-out rows open once after full runs and never tune the
threshold (`configs/wfcllm/v2_retry20_preregistration.json:58-117`). The
postprocess manifest records one held-out opening at
`2026-07-13T21:51:52+08:00` and completion 15 seconds later. Both arms exceeded
5% FPR, and this failure is reported rather than repaired post hoc. The
snapshot lacks an append-only access log or supervisor source, so it proves
consistency with one post-closure opening but cannot independently exclude an
unrecorded earlier read.

## Final-code-only and sidecar isolation

The V2 CLI describes and implements strict final-code-only detection
(`scripts/wfcllm_v2_detect.py:48-65`, `118-134`). Replay publishes the detector
input contract as `id`, `dataset`, `prompt`, and `final_code`, re-scores final
code independently, and separately labels ledger checks as audit evidence
(`scripts/wfcllm_v2_replay.py:223-258`). Sidecars and ledgers are not detector
inputs.

## Schema compatibility: WARN

The preregistration freezes `wfcllm-method/v2` and
`wfcllm-canonical-unit/v2` (`configs/wfcllm/v2_retry20_preregistration.json:52-53`).
Replay rejects retry ledgers whose schema is not the V2 ledger schema and
rejects duplicate selected rows (`scripts/wfcllm_v2_replay.py:112-120`). The
calibration aggregation is described as part of the public V2 artifact. The
snapshot did not include the calibration artifact, loader implementation, or
compatibility test output, so fail-fast v1/v2 isolation was not independently
rerun by the reviewer. Retained versioned outputs and repository test evidence
support the contract, but the snapshot evidence alone is limited.

## Retry and seed audit: seed PASS, retry WARN

The preregistration freezes seed `20260713`, retry 20, and retry seed stride
101 (`configs/wfcllm/v2_retry20_preregistration.json:6`, `44-45`). The complete
postprocess manifest records the same seed and retry
(`postprocess_manifest.json:50-52`). Generation/transition audit recorded 2,680
V2 remaining-ledger rows (134 tasks times 20) and 600 pilot rows. Current and
V2 assemblies each contain 164 unique frozen IDs. The independent snapshot did
not carry the full raw retry ledger, so its reviewer verified retry=20 metadata
and selected records but could not recount all 20 attempts per task locally.

## Secret and public-artifact audit

The integrity tool validates strict detector input counts, public artifact
schemas, Git identity, and exact raw-secret absence without placing the secret
in its report (`scripts/wfcllm_integrity_audit.py:136-200`). It explicitly
rejects a secret file inside the scan set and clears the in-memory secret before
building output (`169-175`).

Independent final scans reported:

- Current: 164/164 strict rows; 708 staged/tracked files and 794 total public
  files scanned; zero matches.
- V2: 164/164 strict rows; 708 staged/tracked files and 794 total public files
  scanned; zero matches.
- Git branch: `codex/watermark-mechanism-v2-retry20`.
- Audit-time HEAD: `559398019e901cc13a1be40816f66640251f481b`.

No model, dataset cache, experiment tree, private key, password, `env.sh`, or
temporary log is staged for the final commit.

## Replay and output closure

- Current final-code SHA-256:
  `fd6f1d4d9ad2bf7dd394c070650a189403865d38323546d1329ed2b44ec7c373`.
- V2 final-code SHA-256:
  `96911e4ebef4bb42caf545f5f0702ab342372403d814a30dc1c48afcf720cec1`.
- Paired comparison SHA-256:
  `01a28b025363e0358c9ad7febe87e55bfd1130c5aacac52e56466cb8d896fdf8`.
- V2 replay SHA-256:
  `31ab6d34c2cfa371206d78c0df5a74ee0c93d036d38394329ded1e8796594925`.
- Final-code, ledger-code, and ledger-score replay: 164/164 each.

## Verification

The final offline test command passed **734 tests** with 27 warnings. The final
`compileall` command passed for `wfcllm`, `run.py`, `scripts`, and `tools`.
Warnings were collection/deprecation warnings, not test failures.

## Claim impact

Final numeric ledger:

| Arm | Pass@1 | Frozen-threshold TPR | Held-out FPR | AUROC |
|---|---:|---:|---:|---:|
| Current-R20 | 84/164 (51.22%) | 42/164 (25.61%) | 4/41 (9.76%) | 0.6226 |
| V2-R20 | 82/164 (50.00%) | 38/164 (23.17%) | 3/41 (7.32%) | 0.7313 |

Supported:

1. V2 is a generation-time, key-conditioned, final-code-only experimental
   watermark implementation.
2. Canonical evidence is deterministically recoverable from final code.
3. On this exploratory seed, V2 increases evidence coverage and AUROC.

Unsupported:

1. V2 improves legal held-out `TPR@5%FPR` over Current-R20.
2. V2 controls held-out FPR at or below 5%.
3. V2 is a successful final/default method.
4. V2 is statistically superior or stable across seeds.

Final audit conclusion: **no evidence of result fabrication or secret/sidecar
leakage; independent evidence scope is WARN; method effectiveness claim is
rejected.**
