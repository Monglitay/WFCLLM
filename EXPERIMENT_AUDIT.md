# V3 Experiment Integrity Audit

## Verdict

`integrity_status: warn` — no fabrication, split breach, raw-pool inequality, or key leak was found, but the audit is provisional because independent reviewer delegation was unavailable by system policy. The mechanism itself failed its exactness gate; that is a scientific negative result rather than concealed integrity failure.

Auditor: primary Codex fallback. Independent GPT/Codex reviewer: not run; `[pending external review]`.

## Evidence checks

| Check | Evidence | Verdict |
|---|---|---|
| Frozen baseline/branch | V2 base `eac59bf...`; dedicated V3 branch | PASS |
| Preregistration before implementation/pilot | commit `4f3d9eb`; implementation `5f8b908`; pilot report `922a12b` | PASS |
| Raw-pool fairness | recomputed 600/600 UTF-8 final-code SHA-256 values | PASS |
| Retry cardinality/order | every pilot ID has attempts 0..19 exactly once | PASS |
| Detector isolation | payload API rejects anything except `final_code`; CLI supplies public config + key file | PASS |
| Exact generation/R3 replay | 27/30 | **FAIL scientific gate; transparently reported** |
| Execution correctness | offline executor: Current 15/30; V3 15/30 | PASS as measured |
| Calibration/held-out separation | 250 calibration used; 250 held-out unopened | PASS |
| Full-run policy | no HumanEval-164 result artifacts; run not executed | PASS |
| Wrong-key control | calibration 1/250; pilot 1/30 | PASS as diagnostic control |
| Secret hygiene | key files mode 0600, 32 bytes; zero raw-key occurrences in tracked/public files; no key hashes/fingerprints published | PASS |
| Test suite | 819 passed, 27 warnings, 30.18 s, offline | PASS |
| Syntax smoke | compileall exit 0 | PASS |
| Encoder provenance | benchmark-derived historical training scope disclosed | WARN |
| Live runtime equivalence | saved trajectories replayed line-by-line; speculative rollback traces unavailable | WARN |
| Independent audit | prohibited absent explicit user delegation | WARN |

## Metric grounding

Every headline value in `RESULTS_V3.md` is derived from hash-bound JSON/JSONL artifacts listed in `PILOT_AUDIT_V3.json`. The pilot audit script recomputed candidate hashes from raw code, paired task IDs, execution outcomes, detector decisions, timings, memory, and calibration decisions. Bootstrap intervals use 20,000 deterministic task resamples and are descriptive for a single frozen seed.

Calibration-set 4.4% is not relabeled as held-out FPR. Pilot TPR values are not relabeled as legal TPR because exact replay failed. The absence of robustness/full results is explicitly reported, not imputed.

## Caveats requiring external review

The local executor designed, implemented, ran, and audited the experiment. Although raw evidence and failure decisions are preserved, this is not reviewer independence. An external reviewer should re-run hash verification, inspect the quantization mismatch rows, and independently confirm that no output-driven mechanism change occurred beyond the single documented repair.
