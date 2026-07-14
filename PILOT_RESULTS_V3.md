# Watermark Mechanism V3 Pilot Results

## Verdict

The preregistered 30-task, retry-20 pilot is a **hard-gate failure and complete negative result**. After the single permitted implementation repair, generation-time observation and R3 final-code-only reconstruction were exactly equal for 27/30 tasks (90.0%), below the required 100%. The mismatches were HumanEval/34, HumanEval/39, HumanEval/71. Per the frozen failure policy, held-out negatives were not opened and HumanEval-164 was not run.

This result invalidates the tested implementation as an exact replayable dynamic semantic watermark. Positive pilot shifts remain diagnostic only because the core replay contract failed.

## Frozen setup

- Formal arm: `Streaming-SharedPool-V3-R20`
- Baseline: `Current-R20`
- Dataset/scope: one frozen 30-task HumanEval pilot, one seed (`20260713`), retry budget 20
- Model: local `deepseek-coder-1.3b-instruct`, bf16, offline
- Semantic encoder: local CodeT5 checkpoint, frozen, float32
- Detector: R3 final-code-only; no prompt, task ID, candidate pool, generation trace, or generation-time metadata
- Candidate pool: the exact saved V2 retry-20 raw trajectories; both selectors consumed the same ordered 20 candidates per task
- Repair budget: one allowed, one used, zero remaining

## Primary results

| Measure | Current-R20 | V3-R20 | Difference / gate |
|---|---:|---:|---:|
| R3 detection positives | 1/30 (3.33%) | 7/30 (23.33%) | +20.00 pp; diagnostic only |
| Mean semantic score | -0.044444 | 0.132585 | +0.177029 |
| HumanEval passes | 15/30 | 15/30 | +0 |
| Exact R3 replay | n/a | 27/30 (90.0%) | **FAIL; required 100%** |
| Candidate-pool hash equality | 600/600 | 600/600 | PASS |
| Wrong-key positives | n/a | 1/30 (3.33%) | no positive-control shift |

The paired score difference had a 20,000-resample task bootstrap 95% interval of [0.094172, 0.258662]. Score changes were positive/negative/tied on 19/4/7 tasks; the two-sided exact sign-test p-value was 0.00259948. These statistics describe the pilot but do not rescue the failed exact-replay contract.

Correctness was neutral in aggregate: four Current failures became V3 passes and four Current passes became V3 failures. V3 execution produced 15 passes, 14 failed tests, and one timeout.

## Cost and calibration diagnostics

- Full dynamic selection plus exactly one selected-final replay: mean 0.101131791 s/task, median 0.096212409 s/task.
- Task-bootstrap 95% interval for the mean: [0.083478103, 0.120440904] s/task.
- Reduction from the preregistered complete-final semantic baseline (0.201412 s/task): 49.79%; the required reduction was 30%.
- Mean semantic encoder calls: 4.733/task; total encoded contexts: 745.
- Peak allocated CUDA memory: 687.954 MiB on an RTX 5090 (32,607 MiB gate).
- Calibration-negative primary-key positives: 11/250 (4.4%); bootstrap 95% interval [2.0%, 7.2%]. This is calibration-set behavior, not held-out FPR.
- Calibration-negative wrong-key positives: 1/250 (0.4%).

## Gate table

| Hard gate | Result |
|---|---|
| Shared ordered candidate pool | PASS — 600/600 hashes |
| Exact R3 replay | **FAIL — 27/30** |
| No EOS all-candidate semantic rescore | PASS — count 0 |
| Mean semantic cost <= 0.1409884 s/task | PASS — 0.101131791 |
| Cost reduction >= 30% | PASS — 49.79% |
| Peak VRAM <= 32,607 MiB | PASS — 687.954 MiB |
| V3 pass count >= 15 | PASS — 15 |
| Pass delta >= -1 | PASS — +0 |
| V3 TPR > Current TPR | PASS diagnostically — 23.33% > 3.33% |
| Positive shift not one outlier | PASS diagnostically — 19 positive paired score shifts |
| Calibration-negative rate <= 5% | PASS on calibration only — 4.4% |
| Key excluded from raw generation | PASS by construction/static audit |

## Escalation decision

The exact-replay failure exhausts the one-repair budget. No threshold, precision, encoder, context definition, statistic, split, or retry-budget change was made after observing pilot outcomes. The held-out split remains unopened; robustness attacks and HumanEval-164 remain intentionally unrun because their prerequisite clean-pilot gate did not pass.

Machine-readable evidence is in `PILOT_AUDIT_V3.json`. Raw experiment outputs remain under ignored `data/experiments/watermark_v3_dynamic_semantic_20260714/` and are hash-bound in that audit.
