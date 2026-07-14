# Dynamic Semantic Watermark V3 Implementation Plan

> Execute in `/root/autodl-tmp/WFCLLM-worktrees/dynamic-semantic-v3` on branch
> `codex/watermark-mechanism-v3-dynamic-semantic-reset`. The implementation starts
> from V2 commit `eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`. Preserve both rejected
> V3 branches and the dirty main worktree. Use TDD for every behavioral change.

Goal: implement, verify, profile, and evaluate the preregistered
`Streaming-SharedPool-V3-R20` mechanism. Stop escalation and close a complete
negative result if a frozen hard gate fails after the one permitted defect repair.

## Task 1: Freeze schemas and configuration validation

Files:

- Create `wfcllm/dynamic_semantic/__init__.py`
- Create `wfcllm/dynamic_semantic/config.py`
- Create `tests/dynamic_semantic/test_config.py`
- Modify `wfcllm/method/config.py`
- Modify `tests/method/test_v2_pipeline_config.py`

Steps:

1. Write failing tests for exact schema strings, immutable dataclasses, all numeric
   bounds, forbidden ordinal keying, float32 official precision, Hamming(7,4)
   dimensions, scheduler bounds, and private-key-file-only input.
2. Run `HF_HUB_OFFLINE=1 ... pytest tests/dynamic_semantic/test_config.py -v` and
   record the expected import/failure output.
3. Implement the smallest frozen config types and JSON loader. Reject unknown schema
   versions and incompatible public artifacts with chained `ValueError`.
4. Add V3 as an explicit method version without changing V1/V2 defaults.
5. Run the targeted tests and the existing method config tests.

## Task 2: Implement public canonical contexts and incremental live-unit tracking

Files:

- Create `wfcllm/dynamic_semantic/context.py`
- Create `tests/dynamic_semantic/test_context.py`
- Create `tests/dynamic_semantic/test_incremental_tracker.py`

Steps:

1. Write failing cases for `ast.unparse` canonicalization, direct function-body
   statements, supported compound regions, public roles, content-derived IDs, BOS
   predecessor, duplicate erasure, token-budget erasure, parse failure, insert,
   delete, reorder, and no global ordinal dependency.
2. Write failing incremental cases that feed one token/text fragment at a time,
   commit only stable units, invalidate live units on rollback, flush the final unit,
   and end with exactly the same unit IDs/context hashes as final-code extraction.
3. Run the new tests and capture RED.
4. Implement pure context extraction first, then a rollback-aware tracker whose live
   state is derived only from the public prefix. Keep secret material out of this
   module.
5. Run the new tests plus `tests/generation/test_boundary.py`.

## Task 3: Implement secret handling, deterministic projection, ECC, and statistics

Files:

- Create `wfcllm/dynamic_semantic/keying.py`
- Create `wfcllm/dynamic_semantic/channel.py`
- Create `tests/dynamic_semantic/test_keying.py`
- Create `tests/dynamic_semantic/test_channel.py`

Steps:

1. Write failing tests for mode-0600 key files, short/empty keys, domain separation,
   no fingerprint API, deterministic integer hyperplanes, fixed quantization, exact
   Hamming(7,4) codewords/syndromes, exact signed numerators, duplicate rejection,
   and wrong-key divergence.
2. Run tests and capture RED.
3. Implement private key loading without logging key-derived values. Generate seven
   secret `{-1,+1}` rows after the public embedding. Derive four content-addressed
   target bits and encode them to seven bits. Aggregate as an exact integer
   numerator/denominator plus a derived float.
4. Run targeted tests and a source scan proving no syntax-codebook import.

## Task 4: Load the frozen encoder and fit public whitening

Files:

- Create `wfcllm/dynamic_semantic/encoder.py`
- Create `wfcllm/dynamic_semantic/whitening.py`
- Create `tests/dynamic_semantic/test_encoder.py`
- Create `tests/dynamic_semantic/test_whitening.py`

Steps:

1. Write failing unit tests with fake tokenizer/model for no truncation, length
   erasure, deterministic batch ordering, float32 output, L2 normalization, fit-only
   whitening, artifact schema/hash validation, and no positive/held-out fitting.
2. Implement a narrow adapter around the existing CodeT5+LoRA checkpoint. Verify the
   checkpoint SHA before load and expose batch-call/context counters.
3. Implement 64-dimensional public whitening with deterministic serialization and
   an explicit fitted-source manifest.
4. Run fake-model tests, then a one-batch offline real-checkpoint smoke on RTX 5090.

## Task 5: Implement the dynamic batching scheduler and trajectory accumulator

Files:

- Create `wfcllm/dynamic_semantic/scheduler.py`
- Create `wfcllm/dynamic_semantic/selection.py`
- Create `tests/dynamic_semantic/test_scheduler.py`
- Create `tests/dynamic_semantic/test_selection.py`

Steps:

1. Write failing tests for deterministic queue order, target batch 32, maximum 128,
   four-attempt latency flush, final residual flush, cache by context hash, rollback
   invalidation, exact surviving-unit accumulation, call counters, and no completed-
   candidate score API.
2. Write paired selection tests for unchanged static quality tier, V3 statistic,
   unit count, erasures, attempt index, and the no-embedding fallback.
3. Implement the queue and accumulator. Embeddings may be computed for invalidated
   contexts, but only current live units may contribute at selection.
4. Prove with a test spy that 100 contexts use bounded micro-batches rather than 100
   calls and that a final-code list cannot be passed to selection.

## Task 6: Wire generation-time observations without changing raw trajectories

Files:

- Modify `wfcllm/generation/generator.py`
- Modify `wfcllm/generation/pipeline.py`
- Modify `wfcllm/method/config.py`
- Create `tests/generation/test_dynamic_semantic_observer.py`
- Modify `tests/generation/test_pipeline.py`

Steps:

1. Write failing tests for an optional public prefix observer called after generated
   text changes and after rollback, with no observer-to-generator control path.
2. Write a failing retry-20 test that shares one scheduler across attempts, flushes
   during the attempt stream, and selects from accumulated live evidence.
3. Write a failing fairness test that runs Current and V3 with deterministic fake
   generation and asserts byte-identical ordered attempt final-code hashes/seeds.
4. Add a read-only observer protocol. It must not return decisions, edit logits,
   choose seeds, stop, roll back, repair, or alter the existing result.
5. Wire V3 retry selection and private audit ledger. Keep public final rows restricted
   to the established four fields.
6. Run all generation and method tests.

## Task 7: Implement the R3 final-code-only detector and conformal calibration

Files:

- Create `wfcllm/dynamic_semantic/detector.py`
- Create `wfcllm/dynamic_semantic/calibration.py`
- Create `wfcllm/dynamic_semantic/artifacts.py`
- Create `tests/dynamic_semantic/test_detector.py`
- Create `tests/dynamic_semantic/test_calibration.py`
- Create `tests/audit/test_dynamic_semantic_integrity.py`

Steps:

1. Write failing contract tests that permit only final code, public config, and key
   file; reject prompt/task/candidate/trace fields; batch-encode once; and reproduce
   exact IDs, context hashes, quantized vectors, bits, votes, rational statistic,
   and decision from synthetic generation evidence.
2. Write failing conformal tests including ties, 250 calibration values, minimum
   units, strict one-sided p-values, and serialization round trips.
3. Implement R3 and public/private artifact separation. Do not serialize target bits,
   hyperplanes, per-unit secret matches, key hashes, or key fingerprints publicly.
4. Add secret scans and detector-input integrity checks.
5. Run dynamic semantic and audit test groups.

## Task 8: Add reproducible CLI workflows

Files:

- Modify `wfcllm/cli/arguments.py`
- Modify `wfcllm/cli/config_resolver.py`
- Modify `wfcllm/cli/runners.py`
- Modify `wfcllm/cli/entry.py`
- Create `scripts/wfcllm_v3_fit.py`
- Create `scripts/wfcllm_v3_detect.py`
- Create `scripts/wfcllm_v3_compare.py`
- Create `tests/integration/test_wfcllm_v3_cli.py`

Steps:

1. Write failing CLI tests for fit, generation, R3 replay, detect, compare, profile,
   resume, wrong-key control, and forbidden raw key arguments.
2. Implement thin CLI wiring; keep all business logic in package modules.
3. Ensure every artifact records schema/config versions, baseline/implementation Git
   heads, model/checkpoint hashes, dataset/split hashes, seed, retry, environment,
   and input/output SHA-256 values.
4. Run CLI tests and `python -m compileall wfcllm run.py scripts tools`.

## Task 9: Full local verification and implementation commit

1. Run `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 ... pytest
   tests/dynamic_semantic tests/generation tests/detection tests/audit tests/method
   tests/integration/test_wfcllm_v3_cli.py -v`.
2. Run the whole offline suite and compare against the frozen `734 passed` baseline.
3. Run forbidden-import, schema, secret, and public-artifact scans.
4. Review the diff for accidental changes to V1/V2 and both rejected branches.
5. Commit exactly `feat: implement semantic encoder lsh watermark v3`.

## Task 10: Freeze private key, fit whitening/calibration, and prove R3

1. Create the dynamic-V3 private key at the preregistered path with mode 0600; never
   display it or any key fingerprint.
2. Fit whitening and conformal calibration only on the frozen 250 calibration rows.
3. Run 8 development tasks to validate exact generation/R3 parity and raw-pool
   equality. Fix implementation defects only; do not tune scientific rules.
4. Produce `ENCODER_LEAKAGE_AUDIT.md`, `R3_EXACT_REPLAY_AUDIT.md`,
   `CANDIDATE_POOL_FAIRNESS_AUDIT.md`, and machine-readable ledgers.
5. If parity is not 100%, apply the single permitted defect repair and rerun. If it
   still fails, stop and close a negative result.

## Task 11: Profile RTX 5090 cost and memory

1. Warm the generation and semantic models; exclude model load from comparable
   per-task semantic timing.
2. Measure encoder calls, contexts/call, total contexts/task, semantic wall time,
   selected-final replay time, total generation time, CUDA allocated/reserved peak,
   CPU/GPU transfer, and output hashes on the same debug tasks.
3. Compare against 0.201412 semantic s/task and require at most 0.1409884 s/task.
4. Confirm static source/call-counter evidence of zero EOS 20-candidate rescoring.
5. Write `PROFILE_V3_RTX5090.md`. A cost or memory failure blocks the pilot/full path
   according to the frozen gates.

## Task 12: Run paired pilot 30 × retry 20

1. Save ordered raw attempt artifacts once and derive Current/V3 selections from the
   identical pool. Record every attempt's code and hash privately/audit-only.
2. Run R3 final-only replay, HumanEval correctness, frozen calibration threshold,
   wrong-key control, and clean robustness attacks.
3. Compute paired Pass@1, TPR, score shift, units/code, bootstrap confidence intervals,
   paired permutation/sign tests where valid, and outlier influence.
4. Run experiment-integrity audit before claims.
5. Evaluate every preregistered pilot gate mechanically. If all pass, continue. If
   a demonstrable implementation defect exists, use the one allowed repair and rerun
   the entire pilot once. Otherwise stop escalation and write the negative report.

## Task 13: Conditional full HumanEval 164 and one-time held-out opening

1. Only if the pilot gate artifact says `all_passed=true`, run HumanEval 164 from the
   frozen shared retry-20 pool contract.
2. Open the frozen 250 MBPP held-out negatives exactly once, run primary and wrong-key
   detectors, and never change calibration afterward.
3. Run correctness, R3, fairness, security, attack, and statistical audits.
4. If any full gate fails, report it plainly; do not replace the method or threshold.

## Task 14: Result-to-claim, final reports, verification, and commits

Files/reports:

- `RESULTS_V3.md`
- `V3_FPR_AUDIT.md`
- `R3_EXACT_REPLAY_AUDIT.md`
- `CANDIDATE_POOL_FAIRNESS_AUDIT.md`
- `PROFILE_V3_RTX5090.md`
- `ENCODER_LEAKAGE_AUDIT.md`
- `ROBUSTNESS_V3.md`
- `FINAL_AUDIT_V3.md`
- `FAILURE_POSTMORTEM_V3.md` when any hard gate fails
- update `MANIFEST.md`

Steps:

1. Use experiment-audit and result-to-claim against raw artifacts, not summaries.
2. Report mechanism boundary, exact R3, FPR, Pass, capacity, bias, wrong key,
   synchronization, ECC, attacks, cost, memory, leakage, and statistical uncertainty.
3. Run the whole offline test suite, compileall, artifact/hash verifier, secret scan,
   Git diff/status, and report-link validation immediately before completion claims.
4. Commit the pilot report exactly `docs: record semantic v3 pilot results`.
5. Commit the conditional full report or closed negative result exactly
   `docs: record semantic v3 full or negative results`.
6. Mark the persistent goal complete only after the implementation or complete
   negative-result branch is reproducible and no required deliverable remains.
