# Batch-Invariant Semantic Watermark V4 Implementation Plan

Date: 2026-07-14

Status: frozen execution plan accompanying formal preregistration.

## Objective and stop rules

Implement `CanonicalStructuralReplay-V4-R20` from the frozen public config and
validate it without changing V3 history. Stop escalation immediately on a
debug, pilot, held-out, or full hard-gate failure. A negative result still
requires complete artifacts, audit, result-to-claim, tests, commits, and a clean
worktree.

## Frozen inputs

- Base: V3 final commit `8693e0879cbd5657b335913b7d49171bd6c77c3b`.
- Branch: `codex/watermark-mechanism-v4-batch-invariant-semantic`.
- Public config: `configs/wfcllm/v4_batch_invariant_public.json`.
- Preregistration: `configs/wfcllm/v4_batch_invariant_preregistration.json`.
- Negative manifest: `configs/wfcllm/v4_batch_invariant_split_manifest.json`.
- Formal retry: 20; minimum independent units: 3; target FPR: 5%.
- Formal evidence is CPU canonical AST/data-flow evidence; the frozen neural
  checkpoint is comparison metadata only.

## Implementation sequence

1. Write failing formal V4 tests for key loading/domain separation, canonical
   structural extraction, nested compound shells, duplicate/oversize/parse
   erasures, final-code-only payload validation, exact evidence, cache identity,
   selection scheduling, calibration, and artifact validation.
2. Implement a focused `wfcllm.batch_invariant_v4` package. Reuse existing
   static-quality and retry/candidate contracts; do not import legacy
   `experiment.*` code.
3. Add a formal key type with redacted representation, 0600 file loading, new
   V4 HMAC header, separate signature/target/wrong-key domains, and no raw-key
   CLI option.
4. Implement canonical recursive statement groups. Represent compound headers
   without child subtrees and child statements separately. Use versioned sorted
   JSON and identity-quantized SHA-256 bytes.
5. Implement exact unit/code evidence records and a comparator that names every
   mismatching discrete field. Store integer numerator/denominator and a
   deterministic empirical p-value representation.
6. Implement generation observer and R3 as two isolated callers of the same
   final-code reconstruction function. R3 payload accepts exactly `final_code`;
   public config and key file are constructor inputs.
7. Implement public cache keys from config identity and context hash only.
   Candidate generation remains key-free. Score a candidate only when it
   completes; selected-final replay occurs exactly once; EOS neural complete-
   candidate rescore count remains zero.
8. Fit calibration only on the frozen 230-row calibration panel. Derive the
   wrong-key control through its frozen domain. Do not read held-out.
9. Run targeted tests, then the full offline suite. Commit the formal tests and
   implementation atomically if separating them would leave an unusable tree.

## Debug and profile gate

Use the three V3 mismatch IDs only for regression plus the five frozen V4 debug
IDs. In at least three cold processes compare generation, R3, cache hit/miss,
caller schedule permutations, and every exact field. Verify candidate-pool
hashes and deterministic runtime state.

Profile model load/warm-up separately even though formal V4 loads no encoder.
The hard metric is dynamic candidate scoring plus one selected-final replay.
Require mean <= 0.1409884 seconds/task, at least 30% reduction from 0.201412,
peak allocated VRAM <= 32607 MiB, and zero EOS neural rescoring. If any gate
fails, write the complete pre-pilot negative result and skip pilot.

## Pilot and upgrade

Run only the 30 preregistered HumanEval IDs with Current-R20 and V4-R20 over the
same ordered 20-candidate trajectories. Store all raw candidate SHA-256 values.
Recompute correctness, detection, paired bootstrap CI, exact sign, and McNemar
statistics. Run one independent read-only integrity audit. Use at most one
targeted repair and only for a demonstrated implementation defect.

If every pilot gate passes, open the new held-out panel once and run the frozen
detector. Never adjust mechanism or threshold after access. Run HumanEval-164
only if held-out primary/wrong-key FPR both pass. Run robustness only if pilot,
held-out, and full clean gates all pass.

## Audit and closure

Generate experiment, result-to-claim, final, secret, file-access, artifact-hash,
failure-postmortem, robustness, and results reports even for a negative result.
Run the required offline full pytest command, compileall, JSON validation,
artifact hash verification, secret scan, `git diff --check`, and clean status.
Commit all allowed artifacts without V3 mutation, then mark the Goal complete
only under the user's success or complete-negative definition.
