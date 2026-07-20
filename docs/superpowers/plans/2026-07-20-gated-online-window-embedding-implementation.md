# Gated Online Window Embedding Implementation Plan

> Execute in the existing isolated branch `codex/nocarrier-gated-semantic`.
> Optimize for the shortest auditable path to a full 164-task result.

**Goal:** Make the trained gate's sequential close decisions control natural
1/2/3-statement embedding windows, and replace unsafe free-form semantic
admission with AST-equivalent rewrites that either hit keyed semantic LSH or
restore the exact original.

**Architecture:** Reuse `WindowPartitioner`'s existing causal per-statement
`CONTINUE/CLOSE` loop as the online wait state. Add explicit audit coverage for
the selected window length. Add a key-blind constrained rewriter that emits
three deterministic lexical/source variants and certifies only candidates whose
parsed AST (including type comments) is identical to the original window. Teach
`GatedGenerator` to accept this rule-specific certificate; neural candidates
remain rejected unless independently certified. Keep detector/calibration and
full-denominator evaluation unchanged.

**Environment:** conda `WFCLLM`, Python, tree-sitter, stdlib `ast`/`tokenize`,
pytest, one RTX 5090 for gate/model inference.

---

## Task 1: Freeze online wait behavior in tests

**Files:**
- Modify: `tests/windowing/test_partitioner.py`
- Modify: `tests/generation/test_gated_generator.py`

1. Add a failing test where the gate returns CONTINUE after statement one and
   CLOSE after statement two; assert one two-statement window, one original LSH
   score, and rewrite request length two.
2. Add a hard-boundary/EOF test proving an incomplete requested window closes
   without consuming a later scope.
3. Run the focused tests and record the expected failure or, if existing code
   already satisfies them, retain them as regression evidence.

## Task 2: Add AST-equivalence certificates

**Files:**
- Modify: `wfcllm/generation/window_rewriter.py`
- Modify: `wfcllm/generation/gated_generator.py`
- Modify: `tests/generation/test_window_rewriter.py`
- Modify: `tests/generation/test_gated_generator.py`

1. Add failing tests for three distinct key-blind variants of one natural
   window and for rejection of behavior-changing candidates.
2. Extend parsed rewrite provenance with a non-secret semantic validation rule.
3. Implement an AST-equivalence validator using `ast.parse(...,
   type_comments=True)` and `ast.dump(..., include_attributes=False)` after
   controlled dedent. It must fail closed on parse errors or AST differences.
4. Implement three fixed variants, preferring: safe constant spelling changes,
   redundant expression parentheses, and AST-unparse normalization. Each must
   preserve the parsed AST and exact statement/scope count; duplicate or
   unavailable variants remain visible failures.
5. Update candidate audit validation so `encoder_cosine/v1` still requires
   cosine >= 0.90, while `python-ast-equivalent/v1` requires an explicit true
   certificate and treats cosine only as diagnostic evidence.
6. Run focused tests.

## Task 3: Wire the constrained backend into official semantic generation

**Files:**
- Modify: `wfcllm/cli/runners.py`
- Modify: `wfcllm/method/config.py` if a config switch is required
- Modify: `tests/integration/test_gated_workflow.py`

1. Add a failing integration test that official `semantic_lsh` generation uses
   the constrained key-blind backend and carries no deployment key into rewrite
   generation.
2. Add one explicit rewrite backend config value and fail on unknown values.
3. Wire the constrained backend; keep the old model backend available only for
   declared experiments.
4. Run integration and method-config tests.

## Task 4: Validate without GPU

1. Run `pytest` for windowing, generation, gate, method, audit, and integration.
2. Run compileall.
3. Run static artifact checks proving no carrier implementation is imported and
   no detector schema changed.
4. Commit the focused implementation only after all checks pass.

## Task 5: Offline feasibility and gate threshold repair

1. Replay the new candidate trajectory on existing non-HumanEval gate-data
   sources and frozen training/holdout key banks.
2. Report, by window length, structural validity, AST-equivalence rate, stable
   LSH rate, original hit rate, R1/R3 hit rate, and suitable labels.
3. Fit thresholds only on development splits. Reject any gate whose unsuitable
   false-positive rate exceeds the declared validation limit or whose accepted
   set is degenerate.
4. Select the smallest declared LSH density/rewrite budget that makes the
   detector statistically viable while retaining independent FPR calibration.

## Task 6: Fixed pilot

1. Freeze code commit, config, key sources, model paths, thresholds, sample IDs,
   and success rules before execution.
2. Run a small public prompt pilot without reading HumanEval hidden tests or
   canonical solutions.
3. Require zero carriers, zero added AST statements, exact rollback, sufficient
   window coverage, materially non-random positive hit rate, and no observed
   rewrite-induced execution regression on permitted public checks.
4. If it fails, make one declared aggregate-level change and repeat; do not
   start the 164-task run until feasible.

## Task 7: Full frozen HumanEval run

1. Run exactly one candidate for all 164 tasks.
2. Run calibration, positive detection, independent held-out negative
   detection, report, audit, and HumanEval execution.
3. Verify raw counts: Pass >= 99, positive detections >= 99, held-out FPR <=
   0.05, carrier count 0, added AST statement count 0, and 164-task denominator.
4. Preserve logs, manifests, sidecars, hashes, failure rows, and resolved config.
5. If a target fails, declare the next change from aggregate/public evidence
   before another frozen run.

## Task 8: Completion and shutdown

1. Write the final result report with exact Pass/TPR/FPR counts and artifact
   paths.
2. Save/verify all artifacts locally or on durable storage.
3. Run remote `sudo shutdown`; use the available equivalent only if necessary
   and verify SSH disconnects.
4. Shut down the local machine under the user's explicit authorization.
