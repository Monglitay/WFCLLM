# Gated Online Natural-Window Embedding Design

Date: 2026-07-20

## Objective

Implement the shortest defensible path to the WFCLLM main method described by
the user: while the base model generates code statement by statement, a trained
gate decides how many adjacent natural statements should form the current
embedding window. The runtime waits until that many complete statements exist,
scores the window with keyed semantic LSH, and rewrites only that same window
when the original does not carry the target bit.

The full HumanEval run remains a real one-candidate evaluation over all 164
tasks. Success requires at least 99/164 passes and 99/164 positive detections,
with target FPR 0.05 and held-out FPR at most 0.05. No carrier, added statement,
unreachable branch, dummy assignment, no-op, comment channel, formatting-only
channel, hidden test, canonical solution, or per-task candidate selection is
allowed.

## Existing Failure and Design Consequence

Attempt 14 used no carriers, but selected the original in 987/1013 windows and
found only 26 keyed-bit-flipping rewrites. Its cosine-only semantic check also
accepted behavior-changing edits. The resulting HumanEval metrics were 87/164
Pass@1 and 3/164 TPR.

Therefore this round does not tune LSH density or merely increase free-form
rewrite samples. It fixes the online window lifecycle and replaces the unsafe
semantic admission decision with conservative, rule-specific validation.

## Runtime State Machine

The runtime operates on complete parser-recognized statements from the single
base-model generation stream.

1. `OPEN`: the first new complete natural statement becomes the provisional
   window start.
2. `GATE`: the trained gate receives only causal information available at that
   point: task prompt, committed prefix, provisional statements, parser
   descriptors, and token representations. It predicts a close length in
   `{1, 2, 3}` and a suitability probability.
3. `WAIT(n)`: if the prediction is `n > current_length`, the runtime commits no
   watermark decision yet and waits for later complete statements in the same
   parent scope. A hard boundary or generation end closes early and marks the
   window unsuitable unless the gate explicitly supports that observed length.
4. `SCORE_ORIGINAL`: once the predicted length is available, the exact adjacent
   statements are scored by the keyed semantic LSH detector contract.
5. `ACCEPT_ORIGINAL`: if the original window already produces a stable target
   hit, it is committed unchanged.
6. `REWRITE`: otherwise the runtime requests a bounded trajectory of rewrites
   for exactly the same 1/2/3 statements. Candidates cannot consume future
   statements, cross scope, change statement count, or add surrounding code.
7. `VALIDATE_AND_SCORE`: each candidate must pass parsing, exact window length,
   same-parent scope, rule-specific semantic validation, stable precision/batch
   scoring, and keyed target-bit scoring.
8. `ACCEPT_REWRITE`: commit the first fully valid target hit.
9. `RESTORE_ORIGINAL`: if the bounded trajectory is exhausted, restore the
   byte-identical original window. Failure to embed never deletes a task or
   substitutes another full program.

Every transition is serialized in audit sidecars. Only final code enters the
official detector.

## Fastest Implementation Boundary

The existing partitioner, gate checkpoint format, retry controller, candidate
sidecar, detector, calibration pipeline, and full-denominator evaluator are
reused. The smallest production changes are:

- expose the gate-selected close length explicitly and make the partitioner
  follow the `OPEN -> WAIT -> CLOSE` lifecycle rather than accepting nearly all
  pre-partitioned windows;
- bind the deployment gate target to actual safe keyed controllability, not a
  broad aggregate hit-rate proxy;
- add a constrained rewrite backend before the existing neural fallback;
- fail closed on candidates for which semantic equivalence cannot be justified;
- record predicted length, observed length, wait/close reason, transformation
  family, validation rule, and rollback outcome.

No new large model is trained. The current gate architecture is retrained only
if its target schema changes, and the existing base generation model is reused.

## Rewrite Strategy

For speed and Pass@1 safety, rewrite candidates are ordered as follows:

1. conservative local equivalence transforms on existing expressions and
   statements;
2. existing neural A/B/C trajectory only when its output can be certified by
   one of the same conservative validators;
3. otherwise restore the original.

Initial transform families are restricted to forms with cheap, explicit
preconditions, such as parenthesization/tuple-form normalization, generator
parentheses, string-literal spelling with identical decoded value, and
name-local augmented assignment equivalents where the left-hand side cannot
trigger attribute/subscript evaluation effects. More families may be enabled
only when a regression test demonstrates their safety contract.

CodeT5 cosine remains diagnostic evidence and may be an additional rejection
condition, but it is never sufficient proof of semantic preservation.

## Gate Supervision

Each training example is one natural causal statement prefix with candidate
close lengths 1, 2, and 3. A length is suitable only if the frozen rewrite
trajectory contains at least one candidate that:

- passes the conservative semantic validator;
- preserves statement count and parent scope;
- is stable across required scoring modes; and
- supplies the desired keyed outcome across the training key bank at the
  declared success threshold.

The gate jointly predicts suitability and close length. Validation must report
per-length confusion matrices and must reject a checkpoint whose unsuitable
false-positive rate exceeds the predeclared limit. The attempt-14 behavior
(`suitable_false_positive_rate = 0.9886`) is explicitly inadmissible.

## Integrity and Metrics

The generation manifest and audit must prove:

- `sample_count = 164`;
- `carrier_count = 0`;
- `finalizer_added_ast_statement_count = 0`;
- exactly one final program per HumanEval task;
- no benchmark result is used for per-task regeneration or selection;
- every selected rewrite identifies its gate decision, target bit,
  transformation family, semantic validator, and source span;
- every failed rewrite restores the original window;
- detector input rows contain exactly `id`, `dataset`, `prompt`, and
  `final_code`.

Reported results include raw counts and rates for Pass@1, positive TPR,
calibration FPR, held-out FPR, insufficient-evidence decisions, reliable-window
counts, original hits, successful rewrites, rollbacks, and failure classes.

## Experiment Sequence

The fastest evidence-driven sequence is:

1. focused unit and integration tests for gate waiting, exact-window rewriting,
   semantic-validator rejection, and rollback;
2. offline gate/rewrite feasibility on existing non-HumanEval development
   artifacts;
3. a fixed small public pilot used only to validate runtime integrity and rough
   signal; no hidden HumanEval tests or canonical solutions are read;
4. one frozen full 164-task run when the pilot passes predeclared feasibility;
5. calibration, positive detection, independent held-out negative detection,
   HumanEval execution, and audit;
6. if a frozen full run misses a target, diagnose only aggregate/public
   artifacts, declare the next single change before rerunning, and retain all
   denominators and failures.

## Shutdown

After the hard goal is achieved and result artifacts are durably saved, shut
down the remote server with `sudo shutdown` (falling back to the platform's
available equivalent only if needed), verify disconnection, and then shut down
the local machine as explicitly authorized by the user.
