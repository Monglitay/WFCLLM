# Result-to-Claim Review Prompt

Judge whether V4 jointly supports batch-invariant exact replay, cost, correctness,
FPR, and strictly better final-code-only detection than Current. Inputs are
`PILOT_AUDIT_V4.json`, `PILOT_RESULTS_V4.md`, `EXPERIMENT_AUDIT_V4.md`, and the
frozen preregistration. Return claim_supported, confidence, integrity status,
supported/unsupported claims, missing evidence, revised claim, next experiments,
and pending external review.
