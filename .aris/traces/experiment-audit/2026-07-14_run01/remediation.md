# Remediation

The read-only snapshot correctly caught three stale reports left from the
preregistered pilot stopping decision. Before final commit, the executor
replaced `FULL_REPORT.md`, `EXPERIMENT_INTEGRITY_AUDIT.md`, and
`RESULT_TO_CLAIM.md` with the actual post-gate exploratory 164-task results.
The raw artifacts, thresholds, held-out scores, and detector were not changed.

After replacement, result existence/numeric agreement is PASS. Overall status
remains WARN solely because one-seed post-gate exploratory scope cannot support
a confirmatory or general claim.
