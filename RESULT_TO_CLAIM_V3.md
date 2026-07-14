# V3 Result-to-Claim Judgment

- `claim_supported`: no
- `confidence`: high for the frozen implementation; pending external review for the broader interpretation
- `integrity_status`: warn
- `what_results_support`: Incremental semantic observation is substantially cheaper on the shared saved pool and creates a diagnostic primary-key score shift, but the frozen implementation fails exact final-code reconstruction.
- `what_results_dont_support`: A valid exact R3 watermark, legal TPR at held-out 5% FPR, full HumanEval improvement, robustness, or generalization.
- `missing_evidence`: A deterministic signed representation, a new untouched calibration/held-out split, a clean pilot, full 164-task evidence, robustness attacks, multiple seeds/models, and independent audit.
- `suggested_claim_revision`: Report a negative engineering result: dynamic neural embedding batches are not sufficiently path-independent for bit-exact final-code watermark evidence under the tested quantization.
- `next_experiments_needed`: Only under a new preregistration—remove batch-neighborhood dependence, prove exact replay first, then repeat calibration/pilot with a new split.
- `routing_action`: stop escalation and close the branch as a complete negative result.

The normal skill workflow calls for an independent secondary Codex reviewer. System policy in this session prohibits spawning subagents unless the user explicitly requests delegation. This judgment is therefore a primary-agent fallback marked `[pending external review]`; no independent review is implied.
