# Diagnostic Upper Bound

`evidence_only_selector_diagnostic` is not an official method. It is a diagnostic selector used to estimate what evidence-aware selection could recover under an explicit non-official analysis setting.

The diagnostic selector may read artifacts that the official detector cannot read. Its outputs must therefore never be presented as official WFCLLM method results.

`diagnostic_only=true` and `not_official_method=true` are required for all selector outputs.

## Required Interpretation

- Diagnostic selector outputs are upper-bound analysis artifacts.
- They are not official generation, calibration, detection, or report artifacts.
- They cannot replace `evidence_retry_seed7x3`.
- They cannot be used to claim official detector performance.
- They may mention SAWR only as legacy naming context when comparing historical artifacts.

## Allowed Location

Diagnostic output should live under diagnostic or analysis directories and should not be mixed into `data/runs/<run_id>/inputs/final_code.jsonl`.

## Review Checklist

- Confirm `diagnostic_only=true`.
- Confirm `not_official_method=true`.
- Confirm the output is excluded from official detector input.
- Confirm reports label the result as diagnostic upper-bound analysis.
