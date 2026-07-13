# WFCLLM Aligned Canonical Signature V2

## Status

This document freezes the V2 design before implementation and pilot generation. The method is experimental until the pre-registered pilot and full gates pass. Failure leaves v1 as the official default.

## Design alternatives considered

1. **Aligned canonical signature selection (chosen).** Score every retry-20 final-code candidate with the same canonical keyed statistic used by detection, after a static quality gate. This directly targets the diagnosed objective mismatch and correlated evidence.
2. **Detector-only re-aggregation.** Deduplicate existing v1 windows but keep accepted-count generation selection. This is low cost but cannot create a stronger correct-key score shift when generation is not selecting for it.
3. **Train a new encoder/signature model.** This could improve representation geometry but requires new train/calibration/test isolation and introduces leakage and reproducibility risks without evidence that the current checkpoint is the primary failure.

V2 chooses option 1 and does not train a new model.

## Canonical unit contract

The canonical schema is `wfcllm-canonical-unit/v2`.

Detection parses only `final_code`, identifies the target top-level function deterministically, and extracts simple statements owned by its function body and scoreable nested compound bodies. Each statement appears at most once. Units are ordered by source byte range and assigned a global zero-based ordinal. Exact duplicate `(normalized_text, node_type, parent_node_type)` units are retained only once, with the first ordinal.

Each unit contains:

- `schema_version`
- `unit_id`
- `normalized_text_hash`
- `normalized_text` in diagnostic memory only
- `node_type`
- `parent_node_type`
- `ordinal`
- `start_line` and `end_line`

The public final detector input remains exactly `id`, `dataset`, `prompt`, and `final_code`. Canonical units are recomputed and are never read from a sidecar.

## Key-conditioned signature

The existing fine-tuned CodeT5 encoder and secret-conditioned LSH projection produce a `d=16` sign vector for every canonical unit. A second 16-bit target is derived with HMAC-SHA256 from:

```text
schema_version || node_type || parent_node_type || ordinal
```

under the secret key. Neither target bits nor the key are public artifacts. A unit score is the fraction of matching bits. The sample raw score is a robust mean of unit scores: ordinary mean below five unique units, and a one-unit trimmed mean at five or more. The statistic is bounded, prevents a single large margin from dominating, and provides 16 recoverable bits even for a one-statement solution.

Wrong keys change both the keyed projection and target derivation. Ordinary negative code and wrong-key positives must be checked against independently calibrated null behavior.

## Generation-time embedding and selection

Both Current-R20 and V2-R20 generate exactly 20 full-completion attempts with the same model, prompt, seed schedule, decoding parameters, token limit, and inner state-machine budgets.

Current-R20 selects with the existing v1 accepted-evidence key. V2-R20 performs:

1. Build the strict final code for every attempt.
2. Apply the public static quality gate.
3. Within the highest valid quality tier, select maximum V2 raw detector score.
4. Break ties by larger unique-unit count, fewer v1 fallbacks, then lower attempt index.
5. If no attempt passes the legal quality tier, emit the earliest best-quality candidate as `no_embedding` rather than select a visibly damaged candidate for watermark score.

The selected score is recomputed from the selected `final_code`; no audit state contributes to it. The generation-time objective and final detector raw score are therefore identical by construction.

## Quality protection

The gate may use only public static properties, not HumanEval tests, hidden labels, candidate correctness, or posthoc Pass reports. A legal candidate must:

- parse as Python;
- contain exactly one target function identified from the prompt;
- preserve the prompt function signature;
- contain an executable target body rather than only `pass`, ellipsis, or a docstring;
- contain no Markdown code fence;
- avoid a syntactically suspicious/truncated tail;
- remain within the configured generation length bound.

HumanEval execution happens only after final selection and cannot affect the output.

## V2 detector and calibration

The detector mode is `wfcllm-aligned-canonical-signature/v2`. The calibration artifact schema is `wfcllm-detect-calibration/v2` and records the public config, key hash, encoder/checkpoint/tokenizer fingerprints, calibration score list, strict threshold, and calibration row count. It never contains the raw key.

For 82 calibration negatives and target FPR 0.05, V2 uses a pre-registered strict upper order statistic:

```text
i = ceil((n + 1) * (1 - alpha)) - 1
threshold = nextafter(sorted_scores[i], +infinity)
```

with zero-based clipping. Detection uses `score >= threshold`. The nextafter step handles ties conservatively. Held-out negatives are never used to select or adjust this threshold.

A result is insufficient only if no canonical unit can be recovered. V2 emits raw score, calibrated empirical p-value, unique-unit count, total signature bits, duplicate count, and threshold. These fields are detector outputs, not inputs.

## Artifact compatibility and fail-fast behavior

V2 calibration loading requires both v2 artifact type and schema. V1 loaders reject v2 artifacts; v2 loaders reject v1 or unversioned artifacts. Public configs include `method_schema_version` and `canonical_unit_schema_version`. There is no auto-upgrade and no silent v1/v2 mixing.

The strict four-field final-code input is method-neutral by design: detector version is selected explicitly and compatibility is enforced by its calibration artifact.

Diagnostic retry ledgers use `wfcllm-v2-retry-ledger/v2`, are marked `audit_only=true` and `detector_input_allowed=false`, and may contain candidate code and scores for mechanism analysis. The official detector never opens them.

## Determinism and replay

The following are frozen compatibility inputs:

- CodeT5 model/tokenizer paths and hashes;
- fine-tuned checkpoint hash;
- encoder precision and device policy;
- signature dimension;
- canonical schema and normalization;
- key-target derivation domain separator;
- retry seed schedule and attempt count.

Phase A requires 100% equality between generation-time selected raw score and a fresh detector process scoring the saved final code. It also requires identical unit hashes and signatures across two independent encoder loads. Failure blocks the pilot.

## Claims and limits

If all hard gates pass, the supported claim is limited to improvement on the frozen single-seed HumanEval setting. V2 does not establish cross-seed stability. If any Pass, FPR, TPR, replay, leakage, or final-code-only gate fails, V2 remains experimental and the report states a negative result.
