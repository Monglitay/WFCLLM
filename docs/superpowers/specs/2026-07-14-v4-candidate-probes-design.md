# V4 Candidate Probe Design

Date: 2026-07-14

## Objective

Implement a bounded diagnostic harness that compares four V4 mechanism families before preregistration. It must answer whether each family can plausibly satisfy exact replay, performance, coverage, signal capacity, and R3 purity without touching V3 history, V3 private keys, or any V4 held-out source.

The user's task already approves autonomous execution and fixes the candidate families and hard constraints. That explicit specification serves as design approval; the harness may not expand the experiment into a pilot.

## Components

`wfcllm/diagnostics/candidate_probe_v4.py` contains pure, testable helpers:

- versioned canonical structural/data-flow feature extraction;
- V4 diagnostic projection/target derivation with separate domains;
- empirical margin/erasure comparison;
- strict evidence comparison;
- candidate-pool score-capacity summaries;
- typed result records and validation.

`scripts/wfcllm_v4_candidate_probe.py` owns I/O and expensive inference:

- loads the frozen Stage A context manifest and formal raw files;
- loads the public V3 encoder/whitening identity and a V4 diagnostic key;
- reads a declared read-only diagnostic slice of the V3 ordered candidate ledger;
- profiles A/B/C/D without model or data downloads;
- writes a public JSON artifact containing no key-derived fingerprint.

## Data flow

Candidate A receives serialized public contexts, executes B=1/L=256 encoder and B=1 whitening, then applies V4 diagnostic projections. Caller batches are decomposed before inference; a public content-addressed cache may bypass recomputation.

Candidate B receives Stage A reference and replay projection dots. A bound frozen from Stage A before signal inspection defines erasures. Both the bit and erasure mask are compared.

Candidate C receives only final code, extracts public units, constructs canonical structural/data-flow records, derives signed bits and independent targets under separate diagnostic domains, and aggregates evidence.

Candidate D reuses C's evidence exactly. A neural score may only break ties after signed scores are computed; it is absent from every detector artifact.

## Failure handling

- Malformed context or raw rows fail fast.
- Missing/incorrectly permissioned diagnostic key fails fast.
- Parse failure, oversize context, and duplicate unit IDs produce typed erasures rather than silent truncation.
- Candidate-pool rows must contain exactly 20 ordered attempts per chosen task and matching content hashes.
- Output creation is exclusive; a probe never overwrites raw evidence.
- The artifact states when a metric is a proxy or projection rather than a formal gate result.

## Tests

Tests are written before implementation and cover:

- identifier/format canonicalization and statement-role sensitivity;
- deterministic feature serialization and unit IDs;
- projection/target domain separation;
- erasure-bound ties and exact-mask comparison;
- strict evidence fields;
- candidate-pool cardinality/order/hash validation;
- score-capacity summaries and outlier checks;
- key redaction and forbidden public fields;
- CLI schema and fail-fast behavior.

Expensive encoder behavior is exercised by the real offline probe after pure tests pass. Existing V3 and Current behavior is not modified.

## Decision output

The public probe artifact feeds `V4_METHOD.md`. The selected candidate must have no failed exactness or purity check and must possess credible cost headroom. Candidate B remains ineligible for a formal “certified” claim while its bound is empirical. Candidate D is rejected unless it shows a benefit over simpler C.
