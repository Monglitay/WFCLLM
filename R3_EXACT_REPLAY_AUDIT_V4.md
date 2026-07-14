# V4 R3 Exact Replay Audit

## Result

The formal pilot result is **30/30 exact in each of three independent cold
processes**. The aggregate is 90/90 selected-final replays with no mismatch field.

The R3 detector received only a `DetectorPayload(final_code=...)`, the frozen
public config, the frozen calibration artifact, and a key loaded from the private
key file. Its payload parser rejects prompt, task ID, attempt index, candidate
pool, ledger, stored embeddings, batch metadata, and any other extra field.

## Exact fields

Comparison was strict equality over:

- sorted unit IDs;
- canonical-context SHA-256 values;
- the 32 representation bytes and identity-quantized values;
- erasure masks;
- signature and target bits;
- match counts and per-unit numerator/denominator;
- aggregate numerator/denominator;
- independent units and eligibility;
- empirical p-value and final decision.

No cosine tolerance or relaxed floating-point comparison was used. The formal
representation contains no floating-point field.

## Additional invariance

For all 600 candidates in each cold process, evidence was repeated under nominal
batch sizes 1, 2, 4, 8, 16, and 32; forward, reverse, deterministic permutation,
short-first, and long-first order; and cross-task adversarial composition. All
conditions matched the reference evidence bit for bit. Cache miss and cache hit
paths were also byte-identical.

The common process-independent identity is
`52b9ac6e6cfcf819b9b5165f06cca1f3055cdddaeaf0fd91881e021ddd4bb63e`.
