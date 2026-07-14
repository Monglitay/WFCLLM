# V4 Cache Fairness Audit

## Result

The cache fairness result is **PASS**. Each cold process observed 226 public
context cache misses followed by 2,298 hits, a measured hit rate of
0.910459587955626. Every miss result, hit result, and direct canonical
representation was byte-identical.

## Cache identity and isolation

The cache key is derived only from:

1. the structural context schema version;
2. the SHA-256 identity of the frozen public V4 config; and
3. the canonical context SHA-256.

It contains no private key, task ID, attempt index, candidate identity, batch
metadata, prompt, or generation trace. A factory result that differs from the
canonical representation fails fast. Flush order is public bookkeeping and does
not change cached bytes.

## Fairness checks

- Same context on cache miss versus cache hit: 100% exact.
- Other candidates before or after the target context: 100% exact.
- Forward, reverse, permuted, short-first, and long-first composition: 100% exact.
- Candidate-pool hashes before and after V4 selection: 100% exact.
- Private key influence on raw generation or cache identity: none.
- Pre-stored generation-time final embeddings: not used; the formal mechanism has
  no neural embedding and R3 reconstructs from final code.

The cache timing is included in the structural selection path where exercised.
No cache behavior was used to skip required candidate evidence or the exactly-one
selected-final replay.
