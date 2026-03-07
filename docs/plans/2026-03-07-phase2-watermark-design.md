# Phase 2 Design: Generation-Time Watermark Embedding

## Overview

Implement the watermark embedding pipeline in `wfcllm/watermark/`. During LLM code generation, intercept completed statement blocks, verify semantic projection against watermark targets, and use rejection sampling with KV-Cache rollback to embed watermark signals.

## Key Decisions

| Decision | Choice |
|----------|--------|
| LLM integration | Direct integration with real LLM (no mock) |
| Target models | CodeLlama-7B, DeepSeek-Coder-7B, StarCoder-7B |
| Generation approach | Custom token-by-token loop using `model.forward()` |
| Encoder E | Original CodeT5 as placeholder (not yet fine-tuned) |
| Node entropy | Heuristic estimation via AST node type lookup table |
| Entropy aggregation | Sum of all AST sub-node entropies in the block |
| Dynamic margin | `m = m_base + alpha * block_entropy` |
| Language support | Python primary, extensible design |
| Key management | Single symmetric key + HMAC-SHA256 |
| Configuration | Dataclass (consistent with encoder) |

## Module Structure

```
wfcllm/watermark/
├── __init__.py          # Public API exports
├── config.py            # WatermarkConfig dataclass
├── generator.py         # WatermarkGenerator - core generation loop
├── interceptor.py       # StatementInterceptor - incremental AST parsing
├── entropy.py           # NodeEntropyEstimator - heuristic entropy estimation
├── keying.py            # WatermarkKeying - key derivation (v, t)
├── verifier.py          # ProjectionVerifier - semantic projection check
└── kv_cache.py          # KVCacheManager - KV-Cache snapshot/rollback
```

## Data Flow

```
User prompt
    |
WatermarkGenerator.generate()
    | token-by-token model.forward()
    | each token fed to StatementInterceptor
    |
StatementInterceptor detects block closure
    |
    +-- NodeEntropyEstimator -> dynamic margin m
    +-- WatermarkKeying -> derive (v, t)
    +-- ProjectionVerifier -> semantic projection check (encoder E)
    |
    +-- Pass -> continue generating
    +-- Fail -> KVCacheManager rollback -> resample (up to n times)
                +-- All retries fail -> preserve original, wait for compound block fallback
```

## Module Designs

### 1. `config.py` - WatermarkConfig

```python
@dataclass
class WatermarkConfig:
    secret_key: str
    encoder_model_path: str = "Salesforce/codet5-base"
    encoder_embed_dim: int = 128
    encoder_device: str = "cuda"
    margin_base: float = 0.1
    margin_alpha: float = 0.05
    max_retries: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    max_new_tokens: int = 512
    eos_token_id: int | None = None
    enable_fallback: bool = True
```

### 2. `interceptor.py` - StatementInterceptor

Feeds tokens incrementally. Uses Tree-sitter to parse accumulated text after each token. Detects when a new error-free simple statement block completes by comparing current vs previous AST state. Leverages `SIMPLE_STATEMENT_TYPES` and `COMPOUND_STATEMENT_TYPES` from `wfcllm/common/ast_parser.py`. Uses Tree-sitter incremental parsing (`parser.parse(new_source, old_tree)`) for performance.

Returns `InterceptEvent` with: block text, block type (simple/compound), AST node type, parent node type, token start index, token count.

### 3. `entropy.py` - NodeEntropyEstimator

Pre-built lookup table from `experiment/node_entropy/results/node_entropy_results.json` (133 AST node types with mean entropy values). For a given statement block: parse its AST sub-tree, traverse all nodes, look up each node type's entropy, sum them. Dynamic margin: `m = m_base + alpha * sum_entropy`.

### 4. `keying.py` - WatermarkKeying

Input: parent_node_type + node_type (local topology feature). Process: HMAC-SHA256(key, "parent_type|node_type") -> 32-byte hash -> seed PRNG -> generate unit vector v in R^embed_dim + target bit t from last byte LSB. Same topology always yields same (v, t) - deterministic and reproducible at extraction time.

### 5. `verifier.py` - ProjectionVerifier

Loads encoder E (SemanticEncoder from `wfcllm/encoder/`). For code text: u = E(code_text), p = cos(u, v), t* = 2t - 1. Pass condition: sign(p) == t* AND |p| > margin. Returns VerifyResult with pass/fail, projection value, target sign, margin.

### 6. `kv_cache.py` - KVCacheManager

Snapshot: record sequence length of each KV layer (no deep copy). Rollback: truncate KV tensors to saved length (`kv[:, :, :saved_len, :]`). Much more efficient than cloning entire KV-Cache tensors (hundreds of MB for 7B models).

### 7. `generator.py` - WatermarkGenerator

Custom token-by-token generation loop using `model.forward()` directly (not `model.generate()`). Orchestrates all other modules. Core flow:

1. Forward pass -> sample next token
2. Feed token to interceptor
3. On simple block closure: verify projection
4. On failure: rollback KV-Cache, retry up to n times
5. On all retries failed: mark for fallback
6. On compound block closure: check pending fallbacks, attempt macro-level embedding

## Test Strategy

```
tests/watermark/
├── test_config.py       # Parameter validation
├── test_interceptor.py  # Block detection with token sequences
├── test_entropy.py      # Lookup table + margin calculation
├── test_keying.py       # Determinism, vector normalization
├── test_verifier.py     # Projection logic with mock encoder outputs
├── test_kv_cache.py     # Truncation rollback
└── test_generator.py    # End-to-end with small model
```
