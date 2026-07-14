# Watermark Mechanism V4 Root-Cause Diagnosis

Date: 2026-07-14

Status: Stage A complete

Conclusion: **A2 — the batch-dependent encoder hypothesis is supported, and an additional downstream batch-shape dependency is independently present.**

## Scope and evidence boundary

This diagnosis tests the V3 replay failures without modifying V3 code, artifacts, reports, branch history, candidate pools, conclusions, or private material. The V4 worktree was created from V3 final commit `8693e0879cbd5657b335913b7d49171bd6c77c3b` on branch `codex/watermark-mechanism-v4-batch-invariant-semantic`. V3 remained clean during the worktree audit.

The controlled input is the exact UTF-8 serialized canonical semantic context. Every context record contains a content SHA-256 that is validated against the serialization before inference. The experiment uses the V3 public encoder and whitening identities only to reproduce the failed numerical path. A separate V4 diagnostic-only key and V4 diagnostic HMAC domain generate projection rows; neither the key nor a key-derived fingerprint is present in public artifacts.

The primary public result is `ROOT_CAUSE_MATRIX_V4.json`. Large raw JSONL files are retained in the ignored directory `data/experiments/watermark_v4_batch_invariant_20260714/diagnosis/` and are not committed. Raw writers use exclusive creation and do not overwrite prior runs.

## Frozen context set

The matrix contains 63 distinct serialized contexts:

| Category | Count | Provenance |
|---|---:|---|
| V3 failures | 3 | Exact mismatch units: HumanEval/34 unit 3, HumanEval/39 unit 6, HumanEval/71 unit 2 |
| V3 exact controls | 20 | Deterministically sampled from exact V3 replay units, varied by role and token length |
| Synthetic | 20 | Public Python programs spanning at least 12 AST role prefixes and varied lengths |
| Boundary-focused | 20 | Interleaved nearest observed quantization-boundary and diagnostic projection-margin sources |

The 63 context hashes are unique. Failure IDs are diagnosis-only and are not eligible to become V4 confirmatory pilot successes.

## Controlled matrix

Each full CUDA cold process used one CPU thread, model `eval()` state, PyTorch 2.11.0+cu130, CUDA 13.0, cuDNN 91900, an RTX 5090, and `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Three independent processes ran the complete matrix. Additional processes tested `:16:8`, an unset workspace variable, and CPU single-thread reference inference.

The orthogonal conditions were:

- batch sizes 1, 2, 4, 8, 16, and 32;
- self-repeat batches and short-, long-, and failure-context mixtures;
- forward, reverse, and seeded-random order;
- dynamic padding and fixed padding to 256;
- `torch.no_grad()` and `torch.inference_mode()`;
- deterministic algorithms enabled and disabled;
- TF32 requested and disabled through the current float32 matmul-precision API;
- float32 matmul precision `highest`, `high`, and `medium`;
- zero, one, and five warm-up passes;
- pad-token and alternate-token values in attention-masked tails;
- 20 same-process B=1 reference repeats;
- CPU B=1 single-thread reference inference.

For every trial, the recorder compares input IDs, attention mask, every T5 encoder block sublayer/output, CLS hidden state, projection pre-normalization, model post-normalization, runtime post-normalization, CPU-centered vector, whitening pre-normalization, whitening post-normalization, quantized vector, diagnostic projection dots, and signature bits. It records stable tensor hashes, maximum absolute and relative differences, maximum ULP distance, cosine similarity, mismatch coordinates/values, quantized mismatch count, and signature-bit mismatch count. Full 64-dimensional quantized vectors and all seven diagnostic projection dots/bits are retained; full continuous embeddings are represented by hashes plus mismatch summaries.

## Main observations

### 1. The first unexpected difference is inside the encoder

Across 22,995 trial rows, the first-divergence counts are:

| First divergent field | Rows |
|---|---:|
| No divergence | 3,969 |
| Expected masked-tail `input_ids` edit | 189 |
| `t5_block_00_layer_00_SelfAttention` | 18,837 |

The 189 `input_ids` differences are the intentional attention-masked-tail manipulation. They do not change the attention mask and are excluded from the causal classification. Every other non-reference numerical divergence begins in the first T5 self-attention sublayer, before CLS pooling, the projection head, normalization, whitening, quantization, or hashing.

The largest observed absolute difference at that first self-attention stage is `0.0028247833251953125`. Downstream observed maxima include `0.00039486587047576904` before projection normalization, `0.0004054233431816101` after model normalization, `0.010920524597167969` before whitening normalization, `0.0012444518506526947` after whitening normalization, five integer quantization levels, and 30 integer projection-dot units. Diagnostic signature bits never changed in this matrix.

### 2. Physical batch size is primary; composition and order are not additional causes

For self-repeat batches relative to B=1, full-CUDA quantized mismatch rates rise with physical batch size:

| Batch size | Quantized mismatch rate |
|---:|---:|
| 2 | 4.7619% |
| 4 | 6.3492% |
| 8 | 9.5238% |
| 16 | 9.5238% |
| 32 | 11.1111% |

At each fixed batch size, short, long, and failure-context mixtures have exactly the same quantized mismatch rate as one another. At B=32, forward, reverse, and seeded-random order are also identical at 11.1111%. In addition, every target copy in a self-repeat batch is strictly identical to the first target copy: 10,773/10,773 comparisons, including all stored continuous-stage hashes and all discrete fields.

Therefore the tested effect is a physical batch-shape/execution-path effect, not contamination by the values or order of other candidates. Composition can still affect padding shape when shape is not controlled, but it has no independent effect in the fixed-shape comparisons run here.

### 3. Fixed sequence length is not sufficient

At B=8, fixed padding to 256 still has a 9.5238% quantized mismatch rate relative to the B=1 reference. Dynamic padding has a 6.3492% rate in the corresponding short-mixture condition. Both paths first diverge in the same self-attention sublayer. Thus fixing sequence length alone cannot satisfy exact replay.

The V3 historical result that fixed B=32 and fixed L=256 restored 30/30 is consistent with the controlled evidence: fixing the physical batch shape selects a repeatable numerical path. It does not make that path equal to B=1, and it does not remove the underlying dependency. The current matrix strengthens this explanation because all 7,308 complete non-reference CUDA conditions are bitwise stable across three cold processes, while changing batch shape reproducibly selects a different result.

### 4. Whitening has a separate batch-shape dependency

The experiment replays normalization and whitening from the exact saved encoder/projection tensor, independently of a fresh encoder forward pass. In 19,026 non-reference rows, the first isolated downstream difference is `whitening_pre_norm`; 3,969 reference rows remain exact.

This establishes a second causal path: even when encoder output is held fixed, applying the CPU float32 whitening matrix to different physical batch shapes changes the result. The observed maximum pre-normalization whitening difference is `0.010920524597167969`. Normalization then propagates or rescales it; quantization converts some perturbations into different integer coordinates. This is why the final classification is A2 rather than pure A1.

### 5. Runtime determinism flags do not remove shape dependence

With B=8 and the same mixture, deterministic algorithms enabled and disabled both produce a 9.5238% quantized mismatch rate. `inference_mode` and `no_grad`, warm-up counts 0/1/5, and masked-tail token choice also have identical quantized mismatch rates within their controlled pairs/triples.

TF32/`high` and `medium` precision materially worsen discrete stability: each changes the quantized vector in 100% of full-CUDA trials on that axis, versus 9.5238% for `highest`. Consequently, precision flags are mechanism identity, not a safe runtime-only repair.

The `:16:8` and unset-workspace runtime profiles have identical aggregate results: 213/504 quantized mismatches and no signature-bit mismatch. PyTorch also emitted a warning that the requested unified CUBLASLT workspace exceeded the `:16:8` limit. Neither workspace setting removes batch-shape dependence. The full `:4096:8` runs remain deterministic across process restarts but still choose different results for different shapes.

### 6. Repetition and process state are not the observed source

- Same-process B=1 repeat exactness: 3,591/3,591 (100%).
- Three-process B=1 reference exactness: 126/126 pairwise comparisons (100%).
- Three-process exactness over all full-matrix conditions, copies, and stored continuous/discrete evidence: 7,308/7,308 groups (100%).
- Self-repeat copies within the same physical batch: 10,773/10,773 (100%).

The effect is therefore reproducible across cold processes and insensitive to the tested warm-up/cache state. It is not random run-to-run nondeterminism under a fixed configuration; it is deterministic selection of different numerical results by physical shape and precision path.

### 7. CPU is not a drop-in equality reference

CPU single-thread B=1 is internally exact. Against the CUDA B=1 reference, 59/63 quantized vectors (93.6508%) and 63/63 diagnostic signature bit-vectors are equal. Four quantized vectors differ across device backends even at B=1. Moving inference to CPU may produce a self-consistent mechanism if frozen end to end, but it is a distinct representation identity and cannot be substituted transparently into V3 evidence.

## Answers to the ten required questions

1. **Where does inconsistency first appear?** In `t5_block_00_layer_00_SelfAttention` for every unexpected full-pipeline divergence. A second isolated dependency first appears at the CPU float32 whitening matrix multiplication.
2. **Batch size or composition?** Physical batch size is primary. At fixed B, short/long/failure composition and forward/reverse/random order have identical rates; repeated target copies are 100% strictly equal.
3. **Is fixed sequence length sufficient?** No. Fixed L=256 at B=8 still changes quantized coordinates relative to B=1.
4. **Why did fixed batch size restore V3 exactness?** It forced generation and replay onto the same repeatable physical attention/GEMM and whitening shapes. The current evidence explains the observation but does not retroactively make V3's unrecorded kernel choices independently auditable.
5. **Encoder or downstream?** Both. Encoder self-attention is the first full-path source; saved-tensor isolation independently proves a whitening batch-shape source. Normalization propagates differences; quantization amplifies them into integer mismatches. Diagnostic projection signs did not cross a boundary in this matrix.
6. **Is it reproducible across processes?** Yes. All 7,308 full-condition strict fingerprints are identical across three cold CUDA processes.
7. **Can deterministic CUDA configuration eliminate it without changing the mechanism?** No tested setting does. Deterministic mode changes repeatability guarantees, not batch invariance. `:4096:8`, `:16:8`, and unset workspace settings retain the effect; TF32/lower matmul precision worsens it.
8. **Is single-sample inference exact?** Yes within the frozen device/runtime: 20 repeats per context in each full cold process are exact, and B=1 references are exact across three cold processes. CPU and GPU B=1 are not fully equal to each other.
9. **Is there a maximum perturbation bound?** There is an observed matrix maximum per recorded stage in `ROOT_CAUSE_MATRIX_V4.json`, including 0.010920524597167969 before whitening normalization. It is empirical for the tested 63 contexts and configurations, not a certified universal bound.
10. **Can an experiment-independent stable discretization rule be constructed now?** Not from Stage A alone. A margin derived only from these maxima would be empirical and cannot be called certified. Candidate B must either use a defensible bound plus erasures or be rejected. Shape-isolated inference and fully canonical discrete representations remain candidates for Stage B probes.

## Root-cause decision

The original V3 hypothesis is **partially but strongly supported**: batch-dependent encoder inference is real and causal, but it is not the only cause. CPU whitening is also batch-shape dependent. The formal Stage A decision is therefore:

> **A2 — batch-dependent encoder inference is supported, with an additional major downstream whitening factor.**

No V4 formal mechanism, margin, threshold, split, or pilot decision is selected by this report. Candidate selection remains frozen until the Stage B literature review and runnable mechanism probes are complete.

## Reproducibility entry points

- Diagnostic library: `wfcllm/diagnostics/numerical_replay_v4.py`
- Offline runner: `scripts/wfcllm_v4_numerical_diagnosis.py`
- Unit/CLI tests: `tests/diagnostics/test_numerical_replay_v4.py`, `tests/diagnostics/test_v4_numerical_diagnosis_cli.py`
- Public matrix: `ROOT_CAUSE_MATRIX_V4.json`
- Raw context manifest: `data/experiments/watermark_v4_batch_invariant_20260714/diagnosis/context_manifest.json`
- Immutable raw processes: three `gpu_cold*_workspace4096.jsonl` files, `gpu_workspace16_runtime_flags.jsonl`, `gpu_workspace_unset_runtime_flags.jsonl`, and `cpu_reference_singlethread.jsonl`
