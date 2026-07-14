# V4 Numerical Replay Audit

Date: 2026-07-14

Scope: Stage A controlled numerical diagnosis

Audit status: **PASS with explicit empirical-bound and independence limitations**

## Audit question

Does the Stage A evidence distinguish input construction, batch size, composition, order, padding, CUDA/runtime state, encoder arithmetic, normalization, whitening, quantization, and projection boundaries without relying only on the final V3 27/30 or fixed-batch 30/30 outcomes?

Yes. The public summary is recomputed from 22,995 raw numerical trial rows across six fresh processes and 63 content-validated canonical contexts. The result supports A2: an encoder self-attention batch-shape effect plus an independently replayed whitening batch-shape effect.

## Provenance and isolation checks

| Check | Result | Evidence |
|---|---|---|
| V4 base | PASS | Worktree branch created from full V3 final SHA `8693e0879cbd5657b335913b7d49171bd6c77c3b` |
| V3 immutability | PASS | V3 branch/HEAD verified and V3 worktree clean before V4 worktree creation; no V3 file was edited |
| Serialized input identity | PASS | Each `ContextCase` validates SHA-256 of exact UTF-8 serialized content before execution |
| Required failure contexts | PASS | Exact HumanEval/34 unit 3, /39 unit 6, /71 unit 2 contexts included |
| Control count | PASS | 20 V3-exact control contexts |
| Synthetic count | PASS | 20 contexts covering varied AST roles and lengths |
| Boundary count | PASS | 20 deterministically selected boundary-focused contexts |
| Duplicate contexts | PASS | 63 unique content hashes |
| Private key separation | PASS | V4 diagnostic-only key and diagnostic domain; V3 key not read or reused |
| Public secret metadata | PASS | Raw/public schemas set `secret_metadata_included=false`; no key, key hash, or key fingerprint field exists |

## Matrix coverage audit

The full CUDA plan contains 40 orthogonal conditions per context: a 20-repeat B=1 reference plus 39 controlled conditions spanning batch size, composition, order, padding, grad mode, deterministic mode, TF32, matmul precision, warm-up, and masked-tail values. It includes every required batch size 1/2/4/8/16/32, every required composition family, dynamic and fixed-256 padding, both `no_grad` and `inference_mode`, and model `eval()` inference.

Three independent full CUDA processes used `CUBLAS_WORKSPACE_CONFIG=:4096:8`. Separate processes used `:16:8`, an explicitly unset workspace variable, and CPU single-thread B=1 reference inference. Runtime metadata records Python, PyTorch, CUDA, cuDNN, device, CPU thread count, profile, restart ID, workspace setting, and the public encoder checkpoint identity.

The context counts and controlled-axis values in `ROOT_CAUSE_MATRIX_V4.json` match the plan. Each full raw file contains 7,309 lines: one process metadata row plus 7,308 numerical trial rows.

## Capture-depth audit

The recorder does not infer encoder causality from final quantized vectors. Forward hooks capture every T5 block's self-attention, layer norms, feed-forward sublayer, and block output. The downstream chain records CLS, projection, both normalization calls, centering, whitening, quantization, projection dots, and signs.

For continuous tensors, public evidence contains deterministic tensor hashes, maximum absolute/relative/ULP differences, cosine similarity, and bounded mismatch coordinates with reference/candidate values. This is sufficient to identify the first unequal layer without storing multi-gigabyte full embeddings. For final discrete evidence, the complete 64-coordinate quantized vector and all diagnostic dot products/signature bits are retained.

An explicit saved-tensor path replays normalization and whitening from the same captured encoder/projection tensor. This is the decisive control that prevents an encoder difference from being incorrectly blamed for every downstream difference.

## Exactness audit

The word “exact” in the Stage A matrix means bitwise equality of all stored candidate tensor hashes and discrete fields, not cosine proximity.

| Exactness check | Result |
|---|---:|
| Same-process B=1 repeat comparisons | 3,591/3,591 |
| Cross-cold-process B=1 reference comparisons | 126/126 |
| Cross-cold-process all-condition strict groups | 7,308/7,308 |
| Within-batch target-copy strict comparisons | 10,773/10,773 |

The cross-cold all-condition fingerprint covers every stored continuous-stage candidate hash, the isolated-downstream hashes, full quantized vector, full projection-dot vector, and signature bits. It is stronger than comparing only the final sign vector.

The 189 intentional masked-tail input-ID changes are labeled by axis and excluded from root-cause classification. The attention mask remains unchanged. No other input construction difference precedes the encoder divergence.

## Causal audit

The first unexpected difference is `t5_block_00_layer_00_SelfAttention` in 18,837 rows. With exact encoder/projection tensors held fixed, 19,026 rows first differ at `whitening_pre_norm`. This supports two independently controlled causal paths.

At each batch size, short, long, and failure-context mixtures have identical quantized mismatch rates. At B=32, all three orders have identical rates. Within-batch duplicate targets are strictly equal. These controls reject batch content and order as primary factors in the tested implementation and identify physical shape as the factor.

Fixed L=256 does not eliminate differences. Deterministic-algorithm mode, `no_grad` versus `inference_mode`, and warm-up count do not change the mismatch rate in their controlled comparisons. TF32/`high`/`medium` precision changes all tested quantized vectors on those axes and therefore cannot be treated as a harmless determinism toggle.

The `:16:8` process emitted a PyTorch warning that requested CUBLASLT workspace exceeded the configured limit. Its aggregate mismatch counts are nevertheless identical to the unset-workspace profile. The warning and the unequal physical resource policies are disclosed; no conclusion relies on treating `:16:8` as equivalent to `:4096:8`.

## CPU reference audit

CPU B=1 is self-consistent but not identical to CUDA B=1: 59/63 quantized vectors and 63/63 diagnostic sign vectors match. This is correctly reported as a backend comparison, not generation-to-detector exact replay. It prevents the report from presenting CPU as a transparent numerical oracle.

## Empirical-bound audit

The matrix reports maximum observed absolute differences by stage. These values are bounded observations over 63 contexts and the enumerated configurations only. Neither the code nor reports label them certified, theoretical, universal, or sufficient for a margin rule. A future erasure margin may use them as exploratory evidence but cannot claim certification without a stronger argument.

## Artifact and failure preservation

The canonical context set and every formal raw run are written with exclusive creation. Earlier engineering failures and exploratory debug runs remain in the ignored diagnosis directory and were not overwritten or deleted. They are excluded from the formal summary input list. `ROOT_CAUSE_MATRIX_V4.json` lists exactly the six formal process metadata records used for the public result.

Large raw artifacts and both V4 private files remain under a gitignored experiment root. Only code, tests, public summary, and reports are candidates for commit.

## Review independence

A separate read-only subagent reviewed the V3 numerical path and the proposed Stage A design before the final matrix. It independently identified that V3 had not isolated encoder output from CPU whitening, had not recorded physical batch/runtime details for the historical fixed-B result, and had overinterpreted quantized mismatches as signature mismatches. Those findings were incorporated into the capture layers and saved-tensor control.

This is an independent design review within the same agent system, not an external replication and not a completed independent result-to-claim audit. The final experiment-integrity and result-to-claim audits remain later V4 deliverables.

## Limitations

- The matrix is finite and cannot establish a universal floating-point perturbation bound.
- Diagnostic projection bits did not flip, so Stage A does not estimate watermark detection power.
- The CPU comparison covers B=1 reference inference, not the full CPU batch matrix.
- The historical V3 fixed-B=32 run lacks enough physical-kernel metadata for independent bit-for-bit reconstruction; V4 does not reinterpret it.
- Stage A does not select a V4 mechanism, threshold, margin, ECC, split, or pilot ID.

## Audit conclusion

Stage A passes its integrity objective. The evidence is sufficient for the limited claim that physical batch shape changes the encoder's first self-attention result and that the existing CPU whitening application is independently batch-shape sensitive. It is not sufficient for a certified margin, a watermark-success claim, or a claim that CUDA is randomly nondeterministic under a fixed configuration.

The formal root-cause label is **A2: encoder and downstream whitening batch dependence**.
