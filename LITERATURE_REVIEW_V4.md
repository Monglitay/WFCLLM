# Watermark Mechanism V4 Literature Review

Date searched: 2026-07-14

Status: formal Stage B review; not a preregistration and not evidence that any V4 candidate passes its gates.

## Search method and scope

The review prioritizes first-party evidence. It examined the first three pages of 18 relevant PDFs already present in the repository paper library, then verified metadata and current claims against official PyTorch and NVIDIA documentation, ACL Anthology, PMLR, ACM DOI, OpenReview, and arXiv author pages. Secondary blogs were not used as evidence. The structured record is `literature_matrix_v4.json`; its 24 entries contain title, authors, year, venue, a canonical DOI/arXiv/official URL, source status, V4 relation, candidate support/opposition, and reproducibility limitations.

The search covers deterministic neural inference, batch-invariant inference, CUDA floating-point behavior, PyTorch/cuBLAS controls, quantization and projection boundaries, randomized or deterministic smoothing, error/erasure coding, semantic text watermarking, code watermarking, canonical program representations, final-artifact detection, perceptual hashing, and attacks based on semantics-preserving code transformations.

## Deterministic inference is not batch-invariant inference

PyTorch's official numerical-accuracy note states that floating-point operations are not associative and explicitly warns that a batched result such as `(A @ B)[0]` is not guaranteed to be bitwise equal to `A[0] @ B[0]`, even though the mathematics is identical. The same note allows CPU and GPU results to differ and documents the reduced mantissa used by TF32. This is a direct first-party explanation for the Stage A pattern: fixed shapes are reproducible, while different physical shapes select different arithmetic paths.

PyTorch's reproducibility and `torch.use_deterministic_algorithms` documentation narrows what its controls promise. They select deterministic implementations where available and can fail on known nondeterministic operations, but they do not guarantee equality across releases, platforms, devices, or different computation shapes. Deterministic operations can also be slower. Thus, setting a flag is necessary runtime hygiene for Candidate A, not a proof of its batch-invariance contract.

NVIDIA's cuBLAS reproducibility section promises repeatable results for a fixed toolkit, architecture, SM count, and supported execution conditions. It documents multiple streams, workspace selection, and atomics as exceptions and offers caller-owned workspace or `CUBLAS_WORKSPACE_CONFIG`. It does not promise equality across distinct GEMM or batch shapes. Stage A agrees: `:4096:8` made every fixed condition repeatable across three processes but did not make different batch shapes equal.

Recent systems papers make the missing ingredient explicit. Zhang et al. align intra- and inter-GPU reduction trees with invariant kernels to obtain bitwise equality across tensor-parallel sizes. LLM-42 instead uses a fixed-shape verify/rollback path around faster dynamic inference. Both support a V4 design principle: equality across schedules requires either invariant arithmetic or an isolated canonical replay path. Neither implementation can be transplanted into the frozen CodeT5/PyTorch stack without a new kernel or mechanism identity.

## Stable discretization requires a justified threat bound

Charikar's random-hyperplane LSH provides an angular-similarity collision law, not stability for a vector whose projection lies near zero. A projection sign is stable only when a perturbation bound and the distance to the hyperplane are connected. Perceptual-hash work based on quantization-step analysis likewise treats boundary placement as central; similarity alone is not an exact-replay criterion.

Cohen et al.'s randomized smoothing and Levine and Feizi's deterministic smoothing illustrate what a real certificate contains: an explicit perturbation model, probability or norm statement, and assumptions that connect the randomized or discretized classifier to a radius. They do not certify GPU implementation noise. Stage A supplies only a finite observed maximum, so Candidate B may be called an empirical-margin probe but not a certified mechanism unless it derives a valid bound under the frozen runtime.

Chao et al.'s robust binary code watermark shows how error-correcting codes, deletions, and formal statistical tests can aggregate weak evidence. ECC can improve recovery after errors or erasures; it cannot make an unstable erasure decision exact. V4 therefore has to establish equality of the erasure mask before applying ECC.

## Semantic watermarking motivates the signal but not replay exactness

SemStamp uses sentence embeddings, random-hyperplane LSH, and rejection sampling to produce paraphrase-resistant semantic watermarks. k-SemStamp replaces arbitrary hyperplanes with a learned clustering partition to improve robustness and sampling efficiency. These works support the goal of semantic evidence and show that partition geometry matters. They do not report bitwise equality across batch schedules, encoder kernels, or CPU/GPU backends. A frozen codebook also adds an artifact identity and Voronoi-boundary problem.

CodeT5 and GraphCodeBERT support the scientific value of neural and data-flow-aware code representations. CodeT5 explicitly uses identifier-aware objectives; GraphCodeBERT uses data-flow edges; SynCoBERT treats source, syntax, and identifiers as complementary. Therefore a V4 mechanism that drops neural evidence must narrow its claim to canonical structural/data-flow semantics rather than silently calling an AST digest equivalent to a learned semantic embedding.

## Code watermarking makes correctness and detector purity first-class

SWEET demonstrates that code's low-entropy regions weaken token watermarks and motivates selective eligibility. CodeIP uses grammar/type constraints for multi-bit code watermarking. STONE shows that high-entropy tokens can be syntax-critical and filters syntax elements to protect correctness. ACW learns AST-guided insertion strategies. These methods reinforce V4's independent-unit and correctness gates, but their token-generation contracts do not establish final-code-only semantic replay.

Final-artifact detection must be evaluated under edits. Suresh et al. show that variable renaming and dead-code insertion can erase existing code watermarks. Disappearing Ink gives both theoretical and empirical negative evidence for n-gram code watermarks under obfuscation. These papers oppose any claim that clean exact replay implies robustness. V4 must report each prescribed attack over all samples and may not condition on successes.

FA-AST augments syntax with control/data-flow edges to distinguish semantic clones better than plain AST structure. It motivates Candidate C's bounded structural-semantic representation. Its learned GNN is not needed for V4 evidence and would reintroduce numerical replay risk. Deterministic AST roles, normalized operators/literals, scoped definition/use relations, and bounded data-flow edges are the transferable part.

## Literature-to-candidate implications

| Candidate | Literature-supported rationale | Literature-imposed limitation |
|---|---|---|
| A: shape-isolated neural inference | Official docs and invariant-inference work support a fixed canonical path | Per-context/fixed-shape execution may miss the cost gate; deterministic flags alone are insufficient |
| B: projection margins with erasures | Random-hyperplane, quantization, smoothing, and ECC work support margin/erasure reasoning | Stage A bound is empirical; erasure masks must themselves be exact; ECC cannot repair replay disagreement |
| C: canonical structural-semantic evidence | FA-AST, GraphCodeBERT, CodeIP, STONE, and ACW support AST/data-flow features | Structural evidence is not equivalent to learned semantics and may fail rename/reorder/insertion/deletion attacks |
| D: neural auxiliary plus canonical signature | Preserves a public semantic quality signal while final evidence remains discrete | Neural output must not enter R3 evidence, keying, cache identity, validity, generation, or a hidden replay input |

## Rejected shortcuts

- “Deterministic CUDA” is not accepted as a synonym for batch invariance.
- Cosine similarity is not accepted as exact replay.
- An observed Stage A maximum is not accepted as a certified radius.
- ECC is not accepted as permission for generation and R3 erasure masks to differ.
- A plain source-text or AST hash is not accepted as a semantic claim without scoped structural/data-flow features.
- Clean-code exactness is not accepted as robustness evidence.
- A CPU representation is not accepted as the same mechanism identity as CUDA merely because both are individually repeatable.

## Open evidence required before preregistration

The literature cannot decide V4 by itself. The frozen Stage B probe must measure each family on the same public contexts and diagnostic candidate pools: strict replay, wall time excluding load/warm-up, model load and warm-up separately, peak VRAM, erasure rate, independent-unit coverage, a clearly labeled selection-signal proxy, implementation complexity, R3 input purity, and newly introduced assumptions. Only measured candidates that can plausibly satisfy the formal cost and exactness gates may proceed to preregistration.

## Primary-source index

The complete 24-entry index is machine-readable in `literature_matrix_v4.json`. Key sources include:

- PyTorch, *Numerical accuracy*, *Reproducibility*, and `torch.use_deterministic_algorithms` official documentation.
- NVIDIA, *cuBLAS Results Reproducibility* official documentation.
- Zhang et al. (2025), *Deterministic Inference across Tensor Parallel Sizes That Eliminates Training-Inference Mismatch*, arXiv:2511.17826.
- Gond et al. (2026), *LLM-42*, arXiv:2601.17768.
- Charikar (2002), *Similarity Estimation Techniques from Rounding Algorithms*, DOI 10.1145/509907.509965.
- Cohen et al. (2019) and Levine & Feizi (2021), PMLR certified-smoothing papers.
- Chao et al. (2024/2026), *Watermarking Language Models with Error Correcting Codes*, arXiv:2406.10281.
- Hou et al. (2024), SemStamp and k-SemStamp, ACL Anthology.
- Lee et al. (2024), SWEET; Guan et al. (2024), CodeIP; Kim et al. (2026), STONE.
- Guo & Cheng (2025), ACW, NeurIPS/OpenReview.
- Wang et al. (2020), FA-AST; Guo et al. (2021), GraphCodeBERT; Wang et al. (2021), SynCoBERT; Wang et al. (2021), CodeT5.
- Suresh et al. (2024) and Zhang et al. (2025), negative robustness evidence for code watermarking.
