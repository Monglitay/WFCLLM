# V4 Candidate Hypotheses

Date: 2026-07-14

These hypotheses are frozen for Stage B engineering probes only. They are not the formal V4 preregistration and cannot be changed in response to probe detection scores.

## Shared null and measurement contract

The shared null is that a candidate cannot simultaneously provide strict final-code-only replay, acceptable cost, sufficient independent evidence, and a plausible positive selection signal under V4's purity constraints.

All candidates consume the same public canonical contexts and, where selection capacity is measured, the same ordered raw candidate pool. The diagnostic key is V4-only, 0600, and excluded from public artifacts. Signal reported before a formal pilot is a capacity/proxy measurement, never TPR or watermark success.

Strict replay compares the complete representation, erasure mask, signature bits, target bits, per-unit counts, aggregate counts, eligibility, and decision inputs available to the probe. A cosine match cannot pass.

## H-A: deterministic shape-isolated neural inference

If every context is encoded at B=1, L=256, `eval()`, `inference_mode`, `highest` precision, deterministic algorithms, and per-row whitening, then the same context will have exact evidence regardless of the caller's batch schedule or cache state.

Falsifiers:

- any strict mismatch across cold processes, cache hit/miss, or adversarial schedules;
- mean projected task overhead incompatible with 0.1409884 seconds;
- hidden dependence on another candidate or a stored generation embedding.

Expected risk: exact but too expensive when many cache misses occur.

## H-B: empirical projection margin with erasures

If a projection dot is farther from zero than a frozen bound on observed dot perturbation, its sign should replay exactly; near-boundary bits are erased before ECC/statistical aggregation.

Falsifiers:

- erasure-mask mismatch between any two schedules;
- insufficient independent bits/units after erasure;
- using the finite Stage A maximum as a “certified” universal bound;
- a useful signal only after tuning the margin to probe outcomes.

Expected risk: the sign can be stable while the erasure boundary is not, and a valid theoretical bound may erase too much.

## H-C: canonical discrete structural-semantic representation

A representation made only from canonical AST role, normalized node/operator/literal structure, scoped identifier definition/use classes, and bounded data-flow relations should be batch-, process-, device-, padding-, and cache-invariant and inexpensive to reconstruct from final code.

Falsifiers:

- non-exact replay for identical final code;
- dependence on prompt, task ID, attempt index, ledger, candidate pool, or neural embedding;
- representation collapse that leaves too few distinct independent units;
- inability to create a positive score shift from a frozen ordered candidate pool;
- a claim broader than the tested structural/data-flow semantics.

Expected risk: exactness is easy, but semantic scope and transformation robustness are narrower than the V3 neural claim.

## H-D: canonical signature with public neural auxiliary ranking

Neural embeddings may be retained as a public, key-independent quality/tie-breaking signal during dynamic selection, while all signed evidence and R3 replay use Candidate C's canonical representation. If the signed structural score is the selection objective and neural output is only a public tie-breaker, D should preserve C's exactness and gain useful selection behavior without neural R3 cost.

Falsifiers:

- any private-key influence on raw generation, logits, seeds, stopping, or candidate validity;
- neural evidence entering unit IDs, signature bits, detector payload, cache key, or replay artifacts;
- different selected result when the raw ordered candidate pool is unchanged but neural batching changes;
- no measurable benefit over simpler C.

Expected risk: D adds complexity without improving detection capacity; if so, C is preferred.

## Decision rule

Prefer the least complex candidate that has 100% strict replay, pure final-code-only R3 input, sufficient coverage, no forbidden generation influence, and a measured cost with credible headroom below the formal gate. Candidate B cannot win while its bound remains empirical. Candidate D cannot win without measured benefit over C.
