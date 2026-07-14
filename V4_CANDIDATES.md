# V4 Mechanism Candidates and Probe Contract

Date: 2026-07-14

Status: candidate design frozen for Stage B probes; winner not yet selected.

## Shared inputs and outputs

The probe uses two engineering-only sources:

1. the 63 public contexts frozen in Stage A; and
2. a small, explicitly diagnostic slice of the ordered V3 pilot attempt ledger, read-only, solely to estimate whether a candidate can choose a higher signed score from an unchanged 20-candidate pool.

V3 private keys and V3 threshold artifacts are forbidden. A V4 diagnostic-only key and V4 probe HMAC header derive projection and target domains separately. Probe IDs and scores are not confirmatory V4 pilot evidence.

Every candidate result records:

- strict replay numerator/denominator;
- cold-process/cache/schedule scope;
- steady-state semantic seconds per context and a task-level cost projection with assumptions;
- model load and warm-up separately;
- peak CUDA allocated memory;
- representation/bit erasure rate;
- eligible independent-unit coverage;
- diagnostic selection-score shift or an explicit “not measured” reason;
- public cache identity and R3 input fields;
- implementation-complexity counts;
- new unverifiable assumptions.

## Candidate A — deterministic shape-isolated neural inference

Each canonical context is encoded independently at B=1 and L=256. Whitening is also applied one row at a time. The cache key is a public tuple of schema identity, encoder checkpoint identity, whitening identity, deterministic runtime identity, and canonical context hash. The private key never enters the cache.

Probe:

- encode all 63 contexts twice through cache-miss and cache-hit paths;
- compare evidence against adversarial caller schedules and three fresh processes;
- time per-context miss, hit, and selected-final replay;
- measure peak VRAM and encoder-call cardinality;
- estimate a task cost from the observed unit-count distribution without treating the estimate as a formal profile.

Pass condition: 100% strict equality and credible task cost at or below 0.1409884 seconds. Exactness without cost headroom rejects A as the formal mechanism.

## Candidate B — empirical margin projection with erasures

This probe deliberately uses the name “empirical,” not “certified.” It freezes the Stage A maximum observed projection-dot change before inspecting signal. A bit is eligible only when its reference and replay margins are on the same side of the frozen safety rule. The erasure mask is itself an exact field.

Probe:

- replay every formal Stage A raw condition;
- measure sign and erasure-mask equality, erasure rate, surviving bits per unit, and independent-unit coverage;
- compare an optimistic observed-dot bound with the conservative coordinate-wise bound;
- apply no threshold or margin tuning based on score shift.

Pass condition: exact erasure masks and adequate coverage under a defensible bound. The observed-only bound cannot satisfy the formal certification requirement, so B can at most motivate future work unless a theoretical/runtime certificate is added before preregistration.

## Candidate C — canonical discrete structural-semantic representation

Each final-code unit is represented by a versioned canonical record containing statement role, AST node/operator/literal categories, scoped identifier definition/use slots, call arity, control constructs, and bounded local data-flow relations. Formatting, comments, global ordinal, prompt, task ID, and candidate metadata are excluded. Oversize, parse-failure, and duplicate units become explicit erasures.

Separate V4 HMAC domains map the public representation to signature bits and the public unit ID to target bits. The key does not affect parsing, unit validity, or the cache key.

Probe:

- reconstruct representations repeatedly and under formatting/identifier transformations;
- measure exactness, CPU time, representation collisions, erasures, and independent units;
- score the same ordered 20-candidate pools and report the selected-versus-first-candidate score shift as a capacity proxy;
- report reorder/insertion/deletion sensitivity without calling clean equality robustness.

Pass condition: 100% replay, sufficient independent units, positive selection-capacity proxy not dominated by one pool, and negligible cost. The supported claim remains bounded structural/data-flow semantics.

## Candidate D — neural auxiliary / canonical signed evidence

D reuses C for every signed field and for R3. The frozen neural encoder may supply a public, key-independent semantic-quality tie-breaker among candidates with equal canonical signed score. It cannot affect raw generation, validity, evidence, unit IDs, cache keys, or R3 payload.

Probe:

- verify D's signed evidence is byte-for-byte identical to C;
- measure incremental dynamic-neural cost and batching sensitivity;
- check whether the auxiliary changes any selection or quality proxy relative to C;
- reject D if it adds cost/assumptions without measured benefit.

## Predeclared comparison

| Property | A | B | C | D |
|---|---|---|---|---|
| Learned semantic evidence in final signature | Yes | Yes | No | No |
| Natural batch invariance | Only through isolation | Only if mask exact | Yes | Yes for signed fields |
| Needs a perturbation certificate | No | Yes | No | No |
| R3 neural inference | Yes | Yes | No | No |
| Likely cost risk | High | Medium | Low | Medium |
| Semantic-claim risk | Low | Low | Medium/high | Medium |
| Added assumptions | Frozen runtime path | Valid bound + stable mask | Canonicalizer semantics | C plus harmless auxiliary |

The winner will be recorded only after runnable probe artifacts are produced. No probe may open V4 held-out data or influence the later frozen negative split.
