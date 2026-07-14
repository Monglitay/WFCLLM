# Watermark Mechanism V3 Candidate Decision

Date: 2026-07-14
Baseline: `eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`
Decision status: frozen before implementation and before any dynamic-V3 outcome

## Non-negotiable interpretation

V3 is a semantic-code watermark. Its official evidence is the frozen CodeT5+LoRA
semantic encoder output. A secret may affect a decision only after that output has
been produced. The detector receives only the final source code and the private
key. It may not read the prompt, generation trace, candidate pool, accepted-event
ledger, token log-probabilities, or hidden metadata. Syntax features are allowed
only for public unit extraction and context reconstruction; they are not an
alternative watermark alphabet.

All candidates below use the same 20 raw generation trajectories as Current-R20,
with the same task, prompt, base model, decoding parameters, per-attempt seeds,
and retry budget. This makes candidate-pool hashes a paired fairness contract.

## Candidate A — Streaming Shared-Pool Semantic Closure (selected)

Generate the 20 Current-R20 trajectories once. While each trajectory grows, a
public incremental Python boundary tracker emits a unit when a top-level statement
or bounded statement block becomes closed. Newly closed public contexts from all
live trajectories are gathered into one encoder micro-batch. A secret-keyed
post-encoder projection maps every context to a 7-bit semantic signature; a
content-addressed Hamming(7,4) target supplies the expected 7 bits. The trajectory
accumulates exact signed bit matches as its units close. At EOS, select among the
same quality-valid trajectories by the accumulated semantic statistic, without
re-encoding all complete candidates. Replay only the selected final code with the
R3 detector.

This is dynamic because semantic evidence is produced and accumulated at generation
time, not retrospectively used to score twenty completed programs. It is also an
honest selection watermark rather than logit steering: the secret changes which
member of the shared stochastic pool becomes public, not how the pool is sampled.

## Candidate B — Block Rollback Semantic Search

At every stable block closure, encode the public context, test its secret target,
and resample or roll back a bounded suffix when the target is missed. This is more
locally causal than A and may increase capacity, but it changes the generation
trajectory and candidate pool relative to Current-R20, complicates KV-cache repair,
creates synchronization state, and risks large quality/cost variance. It is retained
as a future mechanism, not the confirmatory arm.

## Candidate C — Semantic Beam / Suffix Expansion

Maintain several partial programs; at a public boundary, expand each with multiple
suffixes, batch-encode their contexts, and prune with a semantic objective plus a
quality constraint. It offers the clearest accelerator batching but requires a new
beam-search distribution, substantially more GPU memory, and a fairness baseline
with an identically sized unwatermarked beam. It is not compatible with the frozen
Current-R20 pool contract.

## Optional Candidate D — Token-Level Semantic Shift

Following SemanticShift, modify token logits using a dynamic semantic direction
derived from the prefix. This is genuinely generation-time and avoids completed-code
rescoring, but the current project encoder is a code-sequence encoder rather than a
validated token-level semantic controller. Adapting it would require new training
or a different model and would violate the request to reuse the frozen encoder.

## Decision matrix

| Dimension | A: streaming shared pool | B: block rollback | C: semantic beam | D: token shift |
|---|---|---|---|---|
| Official semantic evidence | frozen CodeT5+LoRA | same | same | would need new evidence model |
| Dynamic generation-time use | unit closure accumulation | closure-time accept/resample | closure-time beam pruning | per-token logit change |
| Secret application | post-encoder projection and target | same | same | post-representation logit shift |
| Final-code-only observability | yes | yes if rollback state is discarded | yes if context is reconstructible | uncertain with current encoder |
| Exact R3 replay | direct, selected final only | possible but state repair is fragile | possible | not yet defined |
| Public context reconstruction | bounded canonical predecessor/current unit | same | same | prefix reconstruction required |
| Encoder calls per task | micro-batches during 20 trajectories + one replay | potentially many retries | expansions × boundaries | potentially every token |
| Encoded contexts per task | one per closed unit per trajectory + replay | larger and variable | beam width × expansions | sequence length × candidates |
| Batchability | high across trajectories/boundaries | medium | high | high but latency-sensitive |
| RTX 5090 fit | expected, bounded queue | risky due generation rollback | risky at useful beam width | model-dependent |
| Same raw pool as Current-R20 | **yes** | no | no | no |
| Expected Pass@1 risk | selection-only, lowest | rollback may damage completion | pruning may damage diversity | logit shift may damage code |
| Capacity | 7 votes per recoverable unit | potentially higher | potentially higher | high |
| Selection bias | explicit and measurable | generation-distribution shift | beam-distribution shift | token-distribution shift |
| Wrong-key control | exact domain-separated replay | exact | exact | must be redesigned |
| Synchronization | content-addressed units; duplicates erase | rollback markers required | beam lineage required | prefix/token sync required |
| ECC | Hamming(7,4) per unit | same | same | not frozen |
| Attack handling | unit deletion/insert/reorder bounds | same plus rollback artifacts | same | token-edit dependent |
| Implementation complexity | moderate | high | high | high + training risk |
| New training | none | none | none | likely |
| Detector speed | one batched code replay | one batched replay | one batched replay | unknown |
| Leakage risk | known encoder pretraining/training provenance disclosed | same | same | new model adds unknown leakage |

## Frozen decision

Candidate A, configuration name `Streaming-SharedPool-V3-R20`, is the only formal
main arm. Candidate B and C remain design alternatives, and D remains a literature
informed future direction. The decision is based on detector isomorphism, exact
shared-pool fairness, bounded engineering risk, and the ability to replace twenty
candidate-level encoder forwards with dynamic micro-batching plus a single selected
final replay. No result-dependent candidate switch is permitted.
