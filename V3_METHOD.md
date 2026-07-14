# Watermark Mechanism V3: Dynamic Semantic Closure Watermark

## 1. Method boundary

`Streaming-SharedPool-V3-R20` is a selection-based dynamic semantic watermark for
Python code. Twenty trajectories are sampled exactly as in Current-R20. During
generation, public closed-statement contexts are encoded in cross-trajectory
micro-batches and scored with a secret-keyed post-encoder semantic channel. At EOS,
the quality-valid trajectory with the best accumulated semantic statistic is chosen.
Only that final source is replayed by the official R3 detector.

The method does not use a syntax codebook, identifier-renaming alphabet, formatting
channel, hidden metadata, or prompt-dependent detector. It does not choose a seed,
change sampling logits, stop a trajectory, or repair code using the secret. The raw
candidate pool is therefore identical to Current-R20.

## 2. Literature-to-design mapping

SemStamp and k-SemStamp motivate secret partitioning after semantic embedding;
SimMark and the black-box TMLR watermark motivate calibration of continuous semantic
evidence; SemanticShift motivates generation-time rather than post-hoc semantic
use; SAEMark motivates avoiding one brittle global representation; SeqMark's
region-collapse result motivates multiple locally reconstructible regions; CodeT5,
GraphCodeBERT, and UniXcoder motivate code-aware representations; PICARD/SynCode
motivate public structural boundaries; RBC motivates per-unit ECC; SWEET, STONE,
ACW, CodeTracer, and code-watermark attack studies motivate explicit entropy,
robustness, and final-code-only audits. PagedAttention/vLLM motivates queue-based
batching but is not a runtime dependency.

## 3. Public canonical unit contract

1. Parse the final Python code with the standard-library `ast` parser.
2. Eligible units are complete statements directly in a function body and bounded
   compound-statement regions already supported by the public boundary tracker.
3. Canonicalize each statement with `ast.unparse`, LF newlines, and UTF-8.
4. Define a public role from node type, parent node type, and structural field name;
   do not use a global ordinal as the secret target.
5. Define `unit_id = SHA256(schema || role || canonical_current)`. If the same ID
   appears more than once in one program, all occurrences are erasures so duplicate
   ordering cannot become a hidden ordinal channel.
6. The context is the domain header, public role, canonical previous eligible unit
   (or `<BOS>`), and canonical current unit. The previous unit participates in
   context but not identity.
7. Tokenize with `Salesforce/codet5-base`, `max_length=256`, no silent truncation.
   A current unit over 128 tokenizer tokens or a full context over 256 tokens is an
   erasure. Parse failures are code-level erasures.

The generation tracker may propose a newly closed unit only when the current prefix
parses under the same canonical extraction rule. R3 re-extraction is the authority.

## 4. Frozen semantic evidence

- Base: `data/models/codet5-base`.
- Checkpoint: `data/models/encoder/best_model.pt`.
- Checkpoint SHA-256:
  `9da80dfdbb3f8e682298e588e822f15967e460c9dfd640a243aad8ed1ba51d25`.
- Architecture: existing project CodeT5+LoRA semantic encoder.
- Official precision: float32 encoder/projection inference. BF16 profiling is
  diagnostic only and cannot replace official evidence after preregistration.
- Official embedding is the frozen checkpoint's normalized representation before
  any secret operation.

Public whitening is fitted on calibration negatives only, uses 64 dimensions, and
is frozen before pilot detection. Neither the base encoder nor the LoRA checkpoint
is trained or selected using dynamic-V3 positive, pilot, or held-out outcomes.

## 5. Secret-keyed channel and ECC

The private root key is read from a mode-0600 file and is never accepted on the
command line or written to a public artifact. HMAC-SHA256 uses separate domains for
projection rows, unit targets, wrong-key control, and run-internal audit IDs.

After public whitening and L2 normalization, each 64-vector is quantized with public
scale 4096 to signed integers. Seven deterministic secret hyperplanes with entries
in `{-1,+1}` yield seven signature bits through integer dot-product signs. The
content-addressed target begins with four HMAC-derived data bits and is encoded by
Hamming(7,4). This gives 7 Bernoulli match votes per independent recoverable unit and
allows reporting an ECC syndrome without changing the detector decision.

For unit `i`, `m_i` is the number of matching signature/target bits and
`s_i = (2*m_i - 7)/7`. The code statistic is the untrimmed mean of `s_i` over unique
recoverable unit IDs. Codes below the frozen minimum independent-unit count are
ineligible/erasure, not silently padded.

## 6. Dynamic generation and selection

The twenty trajectories use seeds `seed + attempt * 101`, the same decoding
configuration, and the same quality gate as Current-R20. The public incremental
tracker queues newly stable contexts as tokens are generated; rollback invalidates
the corresponding live units but never changes an already computed embedding. The
scheduler drains at 32 queued contexts, after at most four completed attempts, or at
the final residual flush, up to 128 contexts per batch, in deterministic
`(attempt_index, closure_index)` order. It runs one encoder micro-batch and returns
evidence to the live trajectory accumulators. Thus the queue is shared across the
20-attempt generation stream without requiring 20 causal-model copies, and semantic
evidence is not reconstructed by rescoring completed programs.

At EOS no semantic encoder is called on all complete candidates. Selection is:

1. highest existing static quality tier;
2. highest accumulated V3 statistic;
3. most independent recoverable units;
4. fewer public fallbacks/erasures;
5. lower attempt index.

If no candidate reaches the frozen minimum unit count, use the existing best static
quality candidate and mark `no_embedding`. The only allowed repair after the pilot
is a single implementation repair that restores preregistered behavior; it may not
change the statistic, threshold, split, context, encoder, key domains, or gate.

## 7. R3 exact replay contract

R3 receives exactly `(final_code, public_config, private_key_file)`. It reconstructs
units and contexts, encodes them once as a batch, reproduces quantized signatures,
targets, per-unit votes, the aggregate statistic, and the decision.

Required invariants are exact equality of sorted unit IDs, context SHA-256 values,
quantized embeddings, signature bits, target bits, match counts, aggregate rational
numerator/denominator, and decision. Float embeddings are diagnostic and require
cosine similarity at least 0.99999 and max absolute difference at most `1e-4` across
the official generation/replay devices. Any invariant mismatch is a hard failure;
generation-time ledgers cannot repair detector output.

Public artifacts may contain schema/config versions, code/context hashes, unit
counts, aggregate statistics, decisions, timings, and memory counters. They may not
contain raw keys, key hashes/fingerprints, target bits, hyperplanes, per-unit secret
matches, or seed-selected secret material.

## 8. Calibration and decisions

The frozen negative panel is a new domain-separated 250/250 split of MBPP test,
recorded in `configs/wfcllm/v3_dynamic_semantic_split_manifest.json`. Calibration
uses the first 250 only. For score `x`, the one-sided conformal p-value is
`(1 + count(calibration_score >= x)) / 251`; detection is `p <= 0.05` and the
minimum-unit constraint is also required. Ties therefore remain conservative.

Held-out MBPP is opened once, after implementation, R3, cost, calibration, and pilot
rules are frozen. Its empirical FPR must be at most 5%. A domain-separated wrong key
must not produce a positive shift. No held-out outcome may tune a threshold or unit
rule.

## 9. Evaluation and escalation gates

Development smoke uses 8 non-pilot HumanEval IDs:
`93, 14, 161, 70, 66, 148, 62, 74`. The paired pilot uses the frozen V2 30 IDs and
retry budget 20. Full evaluation is HumanEval 164 and is forbidden unless every
pilot gate passes.

Pilot gates: exact shared-pool hashes for Current/V3; R3 replay 100%; no EOS
20-complete-code semantic rescore; semantic cost at most 0.1409884 s/task under the
comparable profile; peak VRAM within 32,607 MiB; Pass@1 at least 15/30 and no worse
than Current by more than 1 task; V3 TPR strictly above Current under the frozen
threshold; improvement not caused by one outlier; pilot negative FPR at most 5%; no
raw-key leakage; and no secret influence on raw trajectory generation.

Full gates: Pass@1 at least 82/164, held-out FPR at most 5%, V3 TPR strictly above
Current, paired statistics and confidence intervals reported, R3 replay 100%, raw
pool equality 100%, and all integrity/security audits pass.

## 10. Threat and robustness scope

The primary threat is accidental final-code-only recovery failure and false-positive
inflation. Secondary attacks are formatting normalization, identifier renaming,
dead-code insertion, independent-statement deletion, and independent-statement
reordering on the positive pilot only. Public canonicalization should absorb
formatting; content IDs preserve unaffected units under insert/delete/reorder;
changed semantic contexts become erasures or changed votes. The report must give
measured degradation and theoretical independent-unit upper bounds. No robustness
claim is allowed if the base held-out FPR or clean R3 gate fails.

## 11. Frozen environment and reporting scope

Formal runs use the authorized RTX 5090 server, PyTorch 2.11.0+cu130, Transformers
4.46.3, Python 3.11, offline models/datasets, seed 20260713, DeepSeek-Coder-1.3B
Instruct, BF16 generation, max 256 new tokens, temperature 0.25, top-p 0.95,
top-k 0, retry repetition penalty 4.0, retry 20, and seed stride 101. The result is
a single-seed within-benchmark mechanism study. Failure of any gate is reported as
a complete negative result; it is not hidden by switching mechanisms.
