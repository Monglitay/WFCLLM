# Watermark Mechanism V4 Method Selection

Date: 2026-07-14

Status: Candidate C selected after Stage B probes; formal parameter values are
not authoritative until the Stage C preregistration commit.

## Selected family

V4 uses a canonical discrete structural/data-flow representation for signed
evidence. The neural encoder is not part of the signed representation, unit
eligibility, selection score, cache key, or R3 detector. This is a deliberate
scope change from V3: V4 claims batch-invariant structural-semantic replay, not
batch-invariant neural embedding replay.

## Final-code-only reconstruction

The detector parses only `final_code`. For every top-level function it visits
each statement-list field recursively. A compound statement contributes a
header shell in which child statement lists are replaced by explicit `pass`
markers; child statements contribute separate units. This prevents a parent
unit from copying the complete child subtree into a second nominally
independent unit.

Each unit contains:

- statement role: statement type, parent type, and statement-list field;
- canonical previous sibling within the same statement-list field, or `<BOS>`;
- canonical current statement or compound header shell;
- identifier-normalized AST dumps without source locations, comments, or
  formatting;
- bounded definition/use relations between the previous and current unit.

Identifier slots are allocated deterministically within a unit context.
Builtins have explicit stable names. Global ordinals, prompt text, task ID,
attempt index, candidate metadata, and generation ledgers are absent.

Parse failure, oversize unit/context, and duplicate unit ID are explicit
erasures. Source is never truncated. The initial probe limits are 4,096 UTF-8
bytes per unit and 8,192 UTF-8 bytes per local context; Stage C must either
freeze these values or reject the candidate before pilot.

## Discrete evidence

The canonical JSON record is sorted and versioned. Its SHA-256 digest is the
public representation byte vector. The unit ID is a separate SHA-256 over the
schema, role, and representation digest.

The formal V4 keying header and domains must be new and distinct from V3 and
from the Stage B diagnostic domains. Projection/signature bits and target bits
use separate HMAC domains. The representation bytes are already discrete:
there is no float quantization, numerical margin, neural whitening, or
batch-shaped operation. The erasure mask is therefore structural and exact.

The Stage B probe used 32 signature/target bits per unit and minimum independent
units three. A unit numerator is `2 * matches - denominator`; aggregate
numerator and denominator are exact integer sums. Stage C freezes the formal
bit count, calibration statistic, empirical p-value rule, and decision
threshold before any pilot.

## Generation and replay contract

Raw generation produces the same ordered retry=20 candidates as Current. The
private key cannot reach the model, tokenizer sampling path, seed, logits,
stopping rule, static validity, or candidate generation cache.

As each completed candidate becomes available, V4 reconstructs and scores its
public structural units. It does not run a neural encoder and does not wait at
EOS to neural-rescore all complete candidates. Selection may use the private
score only after the raw candidate is complete. The selected final code is
replayed exactly once through the same final-code-only structural detector.

R3 accepts only:

1. `final_code`;
2. the frozen public V4 config; and
3. a 0600 private key file.

Every exact comparison includes sorted unit IDs, canonical context hashes,
representation bytes, identity-quantized values, erasure masks, signature
bits, target bits, unit match/numerator/denominator fields, aggregate fields,
eligibility, p-value, and final decision. Cosine or tolerance comparison is
forbidden.

## Cache and scheduling

The public cache key contains only the formal schema/config identity and the
canonical public context hash. It excludes all key material, task/candidate
identity, batch metadata, and generation state. Because the representation is
pure CPU canonicalization plus cryptographic hashing, caller batch size,
composition, order, padding, CUDA kernels, TF32, warm-up, and process cache
state cannot change it.

## Evidence supporting selection

- Candidate A: exact, but 0.310876 seconds/task and therefore failed cost.
- Candidate B: 18 erasure-mask mismatch rows under the frozen empirical bound.
- Candidate C: all raw candidate hashes preserved; 79/100 diagnostic candidates
  and four of five tasks eligible at minimum units three; three of five task
  pools had a positive capacity proxy; largest positive-delta share 45.11%;
  three cold-process summaries were identical.
- Candidate D: no benefit over C and retained unnecessary neural assumptions.

These are engineering probes, not TPR/FPR evidence. Calibration, confirmatory
pilot, held-out negatives, HumanEval-164, and robustness remain unopened or
unrun at method-selection time.

## Claim boundary

V4 may support a claim about deterministic canonical program structure and
bounded local data-flow evidence. It must not claim neural semantic
generalization, invariance to arbitrary refactoring, or robustness to any
attack not explicitly tested. Formatting/comment and scoped identifier rename
invariance are design properties with tests; insertion, deletion, and reorder
are expected to change some local evidence and require later robustness
measurement.
