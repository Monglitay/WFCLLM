# WFCLLM Mechanism Diagnosis Before V2

**Frozen diagnosis date:** 2026-07-13
**Baseline commit:** `25b3c6c001809f80c5314020e4048a7e68960963`
**Historical evidence:** full HumanEval runs under `data/experiments/` on the new server. These run artifacts are local, untracked evidence and are not detector inputs.

## Scope and invariants

The target remains a generation-time, key-conditioned code watermark. The official detector must receive rows containing exactly `id`, `dataset`, `prompt`, and `final_code`, and the scoring path must use only `final_code`, public detector configuration, and the secret key. Audit events, candidate lists, retry decisions, labels, task IDs, and generation-time scores are diagnostic-only.

The new-server tracked tree exactly matches the locally retained tree for baseline commit `25b3c6c` (`e6ec32855ea8f39636900ff32a60c15ab3cfa53d`). The old GPU endpoints were unavailable, so untracked asset equality could not be checked online. The new server has a complete DeepSeek-Coder-1.3B model, CodeT5-base, the fine-tuned encoder checkpoint, ten epoch checkpoints, and local HumanEval/MBPP caches. The 7B model directory is incomplete and is excluded.

## Reconfirmed historical baseline

The existing full HumanEval runs are not the new retry-20 comparison; they are diagnosis inputs only.

| Run | Pass@1 | TPR@5%FPR | held-out FPR | AUROC | no accepted evidence | detector-insufficient |
|---|---:|---:|---:|---:|---:|---:|
| finetuned LoRA baseline | 91/164 (55.49%) | 6.71% | 3.66% | 0.6287 | 21/164 | 52/164 |
| no4-long | 93/164 (56.71%) | 7.32% | 3.66% | 0.6015 | 26/164 | 52/164 |

For no4-long, positive score quantiles were min/25/50/75/95/max = `0/0.1761/0.3224/0.4949/0.9208/1.3979`; negative quantiles were `0/0.0611/0.2240/0.4260/0.9199/1.3979`. The positive median shifts upward, but the upper tails are nearly identical, which leaves very few positives beyond the legal threshold.

## Module-by-module mechanism analysis

### 1. Modules that affect Pass@1

- `wfcllm.generation.generator.SawrModelContext` controls sampling, stop handling, retry penalties, and rollback. It can change semantics whenever a failed keyed candidate is rolled back.
- `wfcllm.generation.state_machine.SawrStateMachine` decides which statements/windows are retried and when a fallback is committed. Larger retry budgets increase the number of semantic deviations from the first model completion.
- `wfcllm.generation.retry.evidence_retry_key` selects the full completion with the largest accepted-hit count, without a syntax, target-signature, placeholder-body, or truncation gate.
- Prompt mode, stop sequences, model precision, maximum tokens, and prompt stripping affect truncation and malformed code.

Historical evidence shows no simple positive watermark/quality alignment: passed and failed no4-long samples had nearly equal mean detector scores (`0.382` versus `0.376`). Failed samples were longer on average (`773.5` versus `630.5` characters). Accepted-hit count correlates with final-code length (`r=0.420`), while length barely correlates with detector score (`r=0.033`). Thus the current selector has a concrete route to prefer longer candidates without obtaining stronger final evidence.

### 2. Modules that affect evidence density

- Boundary extraction determines the number and type of scoreable statement and compound candidates.
- The state machine's retry budgets determine how often a miss is resampled and whether a layer closes without a hit.
- `max_group_statements` creates overlapping one- and two-statement windows.
- The LSH target rate and margin affect acceptance probability.
- The full-completion retry selector rewards accepted-event count, including repeated candidate hashes.

Historical accepted evidence is sparse: median accepted count is one to two per sample, with 21--26 samples having none. Evidence is also redundant. The mean per-sample unique-hash/accepted-event ratio is `0.7725`; across rule-decision events the repeated-candidate rate is `0.4398`. Two-statement windows comprise 12.44% of decisions and overlap single-statement windows.

### 3. Modules that affect correct-key score shift

- The encoder checkpoint, tokenizer, precision, normalization, LSH projection, and key derivation determine per-window signatures.
- Generation currently optimizes accepted event count, not the final detector statistic.
- Detection extracts proxy-local ordinals and overlapping AST windows, maps each context maximum through a conditional empirical null, then averages context scores.

The accepted-hit objective is weakly aligned with the final score: accepted count versus final score has Pearson `r=0.196`; recovered accepted-hash count versus final score has `r=0.315`. Ninety-three no4-long samples had accepted, hash-matched, sufficient evidence but remained below threshold. Their median score was `0.3010`, compared with `0.7796` for true positives.

### 4. Modules that affect negative FPR

- `WFCLLMDetectionConfig` fixes LSH dimension, effective gamma, margin, evidence mode, structure buckets, statistic, and evidence sufficiency.
- Calibration buckets and fallbacks determine the null population for each context.
- Overlapping proxy windows and maximum-within-context aggregation create a heavy upper tail.
- Calibration threshold construction and ties determine finite-sample realized FPR.

Simple threshold lowering is invalid. A cutoff around `0.2--0.3` would yield 42.7--50.6% positive detections but 31.7--41.5% negative FPR. The bottleneck must be addressed by changing the score distribution, not relabeling the threshold.

### 5. Pass/TPR conflicts

Rollback and retry can increase accepted events while changing functional behavior. Full-completion selection can favor more statements because accepted count grows with length. Conversely, a strict quality gate may output a high-quality candidate with no watermark. The legal objective is therefore lexicographic: first meet public static quality conditions, then maximize a detector-isomorphic keyed score, and fall back to quality when no candidate passes the gate.

### 6. Is generation acceptance monotone in detector score?

No. `evidence_retry_key` is `(accepted_hit_count, -closed_without_hit_count, -fallback_count, candidate_count, -attempt_index)`. The detector uses calibrated context scores. The measured accepted-count/final-score correlation of `0.196` is direct evidence that the two objectives are not monotone.

### 7. Are accepted events recoverable?

Mostly, but accepted-event count is not the same as independent score shift. In no4-long, every sample with accepted sidecar hashes had at least one hash match in final detector proxy windows (138/138). This rules out total recovery failure as the dominant cause, while the unique-hash ratio of `0.7725` shows that nominal accepted count is inflated.

### 8. Does formatting/canonicalization destroy evidence?

Not as the dominant failure in the historical run. Accepted normalized hashes matched final-code proxy-window hashes for all 138 samples with accepted hashes. Nevertheless, V2 must make this a tested schema contract rather than rely on a diagnostic observation.

### 9. Are encoder/checkpoint/precision settings consistent?

The current generation and detection scripts load the same CodeT5 path, fine-tuned checkpoint, embedding dimension, LoRA flag, bf16 flag, and LSH construction. The checkpoint SHA-256 is `9da80dfdbb3f8e682298e588e822f15967e460c9dfd640a243aad8ed1ba51d25`. V2 will record public fingerprints and test two independent loads plus final-code replay. Historical configuration consistency is supported; runtime replay consistency still requires the Phase-A gate.

### 10. Is nominal evidence highly correlated?

Yes. Retry decisions repeat hashes, and length-1/length-2 sliding windows overlap. Nominal accepted count includes repeated hashes. This explains why count can grow without proportional detector separation and motivates a non-overlapping, content-deduplicated V2 unit contract.

### 11. Bottleneck classification

The evidence supports the following ordering:

1. **Recoverable but weak/misaligned score shift:** strongly supported by 93 sufficient, matched false negatives and `r=0.196` objective alignment.
2. **Aggregation and effective-sample problem:** strongly supported by duplicated hashes, overlapping windows, and nearly identical positive/negative upper tails.
3. **No embedding:** secondary; 21--26/164 samples have no accepted event.
4. **Embedding lost during final formatting:** not supported as the main cause.
5. **Threshold alone:** rejected because legal FPR is violated by useful lower thresholds.
6. **Pass degradation from watermark selection:** plausible mechanism-level risk; historical length and pass grouping justify a hard static quality gate, but the new R20 arms must measure it directly.

## Root causes and falsifiable hypotheses

### H1: retry selection optimizes the wrong objective

**Claim.** Current accepted-event selection is too weakly related to the final detector score and is biased toward length.
**Observable.** Per-attempt accepted count/final score correlation, selected final score, accepted/recovered count, and candidate length.
**Prediction.** V2 exact final-code-score selection has replay correlation `1.0`, raises the median and broad positive score distribution, and improves TPR over Current-R20.
**Falsifier.** No positive-score or TPR improvement on the frozen pilot, or improvement driven by one/two outliers.
**Rollback.** Reject aligned selection as the final method and retain its attempt ledger as diagnostic evidence.

### H2: overlapping/repeated windows inflate nominal evidence without independent bits

**Claim.** Repeated hashes and overlapping windows reduce effective sample size and broaden the negative tail.
**Observable.** unique-unit count, duplicate rate, unit-level bit agreement, score quantiles, AUROC, and held-out FPR.
**Prediction.** Deduplicated non-overlapping canonical units with a multi-bit keyed signature increase usable evidence per statement and improve AUROC/TPR without FPR above 5%.
**Falsifier.** V2 has no separation gain, too many zero-unit samples, or held-out FPR exceeds 5%.
**Rollback.** Keep the v2 schema/tests as experimental and restore the default method to v1.

### H3: a static quality gate prevents score-only selection from sacrificing Pass@1

**Claim.** Restricting keyed selection to syntax-valid, signature-compatible, non-placeholder target functions prevents obvious low-quality choices without using HumanEval labels.
**Observable.** gate-pass counts, syntax validity, target signature, completion length, retry index, and posthoc Pass@1.
**Prediction.** V2 pilot Pass@1 is at least 15/30 and no more than one task below Current-R20.
**Falsifier.** Either pilot Pass gate fails or full Pass@1 is below 82/164.
**Rollback.** Reject V2; never relax the pre-registered Pass gate after observing results.

## Changes implied by the evidence

V2 will replace count-based full-completion selection with a detector-isomorphic score computed from the final code, replace overlapping membership windows with deduplicated canonical statement units carrying a key-conditioned multi-bit target, and place a public static quality gate ahead of keyed ranking. It will not train a new model or use tests/correctness labels during selection.
