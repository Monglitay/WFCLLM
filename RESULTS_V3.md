# Watermark Mechanism V3 Results

## Final disposition

**Negative result.** Streaming-SharedPool-V3-R20 does not satisfy the exact generation/detection reconstruction contract. The frozen 30-task retry-20 pilot reached exact R3 replay on 27/30 tasks (90.0%) after the only permitted repair; 100% was required. The experiment therefore stopped before held-out evaluation and HumanEval-164, exactly as preregistered.

The tested idea is not “disproven in all forms.” The supported conclusion is narrower: **this float32 CodeT5 + whitening + integer-quantization dynamic-batching realization is not a valid bit-exact final-code-only watermark mechanism under the frozen settings.**

## Evidence summary

| Dimension | Result | Interpretation |
|---|---:|---|
| Shared raw pool | 600/600 candidate hashes | exact fairness control passed |
| R3 exact replay | 27/30 | hard failure; blocks escalation |
| Current R3 TPR | 3.33% | 1/30 |
| V3 R3 TPR | 23.33% | 7/30; diagnostic only |
| Mean paired score shift | +0.177029 | bootstrap 95% CI [0.094172, 0.258662] |
| HumanEval passes | 15 → 15 | no aggregate change; 4 gains and 4 losses |
| Semantic cost | 0.101132 s/task | 49.79% below complete-final profile baseline |
| Peak CUDA allocation | 687.954 MiB | within 32,607 MiB gate |
| Calibration primary positives | 11/250 (4.4%) | calibration consistency only, not held-out FPR |
| Calibration wrong-key positives | 1/250 (0.4%) | no positive wrong-key shift |

## What is supported

- Incremental closure-event batching can reduce semantic-encoder selection cost by about half relative to the measured all-complete-candidate semantic profile while consuming the identical retry pool.
- V3 selected outputs showed a positive within-pilot score shift under the primary key; the shift was distributed across 19 positive, 4 negative, and 7 tied task pairs.
- Candidate-pool fairness, key isolation from raw generation, minimum correctness, cost, memory, and calibration-set consistency controls passed.
- The negative conclusion is high confidence for the exact frozen implementation because a single exact-replay mismatch is sufficient to violate the formal detector contract, and three were observed after the repair budget was exhausted.

## What is not supported

- No legal TPR@5% held-out-FPR claim: the held-out split was never opened.
- No full HumanEval-164 improvement claim: the full run was not permitted or executed.
- No robustness claim: attacks were not run after the base exactness gate failed.
- No general semantic-watermark claim across seeds, models, encoders, languages, datasets, or natural code.
- No live-generation latency claim: the formal study replays frozen raw trajectories through the read-only observer.
- No independent-review claim: final integrity and result-to-claim review used a transparent primary-agent fallback and remains pending external review.

## Stop decision

The one-repair budget is exhausted. Continuing with a new precision scheme, deterministic kernel, quantization dead zone, per-context batch shape, integerized encoder, or canonical non-neural signature would be a new mechanism and requires a new split and preregistration. This branch is closed as a complete negative result without outcome-driven tuning.
