# Watermark V4 RTX 5090 Profile

## Outcome

The frozen `CanonicalStructuralReplay-V4-R20` mechanism passes the pre-pilot
performance gate. Across three independent cold Python processes and eight debug
tasks per process, the formal metric—dynamic selection over the same ordered 20
candidates plus exactly one selected-final replay—averaged **0.032691975 seconds
per task**. The gate is 0.1409884 seconds per task.

Relative to the preregistered complete-final neural baseline of 0.201412 seconds
per task, this is an **83.7686% reduction**, exceeding the required 30% reduction.
The formal V4 path is CPU-only and allocated **0 MiB CUDA memory** on the server's
NVIDIA GeForce RTX 5090, below the 32607 MiB ceiling.

## Timing decomposition

| Component | Mean seconds |
|---|---:|
| Initialization, config/calibration load, and retry-ledger load | 0.071993347 |
| Warm-up | 0.000561531 |
| Dynamic V4 selection per task | 0.030996002 |
| Exactly one selected-final R3 replay per task | 0.001695973 |
| Formal semantic total per task | **0.032691975** |

Initialization and warm-up are separately reported and are not charged to the
per-task semantic metric, matching the preregistration. Candidate generation is
also excluded and separately classified because Stage E replays the frozen raw
retry-20 pools. This is a saved-pool runtime profile, not an end-to-end live model
generation latency claim.

## Cold-process results

| Process | Mean semantic seconds/task | Exact identity |
|---|---:|---|
| cold0 | 0.031208074 | `a6f89db3…c560f` |
| cold1 | 0.032787075 | `a6f89db3…c560f` |
| cold2 | 0.034080776 | `a6f89db3…c560f` |

All processes used the offline environment and the same frozen public config,
calibration artifact, private key file, and two preserved retry-ledger files.
Every process recorded eight selected-final replays and zero EOS all-candidate
neural semantic rescoring calls.

## Gate decision

- Mean semantic cost: PASS.
- Reduction versus 0.201412: PASS.
- Peak allocated VRAM: PASS.
- Retry=20 and all required contexts evaluated: PASS.
- Exactly one selected-final replay: PASS.
- EOS complete-candidate neural rescore count: PASS (zero).

The raw ignored artifacts are under
`data/experiments/watermark_v4_batch_invariant_20260714/debug/`; their hashes are
frozen in `GPU_PROFILE_V4.json`.
