# Watermark Mechanism V3 Frozen Hypotheses

Date: 2026-07-14
Status: preregistered before implementation outcomes

## H1 — Reconstruction, not encoder weakness, caused the proxy failure

The historical incremental proxy accumulated embeddings over mutable generation
windows that the final-code detector could not reconstruct. Therefore its accepted
event statistic will have weak agreement with final recovery even when the encoder
itself is useful. Replacing it with bounded, canonical, final-code-reconstructible
contexts will produce exact generation/replay unit identities for every eligible
selected unit. A single identity/statistic mismatch rejects H1 and blocks the pilot.

## H2 — Bounded semantic contexts yield a calibratable final-only channel

For public canonical predecessor/current-unit contexts with content-addressed unit
IDs, secret-keyed post-encoder signatures and Hamming-coded targets will yield a
right-shifted statistic on selected watermarked code while retaining held-out
negative FPR at or below 5%. Evidence is the frozen calibration rule, one-time
250-example held-out MBPP panel, wrong-key control, and positive Current-R20/V3-R20
paired comparison. Syntax-only or generation-trace evidence cannot support H2.

## H3 — Dynamic batching removes the complete-candidate semantic bottleneck

Cross-trajectory closure batching plus one selected-final replay will reduce semantic
processing wall time per task by at least 30% relative to the measured rejected route
(0.201412 s/task on the same RTX 5090, excluding model load and generation), i.e.
at most 0.1409884 s/task under a comparable debug profile. It must also avoid any
end-of-sequence loop that semantically re-encodes all 20 complete candidates. Static
inspection, call counters, CUDA memory telemetry, and wall-time profiles jointly
decide H3.

## Scope note

These hypotheses establish a single-seed, within-benchmark mechanism validation.
They do not claim cross-model, cross-language, cross-seed, or natural-code
generalization. The existing encoder checkpoint was trained with project datasets
that include HumanEval/MBPP-derived material; this is treated as disclosed mechanism
validation leakage, not as evidence of unseen-benchmark generalization.
