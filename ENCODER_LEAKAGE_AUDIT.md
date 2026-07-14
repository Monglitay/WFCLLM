# Semantic Encoder Leakage Audit

The frozen semantic encoder checkpoint predates this V3 experiment and is not trained or tuned during the pilot. Its provenance nevertheless includes project training material derived from HumanEval/MBPP-style benchmark data. That creates benchmark leakage risk for any general semantic or out-of-distribution claim.

Consequences:

- This experiment may only be described as a within-benchmark mechanism validation.
- It does not establish cross-dataset, cross-model, cross-language, natural-code, or unseen-code generalization.
- Detection uplift cannot be attributed solely to universally meaningful program semantics.
- The fresh MBPP split prevents direct threshold peeking across calibration/held-out halves, but it does not erase encoder pretraining/fine-tuning provenance.
- The held-out half was never opened, so there is no hidden confirmatory result to report.

The checkpoint path and SHA-256 were frozen before the pilot. The encoder is loaded in inference mode, trainable parameters are disabled, and no pilot label enters fitting. Whitening uses only the frozen calibration negatives. These controls limit experiment-time leakage while leaving the historical checkpoint-provenance limitation explicit.
