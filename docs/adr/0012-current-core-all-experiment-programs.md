---
status: accepted
---

# Current Core Programs for All Server Experiments

The two server worktrees contain both current Full Reproduction work and older
experiment-local scripts for parameter sensitivity, proposal budget, gamma and
dimension checks, deployment-key confirmation, rename robustness, DIPPER
robustness, and Pass@10.  The local repository remains the authority for the
method implementation.  Server code is reference material only.

The five public Full Reproduction Profiles are unchanged.  The canonical method
name remains `gated_semantic_window_v1`, and the canonical chain remains:

`encoder -> gate-data -> gate-train -> generate -> calibrate -> detect -> report -> audit`

Experiments that are already covered by the formal Supplementary Ablation Spec
continue to use `--ablation-spec` and fresh current full runs.  Paper-covered or
posthoc experiments are implemented as supplementary experiment programs.  They
consume current artifacts or prepare strict current detector inputs, but they do
not create new methods, profiles, legacy restore paths, old Bundle readers, or
detector side channels.

The local program surface is:

- `scripts/wfcllm_supplementary_experiments.py plan`
- `scripts/wfcllm_supplementary_experiments.py write-specs`
- `scripts/wfcllm_supplementary_experiments.py prepare-rename`
- `scripts/wfcllm_supplementary_experiments.py aggregate-detection`
- `scripts/wfcllm_supplementary_experiments.py aggregate-passk`

`prepare-rename` writes attacked JSONL rows with exactly `id`, `dataset`,
`prompt`, and `final_code`, so re-detection still uses the strict code-only
detector input contract.  `aggregate-detection` reads detector details after
the current detector has run on each condition.  `aggregate-passk` is strictly
posthoc and marks the result as unused by generation, retry, selection,
calibration, or detection.

No server keys, private resources, model weights, datasets, checkpoints, run
trees, logs, or historical artifacts are imported into the repository.  Remote
paths may appear only in the generated experiment plan as provenance for the
reference scripts that were inspected.
