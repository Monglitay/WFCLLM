# Research Findings

## 2026-07-13: Watermark mechanism V2, retry=20

### Outcome

Target claim rejected. The full run is post-gate exploratory because the
preregistered pilot failed before the user authorized one direct 164-task run.

### What worked

- final-code, selected-code, and selected-score replay were 164/164;
- generation objective and final detector score were effectively identical;
- sufficient-evidence coverage increased from 135/164 to 163/164;
- AUROC increased from 0.6226 to 0.7313;
- V2 met the absolute Pass floor exactly at 82/164;
- exact secret scans returned zero matches.

### What failed

- V2 held-out FPR was 3/41 = 7.32%, above 5%;
- V2 frozen-threshold TPR was 38/164, below Current's 42/164;
- the paired TPR delta was negative and its 95% bootstrap CI crossed zero;
- pilot entry, held-out FPR, TPR-superiority, and paired-evidence gates failed.

### Updated mechanism diagnosis

Objective mismatch and final-format evidence loss were engineering defects, but
not the final bottleneck after V2 fixed them. The remaining failure is calibrated
upper-tail separation: V2 moves positive central scores and improves AUROC,
while ordinary-negative extremes remain too high. Increasing nominal evidence
density alone does not guarantee independent or discriminative evidence.

### Constraints for future work

- do not tune a detector or threshold on the now-open held-out panel;
- do not continue retry or margin sweeps on this preregistration;
- do not claim legal TPR@5%FPR when observed held-out FPR exceeds 5%;
- any future candidate needs new independent calibration and held-out splits;
- keep V2 experimental and preserve all negative artifacts.

## 2026-07-14: Watermark mechanism V3, dynamic semantic retry=20

### Outcome

Target claim rejected for the frozen implementation. The repaired pilot achieved only 27/30 exact generation-to-R3 replay, below the absolute 30/30 gate. Held-out negatives and HumanEval-164 were intentionally not opened/run.

### What worked

- exact shared-pool fairness: 600/600 ordered candidate hashes;
- mean semantic selection/replay cost fell 49.79% relative to the complete-final profile baseline;
- Current and V3 both passed 15/30 tasks;
- V3 primary-key scores shifted upward on 19 task pairs, with no analogous wrong-key shift;
- no tracked/public raw-key leak was found.

### What failed

- float32 neural representations remained batch-composition dependent at quantization boundaries;
- three selected programs had one quantized-embedding mismatch under final-only replay;
- the single repair budget was exhausted, so all downstream confirmatory gates were closed.

### Constraints for future work

- do not reuse this V3 held-out split for a mechanism designed from this failure;
- do not equate cosine closeness with exact discrete-evidence replay;
- remove batch-neighborhood dependence before measuring TPR/FPR or robustness;
- preserve final-only detection and identical raw candidate pools;
- treat any further representation/quantization change as a new preregistered mechanism.
