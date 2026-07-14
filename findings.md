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
