# V3 RTX 5090 Profile

## Result

The optimized dynamic observer met the preregistered cost and memory gates, but performance success does not override the R3 exactness failure.

| Item | Value |
|---|---:|
| GPU | NVIDIA GeForce RTX 5090 |
| Reported GPU memory | 32,607 MiB |
| PyTorch | 2.11.0+cu130 |
| Transformers | 4.46.3 |
| Python | 3.11.15 |
| Pilot tasks / retry | 30 / 20 |
| Dynamic semantic mean | 0.101131791 s/task |
| Dynamic semantic median | 0.096212409 s/task |
| Mean bootstrap 95% interval | [0.083478103, 0.120440904] s/task |
| Complete-final profile baseline | 0.201412 s/task |
| Measured reduction | 49.79% |
| Required reduction | 30% |
| Mean encoder calls | 4.733/task |
| Encoded contexts | 745 total |
| Peak allocated CUDA memory | 687.954 MiB |

The timed region contains dynamic semantic selection over the fixed 20-candidate pool plus one selected-final replay. Model loading and warm-up are excluded consistently. CUDA synchronization brackets timing. Candidate generation itself is excluded because the raw pool is deliberately shared and replayed.

The repair used fixed sequence length 256 with dynamic batch size to reduce padding-dependent representation drift while preserving batching. Fixed batch 32 achieved debug exactness but exceeded the cost gate; it was not adopted as a new mechanism. The final setting recovered cost but remained non-exact on 3/30 tasks.
