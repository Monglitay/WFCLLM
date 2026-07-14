# Watermark Mechanism V3 Final Audit

## Closure

The branch is complete as a preregistered negative result. The pilot failed only the exact R3 replay gate (27/30), after the one allowed repair. Held-out negatives, robustness attacks, and HumanEval-164 were not run.

## Reproducibility ledger

- V2 base: `eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`
- Branch: `codex/watermark-mechanism-v3-dynamic-semantic-reset`
- Preregistration commit: `4f3d9eb`
- Implementation commit: `5f8b908`
- Pilot results commit: `922a12b715ef35e50627ef1f0a1f9a28124d7531`
- Seed: `20260713`; single-seed scope
- Raw candidate equality: 600/600 hashes
- Public/raw artifacts: hash-bound in `PILOT_AUDIT_V3.json` and `ARTIFACT_HASHES_V3.json`
- Private keys: ignored, mode 0600, never printed, no public fingerprints

## Verification

- Full offline test suite: **819 passed, 27 warnings in 30.18 s**.
- Syntax smoke: compileall exit 0 for `wfcllm`, `run.py`, `scripts`, and `tools`.
- Staged diff check: pass before the final commit.
- Secret scan: zero raw-key occurrences in Git-tracked files and public V3 artifacts.
- Access audit: calibration opened; held-out detection not run; HumanEval-164 not run.

## Audit disposition

Experiment integrity is `WARN/provisional`, not because evidence was hidden, but because the same primary agent performed design, execution, and audit and system policy prevented independent subagent review. The raw evidence supports a high-confidence negative verdict for the frozen implementation. Broader interpretation remains `[pending external review]`.

See `RESULTS_V3.md`, `FAILURE_POSTMORTEM_V3.md`, `R3_EXACT_REPLAY_AUDIT.md`, `EXPERIMENT_AUDIT.md`, and their machine-readable companions.
