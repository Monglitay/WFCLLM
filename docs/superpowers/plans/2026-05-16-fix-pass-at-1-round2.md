# Fix pass@1: Round 2 — Relax Watermark Constraints

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve pass@1 from 0.207 to 0.35+ by relaxing watermark constraints that degrade code quality, while maintaining AUROC > 0.75.

**Architecture:** The watermark uses adaptive gamma scheduling to control how much of the vocabulary is in the "green list". High gamma = strong watermark but limited token choices. Lower gamma = weaker watermark but better code quality. We also need to fix the generation speed issue (mean 109 retries per sample).

**Tech Stack:** Python, JSON config

---

### Task 1: Relax adaptive gamma constraints

**Files:**
- Modify: `configs/base_config.json`

- [ ] **Step 1: Lower gamma values to give model more freedom**

In `configs/base_config.json`, change the adaptive_gamma anchors:

```json
"adaptive_gamma": {
  "enabled": true,
  "strategy": "piecewise_quantile",
  "profile_path": "data/calibration/humaneval_entropy_profile.json",
  "profile_id": "humaneval_entropy_profile",
  "gamma_min": 0.20,
  "gamma_max": 0.60,
  "anchors": {
    "p10": 0.60,
    "p50": 0.40,
    "p75": 0.45,
    "p90": 0.30,
    "p95": 0.20
  }
}
```

Rationale: Current gamma_max=0.75 means up to 75% of vocabulary is restricted.
