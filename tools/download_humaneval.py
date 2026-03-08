#!/usr/bin/env python
"""Download openai/openai_humaneval to data/datasets/humaneval/.

Run from project root:
    conda run -n WFCLLM python tools/download_humaneval.py
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
cache_dir = PROJECT_ROOT / "data" / "datasets" / "humaneval"
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading HumanEval to {cache_dir} ...")
from datasets import load_dataset

ds = load_dataset("openai/openai_humaneval", cache_dir=str(cache_dir))
splits = {k: len(v) for k, v in ds.items()}
print(f"HumanEval splits: {splits}")
assert len(ds["test"]) == 164, f"Expected 164 test samples, got {len(ds['test'])}"
print("PASS: HumanEval downloaded to data/datasets/humaneval/")
