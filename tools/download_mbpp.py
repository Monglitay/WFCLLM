#!/usr/bin/env python
"""Download google-research-datasets/mbpp to data/datasets/mbpp/.

Run from project root:
    conda run -n WFCLLM python tools/download_mbpp.py
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
cache_dir = PROJECT_ROOT / "data" / "datasets" / "mbpp"
cache_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading MBPP to {cache_dir} ...")
from datasets import load_dataset

ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir=str(cache_dir))
splits = {k: len(v) for k, v in ds.items()}
print(f"MBPP splits: {splits}")
assert len(ds["train"]) == 374, f"Expected 374 train samples, got {len(ds['train'])}"
print("PASS: MBPP downloaded to data/datasets/mbpp/")
