#!/usr/bin/env python
"""Verify HumanEval dataset loads correctly in offline mode from data/datasets/humaneval/.

Run from project root:
    conda run -n WFCLLM python tools/verify_humaneval.py
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["HF_DATASETS_OFFLINE"] = "1"

PROJECT_ROOT = Path(__file__).parent.parent
cache_dir = PROJECT_ROOT / "data" / "datasets" / "humaneval"

from datasets import load_dataset

ds = load_dataset("openai/openai_humaneval", cache_dir=str(cache_dir))
splits = {k: len(v) for k, v in ds.items()}
print(f"Offline load OK: {splits}")
