#!/usr/bin/env python
"""Download Salesforce/codet5-base model + tokenizer to data/models/codet5-base/.

Run from project root:
    conda run -n WFCLLM python tools/download_codet5.py
"""
from __future__ import annotations

import os
from pathlib import Path

# Resolve project root regardless of cwd
PROJECT_ROOT = Path(__file__).parent.parent
local_dir = PROJECT_ROOT / "data" / "models" / "codet5-base"
local_dir.mkdir(parents=True, exist_ok=True)

print(f"Downloading Salesforce/codet5-base to {local_dir} ...")
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Salesforce/codet5-base",
    local_dir=str(local_dir),
)
print(f"Downloaded to: {path}")
for f in sorted(os.listdir(local_dir)):
    fpath = local_dir / f
    if fpath.is_file():
        size = fpath.stat().st_size
        print(f"  {f}: {size / 1024 / 1024:.1f} MB")
print("DONE")
