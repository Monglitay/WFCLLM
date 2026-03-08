#!/usr/bin/env python
"""Verify Salesforce/codet5-base loads correctly from data/models/codet5-base/.

Run from project root:
    conda run -n WFCLLM python tools/verify_codet5.py
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
local_path = PROJECT_ROOT / "data" / "models" / "codet5-base"

from transformers import AutoTokenizer, T5EncoderModel

tok = AutoTokenizer.from_pretrained(str(local_path))
print(f"Tokenizer OK: vocab_size={tok.vocab_size}")

model = T5EncoderModel.from_pretrained(str(local_path))
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model OK: {params:.1f}M params")

files = os.listdir(local_path)
assert "tokenizer_config.json" in files, "tokenizer_config.json missing!"
assert "vocab.json" in files or "tokenizer.json" in files, "vocab files missing!"
assert "config.json" in files, "config.json missing!"
print("PASS: model + tokenizer both present at data/models/codet5-base/")
