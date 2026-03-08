#!/usr/bin/env python
"""End-to-end offline resource verification."""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent

from transformers import AutoTokenizer, T5EncoderModel
from datasets import load_dataset

# 1. CodeT5 model + tokenizer
tok = AutoTokenizer.from_pretrained(str(PROJECT_ROOT / "data/models/codet5-base"))
print(f"[OK] CodeT5 tokenizer: vocab_size={tok.vocab_size}")
model = T5EncoderModel.from_pretrained(str(PROJECT_ROOT / "data/models/codet5-base"))
print(f"[OK] CodeT5 model: {sum(p.numel() for p in model.parameters())//1_000_000}M params")

# 2. MBPP dataset
ds_mbpp = load_dataset("google-research-datasets/mbpp", "full", cache_dir=str(PROJECT_ROOT / "data/datasets/mbpp"))
print(f"[OK] MBPP: { {k: len(v) for k,v in ds_mbpp.items()} }")

# 3. HumanEval dataset
ds_he = load_dataset("openai/openai_humaneval", cache_dir=str(PROJECT_ROOT / "data/datasets/humaneval"))
print(f"[OK] HumanEval: { {k: len(v) for k,v in ds_he.items()} }")

print("\nALL RESOURCES OK - offline deployment ready")
