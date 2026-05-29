"""Validate ZCA whitening improves LSH region uniformity.

Loads encoder + whitening matrix, runs on N samples from corpus,
prints per-region hit counts before and after whitening.

Usage:
    python scripts/validate_whitening.py \
        --corpus data/negative_corpus.jsonl \
        --whitening data/calibration/whitening.npz \
        --limit 20
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.lang.python.parser import extract_statement_blocks
from wfcllm.watermark.lsh_space import LSHSpace


def load_encoder(checkpoint: str, device: str):
    enc_config = EncoderConfig(embed_dim=128)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists():
        enc_config.model_name = str(local_codet5)
    encoder = SemanticEncoder(config=enc_config)
    ckpt = torch.load(checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["model_state_dict"])
    return encoder.to(device).eval(), AutoTokenizer.from_pretrained(enc_config.model_name)


@torch.no_grad()
def collect_embeddings(corpus, encoder, tokenizer, device, limit):
    embeddings = []
    with open(corpus, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            code = rec.get("generated_code", "")
            for block in extract_statement_blocks(code):
                if block.block_type != "simple":
                    continue
                inputs = tokenizer(block.source, return_tensors="pt",
                                   padding=True, truncation=True, max_length=256)
                u = encoder(inputs["input_ids"].to(device),
                            inputs["attention_mask"].to(device)).squeeze(0).cpu()
                embeddings.append(u)
    return embeddings


def region_counts(embeddings, lsh: LSHSpace) -> Counter:
    return Counter(lsh.sign(u) for u in embeddings)


def entropy_bits(counts: Counter, n: int) -> float:
    import math
    h = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h -= p * math.log2(p)
    return h


def print_distribution(label: str, counts: Counter, n: int, d: int) -> None:
    max_regions = 2 ** d
    print(f"\n{'='*60}")
    print(f"{label}  (n={n} blocks, {max_regions} possible regions)")
    print(f"{'='*60}")
    occupied = len(counts)
    entropy = entropy_bits(counts, n)
    max_entropy = d  # log2(2^d)
    print(f"  Occupied regions : {occupied}/{max_regions}  ({100*occupied/max_regions:.1f}%)")
    print(f"  Entropy          : {entropy:.3f} / {max_entropy:.3f} bits  ({100*entropy/max_entropy:.1f}% of max)")
    top = counts.most_common(5)
    print(f"  Top-5 regions    : {[(str(k), v) for k, v in top]}")
    bottom = counts.most_common()[:-6:-1]
    print(f"  Bottom-5 regions : {[(str(k), v) for k, v in bottom]}")
    # Chi-squared uniformity test
    expected = n / max_regions
    chi2 = sum((c - expected) ** 2 / expected for c in counts.values())
    # Add zero-count regions
    chi2 += (max_regions - occupied) * expected
    print(f"  Chi² (uniform)   : {chi2:.1f}  (lower = more uniform)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/negative_corpus.jsonl")
    parser.add_argument("--whitening", default="data/calibration/whitening.npz")
    parser.add_argument("--checkpoint", default="data/models/encoder/best_model.pt")
    parser.add_argument("--secret-key", default="1010")
    parser.add_argument("--lsh-d", type=int, default=4)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Loading encoder from {args.checkpoint}")
    encoder, tokenizer = load_encoder(args.checkpoint, args.device)

    print(f"Collecting embeddings (limit={args.limit} samples)...")
    embeddings = collect_embeddings(args.corpus, encoder, tokenizer, args.device, args.limit)
    print(f"Got {len(embeddings)} block embeddings")

    lsh_plain = LSHSpace(args.secret_key, 128, args.lsh_d)
    lsh_white = LSHSpace(args.secret_key, 128, args.lsh_d,
                         whitening_path=args.whitening if Path(args.whitening).exists() else None)

    counts_plain = region_counts(embeddings, lsh_plain)
    counts_white = region_counts(embeddings, lsh_white)

    n = len(embeddings)
    print_distribution("WITHOUT whitening", counts_plain, n, args.lsh_d)
    if Path(args.whitening).exists():
        print_distribution("WITH whitening", counts_white, n, args.lsh_d)
    else:
        print(f"\n[!] Whitening file not found: {args.whitening}")
        print("    Run scripts/build_whitening.py first.")


if __name__ == "__main__":
    main()
