"""Compute ZCA whitening matrix from encoder embeddings on a code corpus.

Usage:
    python scripts/build_whitening.py \
        --corpus data/negative_corpus.jsonl \
        --output data/calibration/whitening.npz \
        --limit 164

Output .npz contains:
    mu  : (embed_dim,) float32  — mean of embeddings
    W   : (embed_dim, embed_dim) float32  — ZCA whitening matrix
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

# ── project root on path ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.lang.python.parser import extract_statement_blocks


def load_encoder(model_dir: str, checkpoint: str, device: str) -> tuple:
    enc_config = EncoderConfig(embed_dim=128)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists():
        enc_config.model_name = str(local_codet5)
    encoder = SemanticEncoder(config=enc_config)
    ckpt = torch.load(checkpoint, map_location="cpu")
    encoder.load_state_dict(ckpt["model_state_dict"])
    encoder = encoder.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)
    return encoder, tokenizer


@torch.no_grad()
def collect_embeddings(
    corpus_path: str,
    encoder,
    tokenizer,
    device: str,
    limit: int | None,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    with open(corpus_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            rec = json.loads(line)
            code = rec.get("generated_code", "")
            if not code.strip():
                continue
            blocks = [b for b in extract_statement_blocks(code) if b.block_type == "simple"]
            for block in blocks:
                inputs = tokenizer(
                    block.source,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                )
                u = encoder(
                    inputs["input_ids"].to(device),
                    inputs["attention_mask"].to(device),
                ).squeeze(0).cpu().numpy()
                embeddings.append(u)
    print(f"Collected {len(embeddings)} block embeddings from {i+1} samples")
    return np.stack(embeddings, axis=0)  # (N, embed_dim)


def compute_zca(embeddings: np.ndarray, eps: float = 1e-5) -> tuple[np.ndarray, np.ndarray]:
    """Return (mu, W) where W is the ZCA whitening matrix."""
    mu = embeddings.mean(axis=0)
    centered = embeddings - mu
    cov = (centered.T @ centered) / len(centered)
    U, S, _ = np.linalg.svd(cov)
    W = U @ np.diag(1.0 / np.sqrt(S + eps)) @ U.T
    return mu.astype(np.float32), W.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", default="data/negative_corpus.jsonl")
    parser.add_argument("--output", default="data/calibration/whitening.npz")
    parser.add_argument("--checkpoint", default="data/models/encoder/best_model.pt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--eps", type=float, default=1e-5)
    args = parser.parse_args()

    print(f"Loading encoder from {args.checkpoint} on {args.device}")
    encoder, tokenizer = load_encoder("data/models", args.checkpoint, args.device)

    print(f"Collecting embeddings from {args.corpus} (limit={args.limit})")
    embeddings = collect_embeddings(args.corpus, encoder, tokenizer, args.device, args.limit)

    print("Computing ZCA whitening matrix...")
    mu, W = compute_zca(embeddings, eps=args.eps)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, mu=mu, W=W)
    print(f"Saved whitening matrix to {args.output}  (shape W={W.shape})")


if __name__ == "__main__":
    main()
