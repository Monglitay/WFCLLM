#!/usr/bin/env python
"""CLI tool to calibrate FPR-based watermark detection threshold.

Usage:
    python scripts/calibrate.py \\
        --input data/negative_corpus.jsonl \\
        --output threshold.json \\
        --fpr 0.01 \\
        --secret-key <key> \\
        --model data/models/codet5-base \\
        [--device cpu]

Output threshold.json example:
    {"fpr": 0.01, "fpr_threshold": 2.87, "n_samples": 500}

Set fpr_threshold as ExtractConfig.fpr_threshold before deployment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calibrate FPR-based watermark detection threshold."
    )
    parser.add_argument("--input", required=True, help="Path to negative corpus JSONL")
    parser.add_argument("--output", required=True, help="Path to write threshold JSON")
    parser.add_argument(
        "--fpr", type=float, default=0.01,
        help="Target false positive rate (default: 0.01)",
    )
    parser.add_argument("--secret-key", required=True, help="Watermark secret key")
    parser.add_argument(
        "--model", required=True,
        help="Path to encoder model (e.g. data/models/codet5-base)",
    )
    parser.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    parser.add_argument(
        "--lsh-d", type=int, default=3, help="LSH projection count (default: 3)"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.5,
        help="LSH valid-region fraction gamma (default: 0.5)",
    )
    return parser.parse_args()


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def main() -> None:
    args = _parse_args()

    # Lazy imports to keep startup fast when --help is used
    import torch
    from transformers import AutoModel, AutoTokenizer

    from wfcllm.extract.calibrator import ThresholdCalibrator
    from wfcllm.extract.scorer import BlockScorer
    from wfcllm.watermark.keying import WatermarkKeying
    from wfcllm.watermark.lsh_space import LSHSpace
    from wfcllm.watermark.verifier import ProjectionVerifier

    print(f"Loading model from {args.model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    encoder = AutoModel.from_pretrained(args.model).to(args.device)
    encoder.eval()

    lsh_space = LSHSpace(args.secret_key, args.embed_dim, args.lsh_d)
    keying = WatermarkKeying(args.secret_key, args.lsh_d, args.gamma)
    verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=args.device)
    scorer = BlockScorer(keying, verifier)

    print(f"Loading corpus from {args.input} ...", file=sys.stderr)
    corpus = _load_jsonl(args.input)
    print(f"  {len(corpus)} samples loaded.", file=sys.stderr)

    calibrator = ThresholdCalibrator(scorer, gamma=args.gamma)
    result = calibrator.calibrate(corpus, fpr=args.fpr)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Calibration complete:\n"
        f"  FPR target    : {result['fpr']}\n"
        f"  M_r threshold : {result['fpr_threshold']:.4f}\n"
        f"  Samples used  : {result['n_samples']}\n"
        f"  Output        : {args.output}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
