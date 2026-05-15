#!/usr/bin/env python
"""Calibrate FPR-based watermark detection threshold from a negative corpus."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Calibrate FPR-based watermark detection threshold.",
    )
    parser.add_argument("--input", required=True, help="negative corpus JSONL")
    parser.add_argument("--output", required=True, help="threshold result JSON")
    parser.add_argument("--secret-key", required=True)
    parser.add_argument("--model", required=True, help="encoder model path")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fpr", type=float, default=0.01)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lsh-d", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    calibrate_threshold_from_corpus(
        input=args.input,
        output=args.output,
        secret_key=args.secret_key,
        model=args.model,
        device=args.device,
        fpr=args.fpr,
        embed_dim=args.embed_dim,
        lsh_d=args.lsh_d,
        gamma=args.gamma,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
