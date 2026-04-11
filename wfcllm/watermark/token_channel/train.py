"""Offline entrypoint skeleton for token-channel training assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.watermark.token_channel.teacher import load_teacher_cache
from wfcllm.watermark.token_channel.train_corpus import load_training_cache


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal Task 4 training CLI parser."""

    parser = argparse.ArgumentParser(description="Offline token-channel training loader")
    parser.add_argument("--corpus-cache", type=Path, help="Path to a saved training corpus cache")
    parser.add_argument("--teacher-cache", type=Path, help="Path to a saved teacher cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load cached offline assets without running training loops yet."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.corpus_cache is None and args.teacher_cache is None:
        parser.print_help()
        return 1

    if args.corpus_cache is not None:
        rows = load_training_cache(args.corpus_cache)
        print(f"Loaded {len(rows)} training rows from {args.corpus_cache}")

    if args.teacher_cache is not None:
        rows = load_teacher_cache(args.teacher_cache)
        print(f"Loaded {len(rows)} teacher rows from {args.teacher_cache}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
