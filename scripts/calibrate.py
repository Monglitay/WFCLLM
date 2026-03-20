#!/usr/bin/env python
"""Adaptive watermark calibration utilities.

Legacy usage without a subcommand still maps to `calibrate-threshold`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path


_ENTROPY_PATTERN = re.compile(r"entropy=(?P<entropy>-?\d+(?:\.\d+)?)")
_ENTROPY_SCALE = 10000
_QUANTILES: tuple[tuple[str, float], ...] = (
    ("p10", 0.10),
    ("p50", 0.50),
    ("p75", 0.75),
    ("p90", 0.90),
    ("p95", 0.95),
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build entropy profiles or calibrate watermark detection thresholds."
    )
    subparsers = parser.add_subparsers(dest="command")

    build_profile = subparsers.add_parser(
        "build-entropy-profile",
        help="Parse watermark debug logs into an entropy profile JSON",
    )
    build_profile.add_argument("--input-log", required=True, help="Path to watermark debug log")
    build_profile.add_argument("--output", required=True, help="Path to write profile JSON")
    build_profile.add_argument("--language", required=True, help="Profile language label")
    build_profile.add_argument("--model-family", required=True, help="Profile model-family label")
    build_profile.add_argument(
        "--strategy",
        default="piecewise_quantile",
        help="Adaptive gamma schedule strategy label to persist",
    )
    build_profile.add_argument(
        "--profile-id",
        default=None,
        help="Optional profile identifier to persist alongside the profile",
    )

    calibrate = subparsers.add_parser(
        "calibrate-threshold",
        help="Calibrate FPR-based watermark detection threshold",
    )
    calibrate.add_argument("--input", required=True, help="Path to negative corpus JSONL")
    calibrate.add_argument("--output", required=True, help="Path to write threshold JSON")
    calibrate.add_argument(
        "--fpr", type=float, default=0.01,
        help="Target false positive rate (default: 0.01)",
    )
    calibrate.add_argument("--secret-key", required=True, help="Watermark secret key")
    calibrate.add_argument(
        "--model", required=True,
        help="Path to encoder model (e.g. data/models/codet5-base)",
    )
    calibrate.add_argument("--device", default="cuda", help="Device: cuda or cpu")
    calibrate.add_argument(
        "--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    calibrate.add_argument(
        "--lsh-d", type=int, default=3, help="LSH projection count (default: 3)"
    )
    calibrate.add_argument(
        "--gamma", type=float, default=0.5,
        help="LSH valid-region fraction gamma (default: 0.5)",
    )
    return parser


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_parser()
    if argv is None:
        argv = sys.argv[1:]
    if not argv:
        parser.print_help(sys.stderr)
        parser.exit(2)
    if argv[0] not in {"build-entropy-profile", "calibrate-threshold", "-h", "--help"}:
        argv = ["calibrate-threshold", *argv]
    return parser.parse_args(argv)


def _load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _nearest_rank_quantile(sorted_values: list[int], probability: float) -> int:
    index = max(1, math.ceil(probability * len(sorted_values))) - 1
    return sorted_values[index]


def _build_entropy_profile(args: argparse.Namespace) -> None:
    entropy_units: list[int] = []
    with open(args.input_log, encoding="utf-8") as handle:
        for line in handle:
            match = _ENTROPY_PATTERN.search(line)
            if match is None:
                continue
            entropy_value = float(match.group("entropy"))
            entropy_units.append(max(0, int(round(entropy_value * _ENTROPY_SCALE))))

    if not entropy_units:
        raise SystemExit("No entropy=<float> entries found in input log")

    entropy_units.sort()
    payload = {
        "language": args.language,
        "model_family": args.model_family,
        "strategy": args.strategy,
        "sample_count": len(entropy_units),
        "quantiles_units": {
            name: _nearest_rank_quantile(entropy_units, probability)
            for name, probability in _QUANTILES
        },
    }
    if args.profile_id is not None:
        payload["profile_id"] = args.profile_id

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _calibrate_threshold(args: argparse.Namespace) -> None:

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


def main() -> None:
    args = _parse_args()

    if args.command == "build-entropy-profile":
        _build_entropy_profile(args)
        return
    if args.command == "calibrate-threshold":
        _calibrate_threshold(args)
        return
    raise SystemExit(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    main()
