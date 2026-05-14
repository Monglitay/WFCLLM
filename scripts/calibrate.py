#!/usr/bin/env python
"""Watermark calibration utilities.

(Phase 3 refactor: ``build-entropy-profile`` moved to scripts/build_entropy_profile.py;
this script proxies that subcommand for backwards compatibility and keeps
``calibrate-threshold`` for now.)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Watermark calibration utilities (deprecated wrapper).",
    )
    subparsers = parser.add_subparsers(dest="command")

    build_profile = subparsers.add_parser(
        "build-entropy-profile",
        help="DEPRECATED: use scripts/build_entropy_profile.py",
    )
    build_profile.add_argument("--input-log", required=True)
    build_profile.add_argument("--output", required=True)
    build_profile.add_argument("--language", required=True)
    build_profile.add_argument("--model-family", required=True)
    build_profile.add_argument("--strategy", default="piecewise_quantile")
    build_profile.add_argument("--profile-id", default=None)

    calibrate = subparsers.add_parser(
        "calibrate-threshold",
        help="Calibrate FPR-based watermark detection threshold",
    )
    calibrate.add_argument("--input", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--fpr", type=float, default=0.01)
    calibrate.add_argument("--secret-key", required=True)
    calibrate.add_argument("--model", required=True)
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--embed-dim", type=int, default=128)
    calibrate.add_argument("--lsh-d", type=int, default=3)
    calibrate.add_argument("--gamma", type=float, default=0.5)
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


def _proxy_build_entropy_profile(args: argparse.Namespace) -> int:
    print(
        "[deprecated] scripts/calibrate.py build-entropy-profile is deprecated; "
        "use scripts/build_entropy_profile.py instead.",
        file=sys.stderr,
    )
    from wfcllm.watermark.adaptive_gamma.calibrate import (
        build_entropy_profile_from_log,
    )
    try:
        build_entropy_profile_from_log(
            input_log=args.input_log,
            output=args.output,
            language=args.language,
            model_family=args.model_family,
            strategy=args.strategy,
            profile_id=args.profile_id,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    return 0


def _load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _calibrate_threshold(args: argparse.Namespace) -> int:
    import torch  # noqa: F401  (sanity import; real dependency below)
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
    return 0


def main() -> int:
    args = _parse_args()
    if args.command == "build-entropy-profile":
        return _proxy_build_entropy_profile(args)
    if args.command == "calibrate-threshold":
        return _calibrate_threshold(args)
    raise SystemExit(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
