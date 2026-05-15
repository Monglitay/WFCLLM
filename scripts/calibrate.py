#!/usr/bin/env python
"""Deprecated wrapper. Use:

  scripts/build_entropy_profile.py   for build-entropy-profile
  scripts/calibrate_threshold.py     for calibrate-threshold
"""
from __future__ import annotations

import argparse
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
        help="DEPRECATED: use scripts/calibrate_threshold.py",
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


def _proxy_calibrate_threshold(args: argparse.Namespace) -> int:
    print(
        "[deprecated] scripts/calibrate.py calibrate-threshold is deprecated; "
        "use scripts/calibrate_threshold.py instead.",
        file=sys.stderr,
    )
    from wfcllm.extract.calibration.runner import calibrate_threshold_from_corpus
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


def main() -> int:
    args = _parse_args()
    if args.command == "build-entropy-profile":
        return _proxy_build_entropy_profile(args)
    if args.command == "calibrate-threshold":
        return _proxy_calibrate_threshold(args)
    raise SystemExit(f"Unsupported command: {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())
