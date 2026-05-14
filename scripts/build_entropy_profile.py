#!/usr/bin/env python
"""Build an entropy profile JSON from a watermark debug log.

(Phase 3 refactor: replaces ``scripts/calibrate.py build-entropy-profile``.)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.watermark.adaptive_gamma.calibrate import (  # noqa: E402
    build_entropy_profile_from_log,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an entropy profile JSON from a watermark debug log.",
    )
    parser.add_argument("--input-log", required=True, help="Path to watermark debug log")
    parser.add_argument("--output", required=True, help="Path to write profile JSON")
    parser.add_argument("--language", required=True, help="Profile language label")
    parser.add_argument("--model-family", required=True, help="Profile model-family label")
    parser.add_argument(
        "--strategy",
        default="piecewise_quantile",
        help="Adaptive gamma schedule strategy label to persist",
    )
    parser.add_argument(
        "--profile-id",
        default=None,
        help="Optional profile identifier to persist alongside the profile",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        output_path = build_entropy_profile_from_log(
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
    print(f"[完成] entropy profile 已保存至 {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
