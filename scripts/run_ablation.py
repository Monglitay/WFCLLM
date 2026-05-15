#!/usr/bin/env python
"""Run an ablation sweep described by a YAML file (spec §5.3)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.ablation import AblationRunner, SweepSpec  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an ablation sweep described by a YAML spec.",
    )
    parser.add_argument("yaml", help="Path to sweep YAML file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="No-op flag (kept for spec parity); resume is automatic from results.jsonl",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    yaml_path = Path(args.yaml)
    if not yaml_path.exists():
        print(f"[error] sweep yaml not found: {yaml_path}", file=sys.stderr)
        return 2
    try:
        spec = SweepSpec.from_yaml(yaml_path)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 2
    runner = AblationRunner()
    runner.run(spec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
