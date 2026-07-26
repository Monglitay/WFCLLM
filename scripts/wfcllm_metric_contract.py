"""Emit unified Metric Contract JSONL rows for one or more run directories.

Posthoc reporting CLI (ADR 0003): each run directory yields exactly one JSON
line with a ``schema_version`` field. Output is unconditional — missing or
below-target metrics are recorded as nulls plus caveats, never suppressed.
The tool is read-only over run directories and writes only to stdout or the
explicit ``--output`` file.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from wfcllm.evaluation.metric_contract import extract_metric_contract


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract one Metric Contract record per run directory and emit "
            "JSONL (one line per run) to stdout or --output."
        )
    )
    parser.add_argument(
        "run_dirs",
        nargs="*",
        help="Run directories to extract, one JSONL row each.",
    )
    parser.add_argument(
        "--run-dir",
        action="append",
        default=[],
        dest="run_dir_options",
        help="Run directory to extract; may be given multiple times.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output file; defaults to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_dirs = [*args.run_dir_options, *args.run_dirs]
    if not run_dirs:
        parser.error("at least one run directory is required")

    lines = [
        json.dumps(
            extract_metric_contract(Path(run_dir)),
            ensure_ascii=False,
            sort_keys=True,
        )
        for run_dir in run_dirs
    ]
    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    else:
        for line in lines:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
