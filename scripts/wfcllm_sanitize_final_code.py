#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.code_only import validate_final_code_record_exact  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sanitize legacy rows into WFCLLM final_code JSONL.",
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _sanitize_row(row: dict[str, Any]) -> dict[str, str]:
    final_code = row["final_code"] if "final_code" in row else row.get("generated_code")
    sanitized = {
        "id": str(row["id"]),
        "dataset": str(row.get("dataset", "humaneval")),
        "prompt": str(row.get("prompt", "")),
        "final_code": final_code,
    }
    validate_final_code_record_exact(sanitized)
    return sanitized


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with input_path.open(encoding="utf-8") as source:
            with output_path.open("w", encoding="utf-8") as target:
                for line_number, line in enumerate(source, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError(
                            f"input row must be object at line {line_number}"
                        )
                    target.write(
                        json.dumps(
                            _sanitize_row(row),
                            allow_nan=False,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] WFCLLM final-code input saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
