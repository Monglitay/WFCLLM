#!/usr/bin/env python
"""Legacy WFCLLM script retained as archive guidance."""
from __future__ import annotations

import sys

MESSAGE = "legacy threshold calibration support was archived under archive/legacy_wfcllm_2026_07/code/extract/calibration."

def main(argv: list[str] | None = None) -> int:
    print(MESSAGE, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
