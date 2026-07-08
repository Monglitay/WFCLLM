#!/usr/bin/env python
from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.wfcllm_generate import main as wfcllm_generate_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    warnings.warn(
        "scripts/run_sawr_smoke.py is deprecated; use scripts/wfcllm_generate.py",
        DeprecationWarning,
        stacklevel=2,
    )
    print("[deprecated] use scripts/wfcllm_generate.py", file=sys.stderr)
    return wfcllm_generate_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
