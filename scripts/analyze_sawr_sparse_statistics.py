#!/usr/bin/env python
"""Deprecated wrapper for diagnostics/analyze_sparse_statistics.py."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.warn(
    "scripts/analyze_sawr_sparse_statistics.py is deprecated; use "
    "scripts/diagnostics/analyze_sparse_statistics.py",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.diagnostics import analyze_sparse_statistics as _impl

for _name, _value in vars(_impl).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
