#!/usr/bin/env python
"""Deprecated wrapper for the diagnostic candidate selector."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.warn(
    "scripts/select_sawr_candidates.py is deprecated; use "
    "scripts/diagnostics/select_evidence_only_candidates.py",
    DeprecationWarning,
    stacklevel=2,
)

from scripts.diagnostics import select_evidence_only_candidates as _impl

for _name, _value in vars(_impl).items():
    if not (_name.startswith("__") and _name.endswith("__")):
        globals()[_name] = _value

main = _impl.main


if __name__ == "__main__":
    raise SystemExit(main())
