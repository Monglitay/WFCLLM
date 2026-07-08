#!/usr/bin/env python
"""Compatibility name for quality-proxy diagnostic candidate selection."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.diagnostics.select_evidence_only_candidates import *  # noqa: F403
from scripts.diagnostics.select_evidence_only_candidates import main


if __name__ == "__main__":
    raise SystemExit(main())
