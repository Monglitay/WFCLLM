#!/usr/bin/env python
"""Deprecated wrapper around wfcllm.evaluation.dual_channel.

Use ``scripts/evaluate.py dual ...`` (Phase 2 refactor) or import from
``wfcllm.evaluation.dual_channel`` directly.
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Repo-root path bootstrap so this script keeps working when invoked as
# ``python scripts/evaluate_dual_channel.py`` (i.e. without ``-m``).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

warnings.warn(
    "scripts/evaluate_dual_channel.py is deprecated; "
    "use 'python scripts/evaluate.py dual ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.evaluation.dual_channel import (  # noqa: E402,F401
    CommandRunResult,
    CommandRunner,
    MODES,
    PERTURBATIONS,
    TARGET_FPR,
    build_argument_parser,
    main,
    run_evaluation,
)


if __name__ == "__main__":
    main()
