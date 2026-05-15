"""Deprecated import path. Use wfcllm.lang.python.parser instead."""
from __future__ import annotations

import warnings

warnings.warn(
    "wfcllm.common.ast_parser is deprecated; use wfcllm.lang.python.parser",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.lang.python.parser import *  # noqa: E402, F401, F403
from wfcllm.lang.python.parser import (  # noqa: E402, F401
    PY_LANGUAGE,
    SIMPLE_STATEMENT_TYPES,
    COMPOUND_STATEMENT_TYPES,
    STATEMENT_TYPES,
    PythonParser,
    StatementBlock,
    extract_statement_blocks,
)
