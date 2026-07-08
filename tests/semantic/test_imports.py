from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import wfcllm.semantic",
        "import wfcllm.semantic.rules",
        "import wfcllm.semantic.lsh",
        "import wfcllm.semantic.keying",
        "import wfcllm.semantic.verifier",
    ],
)
def test_semantic_imports_succeed_in_clean_process(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
