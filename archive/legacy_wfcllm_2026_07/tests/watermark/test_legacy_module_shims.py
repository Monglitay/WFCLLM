from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import wfcllm.watermark.keying",
        "import wfcllm.watermark.lsh_space",
        "import wfcllm.watermark.verifier",
        "import wfcllm.watermark.kv_cache",
    ],
)
def test_legacy_watermark_module_imports_succeed_in_clean_process(
    statement: str,
) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
