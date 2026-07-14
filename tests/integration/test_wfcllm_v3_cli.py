from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v3_scripts_expose_key_file_not_raw_secret_cli() -> None:
    for name in ("wfcllm_v3_fit.py", "wfcllm_v3_detect.py", "wfcllm_v3_replay_pool.py"):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / "scripts" / name), "--help"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "--key-file" in result.stdout
        assert "--secret-key" not in result.stdout
