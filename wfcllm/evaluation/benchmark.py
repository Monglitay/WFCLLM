"""Offline benchmark evaluation: Pass@1, Pass@10, AUROC for code watermarking."""
from __future__ import annotations

import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TestExecutor:
    """Execute code + test cases via subprocess, return pass/fail."""

    timeout: float = 5.0

    def run_test(self, code: str, test_code: str) -> bool:
        """Run generated code against test assertions. Returns True if all pass."""
        full_source = f"{code}\n{test_code}\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_source)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def execute_humaneval(
        self, generated_code: str, test_code: str, entry_point: str
    ) -> bool:
        """HumanEval-style execution: code + test + check(entry_point)."""
        full_test = f"{test_code}\ncheck({entry_point})\n"
        return self.run_test(generated_code, full_test)

    def execute_mbpp(self, generated_code: str, test_code: str) -> bool:
        """MBPP-style execution: code + assert statements."""
        return self.run_test(generated_code, test_code)
