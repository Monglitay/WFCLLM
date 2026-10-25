"""Post-generation HumanEvalPack C++ execution with the unchanged test suite."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import signal
import subprocess
import tempfile
import time


# The upstream HumanEvalPack harness supplies these headers and removes a
# generated main before appending its own tests. This never edits detector input.
CPP_EVALUATION_PROTOCOL = "humanevalpack-cpp11-tests/v1"
CPP_TIMEOUT_SECONDS = 60.0
CPP_HEADERS = (
    "stdlib.h", "algorithm", "cmath", "math.h", "numeric", "stdio.h",
    "vector", "set", "map", "queue", "stack", "list", "deque",
    "boost/any.hpp", "string", "climits", "cstring", "iostream",
    "sstream", "fstream",
)


@dataclass(frozen=True)
class CppToolchain:
    compiler: Path
    include_dir: Path


def require_cpp_toolchain() -> CppToolchain:
    compiler = shutil.which("g++")
    if compiler is None:
        raise ValueError("C++ posthoc evaluation requires a real local g++")
    include_dir = Path(os.environ.get("WFCLLM_CPP_INCLUDE_DIR", "/usr/include"))
    if not (include_dir / "boost" / "any.hpp").is_file():
        raise ValueError(
            "C++ posthoc evaluation requires real Boost headers; set "
            "WFCLLM_CPP_INCLUDE_DIR to their include root"
        )
    if not Path("/usr/include/openssl/ssl.h").is_file():
        raise ValueError("C++ posthoc evaluation requires local OpenSSL development headers")
    return CppToolchain(Path(compiler), include_dir.resolve())


def cpp_check_program(final_code: str, test_code: str) -> str:
    if not isinstance(final_code, str) or not isinstance(test_code, str) or not test_code.strip():
        raise ValueError("C++ evaluation requires source text and a nonempty original test")
    headers = "\n".join(f"#include<{name}>" for name in CPP_HEADERS)
    candidate = final_code.split("int main", 1)[0]
    return headers + "\nusing namespace std;\n" + candidate + "\n" + test_code


def _run_process(command: list[str], directory: Path, timeout: float, phase: str) -> tuple[int | None, str]:
    stdout = directory / f"{phase}.stdout"
    stderr = directory / f"{phase}.stderr"
    with stdout.open("wb") as out, stderr.open("wb") as err:
        process = subprocess.Popen(
            command, cwd=directory, stdin=subprocess.DEVNULL, stdout=out,
            stderr=err, start_new_session=os.name == "posix",
        )
        try:
            status = process.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return None, f"{phase} timed out"
    for path in (stderr, stdout):
        with path.open("rb") as handle:
            message = handle.read(2048).decode("utf-8", errors="replace").strip()
        if message:
            return status, message[:500]
    return status, f"exit status {status}"


def evaluate_cpp_row(
    *, final_code: str, test_code: str, toolchain: CppToolchain,
) -> dict[str, object]:
    """Compile and execute original assertions; retain every failed completion."""
    program = cpp_check_program(final_code, test_code)
    common: dict[str, object] = {
        "evaluation_protocol": CPP_EVALUATION_PROTOCOL,
        "original_test_sha256": hashlib.sha256(test_code.encode("utf-8")).hexdigest(),
        "check_program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "compiler": str(toolchain.compiler),
        "compiler_flags": ["-std=c++11", "-lcrypto", "-lssl"],
        "timeout_seconds": CPP_TIMEOUT_SECONDS,
    }
    with tempfile.TemporaryDirectory(prefix="wfcllm-cpp-posthoc-") as temporary:
        directory = Path(temporary)
        (directory / "test.cpp").write_text(program, encoding="utf-8")
        command = [str(toolchain.compiler), "-std=c++11", "-I", str(toolchain.include_dir),
                   "test.cpp", "-lcrypto", "-lssl", "-o", "test-program"]
        deadline = time.monotonic() + CPP_TIMEOUT_SECONDS
        status, message = _run_process(command, directory, CPP_TIMEOUT_SECONDS, "compilation")
        if status != 0:
            return {**common, "is_correct": False, "error": "compilation: " + message}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {**common, "is_correct": False, "error": "execution timed out"}
        status, message = _run_process([str(directory / "test-program")], directory, remaining, "execution")
    return {**common, "is_correct": status == 0, "error": None if status == 0 else "execution: " + message}
