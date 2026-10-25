"""Native, post-generation Java/JavaScript execution of HumanEvalPack tests."""

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


NATIVE_TIMEOUT_SECONDS = 10.0


@dataclass(frozen=True)
class NativeToolchain:
    language: str
    runtime: Path
    compiler: Path | None = None


def require_native_toolchain(language: str) -> NativeToolchain:
    if language not in {"java", "js"}:
        raise ValueError("native HumanEvalPack evaluation requires java or js")
    runtime = shutil.which("java" if language == "java" else "node")
    compiler = shutil.which("javac") if language == "java" else None
    if runtime is None or (language == "java" and compiler is None):
        needed = "javac and java" if language == "java" else "node"
        raise ValueError(f"{language} posthoc evaluation requires real local {needed}")
    return NativeToolchain(language, Path(runtime), Path(compiler) if compiler else None)


def native_check_program(final_code: str, test_code: str, language: str) -> str:
    if language not in {"java", "js"}:
        raise ValueError("native HumanEvalPack evaluation requires java or js")
    if not isinstance(final_code, str) or not isinstance(test_code, str) or not test_code.strip():
        raise ValueError("native evaluation requires source text and a nonempty original test")
    # Match HumanEvalPack's Java harness cleanup, exclusively in the posthoc
    # program. Neither the original tests nor detector final_code are rewritten.
    candidate = final_code.replace("public class Main {\n    }", "") if language == "java" else final_code
    return candidate.strip() + "\n" + test_code


def _execute(command: list[str], directory: Path, timeout: float, phase: str) -> tuple[int | None, str, str]:
    paths = [directory / f"{phase}.{stream}" for stream in ("stdout", "stderr")]
    with paths[0].open("wb") as out, paths[1].open("wb") as err:
        process = subprocess.Popen(
            command, cwd=directory, stdin=subprocess.DEVNULL, stdout=out, stderr=err,
            start_new_session=os.name == "posix",
        )
        try:
            status = process.wait(timeout=max(0.001, timeout))
        except subprocess.TimeoutExpired:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
            process.wait()
            return None, "", f"{phase} timed out"
    messages = []
    for path in paths:
        with path.open("rb") as handle:
            messages.append(handle.read(4096).decode("utf-8", errors="replace"))
    return status, messages[0], messages[1]


def evaluate_native_row(
    *, final_code: str, test_code: str, toolchain: NativeToolchain,
) -> dict[str, object]:
    program = native_check_program(final_code, test_code, toolchain.language)
    common: dict[str, object] = {
        "evaluation_protocol": f"humanevalpack-{toolchain.language}-native-tests/v1",
        "original_test_sha256": hashlib.sha256(test_code.encode("utf-8")).hexdigest(),
        "check_program_sha256": hashlib.sha256(program.encode("utf-8")).hexdigest(),
        "runtime": str(toolchain.runtime),
        "compiler": str(toolchain.compiler) if toolchain.compiler else None,
        "timeout_seconds": NATIVE_TIMEOUT_SECONDS,
    }
    with tempfile.TemporaryDirectory(prefix=f"wfcllm-{toolchain.language}-posthoc-") as temporary:
        directory = Path(temporary)
        filename = "Main.java" if toolchain.language == "java" else "test.js"
        (directory / filename).write_text(program, encoding="utf-8")
        deadline = time.monotonic() + NATIVE_TIMEOUT_SECONDS
        if toolchain.language == "java":
            if toolchain.compiler is None:
                raise ValueError("Java evaluation requires a compiler")
            for _ in range(5):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {**common, "is_correct": False, "error": "compilation timed out"}
                status, stdout, stderr = _execute(
                    [str(toolchain.compiler), filename], directory, min(5.0, remaining), "compilation",
                )
                if status is not None:
                    break
            if status != 0:
                return {**common, "is_correct": False,
                        "error": "compilation: " + (stderr or stdout or f"exit status {status}")[:500]}
            command = [str(toolchain.runtime), "-cp", ".", "Main"]
        else:
            command = [str(toolchain.runtime), filename]
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {**common, "is_correct": False, "error": "execution timed out"}
        status, stdout, stderr = _execute(command, directory, remaining, "execution")
    # Node's console.assert reports a wrong answer on stderr while exiting zero.
    # OctoPack also treats stdout as failure; preserve that evaluation behavior.
    correct = status == 0 and (toolchain.language != "js" or not (stdout or stderr))
    message = stderr or stdout or f"exit status {status}"
    return {**common, "is_correct": correct,
            "error": None if correct else "execution: " + message[:500]}
