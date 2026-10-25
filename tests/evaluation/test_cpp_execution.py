from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from wfcllm.evaluation import cpp_execution


_TEST = "#undef NDEBUG\n#include<assert.h>\nint main(){assert(plus_one(2)==3);}\n"


@pytest.fixture
def toolchain():
    try:
        return cpp_execution.require_cpp_toolchain()
    except ValueError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize(
    ("source", "expected", "phase"),
    [
        ("int plus_one(int x){return x+1;}", True, None),
        ("int plus_one(int x){return x;}", False, "execution"),
        ("int plus_one(int x){this is not valid;}", False, "compilation"),
    ],
)
def test_real_cpp_compiler_checks_original_assertions(toolchain, source, expected, phase):
    result = cpp_execution.evaluate_cpp_row(final_code=source, test_code=_TEST, toolchain=toolchain)
    assert result["is_correct"] is expected
    assert result["original_test_sha256"] == hashlib.sha256(_TEST.encode()).hexdigest()
    if phase is not None:
        assert result["error"].startswith(phase)


def test_cpp_execution_timeout_kills_child(toolchain, monkeypatch):
    original = cpp_execution._run_process

    def short_execution(command, directory, timeout, phase):
        return original(command, directory, 0.1 if phase == "execution" else timeout, phase)

    monkeypatch.setattr(cpp_execution, "_run_process", short_execution)
    result = cpp_execution.evaluate_cpp_row(
        final_code="int plus_one(int x){while(true){} return x;}",
        test_code=_TEST, toolchain=toolchain,
    )
    assert result["is_correct"] is False
    assert "execution timed out" in result["error"]


def test_missing_compiler_fails_closed(monkeypatch):
    monkeypatch.setattr(cpp_execution.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="real local g\\+\\+"):
        cpp_execution.require_cpp_toolchain()


def test_missing_boost_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr(cpp_execution.shutil, "which", lambda _: str(tmp_path / "g++"))
    monkeypatch.setenv("WFCLLM_CPP_INCLUDE_DIR", str(tmp_path))
    with pytest.raises(ValueError, match="real Boost"):
        cpp_execution.require_cpp_toolchain()


def test_cpp_harness_removes_generated_main_without_mutating_original_test():
    source = "int plus_one(int x){return x+1;}\nint main(){return 0;}"
    program = cpp_execution.cpp_check_program(source, _TEST)
    assert program.endswith(_TEST)
    assert "int main(){return 0;}" not in program
    assert source.endswith("int main(){return 0;}")
