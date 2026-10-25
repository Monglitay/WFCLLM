from __future__ import annotations

import hashlib

import pytest

from wfcllm.evaluation import humanevalpack_execution as native


JAVA_TEST = "public class Main { public static void main(String[] a) { if(new Solution().plusOne(2)!=3) throw new AssertionError(); } }"
JS_TEST = "console.assert(plusOne(2) === 3);"


@pytest.fixture(params=["java", "js"])
def toolchain(request):
    try:
        return native.require_native_toolchain(request.param)
    except ValueError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize("kind", ["correct", "wrong", "syntax"])
def test_real_native_runtime_checks_original_tests(toolchain, kind):
    if toolchain.language == "java":
        body = {"correct": "return x+1;", "wrong": "return x;", "syntax": "invalid syntax"}[kind]
        source = "class Solution { int plusOne(int x) { " + body + " } }"
        test = JAVA_TEST
    else:
        source = {"correct": "const plusOne=x=>x+1;", "wrong": "const plusOne=x=>x;", "syntax": "const plusOne = ;"}[kind]
        test = JS_TEST
    result = native.evaluate_native_row(final_code=source, test_code=test, toolchain=toolchain)
    assert result["is_correct"] is (kind == "correct")
    assert result["original_test_sha256"] == hashlib.sha256(test.encode()).hexdigest()
    if kind != "correct":
        assert result["error"]


def test_native_runtime_timeout_is_a_failed_sample(toolchain, monkeypatch):
    original = native._execute

    def short_execution(command, directory, timeout, phase):
        return original(command, directory, 0.1 if phase == "execution" else timeout, phase)

    monkeypatch.setattr(native, "_execute", short_execution)
    if toolchain.language == "java":
        source = "class Solution { int plusOne(int x) { while(true){} } }"
        test = JAVA_TEST
    else:
        source = "const plusOne=x=>{while(true){}};"
        test = JS_TEST
    result = native.evaluate_native_row(final_code=source, test_code=test, toolchain=toolchain)
    assert result["is_correct"] is False
    assert "execution timed out" in result["error"]


def test_js_assertion_and_stdout_are_failures_even_with_zero_exit_status():
    try:
        toolchain = native.require_native_toolchain("js")
    except ValueError as error:
        pytest.skip(str(error))
    for test in ["console.assert(false);", "console.log('unexpected output');"]:
        result = native.evaluate_native_row(final_code="", test_code=test, toolchain=toolchain)
        assert result["is_correct"] is False


@pytest.mark.parametrize("language", ["java", "js"])
def test_missing_native_runtime_fails_closed(language, monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="real local"):
        native.require_native_toolchain(language)


def test_java_runtime_without_compiler_fails_closed(monkeypatch):
    monkeypatch.setattr(native.shutil, "which", lambda name: "/usr/bin/java" if name == "java" else None)
    with pytest.raises(ValueError, match="javac and java"):
        native.require_native_toolchain("java")


def test_native_harness_preserves_original_assertions_and_source():
    source = "class Solution {}\npublic class Main {\n    }"
    program = native.native_check_program(source, JAVA_TEST, "java")
    assert program == "class Solution {}\n" + JAVA_TEST
    assert source.endswith("public class Main {\n    }")
    assert native.native_check_program("", JS_TEST, "js").endswith(JS_TEST)


@pytest.mark.parametrize("language", ["java", "js"])
def test_native_harness_requires_original_tests(language):
    with pytest.raises(ValueError, match="nonempty original test"):
        native.native_check_program("", "", language)
