from __future__ import annotations

import ast
import contextlib
import doctest
import json
import math
import subprocess
import sys
import tempfile
from dataclasses import dataclass, replace
from typing import Any

_DOCTEST_RESULT_MARKER = "__SAWR_PUBLIC_DOCTEST_RESULT__"
RANKING_MODES = {
    "quality_first",
    "public_then_detector",
    "detector_first",
}


@dataclass(frozen=True)
class CandidateSelectionFeatures:
    sample_id: str
    candidate_index: int
    syntax_valid: bool
    target_function_present: bool
    signature_compatible: bool
    prompt_doctest_passed: bool
    prompt_doctest_count: int
    detector_score: float
    public_doctest_passed: bool | None = None
    public_doctest_count: int | None = None
    public_doctest_timeout: bool = False
    public_doctest_parse_error: bool = False
    public_doctest_stdout: str = ""
    public_doctest_stderr: str = ""
    public_doctest_returncode: int | None = None
    code_chars: int = 0
    suspicious_tail: bool = False
    truncation_suspected: bool = False
    scoreable_contexts: int = 0
    proxy_windows: int = 0
    insufficient_evidence: bool = False
    score_delta_vs_baseline: float | None = None
    proxy_delta_vs_baseline: int | None = None
    selection_reason: str = ""

    def __post_init__(self) -> None:
        if self.public_doctest_passed is None:
            object.__setattr__(
                self,
                "public_doctest_passed",
                self.prompt_doctest_passed,
            )
        if self.public_doctest_count is None:
            object.__setattr__(
                self,
                "public_doctest_count",
                self.prompt_doctest_count,
            )


def evaluate_candidate_quality(
    record: dict[str, Any],
    *,
    detector_score: float = 0.0,
    candidate_index: int = 0,
    doctest_timeout_seconds: float = 0.25,
    scoreable_contexts: int = 0,
    proxy_windows: int = 0,
    insufficient_evidence: bool = False,
    baseline_detector_score: float | None = None,
    baseline_proxy_windows: int | None = None,
) -> CandidateSelectionFeatures:
    sample_id = str(record.get("id") or record.get("task_id") or "")
    prompt = str(record.get("prompt") or "")
    code = str(record.get("final_code") or record.get("generated_code") or "")

    try:
        code_tree = ast.parse(code)
        syntax_valid = True
    except SyntaxError:
        code_tree = None
        syntax_valid = False

    prompt_function = _target_function(prompt)
    code_function = (
        _function_by_name(code_tree, prompt_function.name)
        if code_tree is not None and prompt_function is not None
        else None
    )
    target_function_present = code_function is not None
    signature_compatible = (
        prompt_function is not None
        and code_function is not None
        and _signature_dump(prompt_function) == _signature_dump(code_function)
    )
    doctest_result = _run_prompt_doctests(
        code,
        prompt,
        timeout_seconds=doctest_timeout_seconds,
    )
    suspicious_tail = _has_suspicious_tail(code)

    return CandidateSelectionFeatures(
        sample_id=sample_id,
        candidate_index=candidate_index,
        syntax_valid=syntax_valid,
        target_function_present=target_function_present,
        signature_compatible=signature_compatible,
        prompt_doctest_passed=doctest_result.passed,
        prompt_doctest_count=doctest_result.attempted,
        detector_score=float(detector_score),
        public_doctest_passed=doctest_result.passed,
        public_doctest_count=doctest_result.attempted,
        public_doctest_timeout=doctest_result.timeout,
        public_doctest_parse_error=doctest_result.parse_error,
        public_doctest_stdout=doctest_result.stdout,
        public_doctest_stderr=doctest_result.stderr,
        public_doctest_returncode=doctest_result.returncode,
        code_chars=len(code),
        suspicious_tail=suspicious_tail,
        truncation_suspected=not syntax_valid and suspicious_tail,
        scoreable_contexts=int(scoreable_contexts),
        proxy_windows=int(proxy_windows),
        insufficient_evidence=bool(insufficient_evidence),
        score_delta_vs_baseline=(
            float(detector_score) - baseline_detector_score
            if baseline_detector_score is not None
            else None
        ),
        proxy_delta_vs_baseline=(
            int(proxy_windows) - baseline_proxy_windows
            if baseline_proxy_windows is not None
            else None
        ),
    )


def select_best_candidate(
    candidates: list[CandidateSelectionFeatures],
    *,
    ranking_mode: str = "quality_first",
) -> CandidateSelectionFeatures:
    if not candidates:
        raise ValueError("candidates must be non-empty")
    if ranking_mode not in RANKING_MODES:
        raise ValueError(
            f"ranking_mode must be one of {sorted(RANKING_MODES)}, got {ranking_mode!r}"
        )

    selected = max(candidates, key=lambda item: _ranking_key(item, ranking_mode))
    reason = (
        "syntax_signature_doctest_score"
        if ranking_mode == "quality_first"
        else ranking_mode
    )
    return replace(selected, selection_reason=reason)


def _ranking_key(
    item: CandidateSelectionFeatures,
    ranking_mode: str,
) -> tuple[object, ...]:
    hard_quality = (
        item.syntax_valid,
        item.target_function_present,
        item.signature_compatible,
    )
    public_quality = (
        bool(item.public_doctest_passed),
        not item.public_doctest_timeout,
        not item.public_doctest_parse_error,
        not item.suspicious_tail,
    )
    evidence_quality = (
        not item.insufficient_evidence,
        item.proxy_windows,
        item.scoreable_contexts,
    )

    if ranking_mode == "quality_first":
        return (
            *hard_quality,
            *public_quality,
            *evidence_quality,
            item.detector_score,
            -item.candidate_index,
        )
    if ranking_mode == "public_then_detector":
        return (
            *hard_quality,
            *public_quality,
            item.detector_score,
            not item.insufficient_evidence,
            item.scoreable_contexts,
            item.proxy_windows,
            -item.candidate_index,
        )
    if ranking_mode == "detector_first":
        return (
            *hard_quality,
            not item.public_doctest_timeout,
            not item.public_doctest_parse_error,
            not item.suspicious_tail,
            item.detector_score,
            bool(item.public_doctest_passed),
            not item.insufficient_evidence,
            item.scoreable_contexts,
            item.proxy_windows,
            -item.candidate_index,
        )
    raise ValueError(f"unsupported ranking_mode: {ranking_mode!r}")


def _target_function(prompt: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _function_by_name(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    if not isinstance(tree, ast.Module):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _signature_dump(function: ast.FunctionDef) -> str:
    return ast.dump(function.args, annotate_fields=True, include_attributes=False)


@dataclass(frozen=True)
class _PromptDoctestResult:
    passed: bool
    attempted: int
    timeout: bool = False
    parse_error: bool = False
    stdout: str = ""
    stderr: str = ""
    returncode: int | None = None


def _run_prompt_doctests(
    code: str,
    prompt: str,
    *,
    timeout_seconds: float,
) -> _PromptDoctestResult:
    doctest_text = _prompt_docstring(prompt)
    parser = doctest.DocTestParser()
    try:
        examples = parser.get_examples(doctest_text)
    except ValueError as exc:
        return _PromptDoctestResult(
            passed=False,
            attempted=0,
            parse_error=True,
            stderr=str(exc),
            returncode=None,
        )
    if not examples:
        return _PromptDoctestResult(passed=True, attempted=0, returncode=0)

    payload = json.dumps(
        {"code": code, "doctest_text": doctest_text},
        ensure_ascii=False,
        allow_nan=False,
    )
    try:
        with tempfile.TemporaryDirectory(prefix="sawr-public-tests-") as temp_dir:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _PUBLIC_DOCTEST_RUNNER],
                input=payload,
                text=True,
                capture_output=True,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
                cwd=temp_dir,
                check=False,
                preexec_fn=_resource_limiter(timeout_seconds),
            )
    except subprocess.TimeoutExpired as exc:
        return _PromptDoctestResult(
            passed=False,
            attempted=len(examples),
            timeout=True,
            stdout=_safe_output(exc.stdout),
            stderr=_safe_output(exc.stderr),
            returncode=None,
        )
    except OSError as exc:
        return _PromptDoctestResult(
            passed=False,
            attempted=len(examples),
            stderr=str(exc),
            returncode=None,
        )

    result_payload = _parse_child_doctest_result(completed.stdout)
    if result_payload is None:
        return _PromptDoctestResult(
            passed=False,
            attempted=len(examples),
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
        )
    return _PromptDoctestResult(
        passed=bool(result_payload.get("passed", False)),
        attempted=int(result_payload.get("attempted", len(examples)) or 0),
        timeout=False,
        stdout=str(result_payload.get("stdout", "")),
        stderr=str(result_payload.get("stderr", completed.stderr)),
        returncode=completed.returncode,
    )


def _prompt_docstring(prompt: str) -> str:
    function = _target_function(prompt)
    if function is None:
        return prompt
    return ast.get_docstring(function, clean=False) or ""


def _has_suspicious_tail(code: str) -> bool:
    stripped = code.rstrip()
    if not stripped:
        return True
    tail = stripped.rsplit("\n", maxsplit=1)[-1].strip()
    if not tail:
        return True
    if tail.endswith(("\\", ",", ".", ":", "(", "[", "{", "+", "-", "*", "/", "%")):
        return True
    return tail in {
        "return",
        "if",
        "elif",
        "else",
        "for",
        "while",
        "with",
        "try",
        "except",
        "finally",
    }


def _resource_limiter(timeout_seconds: float):
    def _limit_child() -> None:
        with contextlib.suppress(Exception):
            import resource

            cpu_seconds = max(1, int(math.ceil(timeout_seconds if timeout_seconds > 0 else 1)))
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
            resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))

    return _limit_child


def _safe_output(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _parse_child_doctest_result(stdout: str) -> dict[str, Any] | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(_DOCTEST_RESULT_MARKER):
            try:
                payload = json.loads(line[len(_DOCTEST_RESULT_MARKER) :])
            except json.JSONDecodeError:
                return None
            return payload if isinstance(payload, dict) else None
    return None


_PUBLIC_DOCTEST_RUNNER = r'''
import contextlib
import doctest
import io
import json
import sys
import traceback

MARKER = "__SAWR_PUBLIC_DOCTEST_RESULT__"
real_stdout = sys.stdout
payload = json.loads(sys.stdin.read())
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
result = {
    "passed": False,
    "attempted": 0,
    "stdout": "",
    "stderr": "",
}

try:
    parser = doctest.DocTestParser()
    test = parser.get_doctest(
        payload["doctest_text"],
        {},
        name="prompt",
        filename="<prompt>",
        lineno=0,
    )
    result["attempted"] = len(test.examples)
    globs = {"__name__": "__sawr_candidate__"}
    runner = doctest.DocTestRunner(verbose=False)
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        exec(payload["code"], globs)
        test.globs.update(globs)
        run_result = runner.run(test, out=lambda text: stdout_buffer.write(text))
    result["passed"] = run_result.failed == 0
except BaseException:
    result["stderr"] = traceback.format_exc(limit=8)
finally:
    captured_stderr = stderr_buffer.getvalue()
    if captured_stderr:
        result["stderr"] = (result.get("stderr") or "") + captured_stderr
    result["stdout"] = stdout_buffer.getvalue()
    print(MARKER + json.dumps(result, ensure_ascii=False), file=real_stdout)
'''
