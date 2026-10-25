"""Key-blind JavaScript expression candidates within original statement lines."""

from __future__ import annotations

import hashlib
import re
from functools import lru_cache

from wfcllm.gate.data import RewriteCandidate, RewriteRequest
from wfcllm.generation.window_rewriter import (
    ParsedRewrite,
    RewriteGeneration,
    StatementUnitExtractor,
    _parse_generation,
    _validate_call,
    _validate_candidate_indices,
)


def _balanced_expression(text: str) -> bool:
    # Regex literals, comments, templates and ambiguous/multiline expressions
    # remain unavailable; this is not a general-purpose JavaScript parser.
    if not text.strip() or any(ch in text for ch in ("`", "/", "\n", "\r")):
        return False
    stack: list[str] = []
    quote = None
    escaped = False
    for ch in text:
        if quote is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote:
                quote = None
            continue
        if ch in "'\"":
            quote = ch
        elif ch in "([{":
            stack.append(ch)
        elif ch in ")]}":
            if not stack or stack.pop() != {")": "(", "]": "[", "}": "{"}[ch]:
                return False
        elif ch in ",;" and not stack:
            return False
    return quote is None and not stack


def _line_forms(line: str) -> tuple[str, ...]:
    body = line.rstrip("\r\n")
    trailing = line[len(body):]
    patterns = [
        ("return", r"^(\s*return[ \t]+)(.+?)(;[ \t]*)$"),
        ("value", r"^(\s*(?:const|let|var)[ \t]+[A-Za-z_$][A-Za-z0-9_$]*[ \t]*=[ \t]*)(.+?)(;[ \t]*)$"),
        ("value", r"^(\s*[A-Za-z_$][A-Za-z0-9_$]*(?:\.[A-Za-z_$][A-Za-z0-9_$]*)*[ \t]*(?:[+*%&|^-]=|=(?!=|>))[ \t]*)(.+?)(;[ \t]*)$"),
        ("update", r"^(\s*)((?:\+\+|--)[A-Za-z_$][A-Za-z0-9_$]*|[A-Za-z_$][A-Za-z0-9_$]*(?:\+\+|--))(;[ \t]*)$"),
        ("condition", r"^(\s*(?:if|while)[ \t]*\()(.*)(\)[ \t]*\{[ \t]*)$"),
        ("switch", r"^(\s*switch[ \t]*\()(.*)(\)[ \t]*\{[ \t]*)$"),
    ]
    for kind, pattern in patterns:
        match = re.match(pattern, body)
        if match is None:
            continue
        expr = match.group(2)
        if not _balanced_expression(expr):
            return ()
        forms = [f"({expr})", f"(({expr}))", f"((({expr})))"]
        if kind == "return":
            forms += [f"(void 0, ({expr}))", f"(0, ({expr}))"]
        elif kind == "condition":
            forms += [f"true && ({expr})", f"false || ({expr})", f"!!({expr})"]
        return tuple(match.group(1) + form + match.group(3) + trailing for form in forms)
    return ()


class KeyBlindJavaScriptStructuralWindowRewriter:
    """Emit a fixed expression trajectory without keys or semantic certificates."""

    def __init__(self, *, extractor: StatementUnitExtractor, window_contract_version: str) -> None:
        if window_contract_version != "js-statement-window/v1":
            raise ValueError("JavaScript structural rewriting requires the JS window contract")
        self._extractor = extractor
        self._window_contract_version = window_contract_version

    @lru_cache(maxsize=8)
    def _variants(self, request: RewriteRequest) -> tuple[str, ...]:
        lines = request.original_window.splitlines(keepends=True)
        atoms = [(index, forms) for index, line in enumerate(lines) if (forms := _line_forms(line))]
        pool: list[str] = []
        for index, forms in atoms:
            for form in forms:
                changed = list(lines)
                changed[index] = form
                text = "".join(changed)
                if text != request.original_window and text not in pool:
                    pool.append(text)
        if len(atoms) >= 2:
            for ordinal in range(128):
                number = ordinal
                changed = list(lines)
                for index, forms in atoms:
                    number, digit = divmod(number, len(forms))
                    changed[index] = forms[digit]
                text = "".join(changed)
                if text != request.original_window and text not in pool:
                    pool.append(text)
                if len(pool) >= 18:
                    break
        base = pool or [request.original_window]
        return tuple(base[index % len(base)] for index in range(18))

    def rewrite_windows(self, request: RewriteRequest, *, candidate_indices: tuple[int, ...]) -> tuple[ParsedRewrite, ...]:
        _validate_candidate_indices(candidate_indices)
        return tuple(self.rewrite_window(request, candidate_index=index) for index in candidate_indices)

    def rewrite_many(self, request: RewriteRequest, *, candidate_indices: tuple[int, ...]) -> tuple[RewriteCandidate, ...]:
        return tuple(self._candidate(request, row) for row in self.rewrite_windows(request, candidate_indices=candidate_indices))

    def rewrite(self, request: RewriteRequest, *, candidate_index: int) -> RewriteCandidate:
        return self._candidate(request, self.rewrite_window(request, candidate_index=candidate_index))

    @staticmethod
    def _candidate(request: RewriteRequest, parsed: ParsedRewrite) -> RewriteCandidate:
        first = len(request.completed_prefix.encode("utf-8"))
        return RewriteCandidate(code=parsed.text, parse_status=parsed.parse_status, unit_count=parsed.unit_count,
            same_parent_scope=parsed.same_parent_scope, boundary_span=(first, first + len(parsed.text.encode("utf-8"))),
            generation_seed_id=parsed.generation_seed_id, rewrite_config_id=parsed.rewrite_config_id)

    def rewrite_window(self, request: RewriteRequest, *, candidate_index: int) -> ParsedRewrite:
        _validate_call(request, candidate_index)
        text = self._variants(request)[(candidate_index - 1) % 18]
        return _parse_generation(request, RewriteGeneration((), text,
            "public-js-structural:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16],
            f"js-structural-expressions/v1:{candidate_index}"), extractor=self._extractor,
            window_contract_version=self._window_contract_version)
