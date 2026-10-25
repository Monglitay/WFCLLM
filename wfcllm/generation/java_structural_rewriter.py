"""Key-blind Java expression candidates within the original statement lines."""
from __future__ import annotations

import hashlib
from functools import lru_cache

from wfcllm.gate.data import RewriteCandidate, RewriteRequest

from wfcllm.generation.window_rewriter import (
    ParsedRewrite, RewriteGeneration, StatementUnitExtractor,
    _JAVA_EQUIVALENT_TRANSFORMS, _java_parenthesize_condition,
    _java_boolean_object_ternary_return, _validate_call, _validate_candidate_indices,
    _java_parenthesize_initializer_or_assignment, _java_parenthesize_return,
    _parse_generation,
)


def _line_forms(line: str) -> tuple[str, ...]:
    body = line.rstrip('\r\n')
    trailing = line[len(body):]
    forms: list[str] = []
    for transform in _JAVA_EQUIVALENT_TRANSFORMS:
        # A nonconstant boxed condition loses Java's constant narrowing rules.
        if transform is _java_boolean_object_ternary_return:
            continue
        text = transform(body)
        if text != body and text not in forms:
            forms.append(text)
    for parenthesize in (_java_parenthesize_return, _java_parenthesize_condition,
                         _java_parenthesize_initializer_or_assignment):
        text = body
        for depth in range(4):
            changed = parenthesize(text)
            if changed == text:
                break
            text = changed
            if text not in forms:
                forms.append(text)
    return tuple(text + trailing for text in forms)


class KeyBlindJavaStructuralWindowRewriter:
    def __init__(self, *, extractor: StatementUnitExtractor, window_contract_version: str) -> None:
        if window_contract_version != 'java-statement-window/v1':
            raise ValueError('Java structural rewriting requires the Java window contract')
        self.extractor = extractor
        self.contract = window_contract_version

    @lru_cache(maxsize=8)
    def _variant_texts(self, request: RewriteRequest) -> tuple[str, ...]:
        lines = request.original_window.splitlines(keepends=True)
        atoms = [(i, forms) for i, line in enumerate(lines) if (forms := _line_forms(line))]
        pool: list[str] = []
        for index, forms in atoms:
            for form in forms:
                changed = list(lines); changed[index] = form
                text = ''.join(changed)
                if text != request.original_window and text not in pool:
                    pool.append(text)
        if len(atoms) >= 2:
            for ordinal in range(128):
                number = ordinal; changed = list(lines)
                for index, forms in atoms:
                    number, digit = divmod(number, len(forms))
                    changed[index] = forms[digit]
                text = ''.join(changed)
                if text != request.original_window and text not in pool:
                    pool.append(text)
                if len(pool) >= 18:
                    break
        return tuple(pool or [request.original_window])

    def _parsed_variant(self, request: RewriteRequest, index: int, texts: tuple[str, ...]) -> ParsedRewrite:
        text = texts[(index - 1) % len(texts)]
        return _parse_generation(request, RewriteGeneration((), text,
            'public-java-structural:'+hashlib.sha256(text.encode('utf-8')).hexdigest()[:16],
            f'java-structural-expressions/v1:{index}'), extractor=self.extractor,
            window_contract_version=self.contract)

    def rewrite_windows(self, request: RewriteRequest, *, candidate_indices: tuple[int, ...]) -> tuple[ParsedRewrite, ...]:
        _validate_candidate_indices(candidate_indices)
        for index in candidate_indices:
            _validate_call(request, index)
        texts = self._variant_texts(request)
        return tuple(self._parsed_variant(request, index, texts) for index in candidate_indices)

    def rewrite_window(self, request: RewriteRequest, *, candidate_index: int) -> ParsedRewrite:
        _validate_call(request, candidate_index)
        return self._parsed_variant(request, candidate_index, self._variant_texts(request))

    def rewrite_many(self, request: RewriteRequest, *, candidate_indices: tuple[int, ...]) -> tuple[RewriteCandidate, ...]:
        return tuple(self._candidate(request, parsed) for parsed in self.rewrite_windows(
            request, candidate_indices=candidate_indices))

    def rewrite(self, request: RewriteRequest, *, candidate_index: int) -> RewriteCandidate:
        return self._candidate(request, self.rewrite_window(request, candidate_index=candidate_index))

    @staticmethod
    def _candidate(request: RewriteRequest, parsed: ParsedRewrite) -> RewriteCandidate:
        first = len(request.completed_prefix.encode('utf-8'))
        return RewriteCandidate(code=parsed.text, parse_status=parsed.parse_status,
            unit_count=parsed.unit_count, same_parent_scope=parsed.same_parent_scope,
            boundary_span=(first, first + len(parsed.text.encode('utf-8'))),
            generation_seed_id=parsed.generation_seed_id, rewrite_config_id=parsed.rewrite_config_id)
