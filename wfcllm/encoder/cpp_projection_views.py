"""Public C++ projection views from exact scored structural rewrite windows."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from wfcllm.gate.data import _enumerate_windows, _make_request
from wfcllm.generation.cpp_structural_rewriter import KeyBlindCppStructuralWindowRewriter
from wfcllm.semantic.window_lsh import canonical_semantic_window_text
from wfcllm.windowing import get_statement_unit_extractor


def prepare_cpp_runtime_region_blocks(records: Sequence[Any], *, max_variants: int) -> tuple[dict[str, Any], ...]:
    extractor = get_statement_unit_extractor('cpp')
    rewriter = KeyBlindCppStructuralWindowRewriter(
        extractor=extractor, window_contract_version='cpp-statement-window/v1')
    blocks: list[dict[str, Any]] = []
    for record in records:
        raw = record.code.encode('utf-8')
        units = tuple(extractor.extract(record.code))
        block_index = 0
        for start, unit in enumerate(units):
            if not unit.eligible or unit.hard_boundary:
                continue
            for window in _enumerate_windows(units, start):
                request = _make_request(units, start_index=start, window=window,
                    prompt='Public source-only projection training view', source_bytes=raw,
                    window_contract_version='cpp-statement-window/v1')
                original = canonical_semantic_window_text(request.original_window)
                positives: list[str] = []
                for index in range(1, 19):
                    candidate = rewriter.rewrite_window(request, candidate_index=index)
                    if (candidate.parse_status != 'ok' or not candidate.same_parent_scope
                            or candidate.unit_count != request.window_length or not candidate.parser_spans):
                        continue
                    completed = (request.completed_prefix + candidate.text + request.completed_suffix).encode('utf-8')
                    text = canonical_semantic_window_text(completed[
                        candidate.parser_spans[0][0]:candidate.parser_spans[-1][1]].decode('utf-8'))
                    if text != original and text not in positives:
                        positives.append(text)
                    if len(positives) >= max_variants:
                        break
                blocks.append(dict(source=original, source_id=record.source_id,
                    block_index=block_index, positive_variants=positives))
                block_index += 1
    return tuple(blocks)
