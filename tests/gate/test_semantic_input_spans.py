from __future__ import annotations

import pytest

from wfcllm.gate.data import _make_request
from wfcllm.gate.production import _gate_data_semantic_input_span
from wfcllm.semantic.window_lsh import canonical_semantic_window_text
from wfcllm.windowing import get_statement_unit_extractor


@pytest.mark.parametrize(("language", "prefix", "window", "suffix", "candidate"), [
    ("python", "", "value = 7", "\n", "# before\nvalue = 7\n# after"),
    ("cpp", "int f(int x){\n", "return x+1;", "\n}\n", "/* 开发 */ return x+1; /* after */"),
    ("java", "class Dev{int f(int x){\n", "return x+1;", "\n}}\n", "/* before */ return x+1; /* after */"),
    ("js", "function f(x){\n", "return x+1;", "\n}\n", "// before\nreturn x+1;\n// after"),
])
def test_gate_semantics_uses_exact_parser_span_and_preserves_raw_candidate(language, prefix, window, suffix, candidate):
    source = prefix + window + suffix
    extractor = get_statement_unit_extractor(language)
    units = tuple(extractor.extract(source))
    first, last = len(prefix.encode("utf-8")), len((prefix + window).encode("utf-8"))
    selected = tuple(u for u in units if first <= u.start_byte and u.end_byte <= last)
    request = _make_request(units, start_index=units.index(selected[0]), window=selected,
        prompt="Authored scoring contract example", source_bytes=source.encode("utf-8"),
        window_contract_version=f"{language}-statement-window/v1")
    original_candidate = candidate
    text, span = _gate_data_semantic_input_span(request, candidate, extractor=extractor,
        window_contract_version=f"{language}-statement-window/v1")
    assert candidate == original_candidate
    assert candidate.encode("utf-8")[span[0]:span[1]].decode("utf-8") == text
    assert canonical_semantic_window_text(text) == canonical_semantic_window_text(window)
    assert "before" not in text and "after" not in text and "开发" not in text


def test_semantic_span_keeps_comments_inside_the_scored_statement():
    source = "class Dev{int f(int x){\nreturn x+1;\n}}\n"
    extractor = get_statement_unit_extractor("java")
    units = tuple(extractor.extract(source))
    selected = next(u for u in units if u.text == "return x+1;")
    request = _make_request(units, start_index=units.index(selected), window=(selected,),
        prompt="Authored scoring contract example", source_bytes=source.encode("utf-8"),
        window_contract_version="java-statement-window/v1")
    text, _span = _gate_data_semantic_input_span(request, "return (x /* inside */ +1); /* outside */",
        extractor=extractor, window_contract_version="java-statement-window/v1")
    assert "/* inside */" in text
    assert "outside" not in text
