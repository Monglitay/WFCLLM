from __future__ import annotations

import pytest

from wfcllm.gate.data import _make_request
from wfcllm.generation.cpp_structural_rewriter import KeyBlindCppStructuralWindowRewriter
from wfcllm.windowing import get_statement_unit_extractor


@pytest.mark.parametrize(
    ("prefix", "window", "suffix"),
    [
        ("int f(int x){\n", "return x+1;", "\n}\n"),
        ("int f(int x){\n", "int y=x*3;\nreturn y;", "\n}\n"),
        ("int f(int x){\n", "if(x){", "\nreturn 1;\n}\nreturn 0;\n}\n"),
        ("int f(int x){\n", "return x+1;", "\n}\nint main(){int bad=;\n"),
        ("int f(int x){\n", "// original comment\nreturn x;", "\n}\n"),
    ],
)
def test_structural_trajectory_is_key_blind_scoped_and_not_certified(
    prefix: str, window: str, suffix: str,
) -> None:
    extractor = get_statement_unit_extractor("cpp")
    source = prefix + window + suffix
    units = tuple(extractor.extract(source))
    first, last = len(prefix.encode("utf-8")), len((prefix + window).encode("utf-8"))
    selected = tuple(u for u in units if first <= u.start_byte and u.end_byte <= last)
    request = _make_request(
        units, start_index=units.index(selected[0]), window=selected, prompt="Authored test",
        source_bytes=source.encode("utf-8"), window_contract_version="cpp-statement-window/v1",
    )
    rewriter = KeyBlindCppStructuralWindowRewriter(
        extractor=extractor, window_contract_version="cpp-statement-window/v1",
    )
    indices = tuple(range(1, 19))
    trajectory = rewriter.rewrite_windows(request, candidate_indices=indices)
    assert trajectory == tuple(
        rewriter.rewrite_window(request, candidate_index=i) for i in indices
    )
    assert any(row.text != window for row in trajectory)
    assert all(row.parse_status == "ok" and row.same_parent_scope for row in trajectory)
    assert all(row.semantic_validation_rule is None for row in trajectory)
    assert all(row.semantic_equivalence_certified is None for row in trajectory)
    assert all("wfcllm:" not in row.text for row in trajectory)
    candidates = rewriter.rewrite_many(request, candidate_indices=indices)
    assert tuple(row.code for row in candidates) == tuple(row.text for row in trajectory)


def test_class_return_rules_preserve_named_return_candidate() -> None:
    extractor = get_statement_unit_extractor("cpp")
    source = "struct Box{};Box f(){Box value;\nreturn value;\n}\n"
    units = tuple(extractor.extract(source))
    selected = tuple(u for u in units if u.text == "return value;")
    request = _make_request(
        units, start_index=units.index(selected[0]), window=selected, prompt="Authored test",
        source_bytes=source.encode("utf-8"), window_contract_version="cpp-statement-window/v1",
    )
    rewriter = KeyBlindCppStructuralWindowRewriter(
        extractor=extractor, window_contract_version="cpp-statement-window/v1",
    )
    rows = rewriter.rewrite_windows(request, candidate_indices=tuple(range(1, 19)))
    assert all(row.parse_status == "ok" for row in rows)
    assert all(
        row.text.replace("(", "").replace(")", "") == "return value;"
        for row in rows
    )
