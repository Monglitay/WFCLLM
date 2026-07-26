from __future__ import annotations

import pytest

from wfcllm.windowing import (
    GateScores,
    GateThresholds,
    WindowPartitioner,
    get_statement_unit_extractor,
    window_contract_for_language,
)


class _AlwaysClose:
    def predict(self, serialized_input: str) -> GateScores:
        return GateScores(0.9, 0.9, True, 0.0)


@pytest.mark.parametrize(
    ("language", "source", "expected_types"),
    [
        (
            "cpp",
            "int add(int a, int b) {\n  int c = a + b;\n  return c;\n}\n",
            {"function_definition", "declaration", "return_statement"},
        ),
        (
            "java",
            "class Main {\n  int add(int a, int b) {\n    int c = a + b;\n    return c;\n  }\n}\n",
            {"class_declaration", "method_declaration", "local_variable_declaration", "return_statement"},
        ),
    ],
)
def test_multilanguage_extractor_returns_ordered_nonoverlapping_units(
    language: str,
    source: str,
    expected_types: set[str],
) -> None:
    units = get_statement_unit_extractor(language).extract(source)

    assert expected_types.issubset({unit.node_type for unit in units})
    assert all(left.end_byte <= right.start_byte for left, right in zip(units, units[1:]))
    assert [unit.unit_id for unit in units] == [str(index) for index in range(len(units))]
    assert all(source.encode("utf-8")[unit.start_byte:unit.end_byte].decode("utf-8") == unit.text for unit in units)


@pytest.mark.parametrize("language", ["cpp", "java"])
def test_multilanguage_compound_headers_are_singleton_boundaries(language: str) -> None:
    source = (
        "int f() { if (true) { return 1; } return 0; }"
        if language == "cpp"
        else "class A { int f() { if (true) { return 1; } return 0; } }"
    )
    extractor = get_statement_unit_extractor(language)
    units = extractor.extract(source)

    headers = [unit for unit in units if unit.compound_header]
    assert headers
    assert all(unit.end_byte <= len(source.encode("utf-8")) for unit in headers)
    assert any(unit.eligible is False for unit in headers)


def test_cpp_for_initializer_declaration_does_not_overlap_for_header() -> None:
    source = (
        "bool f(vector<int> values) {\n"
        "  for (int i = 0; i < values.size(); i++) {\n"
        "    if (values[i] < 0) return true;\n"
        "  }\n"
        "  return false;\n"
        "}\n"
    )

    units = get_statement_unit_extractor("cpp").extract(source)

    assert all(left.end_byte <= right.start_byte for left, right in zip(units, units[1:]))
    assert not any(
        unit.node_type == "declaration" and "int i = 0" in unit.text
        for unit in units
    )


@pytest.mark.parametrize("language", ["cpp", "java"])
def test_partitioner_uses_language_window_contract(language: str) -> None:
    source = (
        "int f() { int x = 1; return x; }"
        if language == "cpp"
        else "class A { int f() { int x = 1; return x; } }"
    )
    contract = window_contract_for_language(language)
    partitioner = WindowPartitioner(
        predictor=_AlwaysClose(),
        thresholds=GateThresholds(0.4, 0.6, 0.5),
        tokenizer_counter=lambda text: len(text.split()),
        window_contract_version=contract,
    )

    result = partitioner.partition(get_statement_unit_extractor(language).extract(source))

    assert result.windows
    assert {window.parent_descriptor.contract_version for window in result.windows} == {contract}


def test_unknown_language_has_no_window_contract() -> None:
    with pytest.raises(ValueError, match="unsupported window language"):
        window_contract_for_language("rust")
