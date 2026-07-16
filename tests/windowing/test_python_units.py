from __future__ import annotations

from dataclasses import replace

import pytest

from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    PythonStatementUnitExtractor,
    StatementUnit,
)


def texts(source: str) -> list[str]:
    return [unit.text for unit in PythonStatementUnitExtractor().extract(source)]


def test_semicolon_and_inline_if_are_parser_units() -> None:
    assert texts("def f(c):\n    x = 1; y = 2\n    if c: a = 3; b = 4\n") == [
        "def f(c):",
        "x = 1",
        "y = 2",
        "if c:",
        "a = 3",
        "b = 4",
    ]


def test_parenthesized_expression_is_not_split() -> None:
    assert texts("def f(a, b):\n    x = (a +\n         b)\n    return x\n") == [
        "def f(a, b):",
        "x = (a +\n         b)",
        "return x",
    ]


def test_line_metadata_is_one_based_for_multiline_and_inline_units() -> None:
    units = PythonStatementUnitExtractor().extract(
        "def f(c):\n"
        "    total = (1 +\n"
        "             2)\n"
        "    if c: first(); second()\n"
    )

    assert [(unit.text, unit.start_line, unit.end_line) for unit in units] == [
        ("def f(c):", 1, 1),
        ("total = (1 +\n             2)", 2, 3),
        ("if c:", 4, 4),
        ("first()", 4, 4),
        ("second()", 4, 4),
    ]


def test_return_is_eligible_but_fixed_exclusions_are_not() -> None:
    units = PythonStatementUnitExtractor().extract(
        "def f(x):\n    import os\n    assert x\n    return x\n"
    )
    assert [(unit.node_type, unit.eligible) for unit in units] == [
        ("function_definition", False),
        ("import_statement", False),
        ("assert_statement", False),
        ("return_statement", True),
    ]


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "if a:\n    x = 1\nelif b:\n    x = 2\nelse:\n    x = 3\n",
            ["if a:", "x = 1", "elif b:", "x = 2", "else:", "x = 3"],
        ),
        (
            "for x in xs:\n    use(x)\nwhile ready:\n    tick()\n",
            ["for x in xs:", "use(x)", "while ready:", "tick()"],
        ),
        (
            "try:\n    work()\nexcept ValueError as exc:\n    recover(exc)\nfinally:\n    close()\n",
            [
                "try:",
                "work()",
                "except ValueError as exc:",
                "recover(exc)",
                "finally:",
                "close()",
            ],
        ),
        (
            "with open(path) as stream:\n    consume(stream)\n",
            ["with open(path) as stream:", "consume(stream)"],
        ),
        (
            "match value:\n    case 0:\n        zero()\n    case other:\n        use(other)\n",
            ["match value:", "case 0:", "zero()", "case other:", "use(other)"],
        ),
        (
            "async for item in items:\n    await consume(item)\nasync with lock:\n    await commit()\n",
            [
                "async for item in items:",
                "await consume(item)",
                "async with lock:",
                "await commit()",
            ],
        ),
    ],
)
def test_compound_statements_and_clauses_have_header_units(
    source: str, expected: list[str]
) -> None:
    assert texts(source) == expected


@pytest.mark.parametrize(
    "statement",
    [
        "call(arg)",
        "yield item",
        "yield from items",
        "await call()",
        "return item",
    ],
)
def test_call_yield_await_and_return_are_eligible(statement: str) -> None:
    source = f"async def f():\n    {statement}\n"
    units = PythonStatementUnitExtractor().extract(source)
    assert units[-1].text == statement
    assert units[-1].eligible is True


@pytest.mark.parametrize(
    ("source", "node_type"),
    [
        ("def f():\n    return 1\n", "function_definition"),
        ("async def f():\n    return 1\n", "function_definition"),
        ("class C:\n    value = 1\n", "class_definition"),
    ],
)
def test_function_and_class_headers_are_ineligible_hard_boundaries(
    source: str, node_type: str
) -> None:
    header = PythonStatementUnitExtractor().extract(source)[0]
    assert header.node_type == node_type
    assert header.eligible is False
    assert header.hard_boundary is True
    assert header.compound_header is True


def test_nested_units_record_parent_path_depth_and_sibling_ordinal() -> None:
    units = PythonStatementUnitExtractor().extract(
        "def outer(flag):\n"
        "    before = 0\n"
        "    if flag:\n"
        "        first = 1\n"
        "        second = 2\n"
        "    after = 3\n"
    )
    by_text = {unit.text: unit for unit in units}

    assert by_text["def outer(flag):"].parent_path == ("module",)
    assert by_text["def outer(flag):"].direct_parent_type == "module"
    assert by_text["def outer(flag):"].depth == 0
    assert by_text["before = 0"].parent_path == (
        "module",
        "function_definition",
        "block",
    )
    assert by_text["before = 0"].direct_parent_type == "block"
    assert by_text["before = 0"].direct_child_ordinal == 0
    assert by_text["if flag:"].direct_child_ordinal == 1
    assert by_text["after = 3"].direct_child_ordinal == 2
    assert by_text["first = 1"].parent_path == (
        "module",
        "function_definition",
        "block",
        "if_statement",
        "block",
    )
    assert by_text["first = 1"].direct_parent_type == "block"
    assert by_text["first = 1"].depth == 2
    assert by_text["first = 1"].direct_child_ordinal == 0
    assert by_text["second = 2"].direct_child_ordinal == 1


@pytest.mark.parametrize(
    ("statement", "node_type"),
    [
        ("pass", "pass_statement"),
        ("break", "break_statement"),
        ("continue", "continue_statement"),
        ("raise RuntimeError", "raise_statement"),
        ("import os", "import_statement"),
        ("from os import path", "import_from_statement"),
        ("from __future__ import annotations", "future_import_statement"),
        ("global state", "global_statement"),
        ("nonlocal state", "nonlocal_statement"),
        ("del state", "delete_statement"),
        ("assert state", "assert_statement"),
    ],
)
def test_fixed_exclusions_are_retained_as_hard_boundaries(
    statement: str, node_type: str
) -> None:
    units = PythonStatementUnitExtractor().extract(
        f"def f():\n    {statement}\n    return 1\n"
    )
    excluded = units[1]

    assert excluded.text == statement
    assert excluded.node_type == node_type
    assert excluded.eligible is False
    assert excluded.hard_boundary is True
    assert excluded.direct_parent_type == "block"
    assert excluded.direct_child_ordinal == 0
    assert units[2].direct_child_ordinal == 1


def test_parser_recovery_fragment_is_retained_as_a_hard_boundary() -> None:
    units = PythonStatementUnitExtractor().extract(
        "def f():\n    @@@\n    return 1\n"
    )

    assert [(unit.node_type, unit.text) for unit in units] == [
        ("function_definition", "def f():"),
        ("ERROR", "@@@"),
        ("return_statement", "return 1"),
    ]
    recovery = units[1]
    assert recovery.parent_path == ("module", "function_definition")
    assert recovery.direct_parent_type == "function_definition"
    assert recovery.direct_child_ordinal == 0
    assert recovery.eligible is False
    assert recovery.hard_boundary is True


def test_zero_width_missing_node_marks_its_physical_header_ineligible() -> None:
    units = PythonStatementUnitExtractor().extract("if :\n    pass\n")

    assert [(unit.node_type, unit.text) for unit in units] == [
        ("if_statement", "if :"),
        ("pass_statement", "pass"),
    ]
    assert units[0].eligible is False
    assert units[0].hard_boundary is True


def test_incremental_compound_header_remains_an_eligible_singleton_candidate() -> None:
    units = PythonStatementUnitExtractor().extract("if ready:\n")

    assert len(units) == 1
    assert (
        units[0].eligible,
        units[0].hard_boundary,
        units[0].compound_header,
    ) == (True, False, True)


@pytest.mark.parametrize(
    ("definition", "header_type", "body_type", "body_text"),
    [
        (
            "def f():\n    return 1",
            "function_definition",
            "return_statement",
            "return 1",
        ),
        (
            "class C:\n    value = 1",
            "class_definition",
            "expression_statement",
            "value = 1",
        ),
    ],
)
def test_malformed_decorator_keeps_definition_header_as_a_hard_boundary(
    definition: str, header_type: str, body_type: str, body_text: str
) -> None:
    units = PythonStatementUnitExtractor().extract(
        f"before = 1\n@broken(\n{definition}\n"
    )

    assert [(unit.node_type, unit.text) for unit in units] == [
        ("expression_statement", "before = 1"),
        ("ERROR", "@broken("),
        (header_type, definition.splitlines()[0]),
        (body_type, body_text),
    ]
    assert units[0].eligible is True
    assert units[0].hard_boundary is False
    assert all(not unit.eligible and unit.hard_boundary for unit in units[1:3])


def test_expression_internal_error_is_not_promoted_to_a_statement_unit() -> None:
    units = PythonStatementUnitExtractor().extract("value = (1 + )\n")

    assert [(unit.node_type, unit.text) for unit in units] == [
        ("expression_statement", "value = (1 + )")
    ]
    assert units[0].eligible is False
    assert units[0].hard_boundary is True


def test_comments_and_blank_lines_do_not_create_units() -> None:
    assert texts("# module comment\n\nvalue = 1  # inline\n\n# trailing\n") == [
        "value = 1"
    ]


def test_utf8_byte_spans_slice_the_original_source() -> None:
    source = 'def f():\n    message = "你好"\n    return message\n'
    encoded = source.encode("utf-8")

    units = PythonStatementUnitExtractor().extract(source)

    assert [encoded[unit.start_byte : unit.end_byte].decode("utf-8") for unit in units] == [
        unit.text for unit in units
    ]
    assert units[1].text == 'message = "你好"'


def test_compound_headers_are_eligible_singleton_candidates() -> None:
    units = PythonStatementUnitExtractor().extract(
        "if ready:\n    run()\nelse:\n    wait()\n"
    )
    headers = [unit for unit in units if unit.compound_header]

    assert [unit.text for unit in headers] == ["if ready:", "else:"]
    assert all(unit.eligible for unit in headers)
    assert all(not unit.hard_boundary for unit in headers)
    assert all(unit.compound_header for unit in headers)


def test_function_name_limits_extraction_to_the_named_function() -> None:
    source = (
        "def first():\n"
        "    return 1\n"
        "\n"
        "def second():\n"
        "    return 2\n"
    )

    assert [
        unit.text
        for unit in PythonStatementUnitExtractor().extract(
            source, function_name="second"
        )
    ] == ["def second():", "return 2"]


def test_statement_unit_contract_rejects_empty_byte_span() -> None:
    assert WINDOW_CONTRACT_VERSION == "python-statement-window/v1"
    with pytest.raises(ValueError, match="non-empty byte span"):
        StatementUnit(
            unit_id="0",
            node_type="return_statement",
            text="return 1",
            start_byte=3,
            end_byte=3,
            start_line=1,
            end_line=1,
            depth=0,
            parent_path=("module",),
            direct_parent_type="module",
            direct_child_ordinal=0,
            eligible=True,
            hard_boundary=False,
            compound_header=False,
        )


def _valid_statement_unit() -> StatementUnit:
    return StatementUnit(
        unit_id="0",
        node_type="return_statement",
        text="return 1",
        start_byte=0,
        end_byte=8,
        start_line=1,
        end_line=1,
        depth=0,
        parent_path=("module",),
        direct_parent_type="module",
        direct_child_ordinal=0,
        eligible=True,
        hard_boundary=False,
        compound_header=False,
    )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"start_line": 0}, "start_line must be at least 1"),
        ({"start_line": 2, "end_line": 1}, "end_line must not precede start_line"),
        ({"depth": -1}, "depth must be non-negative"),
        ({"direct_child_ordinal": -1}, "direct_child_ordinal must be non-negative"),
        ({"parent_path": ()}, "parent_path must not be empty"),
        (
            {"parent_path": ("module", "block"), "direct_parent_type": "module"},
            "parent_path must end with direct_parent_type",
        ),
    ],
)
def test_statement_unit_contract_rejects_invalid_structure(
    changes: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        replace(_valid_statement_unit(), **changes)


@pytest.mark.parametrize(
    ("eligible", "hard_boundary"),
    [(True, True), (False, False)],
)
def test_statement_unit_contract_does_not_force_gate_state_complements(
    eligible: bool, hard_boundary: bool
) -> None:
    unit = replace(
        _valid_statement_unit(),
        eligible=eligible,
        hard_boundary=hard_boundary,
    )

    assert (unit.eligible, unit.hard_boundary) == (eligible, hard_boundary)
