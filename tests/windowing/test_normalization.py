from __future__ import annotations

from dataclasses import replace
import token
import tokenize

import pytest

import wfcllm.windowing.normalization as normalization
from wfcllm.windowing.contracts import ParentDescriptor
from wfcllm.windowing.normalization import (
    WINDOW_NORMALIZATION_VERSION,
    normalize_unit_text,
)


def test_parent_descriptor_has_one_canonical_string() -> None:
    descriptor = ParentDescriptor(
        contract_version="python-statement-window/v1",
        ancestor_node_types=("module", "function_definition", "if_statement"),
        direct_parent_type="block",
        first_unit_ordinal=2,
        compound_header_role="body",
    )

    assert descriptor.canonical == (
        "python-statement-window/v1|module/function_definition/if_statement|"
        "parent=block|ordinal=2|role=body"
    )


def test_normalization_canonicalizes_newlines_and_nonsemantic_whitespace() -> None:
    lf_text = 'message = "你好  世界"\nnext_value = message\n'
    crlf_text = (
        '\r\nmessage = "你好  世界"  \t\r\n'
        "next_value = message\t\r\n\r\n"
    )

    assert WINDOW_NORMALIZATION_VERSION == "wfcllm-window-normalization/v1"
    assert normalize_unit_text(lf_text) == normalize_unit_text(crlf_text)
    assert normalize_unit_text(crlf_text) == (
        'message = "你好  世界"\nnext_value = message'
    )


def test_normalization_preserves_whitespace_inside_string_literals() -> None:
    source = 'value = "  keep   this  \t spacing  "  \n'

    assert normalize_unit_text(source) == 'value = "  keep   this  \t spacing  "'


@pytest.mark.parametrize(
    "literal",
    [
        "'单引号  \t 内容'",
        '"双引号  \t 内容"',
        '"""三引号第一行  \t\n三引号第二行"""',
        'f"""前缀第一行  \t\n前缀第二行 {name}"""',
    ],
)
def test_normalization_preserves_literal_whitespace_across_string_forms(
    literal: str,
) -> None:
    lf_source = f"value = {literal}  \t\nresult = value  \t\n"
    crlf_source = lf_source.replace("\n", "\r\n")
    expected = f"value = {literal}\nresult = value"

    assert normalize_unit_text(lf_source) == expected
    assert normalize_unit_text(crlf_source) == expected


def test_fstring_whole_token_preserves_expression_but_not_external_whitespace() -> None:
    lf_source = (
        'message = f"""第一行  \t\n'
        "第二行 {\n"
        "    name  \t\n"
        "} 后缀\n"
        '第三行"""  \t\n'
        "result = message  \t\n"
    )
    crlf_source = lf_source.replace("\n", "\r\n")
    expected = (
        'message = f"""第一行  \t\n'
        "第二行 {\n"
        "    name  \t\n"
        "} 后缀\n"
        '第三行"""\n'
        "result = message"
    )

    assert normalize_unit_text(lf_source) == expected
    assert normalize_unit_text(crlf_source) == expected


@pytest.mark.parametrize(
    ("fstring", "expected_fstring"),
    [
        (
            'f"""literal  \t\n{\n    value  \t\n} suffix\nend"""',
            'f"""literal  \t\n{\n    value  \t\n} suffix\nend"""',
        ),
        (
            'f"""{{escaped}} literal  \t\n{\n    value  \t\n}\nend"""',
            'f"""{{escaped}} literal  \t\n{\n    value  \t\n}\nend"""',
        ),
        (
            'f"""nested  \t\n{\n    call((left + right), [1, 2])  \t\n}\nend"""',
            'f"""nested  \t\n{\n    call((left + right), [1, 2])  \t\n}\nend"""',
        ),
        (
            'f"""mapping  \t\n{\n    render({"key": "{literal}"})  \t\n}\nend"""',
            'f"""mapping  \t\n{\n    render({"key": "{literal}"})  \t\n}\nend"""',
        ),
        (
            'f"""{value:\n>10  \t\n}"""',
            'f"""{value:\n>10  \t\n}"""',
        ),
        (
            'f"""{value!r:\nprefix  \t\n{width  \t\n}suffix  \t\n}"""',
            'f"""{value!r:\nprefix  \t\n{width  \t\n}suffix  \t\n}"""',
        ),
        (
            'f"""{value =  \t\n}"""',
            'f"""{value =  \t\n}"""',
        ),
    ],
)
def test_single_string_token_fstring_matches_native_tokenization(
    monkeypatch: pytest.MonkeyPatch,
    fstring: str,
    expected_fstring: str,
) -> None:
    source = f"message = {fstring}  \t\nresult = message  \t\n"
    native = normalize_unit_text(source)
    start_offset = source.index(fstring)
    end_offset = start_offset + len(fstring)
    whole_string_token = tokenize.TokenInfo(
        type=token.STRING,
        string=fstring,
        start=_offset_position(source, start_offset),
        end=_offset_position(source, end_offset),
        line=source.splitlines(keepends=True)[0],
    )
    monkeypatch.setattr(
        normalization.tokenize,
        "generate_tokens",
        lambda _readline: iter((whole_string_token,)),
    )

    normalized = normalize_unit_text(source)

    assert normalized == native == (
        f"message = {expected_fstring}\nresult = message"
    )


def test_single_string_token_fstring_preserves_expression_string_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fstring = 'f"""{len(\'\'\'inside  \t\nnext\'\'\')}"""'
    whole_string_token = tokenize.TokenInfo(
        type=token.STRING,
        string=fstring,
        start=(1, 0),
        end=_offset_position(fstring, len(fstring)),
        line=fstring.splitlines(keepends=True)[0],
    )
    monkeypatch.setattr(
        normalization.tokenize,
        "generate_tokens",
        lambda _readline: iter((whole_string_token,)),
    )

    normalized = normalize_unit_text(fstring)

    assert normalized == fstring
    assert eval(normalized) == eval(fstring)


def test_pep701_comment_keeps_all_fstring_internal_whitespace() -> None:
    source = 'f"""{(\n    value  # { } \'"\' ! : = marker  \t\n)}"""'
    namespace = {"value": 7}

    normalized = normalize_unit_text(source)

    assert normalized == source
    assert eval(normalized, namespace) == eval(source, namespace)


@pytest.mark.parametrize(
    "source",
    [
        'f"""{f\'{value}  \\t\'}  \t\n}"""',
        'f"""{len("inside  \\t")}  \t\n}"""',
        'f"""{len(\'\'\'inside  \t\nnext\'\'\')}"""',
    ],
)
def test_nested_fstring_and_expression_strings_preserve_whole_token(
    source: str,
) -> None:
    assert normalize_unit_text(source) == source


def test_format_spec_literal_is_preserved_semantically() -> None:
    source = 'f"""{value:\n>10  \t\n}"""'
    recorder = _FormatRecorder()

    original_result = eval(source, {"value": recorder})
    normalized_result = eval(normalize_unit_text(source), {"value": recorder})

    assert original_result == normalized_result == "formatted"
    assert recorder.specs == ["\n>10  \t\n", "\n>10  \t\n"]


def test_nested_format_field_preserves_complete_fstring_token() -> None:
    source = 'f"""{value!r:\nprefix  \t\n{width  \t\n}suffix  \t\n}"""'

    assert normalize_unit_text(source) == source


def test_debug_fstring_source_whitespace_is_semantic() -> None:
    source = 'f"""{value =  \t\n}"""'
    namespace = {"value": 7}
    normalized = normalize_unit_text(source)

    assert normalized == source
    assert eval(normalized, namespace) == eval(source, namespace)


def test_incomplete_triple_string_is_normalized_conservatively() -> None:
    source = 'value = """第一行  \t\r\n第二行  \t\r\n'

    assert normalize_unit_text(source) == 'value = """第一行  \t\n第二行  \t'


def test_parent_descriptor_rejects_negative_ordinal() -> None:
    descriptor = ParentDescriptor(
        contract_version="python-statement-window/v1",
        ancestor_node_types=("module",),
        direct_parent_type="block",
        first_unit_ordinal=0,
        compound_header_role="body",
    )

    with pytest.raises(ValueError, match="first_unit_ordinal must be non-negative"):
        replace(descriptor, first_unit_ordinal=-1)


def test_parent_descriptor_rejects_repeated_direct_parent_at_path_end() -> None:
    with pytest.raises(
        ValueError,
        match="direct parent must not be repeated in ancestor path",
    ):
        ParentDescriptor(
            contract_version="python-statement-window/v1",
            ancestor_node_types=("module", "function_definition", "block"),
            direct_parent_type="block",
            first_unit_ordinal=0,
            compound_header_role="body",
        )


def test_parent_descriptor_allows_earlier_repeated_parent_type() -> None:
    descriptor = ParentDescriptor(
        contract_version="python-statement-window/v1",
        ancestor_node_types=("module", "block", "if_statement"),
        direct_parent_type="block",
        first_unit_ordinal=0,
        compound_header_role="body",
    )

    assert descriptor.ancestor_node_types == ("module", "block", "if_statement")


def test_parent_descriptor_requires_tuple_ancestors() -> None:
    with pytest.raises(ValueError, match="ancestor_node_types must be a tuple"):
        ParentDescriptor(
            contract_version="python-statement-window/v1",
            ancestor_node_types=["module"],  # type: ignore[arg-type]
            direct_parent_type="block",
            first_unit_ordinal=0,
            compound_header_role="body",
        )


def test_parent_descriptor_requires_string_ancestors() -> None:
    with pytest.raises(
        ValueError,
        match="ancestor_node_types must contain only strings",
    ):
        ParentDescriptor(
            contract_version="python-statement-window/v1",
            ancestor_node_types=("module", 1),  # type: ignore[arg-type]
            direct_parent_type="block",
            first_unit_ordinal=0,
            compound_header_role="body",
        )


def _offset_position(text: str, offset: int) -> tuple[int, int]:
    before = text[:offset]
    line_number = before.count("\n") + 1
    line_start = before.rfind("\n") + 1
    return line_number, offset - line_start


class _FormatRecorder:
    def __init__(self) -> None:
        self.specs: list[str] = []

    def __format__(self, spec: str) -> str:
        self.specs.append(spec)
        return "formatted"
