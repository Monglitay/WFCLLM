from __future__ import annotations

from wfcllm.dynamic_semantic.config import ContextConfig
from wfcllm.dynamic_semantic.context import DynamicContextExtractor


def _counter(text: str) -> int:
    return len(text.replace("\n", " ").split())


def _extractor(**overrides: int) -> DynamicContextExtractor:
    values = {
        "max_context_tokens": 256,
        "max_current_unit_tokens": 128,
    }
    values.update(overrides)
    return DynamicContextExtractor(ContextConfig(**values), token_counter=_counter)


def test_formatting_equivalent_code_has_identical_context_contract() -> None:
    left = "def f(x):\n    y=x+1\n    return y\n"
    right = "def f( x ):\n  y = x + 1\n  return (y)\n"

    left_result = _extractor().extract(left)
    right_result = _extractor().extract(right)

    assert left_result.parse_ok is True
    assert right_result.parse_ok is True
    assert [item.unit_id for item in left_result.contexts] == [
        item.unit_id for item in right_result.contexts
    ]
    assert [item.context_sha256 for item in left_result.contexts] == [
        item.context_sha256 for item in right_result.contexts
    ]
    assert [item.canonical_current for item in left_result.contexts] == [
        "y = x + 1",
        "return y",
    ]


def test_unit_identity_is_content_addressed_not_global_ordinal() -> None:
    base = _extractor().extract(
        "def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
    )
    inserted = _extractor().extract(
        "def f(x):\n    marker = 0\n    y = x + 1\n    z = y * 2\n    return z\n"
    )

    base_by_code = {item.canonical_current: item for item in base.contexts}
    inserted_by_code = {item.canonical_current: item for item in inserted.contexts}

    for code in ("y = x + 1", "z = y * 2", "return z"):
        assert base_by_code[code].unit_id == inserted_by_code[code].unit_id
    assert base_by_code["z = y * 2"].context_sha256 == inserted_by_code[
        "z = y * 2"
    ].context_sha256
    assert base_by_code["y = x + 1"].context_sha256 != inserted_by_code[
        "y = x + 1"
    ].context_sha256


def test_delete_and_reorder_preserve_content_ids() -> None:
    source = _extractor().extract(
        "def f():\n    a = 1\n    b = 2\n    c = 3\n    return a + b + c\n"
    )
    changed = _extractor().extract(
        "def f():\n    c = 3\n    a = 1\n    return a + c\n"
    )

    source_ids = {item.canonical_current: item.unit_id for item in source.contexts}
    changed_ids = {item.canonical_current: item.unit_id for item in changed.contexts}

    assert changed_ids["a = 1"] == source_ids["a = 1"]
    assert changed_ids["c = 3"] == source_ids["c = 3"]


def test_duplicate_unit_ids_are_all_erased() -> None:
    result = _extractor().extract(
        "def f():\n    x = 1\n    x = 1\n    return x\n"
    )

    assert [item.canonical_current for item in result.contexts] == ["return x"]
    assert result.erasure_counts == {"duplicate_unit_id": 2}


def test_parse_failure_is_code_level_erasure() -> None:
    result = _extractor().extract("def f(:\n")

    assert result.parse_ok is False
    assert result.contexts == ()
    assert result.erasure_counts == {"parse_failure": 1}


def test_unit_and_context_budgets_erase_without_truncating() -> None:
    unit_limited = _extractor(max_current_unit_tokens=3).extract(
        "def f():\n    value = first + second + third\n    return value\n"
    )
    context_limited = _extractor(
        max_context_tokens=5,
        max_current_unit_tokens=5,
    ).extract(
        "def f():\n    first = one + two\n    return first\n"
    )

    assert unit_limited.erasure_counts["current_unit_too_long"] == 1
    assert all("third" not in item.canonical_current for item in unit_limited.contexts)
    assert context_limited.erasure_counts["context_too_long"] >= 1


def test_context_contains_only_public_bounded_fields() -> None:
    result = _extractor().extract("def f(x):\n    y = x + 1\n    return y\n")
    context = result.contexts[1]

    assert context.role == "Return|FunctionDef|body"
    assert context.canonical_previous == "y = x + 1"
    assert "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3" in context.serialized
    assert "previous=y = x + 1" in context.serialized
    assert "current=return y" in context.serialized
    assert "ordinal" not in context.serialized.lower()
    assert "secret" not in context.serialized.lower()


def test_repeated_prefix_extraction_caches_public_token_counts() -> None:
    calls: list[str] = []

    def counter(text: str) -> int:
        calls.append(text)
        return len(text.split())

    extractor = DynamicContextExtractor(ContextConfig(), token_counter=counter)
    code = "def f():\n    x = 1\n    return x\n"
    extractor.extract(code)
    first_count = len(calls)
    extractor.extract(code)

    assert len(calls) == first_count
