from __future__ import annotations

from wfcllm.detection.canonical_v2 import (
    CANONICAL_UNIT_SCHEMA_VERSION,
    extract_canonical_units,
)


def test_extract_canonical_units_orders_and_deduplicates_owned_statements() -> None:
    code = '''
def helper():
    return 99

def target(x):
    total = 0
    if x:
        total = 1
    else:
        total = 1
    return total
'''

    extraction = extract_canonical_units(code)

    assert extraction.parse_valid is True
    assert extraction.target_function_name == "target"
    assert extraction.duplicate_count == 1
    assert [unit.ordinal for unit in extraction.units] == [0, 1, 2]
    assert [unit.normalized_text for unit in extraction.units] == [
        "total = 0",
        "total = 1",
        "return total",
    ]
    assert all(
        unit.schema_version == CANONICAL_UNIT_SCHEMA_VERSION
        for unit in extraction.units
    )
    assert len({unit.unit_id for unit in extraction.units}) == len(extraction.units)


def test_extract_canonical_units_is_stable_under_trailing_whitespace() -> None:
    compact = "def target(x):\n    value = x + 1\n    return value\n"
    padded = "def target(x):   \n    value = x + 1   \n    return value   \n\n"

    left = extract_canonical_units(compact)
    right = extract_canonical_units(padded)

    assert [unit.normalized_text_hash for unit in left.units] == [
        unit.normalized_text_hash for unit in right.units
    ]
    assert [unit.unit_id for unit in left.units] == [
        unit.unit_id for unit in right.units
    ]


def test_extract_canonical_units_returns_zero_evidence_for_malformed_code() -> None:
    extraction = extract_canonical_units("def target(:\n")

    assert extraction.parse_valid is False
    assert extraction.units == ()
    assert extraction.duplicate_count == 0


def test_extract_canonical_units_uses_last_top_level_function_without_prompt() -> None:
    code = "def first():\n    return 1\n\ndef second():\n    return 2\n"

    extraction = extract_canonical_units(code)

    assert extraction.target_function_name == "second"
    assert [unit.normalized_text for unit in extraction.units] == ["return 2"]
