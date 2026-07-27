"""Public non-Python equivalent variants used by encoder training."""
from __future__ import annotations

import pytest

from wfcllm.generation.window_rewriter import public_equivalent_variants


def test_cpp_variants_are_deduplicated_and_differ_from_original() -> None:
    variants = public_equivalent_variants("cpp", "return a + b;", 6)

    assert len(variants) >= 3
    assert len(set(variants)) == len(variants)
    assert "return a + b;" not in variants


def test_java_variants_are_deduplicated_and_differ_from_original() -> None:
    variants = public_equivalent_variants("java", "return a + b;", 3)

    assert len(variants) == 3
    assert len(set(variants)) == 3
    assert "return a + b;" not in variants


def test_java_condition_lines_produce_variants() -> None:
    variants = public_equivalent_variants("java", "if (a > b) {", 3)

    assert len(variants) == 3
    assert all(variant != "if (a > b) {" for variant in variants)


def test_count_bounds_the_result() -> None:
    assert len(public_equivalent_variants("cpp", "return a + b;", 2)) == 2


def test_unproductive_text_returns_empty_tuple() -> None:
    assert public_equivalent_variants("cpp", "break;", 3) == ()


def test_python_is_rejected_in_favor_of_transform_engine() -> None:
    with pytest.raises(ValueError, match="TransformEngine"):
        public_equivalent_variants("python", "return 1", 3)


def test_js_variants_are_dependency_light_and_deduplicated() -> None:
    variants = public_equivalent_variants("js", "return a + b;", 6)

    assert len(variants) == 6
    assert len(set(variants)) == 6
    assert "return a + b;" not in variants
    assert all("wfcllm-equivalent" in variant for variant in variants)


def test_invalid_count_is_rejected() -> None:
    with pytest.raises(ValueError, match="count"):
        public_equivalent_variants("cpp", "return a + b;", 0)


def test_empty_text_is_rejected() -> None:
    with pytest.raises(ValueError, match="text"):
        public_equivalent_variants("cpp", "   ", 3)
