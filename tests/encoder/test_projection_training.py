"""Language-aware group preparation for the per-dataset semantic projection."""
from __future__ import annotations

import pytest

from wfcllm.encoder import projection_training
from wfcllm.gate.production import GateSourceCatalogRecord


_PY_CODE = """def accumulate(values):
    total = 0
    for value in values:
        total = total + value
    if len(values) > 0:
        return total
    return 0
"""

_CPP_CODE = (
    "int add(int a, int b) {\n"
    "    int total = a + b;\n"
    "    return total;\n"
    "}\n"
)

_JAVA_CODE = (
    "class Adder {\n"
    "    int add(int a, int b) {\n"
    "        int total = a + b;\n"
    "        return total;\n"
    "    }\n"
    "}\n"
)

_JS_CODE = "function add(a, b) {\n    return a + b;\n}\n"

_SOURCE_FAMILIES = {
    "python": "oss_python",
    "cpp": "oss_cpp",
    "java": "oss_java",
    "js": "oss_js",
}


def _record(language: str, index: int, code: str) -> GateSourceCatalogRecord:
    return GateSourceCatalogRecord(
        source_family=_SOURCE_FAMILIES[language],
        source_id=f"{language}-source-{index}",
        code=code,
        repository_id=f"repo-{language}-{index}",
        task_id=None,
        function_id=None,
        source_model_id=None,
        license_id="mit",
        contract_or_hard_set=False,
        prompt="",
    )


def test_prepare_region_blocks_python_uses_transform_engine_variants() -> None:
    blocks = projection_training.prepare_region_blocks(
        (_record("python", 0, _PY_CODE),),
        language="python",
        max_variants=8,
        max_perm_len=2,
    )

    assert blocks
    assert all(block["source_id"] == "python-source-0" for block in blocks)
    productive = [
        block
        for block in blocks
        if len(set(block["positive_variants"]) - {block["source"]}) >= 3
    ]
    assert productive


@pytest.mark.parametrize(
    ("language", "code"),
    [("cpp", _CPP_CODE), ("java", _JAVA_CODE)],
)
def test_prepare_region_blocks_cpp_java_use_public_equivalent_variants(
    language: str, code: str
) -> None:
    blocks = projection_training.prepare_region_blocks(
        (_record(language, 0, code),),
        language=language,
        max_variants=8,
        max_perm_len=2,
    )

    by_source = {block["source"]: block for block in blocks}
    assert "return total;" in by_source
    variants = by_source["return total;"]["positive_variants"]
    assert len(variants) >= 3
    assert len(set(variants)) == len(variants)
    assert "return total;" not in variants


@pytest.mark.parametrize("language", ["js", "ruby"])
def test_prepare_region_blocks_rejects_languages_without_public_variants(
    language: str,
) -> None:
    with pytest.raises(ValueError, match="no public equivalent-variant generator"):
        projection_training.prepare_region_blocks(
            (_record("js", 0, _JS_CODE),),
            language=language,
            max_variants=8,
            max_perm_len=2,
        )


def test_js_rejection_explains_missing_encoder_positive_samples() -> None:
    with pytest.raises(ValueError, match="encoder positive samples"):
        projection_training.prepare_region_blocks(
            (_record("js", 0, _JS_CODE),),
            language="js",
            max_variants=8,
            max_perm_len=2,
        )


def test_built_group_counts_report_actual_counts_without_lowering_limits() -> None:
    records = tuple(_record("cpp", index, _CPP_CODE) for index in range(20))

    build = projection_training.build_projection_training_groups(
        records,
        language="cpp",
        max_variants=8,
        max_perm_len=2,
        seed=7,
        max_train_groups=1200,
        max_validation_groups=160,
        max_test_groups=160,
    )

    assert build.requested_group_limits == {
        "train": 1200,
        "validation": 160,
        "test": 160,
    }
    assert build.built_group_counts == {
        name: len(groups) for name, groups in build.split_groups.items()
    }
    assert 0 < build.built_group_counts["train"] < 1200
    assert build.built_group_counts["validation"] >= 1
    assert build.built_group_counts["test"] >= 1


def test_small_train_group_limit_truncates_and_is_reported_truthfully() -> None:
    records = tuple(_record("cpp", index, _CPP_CODE) for index in range(20))

    build = projection_training.build_projection_training_groups(
        records,
        language="cpp",
        max_variants=8,
        max_perm_len=2,
        seed=7,
        max_train_groups=2,
        max_validation_groups=160,
        max_test_groups=160,
    )

    assert build.requested_group_limits["train"] == 2
    assert build.built_group_counts["train"] == 2
    assert len(build.split_groups["train"]) == 2
