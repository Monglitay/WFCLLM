from __future__ import annotations

import importlib
import importlib.util

import pytest


def _module():
    spec = importlib.util.find_spec("wfcllm.datasets.mbpp_interface")
    assert spec is not None, "MBPP interface extraction module is missing"
    return importlib.import_module("wfcllm.datasets.mbpp_interface")


def test_extracts_unique_function_name_and_positional_arity() -> None:
    module = _module()

    interface = module.extract_mbpp_interface(
        [
            "assert calculate_total([1, 2], 3) == 6",
            "assert calculate_total([], 7) == 0",
        ]
    )

    assert interface is not None
    assert interface.function_name == "calculate_total"
    assert interface.positional_arities == (2,)
    assert interface.keyword_names == ()
    assert interface.helper_classes == ()


def test_prompt_exposes_pair_interface_without_test_values_or_assertions() -> None:
    module = _module()

    prompt, interface = module.build_mbpp_generation_prompt(
        "Find the maximum chain length from pairs.",
        [
            "assert max_chain_length([Pair(5, 24)], 4) == 3",
            "assert max_chain_length([Pair(15, 25)], 8) == 9",
        ],
    )

    assert interface is not None
    assert prompt.startswith("class Pair:")
    assert "self.a = a" in prompt
    assert "self.b = b" in prompt
    assert "def max_chain_length(arg1, arg2):" in prompt
    assert '"""Find the maximum chain length from pairs.' in prompt
    assert "Return the result instead of printing it." in prompt
    assert "assert" not in prompt.lower()
    for leaked_literal in ("5", "24", "4", "3", "15", "25", "8", "9"):
        assert leaked_literal not in prompt


def test_ambiguous_or_missing_primary_call_falls_back_to_natural_prompt() -> None:
    module = _module()
    task = "Return whether the supplied object has the requested property."

    prompt, interface = module.build_mbpp_generation_prompt(
        task,
        ["assert value.expected"],
    )

    assert interface is None
    assert prompt == task


def test_prompt_never_accepts_reference_code() -> None:
    module = _module()

    with pytest.raises(TypeError, match="reference_code"):
        module.build_mbpp_generation_prompt(
            "Calculate a scaled total.",
            ["assert calculate_total([5, 24], 4) == 116"],
            reference_code="def calculate_total(numbers, factor): return 116",
        )
