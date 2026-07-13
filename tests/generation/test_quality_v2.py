from __future__ import annotations

import pytest

from wfcllm.generation.quality_v2 import assess_static_quality


PROMPT = '''def target(x: int, step: int = 1) -> int:
    """Return a transformed value."""
'''


def test_quality_gate_accepts_static_valid_completion() -> None:
    final_code = PROMPT + "    value = x + step\n    return value\n"

    quality = assess_static_quality(prompt=PROMPT, final_code=final_code)

    assert quality.eligible is True
    assert quality.syntax_valid is True
    assert quality.target_function_count == 1
    assert quality.signature_compatible is True
    assert quality.executable_body is True
    assert quality.quality_tier == 2


@pytest.mark.parametrize(
    ("suffix", "failed_field"),
    [
        ("    return (\n", "syntax_valid"),
        ("", "executable_body"),
        ("    pass\n", "executable_body"),
        ("    ...\n", "executable_body"),
        ("```python\n    return x\n```\n", "has_markdown_fence"),
    ],
)
def test_quality_gate_rejects_obvious_invalid_candidates(
    suffix: str,
    failed_field: str,
) -> None:
    quality = assess_static_quality(prompt=PROMPT, final_code=PROMPT + suffix)

    assert quality.eligible is False
    assert getattr(quality, failed_field) is False if failed_field != "has_markdown_fence" else True


def test_quality_gate_rejects_duplicate_target_function() -> None:
    final_code = (
        PROMPT
        + "    return x\n\n"
        + "def target(x: int, step: int = 1) -> int:\n    return x + step\n"
    )

    quality = assess_static_quality(prompt=PROMPT, final_code=final_code)

    assert quality.eligible is False
    assert quality.target_function_count == 2


def test_quality_gate_rejects_changed_signature() -> None:
    final_code = "def target(x):\n    return x\n"

    quality = assess_static_quality(prompt=PROMPT, final_code=final_code)

    assert quality.eligible is False
    assert quality.signature_compatible is False


def test_quality_gate_never_executes_candidate_code() -> None:
    final_code = PROMPT + "    raise RuntimeError('must not execute')\n"

    quality = assess_static_quality(prompt=PROMPT, final_code=final_code)

    assert quality.eligible is True
    assert quality.executable_body is True
