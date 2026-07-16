from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from wfcllm.generation.generator import GeneratedToken


def test_generated_token_is_public_and_immutable() -> None:
    token = GeneratedToken(token_id=7, token_text="value")

    assert token.token_id == 7
    assert token.token_text == "value"
    with pytest.raises(FrozenInstanceError):
        token.token_id = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    ("token_id", "token_text"),
    [(True, "x"), (-1, "x"), (1, 3)],
)
def test_generated_token_rejects_malformed_values(
    token_id: object,
    token_text: object,
) -> None:
    with pytest.raises(ValueError):
        GeneratedToken(token_id=token_id, token_text=token_text)  # type: ignore[arg-type]
