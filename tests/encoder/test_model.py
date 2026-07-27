"""Offline unit tests for the retained semantic encoder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from transformers import T5Config, T5EncoderModel

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder, masked_mean_pool


@pytest.fixture(autouse=True)
def tiny_offline_t5(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the heavyweight local model boundary with a tiny T5 fixture."""

    def load(_model_name: str, **kwargs: object) -> T5EncoderModel:
        model = T5EncoderModel(
            T5Config(
                d_model=64,
                d_ff=128,
                num_layers=1,
                num_heads=4,
                vocab_size=128,
            )
        )
        dtype = kwargs.get("torch_dtype")
        if isinstance(dtype, torch.dtype):
            model.to(dtype=dtype)
        return model

    monkeypatch.setattr(T5EncoderModel, "from_pretrained", load)


def _inputs(batch_size: int = 1) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([[1, 4, 8, 2]] * batch_size)
    attention_mask = torch.tensor([[1, 1, 1, 1]] * batch_size)
    return input_ids, attention_mask


def _config(
    *,
    use_lora: bool = False,
    use_bf16: bool = False,
    embed_dim: int = 128,
) -> EncoderConfig:
    return EncoderConfig(
        model_name="offline-tiny-t5",
        use_lora=use_lora,
        use_bf16=use_bf16,
        embed_dim=embed_dim,
    )


def test_masked_mean_pool_ignores_padding_tokens() -> None:
    hidden = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [100.0, 200.0]],
            [[5.0, 7.0], [9.0, 11.0], [13.0, 15.0]],
        ]
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]])

    pooled = masked_mean_pool(hidden, mask)

    assert torch.equal(pooled, torch.tensor([[2.0, 3.0], [5.0, 7.0]]))


def test_full_finetune_output_is_normalized_and_trainable() -> None:
    model = SemanticEncoder(_config())
    input_ids, attention_mask = _inputs(batch_size=3)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output.shape == (3, 128)
    assert torch.allclose(
        torch.norm(output, dim=1),
        torch.ones(3),
        atol=1e-5,
    )
    assert all(parameter.requires_grad for parameter in model.parameters())


def test_lora_freezes_most_encoder_parameters_and_is_deterministic() -> None:
    model = SemanticEncoder(_config(use_lora=True))
    input_ids, attention_mask = _inputs()
    model.eval()

    with torch.no_grad():
        first = model(input_ids, attention_mask)
        second = model(input_ids, attention_mask)

    assert first.shape == (1, 128)
    assert torch.allclose(first, second)
    encoder_parameters = list(model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in encoder_parameters)
    assert any(not parameter.requires_grad for parameter in encoder_parameters)


def test_bf16_encoder_returns_float32_projection() -> None:
    model = SemanticEncoder(_config(use_bf16=True, embed_dim=64))
    input_ids, attention_mask = _inputs()

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert next(model.encoder.parameters()).dtype == torch.bfloat16
    assert output.dtype == torch.float32
    assert output.shape == (1, 64)


@pytest.mark.parametrize(
    ("hidden", "mask", "message"),
    [
        (torch.zeros(2, 3), torch.ones(2, 3), "rank"),
        (torch.zeros(2, 3, 4), torch.ones(2, 2), "match"),
    ],
)
def test_masked_mean_pool_rejects_invalid_shapes(
    hidden: torch.Tensor,
    mask: torch.Tensor,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        masked_mean_pool(hidden, mask)
