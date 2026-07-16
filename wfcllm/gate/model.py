"""Small dual-head model for semantic window close/suitable decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import nn

if TYPE_CHECKING:
    from wfcllm.gate.config import GateTrainConfig


@dataclass(frozen=True)
class GateModelOutput:
    close_logits: torch.Tensor
    suitable_logits: torch.Tensor


class GateModel(nn.Module):
    """Encoder plus masked-mean pooling and independent binary heads."""

    def __init__(self, *, encoder: nn.Module, hidden_size: int) -> None:
        super().__init__()
        if not isinstance(encoder, nn.Module):
            raise ValueError("encoder must be a torch.nn.Module")
        if type(hidden_size) is not int or hidden_size <= 0:
            raise ValueError("hidden_size must be a positive integer")
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.close_head = nn.Linear(hidden_size, 1)
        self.suitable_head = nn.Linear(hidden_size, 1)

    @classmethod
    def from_local_pretrained(cls, config: GateTrainConfig) -> GateModel:
        """Load the configured encoder from local files only.

        ``transformers`` is deliberately imported only when this formal loader
        is called, keeping fake/offline unit tests independent of model files.
        """

        from wfcllm.gate.config import GateTrainConfig

        if not isinstance(config, GateTrainConfig):
            raise ValueError("config must be a GateTrainConfig")

        from transformers import AutoModel

        encoder = AutoModel.from_pretrained(
            str(config.base_model_path), local_files_only=True
        )
        encoder_config = getattr(encoder, "config", None)
        hidden_size = None
        for name in ("hidden_size", "d_model", "dim"):
            candidate = getattr(encoder_config, name, None)
            if type(candidate) is int and candidate > 0:
                hidden_size = candidate
                break
        if hidden_size is None:
            raise ValueError("local encoder config must declare a positive hidden size")
        return cls(encoder=encoder, hidden_size=hidden_size)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> GateModelOutput:
        _validate_inputs(input_ids, attention_mask)
        encoded = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden = _last_hidden_state(encoded)
        expected_shape = (*input_ids.shape, self.hidden_size)
        if tuple(hidden.shape) != expected_shape:
            if hidden.ndim == 3 and hidden.shape[-1] != self.hidden_size:
                raise ValueError(
                    "encoder hidden size does not match configured hidden_size"
                )
            raise ValueError(
                f"encoder last_hidden_state shape must be {expected_shape!r}"
            )
        if not hidden.is_floating_point():
            raise ValueError("encoder last_hidden_state must use a floating dtype")
        if not torch.isfinite(hidden).all():
            raise ValueError("encoder last_hidden_state must be finite")

        mask = attention_mask.unsqueeze(-1).to(device=hidden.device, dtype=hidden.dtype)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        close_logits = self.close_head(pooled).squeeze(-1)
        suitable_logits = self.suitable_head(pooled).squeeze(-1)
        if not torch.isfinite(close_logits).all() or not torch.isfinite(
            suitable_logits
        ).all():
            raise ValueError("gate logits must be finite")
        return GateModelOutput(
            close_logits=close_logits,
            suitable_logits=suitable_logits,
        )


def _validate_inputs(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> None:
    if not isinstance(input_ids, torch.Tensor) or not isinstance(
        attention_mask, torch.Tensor
    ):
        raise ValueError("input_ids and attention_mask must be tensors")
    if input_ids.ndim != 2 or attention_mask.ndim != 2:
        raise ValueError("input_ids and attention_mask must be 2-D")
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids and attention_mask must have the same shape")
    if input_ids.shape[0] == 0 or input_ids.shape[1] == 0:
        raise ValueError("input tensors must have non-empty batch and sequence axes")
    if input_ids.dtype != torch.long:
        if not input_ids.dtype.is_floating_point and input_ids.dtype != torch.bool:
            raise ValueError("input_ids must use torch.long dtype")
        raise ValueError("input_ids must use an integer dtype (torch.long)")
    if attention_mask.dtype not in (torch.bool, torch.long):
        raise ValueError("attention_mask must use torch.bool or torch.long dtype")
    if torch.any(input_ids < 0):
        raise ValueError("input_ids must be non-negative")
    if not torch.all((attention_mask == 0) | (attention_mask == 1)):
        raise ValueError("attention_mask must be binary")


def _last_hidden_state(encoded: Any) -> torch.Tensor:
    hidden = (
        encoded.get("last_hidden_state")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "last_hidden_state", None)
    )
    if not isinstance(hidden, torch.Tensor):
        raise ValueError("encoder output must provide tensor last_hidden_state")
    return hidden
