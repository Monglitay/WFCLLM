"""Semantic encoder model for code representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

from wfcllm.encoder.config import EncoderConfig


class SemanticEncoder(nn.Module):
    """CodeT5 encoder with projection head for contrastive learning.

    Architecture: CodeT5 Encoder → [CLS] pooling → Linear → L2 normalize

    Supports optional LoRA (via peft) and BF16 precision, both configurable
    via EncoderConfig and enabled by default.
    """

    def __init__(self, config: EncoderConfig | None = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()
        self.config = config

        # Load encoder with optional BF16
        load_kwargs = {}
        if config.use_bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.encoder = T5EncoderModel.from_pretrained(
            config.model_name, **load_kwargs
        )
        hidden_size = self.encoder.config.d_model

        # Apply LoRA if enabled
        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        # Projection head (always float32 for stable cosine similarity)
        self.projection = nn.Linear(hidden_size, config.embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode input tokens into L2-normalized semantic vectors.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            (batch_size, embed_dim) L2-normalized float32 vectors.
        """
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Use first token ([CLS] equivalent) as sequence representation
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]
        # Cast to float32 before projection for numerical stability
        cls_hidden = cls_hidden.float()
        projected = self.projection(cls_hidden)
        return F.normalize(projected, p=2, dim=1)
