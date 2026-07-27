"""Architecture configuration for the per-dataset semantic projection."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EncoderConfig:
    """Architecture fields shared by projection training and runtime loading."""

    # Model
    model_name: str = "Salesforce/codet5-base"
    embed_dim: int = 128
    pooling: str = "first"

    # LoRA (optional, default on)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["q", "v"])

    # Precision (optional, default BF16)
    use_bf16: bool = True

    max_seq_length: int = 256
