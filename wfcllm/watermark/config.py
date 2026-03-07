"""Configuration for the watermark embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WatermarkConfig:
    """All parameters for watermark embedding during code generation."""

    # Key
    secret_key: str

    # Encoder
    encoder_model_path: str = "Salesforce/codet5-base"
    encoder_embed_dim: int = 128
    encoder_device: str = "cuda"

    # Margin
    margin_base: float = 0.1
    margin_alpha: float = 0.05

    # Rejection sampling
    max_retries: int = 5
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50

    # Generation
    max_new_tokens: int = 512
    eos_token_id: int | None = None

    # Fallback
    enable_fallback: bool = True
