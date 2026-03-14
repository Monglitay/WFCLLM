"""Configuration for the watermark embedding pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WatermarkConfig:
    """All parameters for watermark embedding during code generation."""

    # Key
    secret_key: str

    # Encoder
    encoder_model_path: str = "data/models/codet5-base"  # 本地路径优先；可传 HF Hub ID 作回退
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

    # Repetition penalty for retry sub-loop
    repetition_penalty: float = 1.3  # 1.0 = disabled; applied to previous retry's tokens

    # LSH parameters
    lsh_d: int = 3
    lsh_gamma: float = 0.5

    # Cascade fallback (compound block re-generation)
    enable_cascade: bool = True
    cascade_max_depth: int = 1

    # Memory management
    cuda_empty_cache_interval: int = 10  # call empty_cache() every N rollbacks

    # Retry budget
    retry_token_budget: int | None = None  # max tokens per retry attempt; None = max_new_tokens // 2
