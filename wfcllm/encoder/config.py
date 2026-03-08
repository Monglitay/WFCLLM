"""Configuration for the semantic encoder pretraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """All hyperparameters and paths for encoder pretraining."""

    # Model
    model_name: str = "Salesforce/codet5-base"
    embed_dim: int = 128

    # LoRA (optional, default on)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["q", "v"])

    # Precision (optional, default BF16)
    use_bf16: bool = True

    # Data
    data_sources: list[str] = field(default_factory=lambda: ["mbpp", "humaneval"])
    max_seq_length: int = 256
    negative_ratio: float = 0.5  # fraction of hard negatives vs random negatives

    # Training
    lr: float = 8e-5
    batch_size: int = 256
    epochs: int = 10
    margin: float = 0.3
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3

    # DataLoader
    num_workers: int = 8
    pin_memory: bool = True

    # Paths
    checkpoint_dir: str = "data/checkpoints/encoder"
    results_dir: str = "data/results"
    local_model_dir: str = "data/models"      # 本地模型根目录（离线部署）
    local_dataset_dir: str = "data/datasets"  # 本地数据集根目录（离线部署）
