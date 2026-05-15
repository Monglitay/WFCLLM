"""Token-channel model, training, and runtime helpers."""

from __future__ import annotations

from wfcllm.watermark.token_channel.core.model import TokenChannelArtifact
from wfcllm.watermark.token_channel.core.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.core.model import TokenChannelCheckpointExport
from wfcllm.watermark.token_channel.core.model import TokenChannelLossWeights
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import TokenChannelModelOutput
from wfcllm.watermark.token_channel.core.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.training.trainer import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.training.trainer import TokenChannelTrainingEvidence
from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch
from wfcllm.watermark.token_channel.training.trainer import run_training_step
from wfcllm.watermark.token_channel.training.trainer import save_training_evidence
from wfcllm.watermark.token_channel.training.trainer import train_one_epoch

__all__ = [
    "TokenChannelArtifact",
    "TokenChannelArtifactMetadata",
    "TokenChannelCheckpointExport",
    "TokenChannelEpochMetrics",
    "TokenChannelLossWeights",
    "TokenChannelModel",
    "TokenChannelModelOutput",
    "TokenChannelTrainingEvidence",
    "build_token_channel_batch",
    "export_token_channel_checkpoint",
    "run_training_step",
    "save_training_evidence",
    "train_one_epoch",
]
