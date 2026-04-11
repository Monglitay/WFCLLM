"""Offline helpers and entrypoint for token-channel training assets."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
import json
from pathlib import Path

import torch

from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.model import TokenChannelLossWeights
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import export_token_channel_checkpoint
from wfcllm.watermark.token_channel.teacher import load_teacher_cache
from wfcllm.watermark.token_channel.train_corpus import load_training_cache

TRAINING_EVIDENCE_FILENAME = "training_evidence.json"


@dataclass(frozen=True)
class TokenChannelEpochMetrics:
    """Per-epoch evidence required by the token-channel training spec."""

    epoch: int
    train_loss: float
    validation_loss: float
    switch_loss: float

    def to_dict(self) -> dict[str, object]:
        return {
            "epoch": self.epoch,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "switch_loss": self.switch_loss,
        }


@dataclass(frozen=True)
class TokenChannelTrainingEvidence:
    """Training/validation evidence for Task 5 verification."""

    switch_target_positive_count: int
    switch_target_negative_count: int
    train_loss: float
    validation_loss: float
    epochs: tuple[TokenChannelEpochMetrics, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "switch_target_positive_count": self.switch_target_positive_count,
            "switch_target_negative_count": self.switch_target_negative_count,
            "train_loss": self.train_loss,
            "validation_loss": self.validation_loss,
            "epochs": [epoch.to_dict() for epoch in self.epochs],
        }


def build_token_channel_batch(
    rows: list[dict[str, object]],
    *,
    context_width: int,
    device: torch.device | str = "cpu",
) -> dict[str, torch.Tensor]:
    """Convert persisted training rows into padded tensor batches."""

    if context_width <= 0:
        raise ValueError("context_width must be > 0")
    if not rows:
        raise ValueError("rows must not be empty")

    prefix_tokens: list[list[int]] = []
    next_tokens: list[int] = []
    teacher_logits: list[list[float]] = []
    switch_targets: list[float] = []
    feature_rows: list[dict[str, object]] = []

    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row must be a mapping")
        prefix = _require_int_list(row.get("prefix_tokens"), "prefix_tokens")
        trimmed_prefix = prefix[-context_width:]
        padded_prefix = [0] * (context_width - len(trimmed_prefix)) + trimmed_prefix
        prefix_tokens.append(padded_prefix)
        next_tokens.append(_require_int(row.get("next_token"), "next_token"))
        teacher_logits.append(_require_numeric_list(row.get("teacher_logits"), "teacher_logits"))
        switch_targets.append(float(_require_binary_int(row.get("switch_target"), "switch_target")))
        feature_rows.append(TokenChannelFeatures.from_mapping(row).to_dict())

    return {
        "prefix_tokens": torch.tensor(prefix_tokens, dtype=torch.long, device=device),
        "next_token": torch.tensor(next_tokens, dtype=torch.long, device=device),
        "teacher_logits": torch.tensor(teacher_logits, dtype=torch.float32, device=device),
        "switch_target": torch.tensor(switch_targets, dtype=torch.float32, device=device),
        "feature_rows": feature_rows,
    }


def run_training_step(
    *,
    model: TokenChannelModel,
    optimizer: torch.optim.Optimizer,
    batch: dict[str, torch.Tensor],
    features: TokenChannelFeatures | None = None,
    loss_weights: TokenChannelLossWeights | None = None,
) -> dict[str, float]:
    """Run one optimizer step and return scalar loss terms."""

    model.train()
    optimizer.zero_grad()
    output = _forward_batch(model=model, batch=batch, features=features)
    losses = model.compute_loss(batch=batch, output=output, loss_weights=loss_weights)
    losses["total_loss"].backward()
    optimizer.step()
    return {name: float(value.detach().cpu().item()) for name, value in losses.items()}


def train_one_epoch(
    *,
    model: TokenChannelModel,
    optimizer: torch.optim.Optimizer,
    train_batches: Iterable[dict[str, torch.Tensor]],
    validation_batches: Iterable[dict[str, torch.Tensor]],
    epoch: int,
    features: TokenChannelFeatures | None = None,
    loss_weights: TokenChannelLossWeights | None = None,
) -> TokenChannelEpochMetrics:
    """Train and evaluate one epoch, returning validation evidence."""

    train_losses: list[float] = []
    switch_losses: list[float] = []
    for batch in train_batches:
        step_losses = run_training_step(
            model=model,
            optimizer=optimizer,
            batch=batch,
            features=features,
            loss_weights=loss_weights,
        )
        train_losses.append(step_losses["total_loss"])
        switch_losses.append(step_losses["switch_loss"])

    validation_losses = [
        evaluate_batch_loss(
            model=model,
            batch=batch,
            features=features,
            loss_weights=loss_weights,
        )["total_loss"]
        for batch in validation_batches
    ]
    if not train_losses:
        raise ValueError("train_batches must not be empty")
    if not validation_losses:
        raise ValueError("validation_batches must not be empty")
    return TokenChannelEpochMetrics(
        epoch=epoch,
        train_loss=sum(train_losses) / len(train_losses),
        validation_loss=sum(validation_losses) / len(validation_losses),
        switch_loss=sum(switch_losses) / len(switch_losses),
    )


def build_training_evidence(
    *,
    rows: list[dict[str, object]],
    epochs: Iterable[TokenChannelEpochMetrics],
) -> TokenChannelTrainingEvidence:
    """Aggregate persisted switch-target counts and epoch losses."""

    epoch_metrics = tuple(epochs)
    if not epoch_metrics:
        raise ValueError("epochs must not be empty")
    positive_count = 0
    negative_count = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("row must be a mapping")
        if "switch_target" not in row:
            raise ValueError("switch_target must be present")
        switch_target = _require_binary_int(row["switch_target"], "switch_target")
        if switch_target == 1:
            positive_count += 1
        else:
            negative_count += 1
    last_epoch = epoch_metrics[-1]
    return TokenChannelTrainingEvidence(
        switch_target_positive_count=positive_count,
        switch_target_negative_count=negative_count,
        train_loss=last_epoch.train_loss,
        validation_loss=last_epoch.validation_loss,
        epochs=epoch_metrics,
    )


def save_training_evidence(
    path: str | Path,
    evidence: TokenChannelTrainingEvidence,
) -> Path:
    """Persist training/validation evidence as JSON."""

    evidence_path = Path(path)
    evidence_path.write_text(
        json.dumps(evidence.to_dict(), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return evidence_path


def save_token_channel_training_artifacts(
    *,
    checkpoint_dir: str | Path,
    model: TokenChannelModel,
    metadata: dict[str, object],
    evidence: TokenChannelTrainingEvidence,
) -> dict[str, Path]:
    """Persist the trained model bundle and required evidence."""

    export = export_token_channel_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        metadata=metadata,
    )
    evidence_path = save_training_evidence(Path(checkpoint_dir) / TRAINING_EVIDENCE_FILENAME, evidence)
    return {
        "checkpoint_path": export.checkpoint_path,
        "metadata_path": export.metadata_path,
        "evidence_path": evidence_path,
    }


def build_parser() -> argparse.ArgumentParser:
    """Build the minimal Task 4 training CLI parser."""

    parser = argparse.ArgumentParser(description="Offline token-channel training loader")
    parser.add_argument("--corpus-cache", type=Path, help="Path to a saved training corpus cache")
    parser.add_argument("--teacher-cache", type=Path, help="Path to a saved teacher cache")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Load cached offline assets without running training loops yet."""

    parser = build_parser()
    args = parser.parse_args(argv)

    if args.corpus_cache is None and args.teacher_cache is None:
        parser.print_help()
        return 1

    if args.corpus_cache is not None:
        rows = load_training_cache(args.corpus_cache)
        print(f"Loaded {len(rows)} training rows from {args.corpus_cache}")

    if args.teacher_cache is not None:
        rows = load_teacher_cache(args.teacher_cache)
        print(f"Loaded {len(rows)} teacher rows from {args.teacher_cache}")

    return 0


def evaluate_batch_loss(
    *,
    model: TokenChannelModel,
    batch: dict[str, torch.Tensor],
    features: TokenChannelFeatures | None = None,
    loss_weights: TokenChannelLossWeights | None = None,
) -> dict[str, float]:
    """Evaluate one batch without mutating model parameters."""

    model.eval()
    with torch.no_grad():
        output = _forward_batch(model=model, batch=batch, features=features)
        losses = model.compute_loss(batch=batch, output=output, loss_weights=loss_weights)
    return {name: float(value.detach().cpu().item()) for name, value in losses.items()}


def _forward_batch(
    *,
    model: TokenChannelModel,
    batch: dict[str, torch.Tensor],
    features: TokenChannelFeatures | None,
) -> dict[str, torch.Tensor]:
    prefix_tokens = batch["prefix_tokens"]
    if prefix_tokens.ndim != 2:
        raise ValueError("prefix_tokens must be a 2D batch tensor")

    row_outputs = []
    feature_rows = batch.get("feature_rows")
    for index, prefix in enumerate(prefix_tokens):
        row_features = features or _feature_for_batch_row(feature_rows, index)
        row_outputs.append(model(prefix, row_features))
    return {
        "switch_logit": torch.stack([row.switch_logit for row in row_outputs]),
        "preference_logits": torch.stack([row.preference_logits for row in row_outputs]),
    }


def _feature_for_batch_row(feature_rows: object, index: int) -> TokenChannelFeatures:
    if not isinstance(feature_rows, list):
        raise ValueError("batch must include feature_rows when features are not provided")
    if index >= len(feature_rows):
        raise ValueError("feature_rows must match prefix_tokens batch size")
    return TokenChannelFeatures.from_mapping(feature_rows[index])


def _require_int_list(value: object, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of integers")
    result: list[int] = []
    for item in value:
        result.append(_require_int(item, field_name))
    return result


def _require_numeric_list(value: object, field_name: str) -> list[float]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of numbers")
    result: list[float] = []
    for item in value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError(f"{field_name} must be a list of numbers")
        result.append(float(item))
    return result


def _require_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_binary_int(value: object, field_name: str) -> int:
    integer_value = _require_int(value, field_name)
    if integer_value not in {0, 1}:
        raise ValueError(f"{field_name} must be 0 or 1")
    return integer_value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a boolean")
    return value


def _require_string(value: object, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string")
    return value


if __name__ == "__main__":
    raise SystemExit(main())
