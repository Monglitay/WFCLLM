"""Workflow config and summary helpers for token-channel training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import os
from pathlib import Path
import random
from typing import Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from wfcllm.common.dataset_loader import load_reference_solutions
from wfcllm.watermark.token_channel.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.model import TokenChannelModel
from wfcllm.watermark.token_channel.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.train import build_token_channel_batch
from wfcllm.watermark.token_channel.train import build_training_evidence
from wfcllm.watermark.token_channel.train import save_token_channel_training_artifacts
from wfcllm.watermark.token_channel.train import train_one_epoch
from wfcllm.watermark.token_channel.train_corpus import build_training_rows
from wfcllm.watermark.token_channel.train_corpus import load_training_cache
from wfcllm.watermark.token_channel.train_corpus import save_training_cache

SupportedTokenChannelDataset = Literal["humaneval", "mbpp"]


@dataclass(frozen=True)
class TokenChannelTrainWorkflowConfig:
    """Validated configuration surface for token-channel training workflows."""

    dataset: SupportedTokenChannelDataset
    dataset_path: Path
    lm_model_path: Path
    model_path: Path
    cache_path: Path
    context_width: int
    hidden_size: int
    batch_size: int
    epochs: int
    lr: float
    entropy_threshold: float
    diversity_threshold: int
    split_ratio: float
    seed: int

    def __post_init__(self) -> None:
        valid_datasets = {"humaneval", "mbpp"}
        if self.dataset not in valid_datasets:
            raise ValueError(f"dataset must be one of {sorted(valid_datasets)}")

        dataset_path = _coerce_path_like(self.dataset_path, "dataset_path")
        lm_model_path = _coerce_path_like(self.lm_model_path, "lm_model_path")
        model_path = _coerce_path_like(self.model_path, "model_path")
        cache_path = _coerce_path_like(self.cache_path, "cache_path")
        context_width = _coerce_int(self.context_width, "context_width")
        hidden_size = _coerce_int(self.hidden_size, "hidden_size")
        batch_size = _coerce_int(self.batch_size, "batch_size")
        epochs = _coerce_int(self.epochs, "epochs")
        diversity_threshold = _coerce_int(self.diversity_threshold, "diversity_threshold")
        seed = _coerce_int(self.seed, "seed")
        lr = _coerce_finite_float(self.lr, "lr")
        entropy_threshold = _coerce_finite_float(self.entropy_threshold, "entropy_threshold")
        split_ratio = _coerce_finite_float(self.split_ratio, "split_ratio")

        object.__setattr__(self, "dataset_path", dataset_path)
        object.__setattr__(self, "lm_model_path", lm_model_path)
        object.__setattr__(self, "model_path", model_path)
        object.__setattr__(self, "cache_path", cache_path)
        object.__setattr__(self, "context_width", context_width)
        object.__setattr__(self, "hidden_size", hidden_size)
        object.__setattr__(self, "batch_size", batch_size)
        object.__setattr__(self, "epochs", epochs)
        object.__setattr__(self, "diversity_threshold", diversity_threshold)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "lr", lr)
        object.__setattr__(self, "entropy_threshold", entropy_threshold)
        object.__setattr__(self, "split_ratio", split_ratio)

        if not self.lm_model_path.exists():
            raise ValueError("lm_model_path must exist")
        if self.context_width <= 0:
            raise ValueError("context_width must be > 0")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if self.epochs <= 0:
            raise ValueError("epochs must be > 0")
        if self.lr <= 0:
            raise ValueError("lr must be > 0")
        if self.entropy_threshold < 0:
            raise ValueError("entropy_threshold must be >= 0")
        if self.diversity_threshold < 1:
            raise ValueError("diversity_threshold must be >= 1")
        if not 0 < self.split_ratio < 1:
            raise ValueError("split_ratio must be between 0 and 1")


def _coerce_path_like(value: str | os.PathLike[str], field_name: str) -> Path:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be path-like")
    try:
        return Path(value)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be path-like") from exc


def _coerce_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _coerce_finite_float(value: object, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a finite number")
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite number") from exc
    if not math.isfinite(coerced):
        raise ValueError(f"{field_name} must be a finite number")
    return coerced


@dataclass(frozen=True)
class TokenChannelTrainWorkflowSummary:
    """Readable summary of token-channel training inputs and outputs."""

    dataset: str
    training_rows: int
    train_rows: int
    validation_rows: int
    artifact_dir: Path
    cache_path: Path
    compatibility_ok: bool
    epochs: tuple[TokenChannelEpochMetrics, ...]
    switch_target_positive_count: int
    switch_target_negative_count: int


def format_token_channel_train_workflow_summary(
    summary: TokenChannelTrainWorkflowSummary,
) -> list[str]:
    """Render workflow summary lines for CLI/log output."""

    lines = [
        f"dataset: {summary.dataset}",
        f"training_rows: {summary.training_rows}",
        f"train_rows: {summary.train_rows}",
        f"validation_rows: {summary.validation_rows}",
        f"artifact_dir: {summary.artifact_dir}",
        f"cache_path: {summary.cache_path}",
        f"compatibility_ok: {'yes' if summary.compatibility_ok else 'no'}",
        f"switch_target_positive_count: {summary.switch_target_positive_count}",
        f"switch_target_negative_count: {summary.switch_target_negative_count}",
    ]
    for epoch in summary.epochs:
        lines.append(
            "epoch "
            f"{epoch.epoch}: train_loss={epoch.train_loss:.4f} "
            f"validation_loss={epoch.validation_loss:.4f} "
            f"switch_loss={epoch.switch_loss:.4f}"
        )
    return lines


def normalize_reference_solution_rows(
    rows: list[dict[str, object]],
    *,
    dataset: SupportedTokenChannelDataset,
) -> list[dict[str, str]]:
    """Normalize dataset-loader rows into token-channel training samples."""

    samples: list[dict[str, str]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("dataset row must be a mapping with generated_code")
        generated_code = row.get("generated_code")
        if not isinstance(generated_code, str) or not generated_code:
            raise ValueError("generated_code must be a non-empty string")
        if dataset == "humaneval":
            prompt = row.get("prompt")
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("prompt must be a non-empty string for humaneval rows")
            samples.append({"source_code": f"{prompt}{generated_code}"})
            continue
        samples.append({"source_code": generated_code})
    return samples


def split_training_rows(
    rows: list[dict[str, object]],
    *,
    split_ratio: float,
    seed: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Shuffle and split rows into disjoint train/validation sets."""

    if not 0 < split_ratio < 1:
        raise ValueError("split_ratio must be between 0 and 1")
    if len(rows) < 2:
        raise ValueError("training rows must include at least 2 rows")

    shuffled_rows = list(rows)
    random.Random(seed).shuffle(shuffled_rows)
    split_index = min(len(shuffled_rows) - 1, max(1, int(len(shuffled_rows) * split_ratio)))
    return shuffled_rows[:split_index], shuffled_rows[split_index:]


def run_token_channel_train_workflow(
    config: TokenChannelTrainWorkflowConfig,
) -> TokenChannelTrainWorkflowSummary:
    """Run the offline token-channel training workflow end to end."""

    reference_rows = load_reference_solutions(config.dataset, str(config.dataset_path))
    if not reference_rows:
        raise ValueError("reference solution rows must not be empty")
    samples = normalize_reference_solution_rows(reference_rows, dataset=config.dataset)
    tokenizer = AutoTokenizer.from_pretrained(config.lm_model_path)
    teacher_model = _load_teacher_model(config.lm_model_path)

    training_rows = build_training_rows(
        samples=samples,
        tokenizer=tokenizer,
        teacher_model=teacher_model,
        context_width=config.context_width,
        entropy_threshold=config.entropy_threshold,
        diversity_threshold=config.diversity_threshold,
    )
    if not training_rows:
        raise ValueError("training corpus rows must not be empty")

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)
    save_training_cache(config.cache_path, training_rows)
    cached_training_rows = load_training_cache(config.cache_path)
    if not cached_training_rows:
        raise ValueError("training corpus rows must not be empty")

    train_rows, validation_rows = split_training_rows(
        cached_training_rows,
        split_ratio=config.split_ratio,
        seed=config.seed,
    )
    train_batches = [
        build_token_channel_batch(batch_rows, context_width=config.context_width)
        for batch_rows in _chunk_rows(train_rows, batch_size=config.batch_size)
    ]
    validation_batches = [
        build_token_channel_batch(batch_rows, context_width=config.context_width)
        for batch_rows in _chunk_rows(validation_rows, batch_size=config.batch_size)
    ]

    _seed_training_runtime(config.seed)
    model = TokenChannelModel(
        vocab_size=_tokenizer_vocab_size(tokenizer),
        context_width=config.context_width,
        hidden_size=config.hidden_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

    epochs: list[TokenChannelEpochMetrics] = []
    for epoch in range(1, config.epochs + 1):
        epochs.append(
            train_one_epoch(
                model=model,
                optimizer=optimizer,
                train_batches=train_batches,
                validation_batches=validation_batches,
                epoch=epoch,
            )
        )

    training_evidence = build_training_evidence(rows=cached_training_rows, epochs=epochs)
    metadata = _build_training_artifact_metadata(config=config, tokenizer=tokenizer)

    config.model_path.mkdir(parents=True, exist_ok=True)
    export_paths = save_token_channel_training_artifacts(
        checkpoint_dir=config.model_path,
        model=model,
        metadata=metadata,
        evidence=training_evidence,
    )
    _require_artifact_outputs(export_paths)

    artifact = load_token_channel_artifact(config.model_path)
    require_token_channel_compatibility(
        artifact.metadata,
        tokenizer_name=metadata["tokenizer_name"],
        tokenizer_vocab_size=metadata["tokenizer_vocab_size"],
        context_width=config.context_width,
        feature_version=metadata["feature_version"],
    )
    return TokenChannelTrainWorkflowSummary(
        dataset=config.dataset,
        training_rows=len(cached_training_rows),
        train_rows=len(train_rows),
        validation_rows=len(validation_rows),
        artifact_dir=config.model_path,
        cache_path=config.cache_path,
        compatibility_ok=True,
        epochs=tuple(epochs),
        switch_target_positive_count=training_evidence.switch_target_positive_count,
        switch_target_negative_count=training_evidence.switch_target_negative_count,
    )


def _seed_training_runtime(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_training_artifact_metadata(
    *,
    config: TokenChannelTrainWorkflowConfig,
    tokenizer: object,
) -> dict[str, object]:
    return {
        "schema_version": "token-channel/v1",
        "tokenizer_name": _tokenizer_name(tokenizer, config.lm_model_path),
        "tokenizer_vocab_size": _tokenizer_vocab_size(tokenizer),
        "context_width": config.context_width,
        "feature_version": FEATURE_VERSION,
        "training_config": {
            "dataset": config.dataset,
            "dataset_path": str(config.dataset_path),
            "cache_path": str(config.cache_path),
            "hidden_size": config.hidden_size,
            "batch_size": config.batch_size,
            "epochs": config.epochs,
            "lr": config.lr,
            "entropy_threshold": config.entropy_threshold,
            "diversity_threshold": config.diversity_threshold,
            "split_ratio": config.split_ratio,
            "seed": config.seed,
            "model_path": str(config.model_path),
        },
    }


def _chunk_rows(
    rows: list[dict[str, object]],
    *,
    batch_size: int,
) -> list[list[dict[str, object]]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    return [rows[index : index + batch_size] for index in range(0, len(rows), batch_size)]


def _require_artifact_outputs(export_paths: object) -> None:
    if not isinstance(export_paths, Mapping):
        raise ValueError("artifact export must return a mapping of output paths")
    for key in ("checkpoint_path", "metadata_path", "evidence_path"):
        value = export_paths.get(key)
        if not isinstance(value, (str, os.PathLike)):
            raise ValueError(f"artifact export must include {key}")
        if not Path(value).exists():
            raise ValueError(f"artifact export path does not exist: {key}")


def _load_teacher_model(lm_model_path: Path) -> object:
    model = AutoModelForCausalLM.from_pretrained(lm_model_path)
    model.eval()
    return model


def _tokenizer_name(tokenizer: object, fallback_path: Path) -> str:
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    return str(fallback_path)


def _tokenizer_vocab_size(tokenizer: object) -> int:
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size
    try:
        length = len(tokenizer)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("tokenizer must provide a positive vocab size") from exc
    if not isinstance(length, int) or length <= 0:
        raise ValueError("tokenizer must provide a positive vocab size")
    return length
