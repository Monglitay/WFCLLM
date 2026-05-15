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

from wfcllm.datasets.loaders.local import load_reference_solutions
from wfcllm.watermark.token_channel.core.features import FEATURE_VERSION
from wfcllm.watermark.token_channel.core.model import TokenChannelModel
from wfcllm.watermark.token_channel.core.model import load_token_channel_artifact
from wfcllm.watermark.token_channel.core.model import load_training_state
from wfcllm.watermark.token_channel.core.model import require_token_channel_compatibility
from wfcllm.watermark.token_channel.training.trainer import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch
from wfcllm.watermark.token_channel.training.trainer import build_training_evidence
from wfcllm.watermark.token_channel.training.trainer import save_token_channel_training_artifacts
from wfcllm.watermark.token_channel.training.trainer import train_one_epoch
from wfcllm.watermark.token_channel.training.corpus import build_training_rows
from wfcllm.watermark.token_channel.training.corpus import load_training_cache
from wfcllm.watermark.token_channel.training.corpus import save_training_cache_streaming
from wfcllm.watermark.token_channel.training.corpus_streaming import count_training_cache_rows
from wfcllm.watermark.token_channel.training.corpus_streaming import load_rows_by_indices
from wfcllm.watermark.token_channel.training.corpus_streaming import split_training_cache_streaming

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
    teacher_batch_size: int = 16
    max_variants: int | None = None
    top_k_logits: int | None = 100
    resume_training: bool = True  # Auto-resume if checkpoint exists

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
        teacher_batch_size = _coerce_int(self.teacher_batch_size, "teacher_batch_size")
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
        object.__setattr__(self, "teacher_batch_size", teacher_batch_size)
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
        if self.teacher_batch_size <= 0:
            raise ValueError("teacher_batch_size must be > 0")


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

    config.cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if cache already exists and is valid
    if config.cache_path.exists():
        print(f"[检测] 发现已存在的缓存: {config.cache_path}")
        try:
            # Try to validate cache by counting rows
            print(f"[验证] 检查缓存完整性...")
            total_rows = count_training_cache_rows(config.cache_path)
            if total_rows > 0:
                print(f"[复用] 缓存有效，包含 {total_rows} 条训练样本，跳过语料生成")
                # Skip corpus generation, go directly to training
            else:
                print(f"[警告] 缓存为空，重新生成")
                raise ValueError("Empty cache")
        except Exception as e:
            print(f"[警告] 缓存验证失败: {e}，重新生成")
            config.cache_path.unlink(missing_ok=True)
            total_rows = None
    else:
        total_rows = None

    # Generate corpus only if cache doesn't exist or is invalid
    if total_rows is None:
        teacher_model = _load_teacher_model(config.lm_model_path)

        # Use streaming to avoid memory buildup
        training_rows_iterator = build_training_rows(
            samples=samples,
            tokenizer=tokenizer,
            teacher_model=teacher_model,
            context_width=config.context_width,
            entropy_threshold=config.entropy_threshold,
            diversity_threshold=config.diversity_threshold,
            teacher_batch_size=config.teacher_batch_size,
            max_variants=config.max_variants,
            top_k_logits=config.top_k_logits,
        )

        print(f"[开始] 生成训练语料...")
        row_count = save_training_cache_streaming(config.cache_path, training_rows_iterator)
        print(f"[完成] 已写入 {row_count} 条训练样本到缓存")

        # Use streaming to avoid loading entire cache into memory
        print(f"[开始] 统计缓存行数...")
        total_rows = count_training_cache_rows(config.cache_path)
    if total_rows == 0:
        raise ValueError("training corpus rows must not be empty")
    print(f"[信息] 缓存包含 {total_rows} 条训练样本")

    print(f"[开始] 划分训练集和验证集...")
    train_indices, validation_indices = split_training_cache_streaming(
        config.cache_path,
        split_ratio=config.split_ratio,
        seed=config.seed,
    )
    print(f"[信息] 训练集: {len(train_indices)} 条, 验证集: {len(validation_indices)} 条")

    # Get vocab_size for batch building
    vocab_size = _tokenizer_vocab_size(tokenizer)

    # Build batches using streaming loader
    print(f"[开始] 构建训练批次（batch_size={config.batch_size}）...")
    train_batches = list(_build_batches_streaming(
        cache_path=config.cache_path,
        indices=train_indices,
        batch_size=config.batch_size,
        context_width=config.context_width,
        vocab_size=vocab_size,
    ))
    print(f"[完成] 训练批次: {len(train_batches)}")

    print(f"[开始] 构建验证批次...")
    validation_batches = list(_build_batches_streaming(
        cache_path=config.cache_path,
        indices=validation_indices,
        batch_size=config.batch_size,
        context_width=config.context_width,
        vocab_size=vocab_size,
    ))
    print(f"[完成] 验证批次: {len(validation_batches)}")

    _seed_training_runtime(config.seed)

    # Try to resume from existing checkpoint
    start_epoch = 1
    existing_model = None
    existing_optimizer_state = None

    if config.resume_training and config.model_path.exists():
        print(f"[检测] 发现已存在的模型目录: {config.model_path}")
        training_state = load_training_state(config.model_path)
        if training_state is not None:
            start_epoch = training_state['epoch'] + 1
            existing_optimizer_state = training_state['optimizer_state_dict']
            if start_epoch <= config.epochs:
                print(f"[恢复] 从 epoch {training_state['epoch']} 继续训练，将训练到 epoch {config.epochs}")
                try:
                    artifact = load_token_channel_artifact(config.model_path)
                    existing_model = artifact.model
                    print(f"[恢复] 成功加载模型权重")
                except Exception as e:
                    print(f"[警告] 加载模型失败: {e}，将从头开始训练")
                    start_epoch = 1
                    existing_model = None
                    existing_optimizer_state = None
            else:
                print(f"[信息] 已完成 {training_state['epoch']} 个 epoch，目标 {config.epochs} 个 epoch，无需继续训练")
                # Load existing artifact and return
                artifact = load_token_channel_artifact(config.model_path)
                # Return early with existing results
                print(f"[完成] 训练已完成")
                return TokenChannelTrainWorkflowSummary(
                    dataset=config.dataset,
                    training_rows=total_rows,
                    train_rows=len(train_indices),
                    validation_rows=len(validation_indices),
                    artifact_dir=config.model_path,
                    cache_path=config.cache_path,
                    compatibility_ok=True,
                    epochs=tuple(),  # No new epochs trained
                    switch_target_positive_count=0,
                    switch_target_negative_count=0,
                )

    print(f"[开始] 初始化模型（hidden_size={config.hidden_size}）...")
    if existing_model is not None:
        model = existing_model
        model.train()
    else:
        model = TokenChannelModel(
            vocab_size=vocab_size,
            context_width=config.context_width,
            hidden_size=config.hidden_size,
        )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    if existing_optimizer_state is not None:
        optimizer.load_state_dict(existing_optimizer_state)
        print(f"[恢复] 成功恢复 optimizer 状态")

    print(f"[信息] 模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print(f"\n[开始] 训练 epoch {start_epoch} 到 {config.epochs}...")
    epochs: list[TokenChannelEpochMetrics] = []
    for epoch in range(start_epoch, config.epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{config.epochs}")
        print(f"{'='*60}")
        epoch_metrics = train_one_epoch(
            model=model,
            optimizer=optimizer,
            train_batches=train_batches,
            validation_batches=validation_batches,
            epoch=epoch,
            total_epochs=config.epochs,
        )
        epochs.append(epoch_metrics)
        print(f"[Epoch {epoch}] train_loss={epoch_metrics.train_loss:.4f}, "
              f"val_loss={epoch_metrics.validation_loss:.4f}, "
              f"switch_loss={epoch_metrics.switch_loss:.4f}")

        # Save checkpoint after each epoch (for resuming)
        print(f"[保存] 保存 epoch {epoch} checkpoint...")
        from wfcllm.watermark.token_channel.core.model import export_token_channel_checkpoint
        metadata = _build_training_artifact_metadata(config=config, tokenizer=tokenizer)
        config.model_path.mkdir(parents=True, exist_ok=True)
        export_token_channel_checkpoint(
            checkpoint_dir=config.model_path,
            model=model,
            metadata=metadata,
            optimizer=optimizer,
            epoch=epoch,
        )

    print(f"\n[完成] 训练结束")

    # Load a sample of rows to build training evidence (avoid loading all)
    print(f"\n[开始] 构建训练证据...")
    evidence_sample_size = min(1000, total_rows)
    evidence_rows = list(_load_sample_rows(config.cache_path, evidence_sample_size))
    training_evidence = build_training_evidence(rows=evidence_rows, epochs=epochs)
    print(f"[信息] switch_target 正样本: {training_evidence.switch_target_positive_count}")
    print(f"[信息] switch_target 负样本: {training_evidence.switch_target_negative_count}")

    print(f"\n[开始] 保存模型产物...")
    metadata = _build_training_artifact_metadata(config=config, tokenizer=tokenizer)

    config.model_path.mkdir(parents=True, exist_ok=True)
    export_paths = save_token_channel_training_artifacts(
        checkpoint_dir=config.model_path,
        model=model,
        metadata=metadata,
        evidence=training_evidence,
    )
    _require_artifact_outputs(export_paths)
    print(f"[完成] 模型已保存到: {config.model_path}")

    print(f"\n[开始] 验证模型兼容性...")
    artifact = load_token_channel_artifact(config.model_path)
    require_token_channel_compatibility(
        artifact.metadata,
        tokenizer_name=metadata["tokenizer_name"],
        tokenizer_vocab_size=metadata["tokenizer_vocab_size"],
        context_width=config.context_width,
        feature_version=metadata["feature_version"],
    )
    print(f"[完成] 兼容性检查通过")
    return TokenChannelTrainWorkflowSummary(
        dataset=config.dataset,
        training_rows=total_rows,
        train_rows=len(train_indices),
        validation_rows=len(validation_indices),
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
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        lm_model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()
    return model


def _tokenizer_name(tokenizer: object, fallback_path: Path) -> str:
    name = getattr(tokenizer, "name_or_path", None)
    if isinstance(name, str) and name:
        return name
    return str(fallback_path)


def _tokenizer_vocab_size(tokenizer: object) -> int:
    # Prefer len(tokenizer) which includes special tokens
    # This matches what the model actually uses
    try:
        length = len(tokenizer)  # type: ignore[arg-type]
        if isinstance(length, int) and length > 0:
            return length
    except TypeError:
        pass

    # Fallback to vocab_size attribute
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if isinstance(vocab_size, int) and vocab_size > 0:
        return vocab_size

    raise ValueError("tokenizer must provide a positive vocab size")


def _build_batches_streaming(
    cache_path: Path,
    indices: list[int],
    batch_size: int,
    context_width: int,
    vocab_size: int,
) -> list[dict[str, object]]:
    """Build batches by streaming rows at specified indices.

    This function loads rows in chunks to avoid loading all rows at once.
    """
    from wfcllm.watermark.token_channel.training.trainer import build_token_channel_batch

    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if not indices:
        return []

    # Load rows in batches
    batch_rows = []
    for row in load_rows_by_indices(cache_path, indices):
        batch_rows.append(row)

        if len(batch_rows) >= batch_size:
            # Yield complete batch
            yield build_token_channel_batch(
                batch_rows,
                context_width=context_width,
                vocab_size=vocab_size,
            )
            batch_rows = []

    # Yield remaining rows as final batch
    if batch_rows:
        yield build_token_channel_batch(
            batch_rows,
            context_width=context_width,
            vocab_size=vocab_size,
        )


def _load_sample_rows(cache_path: Path, sample_size: int) -> list[dict[str, object]]:
    """Load a sample of rows for building training evidence."""
    from wfcllm.watermark.token_channel.training.corpus_streaming import stream_training_cache

    rows = []
    for i, row in enumerate(stream_training_cache(cache_path)):
        if i >= sample_size:
            break
        rows.append(row)
    return rows

