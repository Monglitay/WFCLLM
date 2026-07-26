"""Per-dataset training of the public semantic projection (gated runtime).

Moved out of ``scripts/train_gated_semantic_projection.py`` so the gated
``encoder`` phase can train one projection per dataset catalog.  Block
preparation is language aware: python uses the TransformEngine positive
rules, cpp/java use the public equivalent-variant entry shared with the
window rewriter, and languages without a public generator fail fast.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.region_training import (
    build_region_training_groups_from_blocks,
    semantic_region_diversity_loss,
    split_source_ids,
)
from wfcllm.semantic.lsh_space import LSHSpace


PUBLIC_PLANE_ID = "wfcllm-public-window-plane/v1"
OBJECTIVE_VERSION = "wfcllm-public-region-prototype/v2"
_SUPPORTED_LANGUAGES = ("python", "cpp", "java")


@dataclass(frozen=True)
class ProjectionTrainingSettings:
    """Inputs for one per-dataset semantic projection training run."""

    source_catalog: Path
    model_path: Path
    output_dir: Path
    language: str = "python"
    epochs: int = 10
    batch_size: int = 8
    max_length: int = 128
    max_variants: int = 8
    max_perm_len: int = 2
    max_train_groups: int = 1200
    max_validation_groups: int = 160
    max_test_groups: int = 160
    embed_dim: int = 128
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lr: float = 2e-4
    projection_lr: float = 1e-3
    weight_decay: float = 0.01
    num_workers: int = 2
    seed: int = 731


@dataclass(frozen=True)
class ProjectionGroupBuild:
    """Deterministic split groups plus truthful count bookkeeping."""

    source_splits: dict[str, tuple[str, ...]]
    split_groups: dict[str, tuple[dict[str, Any], ...]]
    block_count: int
    requested_group_limits: dict[str, int]
    built_group_counts: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "built_group_counts",
            {name: len(groups) for name, groups in self.split_groups.items()},
        )


def _require_supported_language(language: str) -> None:
    if language not in _SUPPORTED_LANGUAGES:
        raise ValueError(
            f"language {language!r} has no public equivalent-variant generator; "
            "encoder positive samples cannot be built "
            f"(supported: {', '.join(_SUPPORTED_LANGUAGES)})"
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: object) -> str:
    payload = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _python_region_blocks(
    records: Sequence[Any],
    *,
    max_variants: int,
    max_perm_len: int,
) -> tuple[dict[str, Any], ...]:
    from wfcllm.lang.python.parser import extract_statement_blocks
    from wfcllm.lang.python.transform.engine import TransformEngine
    from wfcllm.lang.python.transform.positive import get_all_positive_rules

    engine = TransformEngine(
        rules=get_all_positive_rules(),
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode="positive",
    )
    blocks: list[dict[str, Any]] = []
    for record in records:
        for block_index, statement_block in enumerate(
            extract_statement_blocks(record.code)
        ):
            variants = engine.generate_variants(statement_block.source)
            blocks.append(
                {
                    "source": statement_block.source,
                    "source_id": record.source_id,
                    "block_index": block_index,
                    "positive_variants": [
                        variant["transformed_source"] for variant in variants
                    ],
                }
            )
    return tuple(blocks)


def _public_variant_region_blocks(
    records: Sequence[Any],
    *,
    language: str,
    max_variants: int,
) -> tuple[dict[str, Any], ...]:
    from wfcllm.generation.window_rewriter import public_equivalent_variants

    if language == "cpp":
        from wfcllm.lang.cpp.parser import extract_statement_blocks
    else:
        from wfcllm.lang.java.parser import extract_statement_blocks

    blocks: list[dict[str, Any]] = []
    for record in records:
        for block_index, statement_block in enumerate(
            extract_statement_blocks(record.code)
        ):
            if not statement_block.source.strip():
                continue
            variants = public_equivalent_variants(
                language, statement_block.source, max_variants
            )
            blocks.append(
                {
                    "source": statement_block.source,
                    "source_id": record.source_id,
                    "block_index": block_index,
                    "positive_variants": list(variants),
                }
            )
    return tuple(blocks)


def prepare_region_blocks(
    records: Sequence[Any],
    *,
    language: str = "python",
    max_variants: int,
    max_perm_len: int,
) -> tuple[dict[str, Any], ...]:
    """Build structure-preserving training views from catalog code only."""

    _require_supported_language(language)
    if language == "python":
        return _python_region_blocks(
            records,
            max_variants=max_variants,
            max_perm_len=max_perm_len,
        )
    return _public_variant_region_blocks(
        records,
        language=language,
        max_variants=max_variants,
    )


def build_projection_training_groups(
    records: Sequence[Any],
    *,
    language: str = "python",
    max_variants: int,
    max_perm_len: int,
    seed: int,
    max_train_groups: int,
    max_validation_groups: int,
    max_test_groups: int,
) -> ProjectionGroupBuild:
    """Split sources and build groups without ever lowering requested limits."""

    source_splits = split_source_ids(
        [record.source_id for record in records],
        seed=seed,
    )
    blocks = prepare_region_blocks(
        records,
        language=language,
        max_variants=max_variants,
        max_perm_len=max_perm_len,
    )
    limits = {
        "train": max_train_groups,
        "validation": max_validation_groups,
        "test": max_test_groups,
    }
    split_groups: dict[str, tuple[dict[str, Any], ...]] = {}
    for split_name, split_ids in source_splits.items():
        allowed = set(split_ids)
        split_blocks = [block for block in blocks if block["source_id"] in allowed]
        split_groups[split_name] = build_region_training_groups_from_blocks(
            split_blocks,
            max_groups=limits[split_name],
            seed=seed,
        )
        if not split_groups[split_name]:
            raise ValueError(f"{split_name} has no region training groups")
    return ProjectionGroupBuild(
        source_splits=source_splits,
        split_groups=split_groups,
        block_count=len(blocks),
        requested_group_limits=limits,
    )


class RegionGroupDataset(Dataset):
    """Pre-tokenized anchor/three-positive/cross-source-negative groups."""

    def __init__(
        self,
        groups: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        max_length: int,
    ) -> None:
        self.groups = tuple(groups)
        texts: list[str] = []
        for group in self.groups:
            positives = group["positives"]
            texts.extend(
                [
                    group["anchor"],
                    positives[0],
                    positives[1],
                    positives[2],
                    group["negative"],
                ]
            )
        encoded = tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        self.input_ids = encoded["input_ids"].reshape(len(self.groups), 5, max_length)
        self.attention_mask = encoded["attention_mask"].reshape(
            len(self.groups), 5, max_length
        )

    def __len__(self) -> int:
        return len(self.groups)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": self.input_ids[index],
            "attention_mask": self.attention_mask[index],
        }


def _region_metrics(group_embeddings: torch.Tensor, planes: torch.Tensor) -> dict[str, float]:
    responses = torch.einsum("bve,de->bvd", group_embeddings.float(), planes.float())
    signatures = responses > 0
    distinct = [len({tuple(row.tolist()) for row in group}) for group in signatures]
    margins = responses.abs().amin(dim=-1)
    return {
        "mean_distinct_signatures": sum(distinct) / max(len(distinct), 1),
        "fraction_at_least_3_signatures": sum(value >= 3 for value in distinct)
        / max(len(distinct), 1),
        "mean_min_plane_margin": float(margins.mean()),
    }


def _run_epoch(
    *,
    model: SemanticEncoder,
    loader: DataLoader,
    planes: torch.Tensor,
    device: torch.device,
    optimizer: AdamW | None,
    scheduler: CosineAnnealingLR | None,
    use_bf16: bool,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals: dict[str, float] = {}
    batches = 0
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        batch_size, views, length = input_ids.shape
        with torch.set_grad_enabled(training):
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16 and device.type == "cuda",
            ):
                embeddings = model(
                    input_ids.reshape(batch_size * views, length),
                    attention_mask.reshape(batch_size * views, length),
                ).reshape(batch_size, views, -1)
                loss, metrics = semantic_region_diversity_loss(
                    embeddings[:, :4, :],
                    embeddings[:, 4, :],
                    planes,
                )
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                max_norm=1.0,
            )
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        row = {"loss": float(loss.detach()), **metrics}
        row.update(_region_metrics(embeddings[:, :4, :].detach(), planes))
        for name, value in row.items():
            totals[name] = totals.get(name, 0.0) + value
        batches += 1
    return {name: value / max(batches, 1) for name, value in totals.items()}


def train_semantic_projection(settings: ProjectionTrainingSettings) -> dict[str, Any]:
    """Train one per-dataset semantic projection and return its manifest."""

    from wfcllm.gate.production import load_source_catalog

    _require_supported_language(settings.language)
    catalog_path = Path(settings.source_catalog).resolve()
    model_path = Path(settings.model_path).resolve()
    output_dir = Path(settings.output_dir).resolve()
    if not catalog_path.is_file() or catalog_path.is_symlink():
        raise ValueError("source catalog must be a local non-symlink file")
    if not model_path.is_dir() or model_path.is_symlink():
        raise ValueError("model path must be a local non-symlink directory")
    if settings.epochs < 10:
        raise ValueError("formal semantic projection training requires at least 10 epochs")
    output_dir.mkdir(parents=True, exist_ok=True)

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)
    records = tuple(load_source_catalog(catalog_path))
    build = build_projection_training_groups(
        records,
        language=settings.language,
        max_variants=settings.max_variants,
        max_perm_len=settings.max_perm_len,
        seed=settings.seed,
        max_train_groups=settings.max_train_groups,
        max_validation_groups=settings.max_validation_groups,
        max_test_groups=settings.max_test_groups,
    )
    split_groups = build.split_groups

    catalog_sha256 = _sha256_file(catalog_path)
    provenance = {
        "schema_version": "wfcllm-semantic-projection-training/v1",
        "objective_version": OBJECTIVE_VERSION,
        "public_plane_id": PUBLIC_PLANE_ID,
        "language": settings.language,
        "catalog_path": str(catalog_path),
        "catalog_sha256": catalog_sha256,
        "source_count": len(records),
        "block_count": build.block_count,
        "source_splits": build.source_splits,
        "requested_group_limits": build.requested_group_limits,
        "built_group_counts": build.built_group_counts,
        "group_selection_sha256": _canonical_hash(split_groups),
        "seed": settings.seed,
        "human_eval_included": False,
        "watermark_key_accessed": False,
        "allowed_region_labels_accessed": False,
        "quality_results_accessed": False,
    }
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    datasets = {
        name: RegionGroupDataset(
            groups,
            tokenizer,
            max_length=settings.max_length,
        )
        for name, groups in split_groups.items()
    }
    generator = torch.Generator().manual_seed(settings.seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=settings.batch_size,
            shuffle=True,
            generator=generator,
            num_workers=settings.num_workers,
            pin_memory=True,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=True,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=settings.batch_size,
            shuffle=False,
            num_workers=settings.num_workers,
            pin_memory=True,
        ),
    }
    config = EncoderConfig(
        model_name=str(model_path),
        embed_dim=settings.embed_dim,
        pooling="masked_mean",
        use_lora=True,
        use_bf16=True,
        lora_r=settings.lora_r,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        lora_target_modules=["q", "v"],
        max_seq_length=settings.max_length,
        lr=settings.lr,
        batch_size=settings.batch_size,
        epochs=settings.epochs,
        num_workers=settings.num_workers,
        checkpoint_dir=str(output_dir / "checkpoints"),
        output_model_dir=str(output_dir),
        results_dir=str(output_dir),
        data_sources=[],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticEncoder(config=config).to(device)
    encoder_parameters = [
        parameter for parameter in model.encoder.parameters() if parameter.requires_grad
    ]
    optimizer = AdamW(
        [
            {"params": encoder_parameters, "lr": settings.lr},
            {"params": model.projection.parameters(), "lr": settings.projection_lr},
        ],
        weight_decay=settings.weight_decay,
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=max(1, len(loaders["train"]) * settings.epochs),
    )
    planes = LSHSpace(
        secret_key=PUBLIC_PLANE_ID,
        embed_dim=settings.embed_dim,
        d=4,
    ).planes.to(device)
    metrics_path = output_dir / "metrics.jsonl"
    best_loss = float("inf")
    best_epoch = 0
    for epoch in range(1, settings.epochs + 1):
        train_metrics = _run_epoch(
            model=model,
            loader=loaders["train"],
            planes=planes,
            device=device,
            optimizer=optimizer,
            scheduler=scheduler,
            use_bf16=config.use_bf16,
        )
        validation_metrics = _run_epoch(
            model=model,
            loader=loaders["validation"],
            planes=planes,
            device=device,
            optimizer=None,
            scheduler=None,
            use_bf16=config.use_bf16,
        )
        row = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, allow_nan=False, sort_keys=True) + "\n")
        print(json.dumps(row, allow_nan=False, sort_keys=True), flush=True)
        if validation_metrics["loss"] < best_loss:
            best_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": dataclasses.asdict(config),
                    "epoch": epoch,
                    "best_metric": best_loss,
                    "provenance": provenance,
                },
                output_dir / "best_model.pt",
            )

    checkpoint_path = output_dir / "best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = _run_epoch(
        model=model,
        loader=loaders["test"],
        planes=planes,
        device=device,
        optimizer=None,
        scheduler=None,
        use_bf16=config.use_bf16,
    )
    report = {
        "schema_version": "wfcllm-semantic-projection-report/v1",
        "language": settings.language,
        "best_epoch": best_epoch,
        "epochs_completed": settings.epochs,
        "best_validation_loss": best_loss,
        "test": test_metrics,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256_file(checkpoint_path),
        "provenance_sha256": _sha256_file(output_dir / "provenance.json"),
        "metrics_sha256": _sha256_file(metrics_path),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, allow_nan=False, sort_keys=True), flush=True)

    manifest = {
        "schema_version": "wfcllm-semantic-encoder-manifest/v1",
        "objective_version": OBJECTIVE_VERSION,
        "public_plane_id": PUBLIC_PLANE_ID,
        "language": settings.language,
        "catalog_sha256": catalog_sha256,
        "built_group_counts": build.built_group_counts,
        "requested_group_limits": build.requested_group_limits,
        "best_model_path": str(checkpoint_path),
        "checkpoint_sha256": report["checkpoint_sha256"],
        "best_epoch": best_epoch,
        "epochs_completed": settings.epochs,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return {**manifest, "manifest_path": str(manifest_path)}
