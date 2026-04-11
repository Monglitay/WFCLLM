"""Offline corpus builders for token-channel training rows."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path

from wfcllm.common.transform.engine import TransformEngine
from wfcllm.common.transform.positive import get_all_positive_rules
from wfcllm.watermark.token_channel.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.teacher import extract_teacher_rows

TRAINING_CACHE_SCHEMA_VERSION = "token-channel-training-corpus/v1"


def build_augmented_variants(
    source_code: str,
    transform_engine: TransformEngine | object | None = None,
) -> list[str]:
    """Return the base sample plus positive semantic-equivalent variants."""

    variants = [source_code]
    engine = transform_engine or TransformEngine(rules=get_all_positive_rules())
    generated_variants = engine.generate_variants(source_code)
    seen_sources = {source_code}
    for variant in generated_variants:
        if variant.get("sample_type") != "positive":
            continue
        transformed_source = variant.get("transformed_source")
        if not isinstance(transformed_source, str):
            continue
        if transformed_source in seen_sources:
            continue
        variants.append(transformed_source)
        seen_sources.add(transformed_source)
    return variants


def build_training_rows(
    samples: list[dict[str, object]],
    tokenizer: object,
    teacher_model: object,
    context_width: int,
    *,
    transform_engine: TransformEngine | object | None = None,
    entropy_threshold: float,
    diversity_threshold: int,
) -> list[dict[str, object]]:
    """Build offline supervised rows from base samples and positive variants."""

    if context_width <= 0:
        raise ValueError("context_width must be > 0")
    if diversity_threshold <= 0:
        raise ValueError("diversity_threshold must be > 0")

    rows: list[dict[str, object]] = []
    for sample in samples:
        source_code = sample.get("source_code")
        if not isinstance(source_code, str) or not source_code:
            raise ValueError("sample source_code must be a non-empty string")

        sample_rows: list[dict[str, object]] = []
        continuation_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)

        for variant_source in build_augmented_variants(
            source_code,
            transform_engine=transform_engine,
        ):
            feature_context = prepare_token_channel_feature_context(variant_source)
            teacher_rows = extract_teacher_rows(
                tokenizer=tokenizer,
                model=teacher_model,
                text=variant_source,
                context_width=context_width,
            )
            for teacher_row in teacher_rows:
                token_start = teacher_row["token_start"]
                token_end = teacher_row["token_end"]
                if not isinstance(token_start, int) or not isinstance(token_end, int):
                    raise ValueError("teacher rows must include integer token spans")
                features = build_token_channel_features_from_context(
                    feature_context,
                    token_start=token_start,
                    token_end=token_end,
                )
                row = {
                    "prefix_tokens": list(teacher_row["prefix_tokens"]),
                    "next_token": teacher_row["next_token"],
                    "teacher_logits": list(teacher_row["teacher_logits"]),
                    "entropy": teacher_row["entropy"],
                    "continuation_diversity": 0,
                    "node_type": features.node_type,
                    "parent_node_type": features.parent_node_type,
                    "block_relative_offset": features.block_relative_offset,
                    "in_code_body": features.in_code_body,
                    "structure_mask": features.structure_mask,
                    "language": features.language,
                    "switch_target": 0,
                }
                sample_rows.append(row)
                continuation_sets[tuple(row["prefix_tokens"])].add(int(row["next_token"]))

        for row in sample_rows:
            continuation_diversity = len(continuation_sets[tuple(row["prefix_tokens"])])
            row["continuation_diversity"] = continuation_diversity
            row["switch_target"] = int(
                bool(row["in_code_body"])
                and
                bool(row["structure_mask"])
                and float(row["entropy"]) >= entropy_threshold
                and continuation_diversity >= diversity_threshold
            )
        rows.extend(sample_rows)

    return rows


def save_training_cache(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Persist corpus rows with a stable schema wrapper."""

    cache_path = Path(path)
    payload = {
        "schema_version": TRAINING_CACHE_SCHEMA_VERSION,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_training_cache(path: str | Path) -> list[dict[str, object]]:
    """Load persisted corpus rows."""

    cache_path = Path(path)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("training cache must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("training cache must contain a payload dictionary")
    if payload.get("schema_version") != TRAINING_CACHE_SCHEMA_VERSION:
        raise ValueError("training cache schema_version is incompatible")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("training cache rows must be a list")
    return rows
