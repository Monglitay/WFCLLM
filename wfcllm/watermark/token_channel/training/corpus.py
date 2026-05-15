"""Offline corpus builders for token-channel training rows."""

from __future__ import annotations

from collections import defaultdict
import gc
import json
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

from wfcllm.common.transform.engine import TransformEngine
from wfcllm.common.transform.positive import get_all_positive_rules
from wfcllm.watermark.token_channel.core.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.core.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.training.teacher import batch_extract_teacher_rows
from wfcllm.watermark.token_channel.training.teacher import extract_teacher_rows

TRAINING_CACHE_SCHEMA_VERSION = "token-channel-training-corpus/v1"


def build_augmented_variants(
    source_code: str,
    transform_engine: TransformEngine | object | None = None,
    max_variants: int | None = None,
) -> list[str]:
    """Return the base sample plus positive semantic-equivalent variants.

    Args:
        source_code: Original source code
        transform_engine: Optional transform engine
        max_variants: Maximum number of variants to generate (including original).
                     None means no limit. Default is None.
    """

    variants = [source_code]
    engine = transform_engine or TransformEngine(rules=get_all_positive_rules())
    generated_variants = engine.generate_variants(source_code)
    seen_sources = {source_code}
    for variant in generated_variants:
        # Stop if we've reached the limit
        if max_variants is not None and len(variants) >= max_variants:
            break

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
    teacher_batch_size: int = 16,
    max_variants: int | None = None,
    top_k_logits: int | None = 100,
):
    """Build offline supervised rows from base samples and positive variants.

    Yields rows incrementally to support streaming writes and avoid memory buildup.

    Args:
        max_variants: Maximum number of variants per sample (including original).
                     None means no limit. Set to a small number (e.g., 5) to reduce
                     corpus size and training time.
        top_k_logits: Only save top-k logits to reduce cache size. None means save all.
                     Default is 100, which reduces cache size by ~99% with minimal
                     performance impact.
    """

    if context_width <= 0:
        raise ValueError("context_width must be > 0")
    if diversity_threshold <= 0:
        raise ValueError("diversity_threshold must be > 0")

    sample_iterator = (
        tqdm(samples, desc="      处理样本", unit="sample", dynamic_ncols=True)
        if tqdm is not None
        else samples
    )

    for sample in sample_iterator:
        source_code = sample.get("source_code")
        if not isinstance(source_code, str) or not source_code:
            raise ValueError("sample source_code must be a non-empty string")

        sample_rows: list[dict[str, object]] = []
        continuation_sets: dict[tuple[int, ...], set[int]] = defaultdict(set)

        variants = build_augmented_variants(
            source_code,
            transform_engine=transform_engine,
            max_variants=max_variants,
        )

        # Filter out syntax-invalid variants and prepare feature contexts
        valid_variants: list[tuple[str, object]] = []
        for variant_idx, variant_source in enumerate(variants, 1):
            # Update progress bar suffix
            if tqdm is not None and hasattr(sample_iterator, 'set_postfix'):
                sample_iterator.set_postfix({'variant': f'{variant_idx}/{len(variants)}'}, refresh=True)

            try:
                feature_context = prepare_token_channel_feature_context(variant_source)
                valid_variants.append((variant_source, feature_context))
            except SyntaxError:
                if tqdm is not None and hasattr(sample_iterator, 'set_postfix'):
                    sample_iterator.set_postfix({'variant': f'{variant_idx}/{len(variants)} (跳过-语法错误)'}, refresh=True)
                continue
            except ValueError as e:
                # Skip variants that fail token alignment or other validation
                if tqdm is not None and hasattr(sample_iterator, 'set_postfix'):
                    error_type = "对齐错误" if "align" in str(e).lower() else "验证错误"
                    sample_iterator.set_postfix({'variant': f'{variant_idx}/{len(variants)} (跳过-{error_type})'}, refresh=True)
                continue

        # Determine if we can use batch inference
        # Batch inference requires forward-pass models with bos_token_id
        can_use_batch = (
            not hasattr(teacher_model, "score_next")
            and hasattr(tokenizer, "bos_token_id")
            and isinstance(getattr(tokenizer, "bos_token_id", None), int)
            and not isinstance(getattr(tokenizer, "bos_token_id", None), bool)
        )

        if can_use_batch and valid_variants:
            # Use batch inference for all valid variants
            variant_texts = [variant_source for variant_source, _ in valid_variants]
            batch_teacher_rows = batch_extract_teacher_rows(
                tokenizer=tokenizer,
                model=teacher_model,
                texts=variant_texts,
                context_width=context_width,
                batch_size=teacher_batch_size,
                show_progress=True,
            )

            # Process results for each variant
            for (variant_source, feature_context), teacher_rows in zip(valid_variants, batch_teacher_rows):
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

                    # Optionally save only top-k logits (by value) to reduce cache size
                    teacher_logits_full = teacher_row["teacher_logits"]
                    if top_k_logits is not None and len(teacher_logits_full) > top_k_logits:
                        # Save top-k by value (not by position)
                        import torch
                        logits_tensor = torch.tensor(teacher_logits_full, dtype=torch.float32)
                        topk_values, topk_indices = torch.topk(logits_tensor, k=top_k_logits)
                        teacher_logits_values = topk_values.tolist()
                        teacher_logits_indices = topk_indices.tolist()
                    else:
                        teacher_logits_values = None
                        teacher_logits_indices = None

                    row = {
                        "prefix_tokens": list(teacher_row["prefix_tokens"]),
                        "next_token": teacher_row["next_token"],
                        "teacher_logits": list(teacher_logits_full) if top_k_logits is None or len(teacher_logits_full) <= top_k_logits else None,
                        "teacher_logits_topk_values": teacher_logits_values,
                        "teacher_logits_topk_indices": teacher_logits_indices,
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

            # Explicitly release batch_teacher_rows memory after processing
            del batch_teacher_rows
        else:
            # Fall back to sequential extraction for score_next models
            for variant_source, feature_context in valid_variants:
                teacher_rows = extract_teacher_rows(
                    tokenizer=tokenizer,
                    model=teacher_model,
                    text=variant_source,
                    context_width=context_width,
                    show_progress=True,
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

                    # Optionally save only top-k logits (by value) to reduce cache size
                    teacher_logits_full = teacher_row["teacher_logits"]
                    if top_k_logits is not None and len(teacher_logits_full) > top_k_logits:
                        # Save top-k by value (not by position)
                        import torch
                        logits_tensor = torch.tensor(teacher_logits_full, dtype=torch.float32)
                        topk_values, topk_indices = torch.topk(logits_tensor, k=top_k_logits)
                        teacher_logits_values = topk_values.tolist()
                        teacher_logits_indices = topk_indices.tolist()
                    else:
                        teacher_logits_values = None
                        teacher_logits_indices = None

                    row = {
                        "prefix_tokens": list(teacher_row["prefix_tokens"]),
                        "next_token": teacher_row["next_token"],
                        "teacher_logits": list(teacher_logits_full) if top_k_logits is None or len(teacher_logits_full) <= top_k_logits else None,
                        "teacher_logits_topk_values": teacher_logits_values,
                        "teacher_logits_topk_indices": teacher_logits_indices,
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
            # Yield each row immediately instead of accumulating
            yield row

        # Explicitly release sample-level memory after processing
        del sample_rows
        del continuation_sets
        del valid_variants

        # Force garbage collection every 10 samples to prevent memory buildup
        if (samples.index(sample) + 1) % 10 == 0:
            gc.collect()


def save_training_cache(path: str | Path, rows: list[dict[str, object]]) -> None:
    """Persist corpus rows with a stable schema wrapper."""

    cache_path = Path(path)
    payload = {
        "schema_version": TRAINING_CACHE_SCHEMA_VERSION,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_training_cache_streaming(
    path: str | Path,
    rows_iterator,
    *,
    flush_interval: int = 500,  # Reduced from 5000 to 500 for better memory control
) -> int:
    """Persist corpus rows incrementally to avoid memory buildup.

    Args:
        path: Output cache file path
        rows_iterator: Iterator yielding row dicts
        flush_interval: Write to disk every N rows (default 500 for balanced performance/memory)

    Returns:
        Total number of rows written
    """
    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Write schema header
    temp_path = cache_path.with_suffix('.tmp')
    with open(temp_path, 'w', encoding='utf-8') as f:
        f.write('{\n')
        f.write(f'  "schema_version": "{TRAINING_CACHE_SCHEMA_VERSION}",\n')
        f.write('  "rows": [\n')

        row_count = 0
        buffer = []

        # Try to import tqdm for progress bar
        try:
            from tqdm import tqdm
            rows_iterator = tqdm(rows_iterator, desc="写入训练缓存", unit="行", dynamic_ncols=True)
        except ImportError:
            pass

        for row in rows_iterator:
            buffer.append(row)
            row_count += 1

            # Flush buffer periodically
            if len(buffer) >= flush_interval:
                for i, buffered_row in enumerate(buffer):
                    # Use compact JSON (no indent) for speed
                    row_json = json.dumps(buffered_row, ensure_ascii=False, separators=(',', ':'))
                    if row_count - len(buffer) + i > 0:
                        f.write(',\n    ')
                    else:
                        f.write('    ')
                    f.write(row_json)
                f.flush()
                buffer.clear()
                # GC every flush to keep memory under control
                gc.collect()

        # Flush remaining buffer
        for i, buffered_row in enumerate(buffer):
            row_json = json.dumps(buffered_row, ensure_ascii=False, separators=(',', ':'))
            if row_count - len(buffer) + i > 0:
                f.write(',\n    ')
            else:
                f.write('    ')
            f.write(row_json)

        f.write('\n  ]\n')
        f.write('}\n')

    # Atomic rename
    temp_path.replace(cache_path)
    return row_count


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
