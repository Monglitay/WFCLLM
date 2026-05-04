"""Offline teacher-row extraction helpers for token-channel training."""

from __future__ import annotations

import json
from pathlib import Path

import torch

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

TEACHER_CACHE_SCHEMA_VERSION = "token-channel-teacher-cache/v1"


def extract_teacher_rows(
    tokenizer: object,
    model: object,
    text: str,
    context_width: int,
    *,
    show_progress: bool = False,
) -> list[dict[str, object]]:
    """Extract offline teacher rows for every token position in text."""

    if context_width <= 0:
        raise ValueError("context_width must be > 0")

    token_ids = list(_encode_text(tokenizer, text))
    token_spans = _align_token_spans(text, token_ids, tokenizer)

    rows: list[dict[str, object]] = []

    token_iterator = (
        tqdm(enumerate(token_ids), total=len(token_ids), desc="        提取 teacher logits", leave=False, dynamic_ncols=True)
        if show_progress and tqdm is not None
        else enumerate(token_ids)
    )

    for index, token_id in token_iterator:
        prefix_tokens = token_ids[max(0, index - context_width) : index]
        teacher_logits = _score_next(model, prefix_tokens, tokenizer=tokenizer)
        token_start, token_end = token_spans[index]
        rows.append(
            {
                "prefix_tokens": list(prefix_tokens),
                "next_token": token_id,
                "teacher_logits": teacher_logits.tolist(),
                "entropy": _compute_entropy(teacher_logits),
                "token_text": text[token_start:token_end],
                "token_start": token_start,
                "token_end": token_end,
                "token_index": index,
            }
        )

    return rows


def save_teacher_cache(path: str | Path, rows: list[dict[str, object]]) -> None:
    cache_path = Path(path)
    payload = {
        "schema_version": TEACHER_CACHE_SCHEMA_VERSION,
        "rows": rows,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_teacher_cache(path: str | Path) -> list[dict[str, object]]:
    cache_path = Path(path)
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("teacher cache must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("teacher cache must contain a payload dictionary")
    if payload.get("schema_version") != TEACHER_CACHE_SCHEMA_VERSION:
        raise ValueError("teacher cache schema_version is incompatible")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError("teacher cache rows must be a list")
    return rows


def _encode_text(tokenizer: object, text: str) -> list[int]:
    encode = getattr(tokenizer, "encode", None)
    if encode is None:
        raise ValueError("tokenizer must provide an encode() method")
    token_ids = encode(text, add_special_tokens=False)
    if not isinstance(token_ids, list):
        raise ValueError("tokenizer.encode() must return a list of token ids")
    return token_ids


def _align_token_spans(text: str, token_ids: list[int], tokenizer: object) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    previous_prefix = ""
    for prefix_index in range(1, len(token_ids) + 1):
        rendered_prefix = _render_token_prefix(tokenizer, token_ids[:prefix_index])
        if not text.startswith(rendered_prefix):
            raise ValueError(
                f"Unable to align rendered prefix {rendered_prefix!r} with source"
            )
        start = len(previous_prefix)
        end = len(rendered_prefix)
        if end < start:
            raise ValueError("token alignment prefix lengths must be non-decreasing")
        if start == end:
            raise ValueError("zero-length aligned token spans are not supported")
        spans.append((start, end))
        previous_prefix = rendered_prefix
    return spans


def _render_token_prefix(tokenizer: object, token_ids: list[int]) -> str:
    convert_ids_to_tokens = getattr(tokenizer, "convert_ids_to_tokens", None)
    convert_tokens_to_string = getattr(tokenizer, "convert_tokens_to_string", None)
    if convert_ids_to_tokens is not None and convert_tokens_to_string is not None:
        tokens = convert_ids_to_tokens(token_ids)
        rendered = convert_tokens_to_string(tokens)
        if not isinstance(rendered, str):
            raise ValueError("tokenizer.convert_tokens_to_string() must return a string")
        return rendered

    decode = getattr(tokenizer, "decode", None)
    if decode is None:
        raise ValueError(
            "tokenizer must provide decode() or convert_ids_to_tokens()/convert_tokens_to_string()"
        )
    rendered = decode(token_ids, skip_special_tokens=True)
    if not isinstance(rendered, str):
        raise ValueError("tokenizer.decode() must return a string")
    return rendered


def _score_next(model: object, prefix_tokens: list[int], *, tokenizer: object) -> torch.Tensor:
    if hasattr(model, "score_next"):
        with torch.no_grad():
            logits = model.score_next(tuple(prefix_tokens))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise ValueError("model.score_next() must return a 1D tensor")
        return logits.detach().cpu().to(dtype=torch.float32)

    input_tokens = prefix_tokens
    if not input_tokens:
        bos_token_id = getattr(tokenizer, "bos_token_id", None)
        if not isinstance(bos_token_id, int) or isinstance(bos_token_id, bool):
            raise ValueError(
                "forward teacher models require tokenizer.bos_token_id for empty prefixes"
            )
        input_tokens = [bos_token_id]

    input_ids = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=_resolve_module_device(model),
    )
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()
    try:
        with torch.no_grad():
            output = model(input_ids)
    finally:
        if was_training is not None and callable(train_method):
            train_method(was_training)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("teacher model must expose 3D logits output")
    return logits[0, -1].detach().cpu().to(dtype=torch.float32)


def _resolve_module_device(model: object) -> torch.device:
    if isinstance(model, torch.nn.Module):
        parameter = next(model.parameters(), None)
        if parameter is not None:
            return parameter.device
        buffer = next(model.buffers(), None)
        if buffer is not None:
            return buffer.device
    return torch.device("cpu")


def _compute_entropy(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    entropy = -(probabilities * log_probabilities).sum()
    return float(entropy.item())


def _get_vocab_size(model: object) -> int:
    """Get vocabulary size from model."""
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        vocab_size = model.config.vocab_size
        if isinstance(vocab_size, int):
            return vocab_size
    if hasattr(model, "vocab_size"):
        vocab_size = model.vocab_size
        if isinstance(vocab_size, int):
            return vocab_size
    raise ValueError("model must expose vocab_size via config.vocab_size or vocab_size attribute")


def _extract_single_text_all_positions(
    model: object,
    token_ids: list[int],
    tokenizer: object,
) -> torch.Tensor:
    """Extract logits for all token positions in a single text using one forward pass.

    Args:
        model: Language model that accepts input_ids and returns logits
        token_ids: List of token IDs for the text
        tokenizer: Tokenizer with optional bos_token_id attribute

    Returns:
        Tensor of shape [seq_len, vocab_size] with float32 dtype.
        For position i, returns logits predicting token_ids[i] given token_ids[:i].
    """
    if not token_ids:
        vocab_size = _get_vocab_size(model)
        return torch.empty((0, vocab_size), dtype=torch.float32)

    # Prepend bos_token if tokenizer has one
    input_tokens = token_ids.copy()
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if isinstance(bos_token_id, int) and not isinstance(bos_token_id, bool):
        input_tokens = [bos_token_id] + input_tokens

    # Prepare input tensor
    input_ids = torch.tensor(
        [input_tokens],
        dtype=torch.long,
        device=_resolve_module_device(model),
    )

    # Set model to eval mode
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()

    try:
        # Forward pass with no gradient
        with torch.no_grad():
            output = model(input_ids)
    finally:
        # Restore training mode
        if was_training is not None and callable(train_method):
            train_method(was_training)

    # Extract logits from output
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("teacher model must expose 3D logits output")

    # Extract logits for each position
    # logits shape: [batch_size=1, seq_len, vocab_size]
    # For position i in token_ids, we want logits[0, i] which predicts token_ids[i]
    # If we prepended bos_token, logits[0, 0] predicts token_ids[0], etc.
    batch_logits = logits[0]  # [seq_len, vocab_size]

    # Extract the relevant positions
    if isinstance(bos_token_id, int) and not isinstance(bos_token_id, bool):
        # We prepended bos, so positions 0..len(token_ids)-1 predict token_ids[0..len(token_ids)-1]
        result_logits = batch_logits[:len(token_ids)]
    else:
        # No bos prepended - this path should not be used for batch extraction
        # because we cannot predict the first token without a prefix
        raise ValueError(
            "forward teacher models require tokenizer.bos_token_id for extracting all positions"
        )

    return result_logits.detach().cpu().to(dtype=torch.float32)


def _batch_forward_all_positions(
    model: object,
    batch_token_ids: list[list[int]],
    tokenizer: object,
) -> list[torch.Tensor]:
    """Extract logits for all positions in multiple texts using batch forward pass.

    Args:
        model: Language model that accepts input_ids and returns logits
        batch_token_ids: List of token ID lists (variable length sequences)
        tokenizer: Tokenizer with optional bos_token_id attribute

    Returns:
        List of tensors, each with shape [seq_len, vocab_size] and float32 dtype.
        For text i at position j, returns logits predicting batch_token_ids[i][j]
        given batch_token_ids[i][:j].
    """
    if not batch_token_ids:
        return []

    # Get BOS token
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if not isinstance(bos_token_id, int) or isinstance(bos_token_id, bool):
        raise ValueError(
            "forward teacher models require tokenizer.bos_token_id for batch extraction"
        )

    # Get vocab size for empty sequences
    vocab_size = _get_vocab_size(model)

    # Handle empty sequences separately
    results: list[torch.Tensor] = []
    non_empty_indices: list[int] = []
    non_empty_token_ids: list[list[int]] = []

    for i, token_ids in enumerate(batch_token_ids):
        if not token_ids:
            results.append(torch.empty((0, vocab_size), dtype=torch.float32))
        else:
            results.append(None)  # Placeholder
            non_empty_indices.append(i)
            non_empty_token_ids.append(token_ids)

    # If all sequences are empty, return early
    if not non_empty_token_ids:
        return results

    # Prepend BOS token to each sequence
    input_sequences = [[bos_token_id] + token_ids for token_ids in non_empty_token_ids]

    # Find max length for padding
    max_len = max(len(seq) for seq in input_sequences)

    # Get pad token ID (use 0 as default, common for most tokenizers)
    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if not isinstance(pad_token_id, int) or isinstance(pad_token_id, bool):
        pad_token_id = 0

    # Apply left padding to preserve causal structure
    padded_sequences = []
    attention_masks = []
    original_lengths = []

    for seq in input_sequences:
        seq_len = len(seq)
        original_lengths.append(seq_len)
        padding_len = max_len - seq_len

        # Left padding
        padded_seq = [pad_token_id] * padding_len + seq
        attention_mask = [0] * padding_len + [1] * seq_len

        padded_sequences.append(padded_seq)
        attention_masks.append(attention_mask)

    # Convert to tensors
    device = _resolve_module_device(model)
    input_ids = torch.tensor(padded_sequences, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long, device=device)

    # Set model to eval mode
    was_training = getattr(model, "training", None)
    eval_method = getattr(model, "eval", None)
    train_method = getattr(model, "train", None)
    if callable(eval_method):
        eval_method()

    try:
        # Forward pass with no gradient
        with torch.no_grad():
            # Try passing attention_mask
            try:
                output = model(input_ids, attention_mask=attention_mask)
            except TypeError:
                # Model doesn't accept attention_mask, try without it
                output = model(input_ids)
    finally:
        # Restore training mode
        if was_training is not None and callable(train_method):
            train_method(was_training)

    # Extract logits from output
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("teacher model must expose 3D logits output")

    # Extract logits for each sequence
    # logits shape: [batch_size, max_len, vocab_size]
    for i, (batch_idx, token_ids) in enumerate(zip(non_empty_indices, non_empty_token_ids)):
        seq_len = len(token_ids)
        padding_len = max_len - original_lengths[i]

        # Extract the relevant positions (skip padding, skip BOS, take seq_len positions)
        # After padding: [pad, pad, ..., bos, tok1, tok2, ...]
        # We want logits at positions [padding_len, padding_len+1, ..., padding_len+seq_len-1]
        # These predict [tok1, tok2, ..., tokN]
        start_pos = padding_len
        end_pos = padding_len + seq_len

        seq_logits = logits[i, start_pos:end_pos]  # [seq_len, vocab_size]
        results[batch_idx] = seq_logits.detach().cpu().to(dtype=torch.float32)

    return results
