"""Offline teacher-row extraction helpers for token-channel training."""

from __future__ import annotations

from pathlib import Path
import pickle

import torch

TEACHER_CACHE_SCHEMA_VERSION = "token-channel-teacher-cache/v1"


def extract_teacher_rows(
    tokenizer: object,
    model: object,
    text: str,
    context_width: int,
) -> list[dict[str, object]]:
    """Extract offline teacher rows for every token position in text."""

    if context_width <= 0:
        raise ValueError("context_width must be > 0")

    token_ids = list(_encode_text(tokenizer, text))
    token_spans = _align_token_spans(text, token_ids, tokenizer)

    rows: list[dict[str, object]] = []
    for index, token_id in enumerate(token_ids):
        prefix_tokens = token_ids[max(0, index - context_width) : index]
        teacher_logits = _score_next(model, prefix_tokens)
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
    with cache_path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_teacher_cache(path: str | Path) -> list[dict[str, object]]:
    cache_path = Path(path)
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, list):
        return payload
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
            spans.append((start, end))
            previous_prefix = rendered_prefix
            continue
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


def _score_next(model: object, prefix_tokens: list[int]) -> torch.Tensor:
    if hasattr(model, "score_next"):
        logits = model.score_next(tuple(prefix_tokens))
        if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
            raise ValueError("model.score_next() must return a 1D tensor")
        return logits.detach().cpu().to(dtype=torch.float32)

    input_ids = torch.tensor([prefix_tokens], dtype=torch.long)
    output = model(input_ids)
    logits = getattr(output, "logits", None)
    if logits is None and isinstance(output, dict):
        logits = output.get("logits")
    if logits is None:
        logits = output if isinstance(output, torch.Tensor) else None
    if not isinstance(logits, torch.Tensor) or logits.ndim != 3:
        raise ValueError("teacher model must expose 3D logits output")
    return logits[0, -1].detach().cpu().to(dtype=torch.float32)


def _compute_entropy(logits: torch.Tensor) -> float:
    probabilities = torch.softmax(logits, dim=-1)
    log_probabilities = torch.log(probabilities.clamp_min(1e-12))
    entropy = -(probabilities * log_probabilities).sum()
    return float(entropy.item())
