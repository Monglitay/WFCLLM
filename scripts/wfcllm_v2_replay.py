#!/usr/bin/env python
"""Audit deterministic WFCLLM V2 scoring from saved final code only."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.detection.pipeline import load_jsonl_records  # noqa: E402
from wfcllm.detection.signature_v2 import load_v2_signature_scorer  # noqa: E402
from wfcllm.generation.selection_v2 import (  # noqa: E402
    V2_RETRY_LEDGER_SCHEMA_VERSION,
)

SECRET_KEY_ENV_VAR = "WFCLLM_SECRET_KEY"
REPLAY_SCHEMA_VERSION = "wfcllm-v2-replay/v2"
_WRONG_KEY_DOMAIN = b"wfcllm-v2-replay-wrong-key\x00"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Replay V2 evidence using saved final code and private keys.",
    )
    parser.add_argument("--positive-input", required=True)
    parser.add_argument("--negative-input", required=True)
    parser.add_argument("--retry-ledger", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--secret-key", default=None)
    parser.add_argument("--encoder-model-path", required=True)
    parser.add_argument("--encoder-checkpoint-path", default=None)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--encoder-use-lora", action="store_true")
    parser.add_argument("--encoder-use-bf16", action="store_true")
    parser.add_argument("--lsh-whitening-path", default=None)
    parser.add_argument("--signature-bits", type=int, default=16)
    return parser


def _resolve_secret_key(cli_secret_key: str | None) -> str:
    if cli_secret_key:
        return cli_secret_key
    value = os.environ.get(SECRET_KEY_ENV_VAR)
    if value:
        return value
    raise ValueError(
        f"secret key must be provided via --secret-key or {SECRET_KEY_ENV_VAR}"
    )


def _derive_wrong_key(secret_key: str) -> str:
    return hashlib.sha256(
        _WRONG_KEY_DOMAIN + secret_key.encode("utf-8")
    ).hexdigest()


def _scorer_args(args: argparse.Namespace, *, secret_key: str) -> dict[str, Any]:
    return {
        "encoder_model_path": args.encoder_model_path,
        "encoder_checkpoint_path": args.encoder_checkpoint_path,
        "embed_dim": args.encoder_embed_dim,
        "device": args.device,
        "use_lora": args.encoder_use_lora,
        "use_bf16": args.encoder_use_bf16,
        "secret_key": secret_key,
        "signature_bits": args.signature_bits,
        "whitening_path": args.lsh_whitening_path,
    }


def _score_summary(score: Any) -> dict[str, int | float]:
    return {
        "score": float(score.raw_score),
        "unit_count": int(score.unit_count),
        "duplicate_units": int(score.duplicate_count),
        "total_signature_bits": int(score.total_bits),
        "matched_signature_bits": int(score.matched_bits),
    }


def _scores_equal(left: Any, right: Any) -> bool:
    return _score_summary(left) == _score_summary(right)


def _load_ledger(path: str | Path | None) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    selected: dict[str, dict[str, Any]] = {}
    for row in _load_generic_jsonl(path):
        if row.get("schema_version") != V2_RETRY_LEDGER_SCHEMA_VERSION:
            raise ValueError("retry ledger schema_version mismatch")
        if row.get("selected") is not True:
            continue
        sample_id = row.get("id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("selected retry ledger row must contain an id")
        if sample_id in selected:
            raise ValueError(f"multiple selected retry rows for {sample_id}")
        selected[sample_id] = row
    return selected


def _load_generic_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"JSONL row {line_number} must be an object")
                rows.append(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL at line {exc.lineno}") from exc
    return rows


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    ordered = sorted(values)

    def pick(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "q00": pick(0.0),
        "q25": pick(0.25),
        "q50": pick(0.5),
        "q75": pick(0.75),
        "q95": pick(0.95),
        "q100": pick(1.0),
    }


def _pearson(left: list[float], right: list[float]) -> float | None:
    if len(left) < 2 or len(left) != len(right):
        return None
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    numerator = sum(
        (left_value - left_mean) * (right_value - right_mean)
        for left_value, right_value in zip(left, right)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right))
    if left_scale == 0 or right_scale == 0:
        return None
    return numerator / (left_scale * right_scale)


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.limit is not None and args.limit <= 0:
        raise ValueError("limit must be positive")
    secret_key = _resolve_secret_key(args.secret_key)
    correct_scorer = load_v2_signature_scorer(
        **_scorer_args(args, secret_key=secret_key)
    )
    wrong_scorer = load_v2_signature_scorer(
        **_scorer_args(args, secret_key=_derive_wrong_key(secret_key))
    )
    positive_rows = load_jsonl_records(args.positive_input)
    negative_rows = load_jsonl_records(args.negative_input)
    if args.limit is not None:
        positive_rows = positive_rows[: args.limit]
        negative_rows = negative_rows[: args.limit]
    ledger_by_id = _load_ledger(args.retry_ledger)

    per_positive: list[dict[str, Any]] = []
    correct_scores: list[float] = []
    wrong_scores: list[float] = []
    final_replay: list[bool] = []
    ledger_code_matches: list[bool] = []
    ledger_score_matches: list[bool] = []
    ledger_generation_scores: list[float] = []
    ledger_recovered_scores: list[float] = []
    for row in positive_rows:
        final_code = str(row["final_code"])
        first = correct_scorer.score_code(final_code)
        replay = correct_scorer.score_code(final_code)
        wrong = wrong_scorer.score_code(final_code)
        replay_equal = _scores_equal(first, replay)
        sample_id = str(row["id"])
        item: dict[str, Any] = {
            "id": sample_id,
            **_score_summary(first),
            "wrong_key_score": float(wrong.raw_score),
            "final_code_replay_equal": replay_equal,
        }
        correct_scores.append(float(first.raw_score))
        wrong_scores.append(float(wrong.raw_score))
        final_replay.append(replay_equal)
        ledger = ledger_by_id.get(sample_id)
        if ledger is not None:
            generation_score = float(ledger["generation_score"])
            code_match = ledger.get("final_code_sha256") == _sha256(final_code)
            score_match = generation_score == float(first.raw_score)
            item["ledger_final_code_match"] = code_match
            item["ledger_score_replay_equal"] = score_match
            ledger_code_matches.append(code_match)
            ledger_score_matches.append(score_match)
            ledger_generation_scores.append(generation_score)
            ledger_recovered_scores.append(float(first.raw_score))
        per_positive.append(item)

    negative_scores = [
        float(correct_scorer.score_code(str(row["final_code"])).raw_score)
        for row in negative_rows
    ]
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "detector_input_contract": ["id", "dataset", "prompt", "final_code"],
        "final_code_only_scoring": True,
        "encoder": {
            "model_path": args.encoder_model_path,
            "checkpoint_path": args.encoder_checkpoint_path,
            "checkpoint_sha256": _file_sha256(args.encoder_checkpoint_path),
            "embed_dim": args.encoder_embed_dim,
            "device": args.device,
            "use_lora": args.encoder_use_lora,
            "use_bf16": args.encoder_use_bf16,
            "signature_bits": args.signature_bits,
            "whitening_path": args.lsh_whitening_path,
        },
        "positive_sample_count": len(positive_rows),
        "negative_sample_count": len(negative_rows),
        "final_code_replay_rate": (
            sum(final_replay) / len(final_replay) if final_replay else 0.0
        ),
        "ledger_selected_count": len(ledger_code_matches),
        "ledger_final_code_match_rate": (
            sum(ledger_code_matches) / len(ledger_code_matches)
            if ledger_code_matches
            else None
        ),
        "ledger_score_replay_rate": (
            sum(ledger_score_matches) / len(ledger_score_matches)
            if ledger_score_matches
            else None
        ),
        "generation_to_final_score_pearson": _pearson(
            ledger_generation_scores,
            ledger_recovered_scores,
        ),
        "positive_correct_key_scores": correct_scores,
        "positive_wrong_key_scores": wrong_scores,
        "negative_scores": negative_scores,
        "score_quantiles": {
            "positive_correct_key": _quantiles(correct_scores),
            "positive_wrong_key": _quantiles(wrong_scores),
            "negative": _quantiles(negative_scores),
        },
        "per_positive": per_positive,
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        report = _build_report(args)
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, allow_nan=False, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] WFCLLM V2 replay report saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
