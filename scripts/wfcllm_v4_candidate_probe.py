#!/usr/bin/env python3
"""Run frozen pre-preregistration probes for V4 mechanism candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.candidate_probe_v4 import (
    load_candidate_ledger,
    load_probe_contexts,
    load_probe_secret,
    derive_projection_bits,
    derive_target_bits,
    probe_shape_isolated,
    structural_pool_capacity,
    summarize_margin_rows,
    write_public_probe_artifact,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--root-cause-matrix", type=Path, required=True)
    parser.add_argument("--margin-rows", type=Path, action="append", default=[])
    parser.add_argument("--candidate-ledger", type=Path, required=True)
    parser.add_argument("--diagnostic-key-file", type=Path, required=True)
    parser.add_argument("--public-config", type=Path)
    parser.add_argument("--encoder-model", type=Path, required=True)
    parser.add_argument("--whitening", type=Path, required=True)
    parser.add_argument("--task-id", action="append", default=[])
    parser.add_argument("--retry", type=int, default=20)
    parser.add_argument("--skip-neural", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load {label}: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _iter_margin_rows(paths: Iterable[Path]) -> Iterable[dict[str, Any]]:
    for path in paths:
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ValueError(f"failed to read margin rows: {path}") from exc
        with handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"invalid margin JSON in {path} at line {line_number}"
                    ) from exc
                if not isinstance(row, dict):
                    raise ValueError("margin row must be a JSON object")
                if "discrete" in row:
                    yield row


def _run_non_neural(args: argparse.Namespace) -> dict[str, Any]:
    contexts = load_probe_contexts(args.contexts)
    matrix = _load_json_object(args.root_cause_matrix, label="root-cause matrix")
    try:
        raw_bound = matrix["maximum_observed_absolute_difference_by_stage"][
            "projection_dots"
        ]
        bound_scope = matrix["bound_scope"]
    except (KeyError, TypeError) as exc:
        raise ValueError("root-cause matrix is missing the projection bound") from exc
    if isinstance(raw_bound, bool) or not isinstance(raw_bound, (int, float)):
        raise ValueError("projection bound must be numeric")
    absolute_dot_bound = int(raw_bound)
    if float(absolute_dot_bound) != float(raw_bound) or absolute_dot_bound < 0:
        raise ValueError("projection bound must be a non-negative integer")
    if not isinstance(bound_scope, str) or "empirical" not in bound_scope.lower():
        raise ValueError("candidate B requires an explicit empirical_only bound scope")
    if "certified" in bound_scope.lower() and "not" not in bound_scope.lower():
        raise ValueError("candidate B must not treat the empirical bound as certified")
    if not args.margin_rows:
        raise ValueError("at least one --margin-rows artifact is required")
    margin = summarize_margin_rows(
        _iter_margin_rows(args.margin_rows),
        absolute_dot_bound=absolute_dot_bound,
    )
    records = load_candidate_ledger(
        args.candidate_ledger,
        retry=args.retry,
        allowed_task_ids=tuple(args.task_id),
    )
    secret = load_probe_secret(args.diagnostic_key_file)
    structural = structural_pool_capacity(
        records,
        secret,
        retry=args.retry,
        bit_count=32,
        minimum_independent_units=3,
    )
    candidate_a = (
        {"status": "skipped_by_cli"}
        if args.skip_neural
        else {"status": "pending_neural_runtime"}
    )
    return {
        "artifact_type": "wfcllm_v4_candidate_probe",
        "secret_metadata_included": False,
        "root_cause_bound_scope": bound_scope,
        "frozen_context_count": len(contexts),
        "diagnostic_task_count": structural.capacity.task_count,
        "retry": args.retry,
        "candidate_results": {
            "A": candidate_a,
            "B": {
                **asdict(margin),
                "absolute_dot_bound": absolute_dot_bound,
                "bound_is_certified": False,
            },
            "C": asdict(structural),
            "D": {
                "signed_evidence_identical_to_C": True,
                "neural_auxiliary_used": False,
                "status": "structural_only_probe",
            },
        },
    }


def _quantized_message(values: tuple[int, ...]) -> bytes:
    payload = bytearray()
    for value in values:
        payload.extend(int(value).to_bytes(8, "big", signed=True))
    return bytes(payload)


def _neural_unit_numerator(
    *,
    secret: Any,
    unit_id: str,
    quantized: tuple[int, ...],
    bit_count: int,
) -> int:
    signature = derive_projection_bits(
        secret,
        _quantized_message(quantized),
        bit_count=bit_count,
    )
    target = derive_target_bits(
        secret,
        unit_id.encode("ascii"),
        bit_count=bit_count,
    )
    matches = sum(
        left == right for left, right in zip(signature, target, strict=True)
    )
    return 2 * matches - bit_count


def _run_neural(args: argparse.Namespace) -> dict[str, Any]:
    if args.public_config is None:
        raise ValueError("--public-config is required unless --skip-neural is used")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch

    from wfcllm.dynamic_semantic.channel import quantize_vector
    from wfcllm.dynamic_semantic.config import load_public_config
    from wfcllm.dynamic_semantic.context import DynamicContextExtractor
    from wfcllm.dynamic_semantic.encoder import SemanticEncoderRuntime
    from wfcllm.dynamic_semantic.whitening import load_whitening

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    config = load_public_config(args.public_config)
    if Path(config.encoder.model_path) != args.encoder_model:
        raise ValueError("--encoder-model must match the frozen public config")
    contexts = load_probe_contexts(args.contexts)
    whitening = load_whitening(args.whitening)
    load_started = time.perf_counter()
    runtime = SemanticEncoderRuntime.load(
        config.encoder,
        max_tokens=config.context.max_context_tokens,
        device="cuda",
        fixed_batch_size=1,
        fixed_sequence_length=True,
    )
    model_load_seconds = time.perf_counter() - load_started
    if whitening.input_dimensions != config.encoder.embedding_dimensions:
        raise ValueError("whitening input dimensions do not match the encoder")
    torch.cuda.reset_peak_memory_stats()

    def encode_one(serialized: str) -> tuple[int, ...]:
        embedding = runtime.encode((serialized,))
        whitened = whitening.transform(embedding)[0]
        return quantize_vector(
            whitened.tolist(),
            config.channel.quantization_scale,
        )

    warmup_started = time.perf_counter()
    warmup_reference = encode_one(contexts[0].serialized)
    warmup_candidate = encode_one(contexts[0].serialized)
    warmup_seconds = time.perf_counter() - warmup_started
    if warmup_reference != warmup_candidate:
        raise ValueError("shape-isolated warm-up replay is not exact")
    shape_summary = probe_shape_isolated(
        tuple(context.serialized for context in contexts),
        encode_one=encode_one,
    )

    records = load_candidate_ledger(
        args.candidate_ledger,
        retry=args.retry,
        allowed_task_ids=tuple(args.task_id),
    )
    secret = load_probe_secret(args.diagnostic_key_file)
    extractor = DynamicContextExtractor(
        config.context,
        token_counter=runtime.token_count,
    )
    cache: dict[str, tuple[int, ...]] = {}
    cache_serialized: dict[str, str] = {}
    candidate_evidence: dict[tuple[str, int], tuple[tuple[Any, ...], ...]] = {}
    candidate_scores: dict[tuple[str, int], float] = {}
    cache_hits = 0
    cache_misses = 0
    dynamic_selection_started = time.perf_counter()
    for record in records:
        extraction = extractor.extract(record.final_code)
        evidence_rows: list[tuple[Any, ...]] = []
        numerator = 0
        denominator = 0
        for context in extraction.contexts:
            if context.context_sha256 in cache:
                if cache_serialized[context.context_sha256] != context.serialized:
                    raise ValueError("public context cache hash collision")
                quantized = cache[context.context_sha256]
                cache_hits += 1
            else:
                quantized = encode_one(context.serialized)
                cache[context.context_sha256] = quantized
                cache_serialized[context.context_sha256] = context.serialized
                cache_misses += 1
            unit_numerator = _neural_unit_numerator(
                secret=secret,
                unit_id=context.unit_id,
                quantized=quantized,
                bit_count=config.channel.projection_rows,
            )
            numerator += unit_numerator
            denominator += config.channel.projection_rows
            evidence_rows.append(
                (
                    context.unit_id,
                    context.context_sha256,
                    quantized,
                    unit_numerator,
                    config.channel.projection_rows,
                )
            )
        key = (record.task_id, record.attempt_index)
        candidate_evidence[key] = tuple(evidence_rows)
        candidate_scores[key] = numerator / denominator if denominator else 0.0
    dynamic_selection_seconds = time.perf_counter() - dynamic_selection_started

    grouped: defaultdict[str, list[Any]] = defaultdict(list)
    task_order: list[str] = []
    for record in records:
        if record.task_id not in grouped:
            task_order.append(record.task_id)
        grouped[record.task_id].append(record)
    selected_records = []
    per_task_deltas = []
    for task_id in task_order:
        items = grouped[task_id]
        selected = max(
            items,
            key=lambda item: (
                candidate_scores[(item.task_id, item.attempt_index)],
                -item.attempt_index,
            ),
        )
        selected_records.append(selected)
        per_task_deltas.append(
            candidate_scores[(selected.task_id, selected.attempt_index)]
            - candidate_scores[(items[0].task_id, items[0].attempt_index)]
        )

    replay_started = time.perf_counter()
    replay_exact = 0
    replay_mismatch_fields: defaultdict[str, int] = defaultdict(int)
    replay_contexts = 0
    for record in selected_records:
        extraction = extractor.extract(record.final_code)
        replay_rows: list[tuple[Any, ...]] = []
        for context in extraction.contexts:
            quantized = encode_one(context.serialized)
            unit_numerator = _neural_unit_numerator(
                secret=secret,
                unit_id=context.unit_id,
                quantized=quantized,
                bit_count=config.channel.projection_rows,
            )
            replay_rows.append(
                (
                    context.unit_id,
                    context.context_sha256,
                    quantized,
                    unit_numerator,
                    config.channel.projection_rows,
                )
            )
            replay_contexts += 1
        expected = candidate_evidence[(record.task_id, record.attempt_index)]
        candidate = tuple(replay_rows)
        if expected == candidate:
            replay_exact += 1
        else:
            if tuple(row[0] for row in expected) != tuple(row[0] for row in candidate):
                replay_mismatch_fields["unit_ids"] += 1
            if tuple(row[1] for row in expected) != tuple(row[1] for row in candidate):
                replay_mismatch_fields["context_sha256"] += 1
            if tuple(row[2] for row in expected) != tuple(row[2] for row in candidate):
                replay_mismatch_fields["quantized"] += 1
            if tuple(row[3:] for row in expected) != tuple(row[3:] for row in candidate):
                replay_mismatch_fields["score_fields"] += 1
    selected_final_replay_seconds = time.perf_counter() - replay_started
    total_semantic = dynamic_selection_seconds + selected_final_replay_seconds
    mean_seconds_per_task = total_semantic / len(task_order)
    positive_deltas = [value for value in per_task_deltas if value > 0]
    positive_total = sum(positive_deltas)
    return {
        "status": "measured",
        "mechanism": "B1_fixed_L256_per_context_with_public_cache",
        "model_load_seconds": model_load_seconds,
        "warmup_seconds": warmup_seconds,
        "shape_invariance": asdict(shape_summary),
        "dynamic_selection_seconds": dynamic_selection_seconds,
        "selected_final_replay_seconds": selected_final_replay_seconds,
        "selected_final_replay_count": len(task_order),
        "selected_final_replay_contexts": replay_contexts,
        "generation_to_r3_exact_tasks": replay_exact,
        "generation_to_r3_total_tasks": len(task_order),
        "generation_to_r3_exact_rate": replay_exact / len(task_order),
        "replay_mismatch_fields": dict(sorted(replay_mismatch_fields.items())),
        "mean_semantic_seconds_per_task": mean_seconds_per_task,
        "cost_gate_seconds_per_task": 0.1409884,
        "cost_gate_pass": mean_seconds_per_task <= 0.1409884,
        "cost_reduction_vs_complete_final": 1.0 - mean_seconds_per_task / 0.201412,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 * 1024),
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "cache_hit_rate": cache_hits / (cache_hits + cache_misses),
        "candidate_count": len(records),
        "task_count": len(task_order),
        "positive_delta_tasks": sum(value > 0 for value in per_task_deltas),
        "mean_score_delta": sum(per_task_deltas) / len(per_task_deltas),
        "maximum_positive_delta_share": (
            max(positive_deltas) / positive_total if positive_total else 0.0
        ),
        "encoder_calls": runtime.encoder_calls,
        "encoded_contexts": runtime.encoded_contexts,
        "physical_batch_size": 1,
        "sequence_length": config.context.max_context_tokens,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32": False,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "model_eval": True,
        "offline_saved_pool_probe_only": True,
        "formal_eos_all_candidate_rescore_count": None,
        "encoder_checkpoint_sha256": config.encoder.checkpoint_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = _run_non_neural(args)
    if not args.skip_neural:
        payload["candidate_results"]["A"] = _run_neural(args)
    write_public_probe_artifact(args.output, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
