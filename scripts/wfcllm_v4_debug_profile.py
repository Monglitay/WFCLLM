#!/usr/bin/env python3
"""Run one cold-process V4 debug, invariance, cache, and cost profile."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.batch_invariant_v4.audit import (  # noqa: E402
    compare_evidence_maps,
    evidence_sha256,
    load_retry_ledgers,
    replay_schedule,
    rows_to_candidates,
    scheduled_indices,
)
from wfcllm.batch_invariant_v4.cache import PublicContextCache  # noqa: E402
from wfcllm.batch_invariant_v4.calibration import load_calibration  # noqa: E402
from wfcllm.batch_invariant_v4.channel import StructuralChannel  # noqa: E402
from wfcllm.batch_invariant_v4.config import load_public_config  # noqa: E402
from wfcllm.batch_invariant_v4.context import (  # noqa: E402
    ContextConfig,
    StructuralContextExtractor,
)
from wfcllm.batch_invariant_v4.detector import (  # noqa: E402
    DetectorPayload,
    V4Detector,
    exact_code_evidence_mismatches,
)
from wfcllm.batch_invariant_v4.keying import load_secret_key  # noqa: E402
from wfcllm.batch_invariant_v4.runtime import CandidateRuntime  # noqa: E402


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public-config", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--key-file", type=Path, required=True)
    parser.add_argument("--retry-ledger", type=Path, action="append", required=True)
    parser.add_argument("--sample-id", action="append", required=True)
    parser.add_argument("--process-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _build_detector(args: argparse.Namespace) -> tuple[Any, Any, V4Detector, str]:
    public_config = load_public_config(args.public_config)
    calibration = load_calibration(args.calibration)
    key = load_secret_key(args.key_file)
    public_sha256 = _sha256_file(args.public_config)
    extractor = StructuralContextExtractor(
        ContextConfig(
            schema_version=public_config.canonical_context.schema_version,
            max_unit_bytes=public_config.canonical_context.max_unit_bytes,
            max_context_bytes=public_config.canonical_context.max_context_bytes,
            global_ordinal_keying=public_config.canonical_context.global_ordinal_keying,
        )
    )
    detector = V4Detector(
        extractor=extractor,
        channel=StructuralChannel(
            key,
            bit_count=public_config.channel.bit_count_per_unit,
            minimum_independent_units=public_config.channel.minimum_independent_units,
        ),
        calibration=calibration,
    )
    return public_config, extractor, detector, public_sha256


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        import torch

        load_started = time.perf_counter()
        public_config, extractor, detector, public_sha256 = _build_detector(args)
        grouped = load_retry_ledgers(args.retry_ledger)
        missing = sorted(set(args.sample_id) - set(grouped))
        if missing:
            raise ValueError(f"retry ledger missing sample IDs: {missing}")
        model_load_seconds = time.perf_counter() - load_started

        warmup_started = time.perf_counter()
        detector.detect(DetectorPayload(final_code="def warmup():\n    return 0\n"))
        warmup_seconds = time.perf_counter() - warmup_started
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        cache = PublicContextCache(public_config_sha256=public_sha256)
        tasks: list[dict[str, Any]] = []
        reference_global: dict[str, str] = {}
        global_candidates: list[tuple[str, int, str]] = []
        batch_sizes = (1, 2, 4, 8, 16, 32)
        orders = ("forward", "reverse", "permutation", "short_first", "long_first")
        for sample_id in args.sample_id:
            rows = grouped[sample_id]
            candidates = rows_to_candidates(rows, retry=20)
            runtime = CandidateRuntime(detector, public_config_sha256=public_sha256)

            selection_started = time.perf_counter()
            selection = runtime.select(candidates, retry=20)
            dynamic_selection_seconds = time.perf_counter() - selection_started
            replay_started = time.perf_counter()
            replay = runtime.replay_selected(selection.selected.final_code)
            selected_final_replay_seconds = time.perf_counter() - replay_started
            replay_mismatches = exact_code_evidence_mismatches(
                selection.selected_generation_evidence,
                replay,
            )

            reference = selection.evidence_by_attempt
            schedule_rows: list[dict[str, Any]] = []
            for batch_size in batch_sizes:
                for order in orders:
                    scheduled = replay_schedule(
                        detector,
                        candidates,
                        scheduled_indices(
                            candidates,
                            batch_size=batch_size,
                            order=order,
                        ),
                    )
                    mismatches = compare_evidence_maps(reference, scheduled)
                    schedule_rows.append(
                        {
                            "batch_size": batch_size,
                            "order": order,
                            "exact": not mismatches,
                            "mismatches": list(mismatches),
                        }
                    )

            task_cache_exact = True
            task_context_count = 0
            for candidate in candidates:
                extraction = extractor.extract(candidate.final_code)
                cache.flush_order(
                    tuple(context.context_sha256 for context in extraction.contexts)
                )
                for context in extraction.contexts:
                    miss = cache.get_or_create(
                        context,
                        lambda context=context: context.representation_bytes,
                    )
                    hit = cache.get_or_create(
                        context,
                        lambda: (_ for _ in ()).throw(
                            AssertionError("cache hit invoked factory")
                        ),
                    )
                    task_cache_exact = task_cache_exact and (
                        miss == hit == context.representation_bytes
                    )
                    task_context_count += 1

            current_rows = [row for row in rows if bool(row.get("selected"))]
            if len(current_rows) != 1:
                raise ValueError(f"{sample_id} must have one frozen Current selection")
            candidate_hashes = [candidate.final_code_sha256 for candidate in candidates]
            evidence_hashes = {
                str(attempt): evidence_sha256(evidence)
                for attempt, evidence in sorted(reference.items())
            }
            for candidate in candidates:
                identity = f"{sample_id}/{candidate.attempt_index}"
                reference_global[identity] = evidence_hashes[str(candidate.attempt_index)]
                global_candidates.append(
                    (identity, candidate.attempt_index, candidate.final_code)
                )
            tasks.append(
                {
                    "id": sample_id,
                    "retry": 20,
                    "candidate_sha256": candidate_hashes,
                    "candidate_pool_hash_match_rate": selection.candidate_pool_match_rate,
                    "input_pool_sha256": selection.input_pool_sha256,
                    "output_pool_sha256": selection.output_pool_sha256,
                    "current_selected_attempt": int(current_rows[0]["attempt_index"]),
                    "v4_selected_attempt": selection.selected.attempt_index,
                    "selected_generation_evidence": asdict(
                        selection.selected_generation_evidence
                    ),
                    "selected_r3_evidence": asdict(replay),
                    "selected_generation_to_r3_exact": not replay_mismatches,
                    "selected_generation_to_r3_mismatches": list(replay_mismatches),
                    "candidate_evidence_sha256": evidence_hashes,
                    "schedule_conditions": schedule_rows,
                    "all_schedule_conditions_exact": all(
                        row["exact"] for row in schedule_rows
                    ),
                    "cache_hit_miss_exact": task_cache_exact,
                    "context_count": task_context_count,
                    "dynamic_selection_seconds": dynamic_selection_seconds,
                    "selected_final_replay_seconds": selected_final_replay_seconds,
                    "semantic_seconds": (
                        dynamic_selection_seconds + selected_final_replay_seconds
                    ),
                    "selected_final_replay_count": runtime.selected_final_replay_count,
                    "eos_all_candidate_neural_rescore_count": (
                        runtime.eos_all_candidate_neural_rescore_count
                    ),
                }
            )

        # Cross-task adversarial composition: failure and new debug contexts are
        # intentionally interleaved, reversed, length-sorted, and permuted.
        global_orders: dict[str, list[tuple[str, int, str]]] = {
            "forward": list(global_candidates),
            "reverse": list(reversed(global_candidates)),
            "short_first": sorted(global_candidates, key=lambda row: (len(row[2]), row[0])),
            "long_first": sorted(global_candidates, key=lambda row: (-len(row[2]), row[0])),
        }
        permuted = list(global_candidates)
        random.Random(20260714).shuffle(permuted)
        global_orders["permutation"] = permuted
        composition_rows: list[dict[str, Any]] = []
        for order, values in global_orders.items():
            mismatches: list[str] = []
            for identity, _attempt, final_code in values:
                candidate = detector.detect(DetectorPayload(final_code=final_code))
                if evidence_sha256(candidate) != reference_global[identity]:
                    mismatches.append(identity)
            composition_rows.append(
                {
                    "order": order,
                    "candidate_count": len(values),
                    "exact": not mismatches,
                    "mismatches": mismatches,
                }
            )

        exact_identity = {
            "task_order": list(args.sample_id),
            "tasks": [
                {
                    "id": row["id"],
                    "candidate_sha256": row["candidate_sha256"],
                    "current_selected_attempt": row["current_selected_attempt"],
                    "v4_selected_attempt": row["v4_selected_attempt"],
                    "candidate_evidence_sha256": row["candidate_evidence_sha256"],
                    "selected_generation_evidence": row[
                        "selected_generation_evidence"
                    ],
                    "selected_r3_evidence": row["selected_r3_evidence"],
                }
                for row in tasks
            ],
        }
        semantic_seconds = [row["semantic_seconds"] for row in tasks]
        peak_allocated_mib = (
            torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        )
        result = {
            "artifact_type": "wfcllm_v4_debug_profile",
            "schema_version": "wfcllm-v4-debug-profile/v1",
            "process_id": args.process_id,
            "secret_metadata_included": False,
            "public_config_sha256": public_sha256,
            "calibration_sha256": _sha256_file(args.calibration),
            "retry_ledger_sha256": [_sha256_file(path) for path in args.retry_ledger],
            "runtime_contract": {
                "device": public_config.runtime.device,
                "encoder_used": public_config.runtime.encoder_used,
                "neural_auxiliary_used": public_config.runtime.neural_auxiliary_used,
                "deterministic_algorithms": True,
                "tf32": False,
                "float32_matmul_precision": "not_applicable",
                "model_eval": "not_applicable",
                "cuda_required": public_config.runtime.cuda_required,
            },
            "environment": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "gpu_name": (
                    torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
                ),
            },
            "model_load_seconds": model_load_seconds,
            "warmup_seconds": warmup_seconds,
            "tasks": tasks,
            "cross_task_composition": composition_rows,
            "cache": {
                "hits": cache.hits,
                "misses": cache.misses,
                "hit_rate": cache.hits / (cache.hits + cache.misses),
                "key_is_public_only": True,
            },
            "summary": {
                "task_count": len(tasks),
                "candidate_count": sum(row["retry"] for row in tasks),
                "candidate_pool_match_rate": sum(
                    row["candidate_pool_hash_match_rate"] for row in tasks
                )
                / len(tasks),
                "generation_to_r3_exact_rate": sum(
                    row["selected_generation_to_r3_exact"] for row in tasks
                )
                / len(tasks),
                "schedule_exact_rate": sum(
                    row["all_schedule_conditions_exact"] for row in tasks
                )
                / len(tasks),
                "cache_hit_miss_exact_rate": sum(
                    row["cache_hit_miss_exact"] for row in tasks
                )
                / len(tasks),
                "cross_task_composition_exact_rate": sum(
                    row["exact"] for row in composition_rows
                )
                / len(composition_rows),
                "mean_semantic_seconds_per_task": sum(semantic_seconds)
                / len(semantic_seconds),
                "mean_dynamic_selection_seconds_per_task": sum(
                    row["dynamic_selection_seconds"] for row in tasks
                )
                / len(tasks),
                "mean_selected_final_replay_seconds_per_task": sum(
                    row["selected_final_replay_seconds"] for row in tasks
                )
                / len(tasks),
                "peak_allocated_mib": peak_allocated_mib,
                "selected_final_replay_count": sum(
                    row["selected_final_replay_count"] for row in tasks
                ),
                "eos_all_candidate_neural_rescore_count": sum(
                    row["eos_all_candidate_neural_rescore_count"] for row in tasks
                ),
                "exact_identity_sha256": _canonical_sha256(exact_identity),
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    except (AssertionError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
