#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.datasets.loaders.local import load_prompts
from wfcllm.dynamic_semantic.calibration import load_calibration
from wfcllm.dynamic_semantic.channel import SemanticChannel
from wfcllm.dynamic_semantic.config import load_public_config
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.controller import DynamicSemanticController
from wfcllm.dynamic_semantic.detector import R3Detector, compare_exact_replay
from wfcllm.dynamic_semantic.encoder import SemanticEncoderRuntime
from wfcllm.dynamic_semantic.keying import load_secret_key
from wfcllm.dynamic_semantic.scheduler import DynamicSemanticScheduler
from wfcllm.dynamic_semantic.whitening import load_whitening
from wfcllm.generation.selection_v2 import RetryAttempt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Replay frozen retry pools through generation-time V3 observers.")
    parser.add_argument("--public-config", required=True)
    parser.add_argument("--whitening", required=True)
    parser.add_argument("--calibration", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--retry-ledger", action="append", required=True)
    parser.add_argument("--dataset-path", default="data/datasets")
    parser.add_argument("--sample-id", action="append")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    return parser


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_ledgers(paths: list[str]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for path in paths:
        for raw in Path(path).read_text(encoding="utf-8").splitlines():
            if raw:
                row = json.loads(raw)
                grouped[str(row["id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["attempt_index"]))
    return grouped


def _attempt(row: dict) -> RetryAttempt:
    result = SimpleNamespace(
        final_code=str(row["final_code"]),
        accepted_hit_count=int(row["v1_accepted_hit_count"]),
        closed_without_hit_count=int(row["v1_closed_without_hit_count"]),
        fallback_count=int(row["v1_fallback_count"]),
        candidate_count=int(row["v1_candidate_count"]),
    )
    return RetryAttempt(
        attempt_index=int(row["attempt_index"]),
        seed=int(row["seed"]),
        result=result,
    )


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_public_config(args.public_config)
    runtime = SemanticEncoderRuntime.load(config.encoder, max_tokens=config.context.max_context_tokens, device=args.device)
    extractor = DynamicContextExtractor(config.context, token_counter=runtime.token_count)
    whitening = load_whitening(args.whitening)
    channel = SemanticChannel(load_secret_key(args.key_file), config.channel)
    calibration = load_calibration(args.calibration)
    warmup_text = "WFCLLM_DYNAMIC_SEMANTIC_CONTEXT_V3\nrole=Return|FunctionDef|body\nprevious=<BOS>\ncurrent=return 0"
    warmup_embedding = runtime.encode((warmup_text,))
    warmup_vector = whitening.transform(warmup_embedding)[0]
    channel.score("warmup-unit", "warmup-context", warmup_vector.tolist())
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    grouped = _read_ledgers(args.retry_ledger)
    prompt_rows = load_prompts("humaneval", args.dataset_path)
    prompts = {str(row["id"]): str(row["prompt"]) for row in prompt_rows}
    ids = args.sample_id or sorted(grouped, key=lambda item: int(item.split("/")[-1]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    v3_file = (output_dir / "v3_final_code.jsonl").open("w", encoding="utf-8")
    current_file = (output_dir / "current_final_code.jsonl").open("w", encoding="utf-8")
    ledger_file = (output_dir / "v3_attempt_ledger.jsonl").open("w", encoding="utf-8")
    replay_file = (output_dir / "r3_replay.jsonl").open("w", encoding="utf-8")
    summaries = []
    try:
        for sample_id in ids:
            rows = grouped[sample_id]
            if len(rows) != 20 or [int(row["attempt_index"]) for row in rows] != list(range(20)):
                raise ValueError(f"{sample_id} does not have exactly ordered retry-20 rows")
            before_calls = runtime.encoder_calls
            before_contexts = runtime.encoded_contexts
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            scheduler = DynamicSemanticScheduler(
                encoder=runtime,
                whitening=whitening,
                channel=channel,
                config=config.scheduler,
            )
            controller = DynamicSemanticController(
                extractor=extractor,
                scheduler=scheduler,
                minimum_independent_units=config.channel.minimum_independent_units,
            )
            attempts = tuple(_attempt(row) for row in rows)
            for attempt in attempts:
                observer = controller.observer_for_attempt(attempt.attempt_index)
                code = str(attempt.result.final_code)
                prefix = ""
                for line in code.splitlines(keepends=True):
                    prefix += line
                    observer.observe_prefix(prefix)
                observer.flush(code)
                controller.attempt_completed(attempt.attempt_index)
            selection = controller.select(sample_id=sample_id, prompt=prompts[sample_id], attempts=attempts)
            selection_seconds = time.perf_counter() - started
            detector = R3Detector(
                extractor=extractor,
                encoder=runtime,
                whitening=whitening,
                channel=channel,
                calibration=calibration,
                minimum_independent_units=config.channel.minimum_independent_units,
            )
            replay = detector.detect(str(selection.result.final_code))
            semantic_seconds = time.perf_counter() - started
            exact = compare_exact_replay(controller.selected_evidence, replay.evidence)
            current_row = next(row for row in rows if bool(row["selected"]))
            v3_file.write(json.dumps({"id": sample_id, "dataset": "humaneval", "prompt": prompts[sample_id], "final_code": selection.result.final_code}, ensure_ascii=False) + "\n")
            current_file.write(json.dumps({"id": sample_id, "dataset": "humaneval", "prompt": prompts[sample_id], "final_code": current_row["final_code"]}, ensure_ascii=False) + "\n")
            for row in selection.ledger_rows:
                ledger_file.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            replay_public = replay.public_dict()
            replay_public.update({"id": sample_id, "exact_replay": exact.exact, "mismatches": list(exact.mismatches)})
            replay_file.write(json.dumps(replay_public, ensure_ascii=False, sort_keys=True) + "\n")
            summaries.append({
                "id": sample_id,
                "candidate_pool_hashes": [_hash(str(row["final_code"])) for row in rows],
                "selected_attempt": selection.attempt_index,
                "current_selected_attempt": int(current_row["attempt_index"]),
                "r3_exact": exact.exact,
                "semantic_seconds_including_replay": semantic_seconds,
                "dynamic_selection_seconds": selection_seconds,
                "selected_final_replay_seconds": semantic_seconds - selection_seconds,
                "encoder_calls": runtime.encoder_calls - before_calls,
                "encoded_contexts": runtime.encoded_contexts - before_contexts,
                "batch_sizes": scheduler.encoded_batch_sizes,
                "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0,
            })
    finally:
        v3_file.close()
        current_file.close()
        ledger_file.close()
        replay_file.close()
    (output_dir / "run_summary.json").write_text(json.dumps({"schema_version": "wfcllm-dynamic-semantic-pool-replay/v3", "tasks": summaries}, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
