#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.dynamic_semantic.calibration import fit_calibration, save_calibration
from wfcllm.dynamic_semantic.channel import SemanticChannel, aggregate_unit_evidence
from wfcllm.dynamic_semantic.config import load_public_config
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.encoder import SemanticEncoderRuntime
from wfcllm.dynamic_semantic.keying import load_secret_key
from wfcllm.dynamic_semantic.whitening import fit_whitening, save_whitening


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fit frozen Dynamic Semantic V3 artifacts.")
    parser.add_argument("--public-config", required=True)
    parser.add_argument("--split-manifest", required=True)
    parser.add_argument("--calibration-jsonl", required=True)
    parser.add_argument("--key-file", required=True)
    parser.add_argument("--whitening-output", required=True)
    parser.add_argument("--calibration-output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    return parser


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = load_public_config(args.public_config)
    manifest = json.loads(Path(args.split_manifest).read_text(encoding="utf-8"))
    input_path = Path(args.calibration_jsonl)
    input_sha = _sha256(input_path)
    if input_sha != manifest["calibration"]["jsonl_sha256"]:
        raise ValueError("calibration JSONL does not match frozen split manifest")
    rows = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line]
    runtime = SemanticEncoderRuntime.load(
        config.encoder,
        max_tokens=config.context.max_context_tokens,
        device=args.device,
    )
    extractor = DynamicContextExtractor(config.context, token_counter=runtime.token_count)
    extracted = [extractor.extract(str(row["generated_code"])) for row in rows]
    contexts = [context for result in extracted for context in result.contexts]
    embeddings = []
    for start in range(0, len(contexts), args.batch_size):
        batch = contexts[start : start + args.batch_size]
        embeddings.append(runtime.encode(tuple(item.serialized for item in batch)).cpu())
    if not embeddings:
        raise ValueError("calibration corpus produced no semantic contexts")
    all_embeddings = torch.cat(embeddings, dim=0)
    whitening = fit_whitening(
        all_embeddings,
        output_dimensions=config.channel.whitening_dimensions,
        source_role="calibration_negative",
        source_manifest_sha256=input_sha,
    )
    save_whitening(whitening, args.whitening_output)
    transformed = whitening.transform(all_embeddings)
    channel = SemanticChannel(load_secret_key(args.key_file), config.channel)
    scores = []
    offset = 0
    for result in extracted:
        count = len(result.contexts)
        evidence = tuple(
            channel.score(context.unit_id, context.context_sha256, vector.tolist())
            for context, vector in zip(result.contexts, transformed[offset : offset + count], strict=True)
        )
        offset += count
        aggregate = aggregate_unit_evidence(
            evidence,
            minimum_independent_units=config.channel.minimum_independent_units,
        )
        scores.append(aggregate.score if aggregate.eligible else -1.0)
    calibration = fit_calibration(
        scores,
        source_role="calibration_negative",
        source_manifest_sha256=input_sha,
        target_fpr=0.05,
    )
    save_calibration(calibration, args.calibration_output)
    summary = {
        "schema_version": "wfcllm-dynamic-semantic-fit-summary/v3",
        "calibration_rows": len(rows),
        "semantic_contexts": len(contexts),
        "encoder_calls": runtime.encoder_calls,
        "calibration_jsonl_sha256": input_sha,
        "whitening_sha256": _sha256(Path(args.whitening_output)),
        "calibration_sha256": _sha256(Path(args.calibration_output)),
    }
    output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
