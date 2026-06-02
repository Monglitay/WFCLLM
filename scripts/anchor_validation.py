#!/usr/bin/env python
"""Anchor effectiveness diagnostic CLI."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.evaluation.anchor_validation.candidate_generation import (  # noqa: E402
    GenerationContextSource,
    build_hf_sampler,
    generate_candidate_rows,
)
from wfcllm.evaluation.anchor_validation.io import (  # noqa: E402
    read_jsonl,
    write_candidate_contexts,
    write_jsonl,
)
from wfcllm.evaluation.anchor_validation.pool_builder import (  # noqa: E402
    build_candidate_contexts_from_records,
)
from wfcllm.evaluation.anchor_validation.runner import (  # noqa: E402
    AnchorValidationConfig,
    AnchorValidationRunner,
)

DEFAULT_MAIN_METHODS = (
    "vanilla",
    "random",
    "context",
    "slot_context",
    "slot_context_skeleton",
    "role_aware_slot_context",
    "role_aware_slot_context_skeleton",
    "seqmark_oracle",
)


def _cmd_generate_pool(args: argparse.Namespace) -> int:
    sources = _load_generation_sources(tuple(Path(path) for path in args.source_jsonl))
    if args.sampler_mode == "echo":
        sampler = lambda prompt, temperature, sample_index: "pass"
    else:
        if not args.lm_model_path:
            raise ValueError("--lm-model-path is required with --sampler-mode hf")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
        model = AutoModelForCausalLM.from_pretrained(
            args.lm_model_path,
            device_map=args.device_map,
        )
        sampler = build_hf_sampler(model, tokenizer, max_new_tokens=args.max_new_tokens)

    rows = generate_candidate_rows(
        sources=tuple(sources),
        sampler=sampler,
        temperatures=tuple(float(value) for value in args.temperatures),
        candidates_per_temperature=args.candidates_per_temperature,
        max_contexts_per_source=args.max_contexts_per_source,
    )
    write_jsonl(Path(args.output), rows)
    print(f"[anchor-validation] wrote {len(rows)} per-block candidate rows to {args.output}")
    return 0


def _load_generation_sources(paths: tuple[Path, ...]) -> list[GenerationContextSource]:
    sources: list[GenerationContextSource] = []
    for path in paths:
        for row in read_jsonl(path):
            prompt = str(row.get("prompt", ""))
            generated = str(row.get("generated_code", row.get("solution", "")))
            source_code = str(row.get("source_code", prompt + generated))
            sources.append(
                GenerationContextSource(
                    dataset=str(row.get("dataset", "unknown")),
                    task_id=str(row.get("id", row.get("task_id", ""))),
                    prompt=prompt,
                    source_code=source_code,
                )
            )
    return sources


def _cmd_build_pool(args: argparse.Namespace) -> int:
    records = []
    for path in args.input_jsonl:
        records.extend(read_jsonl(Path(path)))
    contexts = build_candidate_contexts_from_records(
        records,
        min_candidates=args.min_candidates,
        max_contexts_per_task=args.max_contexts_per_task,
    )
    write_candidate_contexts(Path(args.output), contexts)
    print(f"[anchor-validation] wrote {len(contexts)} contexts to {args.output}")
    return 0


def _cmd_run_diagnostics(args: argparse.Namespace) -> int:
    config = AnchorValidationConfig(
        pool_path=Path(args.pool),
        output_dir=Path(args.output_dir),
        secret_keys=tuple(args.secret_key),
        gammas=tuple(float(value) for value in args.gammas),
        methods=tuple(args.methods),
        retry_budgets=tuple(int(value) for value in args.retry_budgets),
        lsh_d=args.lsh_d,
        embed_dim=args.embed_dim,
        embedding_mode=args.embedding_mode,
        encoder_model_path=args.encoder_model_path,
        encoder_checkpoint=Path(args.encoder_checkpoint) if args.encoder_checkpoint else None,
        encoder_device=args.encoder_device,
        max_length=args.max_length,
        use_ordinal_keying=not args.legacy_parent_keying,
    )
    result = AnchorValidationRunner(config).run()
    print(f"[anchor-validation] metrics: {result.metrics_path}")
    print(f"[anchor-validation] selection: {result.selection_path}")
    print(f"[anchor-validation] summary: {result.summary_path}")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Anchor effectiveness diagnostics")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-pool")
    generate.add_argument("--source-jsonl", nargs="+", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--sampler-mode", choices=["hf", "echo"], default="hf")
    generate.add_argument("--lm-model-path", default=None)
    generate.add_argument("--device-map", default="auto")
    generate.add_argument("--max-new-tokens", type=int, default=64)
    generate.add_argument("--temperatures", nargs="+", default=["0.2", "0.4", "0.7"])
    generate.add_argument("--candidates-per-temperature", type=int, default=16)
    generate.add_argument("--max-contexts-per-source", type=int, default=None)
    generate.set_defaults(func=_cmd_generate_pool)

    build = subparsers.add_parser("build-pool")
    build.add_argument("--input-jsonl", nargs="+", required=True)
    build.add_argument("--output", required=True)
    build.add_argument("--min-candidates", type=int, default=2)
    build.add_argument("--max-contexts-per-task", type=int, default=None)
    build.set_defaults(func=_cmd_build_pool)

    run = subparsers.add_parser("run-diagnostics")
    run.add_argument("--pool", required=True)
    run.add_argument("--output-dir", default="data/diagnostics/anchor_validation")
    run.add_argument("--embedding-mode", choices=["hash", "encoder"], default="hash")
    run.add_argument("--embed-dim", type=int, default=128)
    run.add_argument("--encoder-model-path", default="data/models/codet5-base")
    run.add_argument("--encoder-checkpoint", default=None)
    run.add_argument("--encoder-device", default="cpu")
    run.add_argument("--max-length", type=int, default=256)
    run.add_argument("--lsh-d", type=int, default=3)
    run.add_argument("--secret-key", nargs="+", required=True)
    run.add_argument(
        "--legacy-parent-keying",
        action="store_true",
        help="derive valid sets from parent node type only; default uses block ordinal",
    )
    run.add_argument("--methods", nargs="+", default=list(DEFAULT_MAIN_METHODS))
    run.add_argument("--gammas", nargs="+", default=["0.5"])
    run.add_argument("--retry-budgets", nargs="+", default=["1", "4", "8", "16"])
    run.set_defaults(func=_cmd_run_diagnostics)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
