#!/usr/bin/env python
"""Run the isolated SAWR generation-time embedding smoke pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.sawr.config import (  # noqa: E402
    SawrGenerationConfig,
    SawrPipelineConfig,
    SawrRuleConfig,
)
from wfcllm.sawr.generator import SawrGenerator  # noqa: E402
from wfcllm.sawr.pipeline import SawrPipeline  # noqa: E402
from wfcllm.sawr.rules import HashEmbeddingRule  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run isolated SAWR generation-time embedding smoke.",
    )
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--dataset-path", default="data/datasets")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="data/sawr")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--torch-dtype",
        default="auto",
        choices=["auto", "fp32", "fp16", "bf16"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        default="completion",
        choices=["completion", "chat"],
        help="Prompt formatting mode. Use completion for official HumanEval prompt completion.",
    )
    parser.add_argument("--max-group-statements", type=int, default=2)
    parser.add_argument("--retry-budget", type=int, default=1)
    parser.add_argument("--target-accept-rate", type=float, default=0.5)
    parser.add_argument("--resume", choices=["latest"], default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        generation_config = SawrGenerationConfig(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            torch_dtype=args.torch_dtype,
            device=args.device,
            seed=args.seed,
            load_in_4bit=args.load_in_4bit,
            eos_token_id=args.eos_token_id,
            prompt_mode=args.prompt_mode,
        )
        rule_config = SawrRuleConfig(
            rule_name="hash",
            target_accept_rate=args.target_accept_rate,
        )
        pipeline_config = SawrPipelineConfig(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            generation=generation_config,
            rule=rule_config,
            sample_limit=args.sample_limit,
            sample_offset=args.sample_offset,
            max_group_statements=args.max_group_statements,
            retry_budget=args.retry_budget,
            resume=args.resume,
        )
        rule = HashEmbeddingRule(target_accept_rate=rule_config.target_accept_rate)
        generator = SawrGenerator(config=generation_config, rule=rule)
        pipeline = SawrPipeline(generator=generator, config=pipeline_config)
        output_path = pipeline.run()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] SAWR smoke final rows saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
