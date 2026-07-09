#!/usr/bin/env python
"""Run the official WFCLLM generation-time embedding pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.generation.generator import WFCLLMGenerator  # noqa: E402
from wfcllm.generation.pipeline import WFCLLMGenerationPipeline  # noqa: E402
from wfcllm.method.config import (  # noqa: E402
    WFCLLMGenerationConfig,
    WFCLLMPipelineConfig,
    WFCLLMRuleConfig,
)
from wfcllm.semantic.lsh import load_semantic_lsh_rule  # noqa: E402
from wfcllm.semantic.rules import HashEmbeddingRule  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run official WFCLLM generation-time embedding.",
    )
    parser.add_argument("--dataset", default="humaneval", choices=["humaneval", "mbpp"])
    parser.add_argument("--dataset-path", default="data/datasets")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", default="data/wfcllm")
    parser.add_argument("--sample-limit", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.25)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--retry-repetition-penalty",
        type=float,
        default=4.0,
        help=(
            "Custom retry-aware prefix penalty. 1.0 disables it; values above "
            "1.0 penalize the next token that would repeat a rolled-back attempt."
        ),
    )
    parser.add_argument(
        "--torch-dtype",
        default="bf16",
        choices=["auto", "fp32", "fp16", "bf16"],
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--load-in-4bit",
        dest="load_in_4bit",
        action="store_true",
        default=True,
    )
    parser.add_argument("--no-load-in-4bit", dest="load_in_4bit", action="store_false")
    parser.add_argument("--eos-token-id", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        default="completion",
        choices=["completion", "chat"],
        help="Prompt formatting mode. Use completion for official HumanEval prompt completion.",
    )
    parser.add_argument("--max-group-statements", type=int, default=2)
    parser.add_argument("--retry-budget", type=int, default=2)
    parser.add_argument("--statement-retry-budget", type=int, default=4)
    parser.add_argument("--window-retry-budget", type=int, default=3)
    parser.add_argument("--compound-retry-budget", type=int, default=2)
    parser.add_argument("--global-rollback-budget", type=int, default=180)
    parser.add_argument("--max-total-sampled-tokens", type=int, default=32768)
    parser.add_argument(
        "--evidence-retry-attempts",
        type=int,
        default=3,
        help=(
            "Number of generation attempts per sample for evidence-only retry. "
            "Attempts are selected only by WFCLLM audit evidence, never tests or "
            "static correctness proxies."
        ),
    )
    parser.add_argument(
        "--evidence-retry-seed-stride",
        type=int,
        default=101,
        help="Seed stride between evidence-only retry attempts.",
    )
    parser.add_argument(
        "--rule-name",
        default="semantic_lsh",
        choices=["hash", "semantic_lsh"],
    )
    parser.add_argument("--target-accept-rate", type=float, default=0.5)
    parser.add_argument("--encoder-model-path", default="data/models/codet5-base")
    parser.add_argument("--encoder-checkpoint-path", default=None)
    parser.add_argument("--encoder-embed-dim", type=int, default=128)
    parser.add_argument("--encoder-device", default=None)
    parser.add_argument("--encoder-use-lora", action="store_true")
    parser.add_argument("--encoder-use-bf16", action="store_true")
    parser.add_argument("--secret-key", default="1010")
    parser.add_argument("--lsh-d", type=int, default=4)
    parser.add_argument("--lsh-gamma", type=float, default=0.25)
    parser.add_argument("--semantic-margin", type=float, default=0.0)
    parser.add_argument("--lsh-whitening-path", default=None)
    parser.add_argument("--use-ordinal-keying", action="store_true")
    parser.add_argument("--resume", choices=["latest"], default=None)
    parser.add_argument(
        "--candidate-sidecar-output",
        default=None,
        help=(
            "Optional diagnostic-only JSONL path for generation candidate "
            "text/hash sidecar rows. The final JSONL schema is unchanged."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        generation_config = WFCLLMGenerationConfig(
            model_path=args.model_path,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            retry_repetition_penalty=args.retry_repetition_penalty,
            torch_dtype=args.torch_dtype,
            device=args.device,
            seed=args.seed,
            load_in_4bit=args.load_in_4bit,
            eos_token_id=args.eos_token_id,
            prompt_mode=args.prompt_mode,
        )
        rule_parameters: dict[str, object] = {}
        if args.rule_name == "semantic_lsh":
            encoder_device = args.encoder_device or args.device
            rule_parameters = {
                "encoder_model_path": args.encoder_model_path,
                "encoder_checkpoint_path": args.encoder_checkpoint_path,
                "encoder_embed_dim": args.encoder_embed_dim,
                "encoder_device": encoder_device,
                "encoder_use_lora": args.encoder_use_lora,
                "encoder_use_bf16": args.encoder_use_bf16,
                "secret_key": args.secret_key,
                "lsh_d": args.lsh_d,
                "lsh_gamma": args.lsh_gamma,
                "semantic_margin": args.semantic_margin,
                "lsh_whitening_path": args.lsh_whitening_path,
                "use_ordinal_keying": args.use_ordinal_keying,
            }
        rule_config = WFCLLMRuleConfig(
            rule_name=args.rule_name,
            target_accept_rate=args.target_accept_rate,
            parameters=rule_parameters,
        )
        pipeline_config = WFCLLMPipelineConfig(
            dataset=args.dataset,
            dataset_path=args.dataset_path,
            output_dir=args.output_dir,
            generation=generation_config,
            rule=rule_config,
            sample_limit=args.sample_limit,
            sample_offset=args.sample_offset,
            max_group_statements=args.max_group_statements,
            retry_budget=args.retry_budget,
            statement_retry_budget=args.statement_retry_budget,
            window_retry_budget=args.window_retry_budget,
            compound_retry_budget=args.compound_retry_budget,
            global_rollback_budget=args.global_rollback_budget,
            max_total_sampled_tokens=args.max_total_sampled_tokens,
            evidence_retry_attempts=args.evidence_retry_attempts,
            evidence_retry_seed_stride=args.evidence_retry_seed_stride,
            resume=args.resume,
            candidate_sidecar_output=args.candidate_sidecar_output,
        )
        if rule_config.rule_name == "semantic_lsh":
            encoder_device = str(rule_config.parameters["encoder_device"])
            rule = load_semantic_lsh_rule(
                encoder_model_path=str(rule_config.parameters["encoder_model_path"]),
                encoder_checkpoint_path=rule_config.parameters["encoder_checkpoint_path"],  # type: ignore[arg-type]
                embed_dim=int(rule_config.parameters["encoder_embed_dim"]),
                device=encoder_device,
                use_lora=bool(rule_config.parameters["encoder_use_lora"]),
                use_bf16=bool(rule_config.parameters["encoder_use_bf16"]),
                secret_key=str(rule_config.parameters["secret_key"]),
                lsh_d=int(rule_config.parameters["lsh_d"]),
                lsh_gamma=float(rule_config.parameters["lsh_gamma"]),
                margin=float(rule_config.parameters["semantic_margin"]),
                whitening_path=rule_config.parameters["lsh_whitening_path"],  # type: ignore[arg-type]
                use_ordinal_keying=bool(rule_config.parameters["use_ordinal_keying"]),
            )
        else:
            rule = HashEmbeddingRule(target_accept_rate=rule_config.target_accept_rate)
        generator = WFCLLMGenerator(config=generation_config, rule=rule)
        pipeline = WFCLLMGenerationPipeline(generator=generator, config=pipeline_config)
        output_path = pipeline.run()
    except (FileNotFoundError, OSError, ValueError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] WFCLLM final-code rows saved to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
