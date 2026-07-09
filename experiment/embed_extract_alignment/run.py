"""CLI entry point for embed-extract alignment diagnostic experiment.

Usage:
    conda run -n WFCLLM python -m experiment.embed_extract_alignment.run \\
        --secret_key my-key \\
        --model_path /path/to/llm \\
        --n_samples 20 \\
        --output_dir data/diag_reports
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, RobertaTokenizer

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.dataset_loader import load_prompts
from wfcllm.encoder import EncoderConfig, SemanticEncoder
from wfcllm.extract.config import ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.watermark.config import WatermarkConfig

from experiment.embed_extract_alignment.aligner import Aligner
from experiment.embed_extract_alignment.diagnostic_generator import DiagnosticGenerator
from experiment.embed_extract_alignment.models import PromptReport
from experiment.embed_extract_alignment.report import (
    build_summary,
    print_prompt_summary,
    print_summary,
    save_reports,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Embed-extract alignment diagnostic")
    p.add_argument("--secret_key", required=True, help="Watermark secret key")
    p.add_argument("--model_path", required=True, help="Path to LLM (HuggingFace format)")
    p.add_argument("--n_samples", type=int, default=20, help="Number of HumanEval prompts")
    p.add_argument("--output_dir", default="data/diag_reports", help="Output directory")
    p.add_argument("--dataset_path", default="data/datasets", help="Local datasets root")
    p.add_argument("--encoder_path", default="data/models/codet5-base", help="Encoder model path")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--verbose", action="store_true", help="Enable DEBUG logging")
    return p.parse_args()


def load_encoder(encoder_path: str, device: str) -> tuple[SemanticEncoder, RobertaTokenizer]:
    """Load SemanticEncoder (with projection head) in eval mode."""
    enc_config = EncoderConfig(model_name=encoder_path, use_lora=False)
    encoder = SemanticEncoder(config=enc_config).to(device).eval()
    tokenizer = RobertaTokenizer.from_pretrained(encoder_path)
    return encoder, tokenizer


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        stream=sys.stderr,
    )

    print(f"Loading LLM from {args.model_path} ...", file=sys.stderr)
    lm = AutoModelForCausalLM.from_pretrained(args.model_path).to(args.device).eval()
    lm_tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    print(f"Loading encoder from {args.encoder_path} ...", file=sys.stderr)
    encoder, enc_tokenizer = load_encoder(args.encoder_path, args.device)

    wm_config = WatermarkConfig(
        secret_key=args.secret_key,
        encoder_model_path=args.encoder_path,
        encoder_device=args.device,
    )
    extract_config = ExtractConfig(
        secret_key=args.secret_key,
        embed_dim=wm_config.encoder_embed_dim,
        lsh_d=wm_config.lsh_d,
        lsh_gamma=wm_config.lsh_gamma,
    )

    gen = DiagnosticGenerator(lm, lm_tokenizer, encoder, enc_tokenizer, wm_config)
    detector = WatermarkDetector(extract_config, encoder, enc_tokenizer, device=args.device)

    prompts = load_prompts("humaneval", args.dataset_path)[: args.n_samples]
    print(f"Loaded {len(prompts)} prompts. Starting diagnostic run...\n", file=sys.stderr)

    reports: list[PromptReport] = []
    for item in prompts:
        result = gen.generate(item["prompt"])
        embed_events = gen.embed_events

        detection_result = detector.detect(result.code)
        all_blocks = extract_statement_blocks(result.code)
        simple_blocks = [b for b in all_blocks if b.block_type == "simple"]

        report = Aligner.align(
            embed_events=embed_events,
            simple_blocks=simple_blocks,
            all_blocks=all_blocks,
            block_scores=detection_result.block_details,
            generated_code=result.code,
            prompt_id=item["id"],
            detect_z_score=detection_result.z_score,
            detect_is_watermarked=detection_result.is_watermarked,
        )
        reports.append(report)
        print_prompt_summary(report)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    summary = build_summary(reports)
    print_summary(summary)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path, details_path = save_reports(reports, summary, args.output_dir, timestamp)
    print(f"\nSaved: {summary_path}", file=sys.stderr)
    print(f"Saved: {details_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
