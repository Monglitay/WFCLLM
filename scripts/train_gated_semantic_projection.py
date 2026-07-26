#!/usr/bin/env python3
"""Train the public semantic projection used by the gated WFCLLM runtime.

Thin CLI shim; the training logic lives in
``wfcllm.encoder.projection_training`` so the gated ``encoder`` phase can
train one projection per dataset catalog.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.encoder.projection_training import (
    ProjectionTrainingSettings,
    train_semantic_projection,
)


def train(args: argparse.Namespace) -> None:
    train_semantic_projection(
        ProjectionTrainingSettings(
            source_catalog=Path(args.source_catalog),
            model_path=Path(args.model_path),
            output_dir=Path(args.output_dir),
            language=args.language,
            epochs=args.epochs,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_variants=args.max_variants,
            max_perm_len=args.max_perm_len,
            max_train_groups=args.max_train_groups,
            max_validation_groups=args.max_validation_groups,
            max_test_groups=args.max_test_groups,
            embed_dim=args.embed_dim,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            lr=args.lr,
            projection_lr=args.projection_lr,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            seed=args.seed,
        )
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-catalog", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--language", default="python")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--max-variants", type=int, default=8)
    parser.add_argument("--max-perm-len", type=int, default=2)
    parser.add_argument("--max-train-groups", type=int, default=1200)
    parser.add_argument("--max-validation-groups", type=int, default=160)
    parser.add_argument("--max-test-groups", type=int, default=160)
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--projection-lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=731)
    return parser


if __name__ == "__main__":
    train(_parser().parse_args())
