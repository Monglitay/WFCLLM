"""Training entry point for the semantic encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None  # type: ignore[assignment]

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.transform.engine import TransformEngine
from wfcllm.common.transform.positive import get_all_positive_rules
from wfcllm.common.transform.negative import get_all_negative_rules
from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.dataset import TripletCodeDataset, build_triplets_from_blocks
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.trainer import ContrastiveTrainer
from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    watermark_sign_consistency,
    mean_reciprocal_rank,
    mean_average_precision,
    pair_f1_metrics,
    save_evaluation_report,
)


def load_code_samples(
    data_sources: list[str],
    local_dataset_dir: str | None = None,
) -> list[dict]:
    """Load code samples from HuggingFace datasets.

    Args:
        data_sources: Dataset names ("mbpp", "humaneval").
        local_dataset_dir: Optional local cache root, e.g. "data/datasets".
            Each dataset lives at <local_dataset_dir>/<name>/.
            If None, uses HF default global cache.
    """
    samples: list[dict] = []

    for source in data_sources:
        if source == "mbpp":
            cache_dir = str(Path(local_dataset_dir) / "mbpp") if local_dataset_dir else None
            ds = load_dataset(
                "google-research-datasets/mbpp", "full",
                cache_dir=cache_dir,
                download_mode="reuse_cache_if_exists",
            )
            for split in ds:
                for item in ds[split]:
                    samples.append({"code": item["code"], "source": "mbpp"})
        elif source == "humaneval":
            cache_dir = str(Path(local_dataset_dir) / "humaneval") if local_dataset_dir else None
            ds = load_dataset(
                "openai/openai_humaneval",
                cache_dir=cache_dir,
                download_mode="reuse_cache_if_exists",
            )
            for split in ds:
                for item in ds[split]:
                    canonical = item.get("canonical_solution", "")
                    prompt = item.get("prompt", "")
                    code = prompt + canonical
                    samples.append({"code": code, "source": "humaneval"})

    return samples


def prepare_blocks_with_variants(
    code_samples: list[dict],
    max_variants: int = 100,
    max_perm_len: int = 3,
) -> list[dict]:
    """Extract statement blocks and generate positive/negative variants.

    Returns list of dicts with keys: source, positive_variants, negative_variants.
    """
    pos_engine = TransformEngine(
        rules=get_all_positive_rules(),
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode="positive",
    )
    neg_engine = TransformEngine(
        rules=get_all_negative_rules(),
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode="negative",
    )

    blocks: list[dict] = []

    for sample in code_samples:
        code = sample["code"]
        stmt_blocks = extract_statement_blocks(code)

        for sb in stmt_blocks:
            pos_variants = pos_engine.generate_variants(sb.source)
            neg_variants = neg_engine.generate_variants(sb.source)

            blocks.append({
                "source": sb.source,
                "positive_variants": [v["transformed_source"] for v in pos_variants],
                "negative_variants": [v["transformed_source"] for v in neg_variants],
            })

    return blocks


def main(config: EncoderConfig | None = None) -> None:
    """Full training pipeline."""
    if config is None:
        config = EncoderConfig()

    print("=== Phase 1: Semantic Encoder Pretraining ===")

    # Resolve model path: prefer local if available
    local_codet5 = Path(config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        effective_model = str(local_codet5)
        print(f"  Using local model: {effective_model}")
    else:
        effective_model = config.model_name
        print(f"  Using HF Hub model: {effective_model}")

    # 1. Load data
    print(f"Loading data from {config.data_sources}...")
    code_samples = load_code_samples(config.data_sources, local_dataset_dir=config.local_dataset_dir)
    print(f"  Loaded {len(code_samples)} code samples")

    # 2. Prepare blocks with variants
    print("Extracting blocks and generating variants...")
    blocks = prepare_blocks_with_variants(code_samples)
    print(f"  Generated {len(blocks)} blocks")

    # 3. Build triplets
    print("Building triplets...")
    triplets = build_triplets_from_blocks(
        blocks, negative_ratio=config.negative_ratio, seed=42
    )
    print(f"  Built {len(triplets)} triplets")

    # 4. Create datasets
    tokenizer = AutoTokenizer.from_pretrained(effective_model)
    dataset = TripletCodeDataset(triplets, tokenizer, max_length=config.max_seq_length)

    # Split: 80% train, 10% val, 10% test
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    _split_gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=_split_gen)

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 5. Train
    from dataclasses import replace
    config_for_model = replace(config, model_name=effective_model)
    model = SemanticEncoder(config=config_for_model)
    trainer = ContrastiveTrainer(model, train_loader, val_loader, config_for_model)

    print("Starting training...")
    best_metrics = trainer.train()
    print(f"Training complete. Best val_loss: {best_metrics.get('val_loss', 'N/A')}")

    # 6. Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    device = trainer.device

    all_anchor, all_positive, all_negative = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            a = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            n = model(batch["negative_input_ids"].to(device), batch["negative_attention_mask"].to(device))
            all_anchor.append(a.cpu())
            all_positive.append(p.cpu())
            all_negative.append(n.cpu())

    anchor_embs = torch.cat(all_anchor)
    pos_embs = torch.cat(all_positive)
    neg_embs = torch.cat(all_negative)

    sep_metrics = cosine_separation(anchor_embs, pos_embs, neg_embs)
    r1 = recall_at_k(anchor_embs, pos_embs, k=1)
    r5 = recall_at_k(anchor_embs, pos_embs, k=5)
    r10 = recall_at_k(anchor_embs, pos_embs, k=10)
    mrr = mean_reciprocal_rank(anchor_embs, pos_embs)
    map_score = mean_average_precision(anchor_embs, pos_embs)
    wsc = watermark_sign_consistency(anchor_embs, pos_embs, num_directions=64, seed=42)

    # pair_f1: positive pairs vs negative pairs
    pos_cos = torch.nn.functional.cosine_similarity(anchor_embs, pos_embs, dim=1)
    neg_cos = torch.nn.functional.cosine_similarity(anchor_embs, neg_embs, dim=1)
    f1_metrics = pair_f1_metrics(pos_cos, neg_cos)

    eval_metrics = {
        **sep_metrics,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": mrr,
        "map": map_score,
        "watermark_sign_consistency": wsc,
        **f1_metrics,
        **best_metrics,
    }

    print("\n=== Evaluation Results ===")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    report_path = save_evaluation_report(eval_metrics, config.results_dir)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the semantic encoder")
    parser.add_argument("--model-name", default="Salesforce/codet5-base")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetune)")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 (use FP32)")
    args = parser.parse_args()

    config = EncoderConfig(
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        margin=args.margin,
        max_seq_length=args.max_seq_length,
        use_lora=not args.no_lora,
        use_bf16=not args.no_bf16,
    )
    main(config)
