"""Contrastive learning trainer for the semantic encoder."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder


def triplet_cosine_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Triplet margin loss using cosine similarity.

    L = max(0, margin - cos(anchor, positive) + cos(anchor, negative))
    """
    cos_pos = F.cosine_similarity(anchor, positive, dim=1)
    cos_neg = F.cosine_similarity(anchor, negative, dim=1)
    loss = torch.clamp(margin - cos_pos + cos_neg, min=0.0)
    return loss.mean()


class ContrastiveTrainer:
    """Training loop for contrastive encoder pretraining.

    Supports optional BF16 mixed precision via config.use_bf16.
    """

    def __init__(
        self,
        model: SemanticEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: EncoderConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Only optimize parameters that require grad (respects LoRA frozen params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.lr)
        total_steps = len(train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=max(total_steps, 1))

        # BF16 autocast context
        self._autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if config.use_bf16 and self.device.type == "cuda"
            else nullcontext()
        )

    def _encode_batch(self, batch: dict, prefix: str) -> torch.Tensor:
        input_ids = batch[f"{prefix}_input_ids"].to(self.device)
        attention_mask = batch[f"{prefix}_attention_mask"].to(self.device)
        return self.model(input_ids, attention_mask)

    def train_epoch(self, epoch: int) -> dict:
        """Run one training epoch. Returns {"loss": float}."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [train]",
            leave=True,
            dynamic_ncols=True,
        )
        for batch in pbar:
            with self._autocast_ctx:
                anchor_emb = self._encode_batch(batch, "anchor")
                positive_emb = self._encode_batch(batch, "positive")
                negative_emb = self._encode_batch(batch, "negative")

                loss = triplet_cosine_loss(
                    anchor_emb, positive_emb, negative_emb, margin=self.config.margin
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{total_loss / n_batches:.4f}")

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self, epoch: int) -> dict:
        """Run validation. Returns {"val_loss": float}."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch}/{self.config.epochs} [val]  ",
            leave=False,
            dynamic_ncols=True,
        )
        for batch in pbar:
            with self._autocast_ctx:
                anchor_emb = self._encode_batch(batch, "anchor")
                positive_emb = self._encode_batch(batch, "positive")
                negative_emb = self._encode_batch(batch, "negative")

                loss = triplet_cosine_loss(
                    anchor_emb, positive_emb, negative_emb, margin=self.config.margin
                )
            total_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(val_loss=f"{total_loss / n_batches:.4f}")

        return {"val_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, epoch: int, metrics: dict) -> Path:
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"encoder_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        return path

    def _export_best_model(self, epoch: int, best_val_loss: float) -> Path:
        """Export best model weights to output_model_dir/best_model.pt."""
        import dataclasses
        output_dir = Path(self.config.output_model_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / "best_model.pt"
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": dataclasses.asdict(self.config),
                "best_metric": best_val_loss,
                "epoch": epoch,
            },
            path,
        )
        return path

    def train(self) -> dict:
        """Full training loop with early stopping and checkpointing."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_metrics: dict = {}

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate(epoch)
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}

            print(
                f"Epoch {epoch}/{self.config.epochs} — "
                f"loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}"
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_metrics = metrics
                self.save_checkpoint(epoch, metrics)
                export_path = self._export_best_model(epoch, best_val_loss)
                print(f"[导出] 最优模型已保存至 {export_path}")
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_metrics
