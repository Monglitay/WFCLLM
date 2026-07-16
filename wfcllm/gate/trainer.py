"""Deterministic, fail-closed training for the first semantic gate."""

from __future__ import annotations

import copy
import json
import math
import os
import random
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Callable

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from wfcllm.gate.dataset import (
    GateCollator,
    GateDataset,
    GateExample,
    GroupConsistencyBatchSampler,
)
from wfcllm.gate.losses import GateLoss

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_CONSISTENCY_EXTRA_PADDING = 7
_INVARIANCE_ATOL = 1e-6
_CHECKPOINT_CONTRACT = "wfcllm-gate-training-checkpoint/v1"
_METRICS_CONTRACT = "wfcllm-gate-training-metrics/v1"
_SUMMARY_CONTRACT = "wfcllm-gate-development-summary/v1"
_LOSS_NAMES = (
    "close_bce",
    "suitable_bce",
    "dangerous_negative_fp",
    "context_consistency",
    "batch_consistency",
    "quantization_consistency",
)
_VALIDATION_KEYS = {
    "total_count",
    "evaluable_count",
    "overflow_count",
    "negative_count",
    "comparable_group_count",
    "coverage",
    "suitable_false_positive_rate",
    "decision_consistency",
}
_CHECKPOINT_KEYS = {
    "contract_version",
    "config_hash",
    "dataset_manifest_hash",
    "epoch",
    "model_state",
    "optimizer_state",
    "scheduler_state",
    "validation",
    "training_state",
}
_TRAINING_STATE_KEYS = {
    "best_objective",
    "best_epoch",
    "best_validation",
    "patience",
    "status",
    "epochs_completed",
    "optimizer_steps",
}
_METRIC_ROW_KEYS = {
    "contract_version",
    "config_hash",
    "dataset_manifest_hash",
    "epoch",
    "total_loss",
    "loss_components",
    "validation",
    "best",
    "batch_consistency_audit",
    "optimizer_steps",
    "epoch_status",
}
_SUMMARY_KEYS = {
    "contract_version",
    "config_hash",
    "dataset_manifest_hash",
    "epochs_completed",
    "best_epoch",
    "best_validation",
    "early_stopped",
    "status",
}
_ALLOWED_FILES = {
    "checkpoints/last.pt",
    "checkpoints/best.pt",
    "training_metrics.jsonl",
    "development_summary.json",
}
_SENSITIVE_TOKENS = {
    "raw",
    "key",
    "keys",
    "secret",
    "code",
    "source",
    "input",
    "extra",
}


def seed_gate_training(seed: int) -> None:
    """Public seed entry point intended to run before model construction."""

    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass(frozen=True)
class GateTrainerConfig:
    epochs: int = 10
    batch_size: int = 9
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    seed: int = 7
    early_stopping_patience: int = 3
    decision_threshold: float = 0.5
    max_tokens: int = 512

    def __post_init__(self) -> None:
        for name in ("epochs", "batch_size", "early_stopping_patience", "max_tokens"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size < 3:
            raise ValueError("batch_size must fit one three-context cohort")
        if self.max_tokens > 512:
            raise ValueError("max_tokens must not exceed 512")
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("learning_rate", "weight_decay"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.learning_rate == 0:
            raise ValueError("learning_rate must be positive")
        if (
            type(self.decision_threshold) not in (int, float)
            or not math.isfinite(self.decision_threshold)
            or not 0 < self.decision_threshold < 1
        ):
            raise ValueError("decision_threshold must be strictly between 0 and 1")


@dataclass(frozen=True)
class _AlignedGateOutput:
    close_logits: torch.Tensor
    suitable_logits: torch.Tensor


@dataclass(frozen=True)
class _Neighbor:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class GateTrainer:
    """Train while persisting only strict, provenance-bound artifacts."""

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: Any,
        output_dir: Path,
        config_hash: str,
        dataset_manifest_hash: str,
        config: GateTrainerConfig | None = None,
        loss_fn: GateLoss | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if not isinstance(model, nn.Module):
            raise ValueError("model must be a torch.nn.Module")
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("tokenizer must be callable")
        if not isinstance(output_dir, Path):
            raise ValueError("output_dir must be a pathlib.Path")
        _validate_digest("config_hash", config_hash)
        _validate_digest("dataset_manifest_hash", dataset_manifest_hash)
        self.config = config or GateTrainerConfig()
        if not isinstance(self.config, GateTrainerConfig):
            raise ValueError("config must be GateTrainerConfig")
        self.loss_fn = loss_fn or GateLoss()
        if not isinstance(self.loss_fn, GateLoss):
            raise ValueError("loss_fn must be GateLoss")
        self.model = model
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.config_hash = config_hash
        self.dataset_manifest_hash = dataset_manifest_hash
        self.device = torch.device(device)
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: max(0.0, 1.0 - step / max(1, self.config.epochs)),
        )

    def fit(
        self,
        training_examples: Sequence[GateExample] | GateDataset,
        validation_examples: Sequence[GateExample] | GateDataset,
        *,
        resume_from: Path | None = None,
        stop_after_epoch: int | None = None,
    ) -> dict[str, Any]:
        training = _snapshot_examples("training_examples", training_examples)
        validation = _snapshot_examples("validation_examples", validation_examples)
        # Constructing GateDataset is an upfront duplicate/integrity check.
        training_dataset = GateDataset(training)
        validation_dataset = GateDataset(validation)
        collator = GateCollator(self.tokenizer, max_tokens=self.config.max_tokens)
        train_batches = self._all_training_batch_indices(training)
        validation_batches = _sequential_batches(len(validation), self.config.batch_size)
        dry_validation = self._dry_run_validation(
            validation, validation_batches, collator
        )
        # Training-time consistency neighbors must come from training data;
        # validation examples are never introduced into an optimizer graph.
        training_neighbors = self._collect_valid_neighbors(training, collator)
        validation_neighbors = self._collect_valid_neighbors(validation, collator)
        _validate_model_state(
            self.model.state_dict(), expected=self.model.state_dict()
        )

        start_epoch = 0
        best_objective: tuple[float, float] | None = None
        best_epoch: int | None = None
        best_validation: dict[str, float] | None = None
        patience = 0
        optimizer_steps = 0
        existing_metric_rows: list[dict[str, Any]] = []
        if resume_from is None:
            self._validate_fresh_output()
        else:
            self._validate_resume_layout(resume_from)
            existing_metric_rows = self._read_metrics()
            payload = self._read_checkpoint_payload(resume_from)
            checkpoint_epoch = payload["epoch"]
            training_state = dict(payload["training_state"])
            replayed = _replay_metrics(
                existing_metric_rows,
                configured_epochs=self.config.epochs,
                early_stopping_patience=self.config.early_stopping_patience,
            )
            for name in (
                "best_objective",
                "best_epoch",
                "best_validation",
                "patience",
                "optimizer_steps",
            ):
                if training_state[name] != replayed[name]:
                    raise ValueError(
                        f"checkpoint training state {name} does not match metrics history"
                    )
            existing_summary = self._read_summary()
            best_payload = self._read_checkpoint_payload(self._best_checkpoint)
            if best_payload["epoch"] != training_state["best_epoch"]:
                raise ValueError("best checkpoint epoch does not match training state")
            if best_payload["validation"] != training_state["best_validation"]:
                raise ValueError("best checkpoint validation does not match training state")
            best_replayed = _replay_metrics(
                existing_metric_rows[: best_payload["epoch"] + 1],
                configured_epochs=self.config.epochs,
                early_stopping_patience=self.config.early_stopping_patience,
            )
            best_checkpoint_state = best_payload["training_state"]
            for name in (
                "best_objective",
                "best_epoch",
                "best_validation",
                "patience",
                "optimizer_steps",
            ):
                if best_checkpoint_state[name] != best_replayed[name]:
                    raise ValueError(
                        f"best checkpoint {name} does not match metrics prefix"
                    )
            if best_checkpoint_state["epochs_completed"] != best_payload["epoch"] + 1:
                raise ValueError("best checkpoint epoch does not match metrics prefix")
            if best_checkpoint_state["status"] != best_replayed["epoch_status"]:
                raise ValueError("best checkpoint status does not match metrics prefix")
            if (
                existing_summary["epochs_completed"]
                != training_state["epochs_completed"]
                or existing_summary["best_epoch"] != training_state["best_epoch"]
                or existing_summary["best_validation"]
                != training_state["best_validation"]
                or existing_summary["status"] != training_state["status"]
            ):
                raise ValueError("development summary does not match checkpoint state")
            if training_state["status"] in {"early_stopped", "completed"}:
                raise ValueError(
                    f"cannot resume checkpoint with terminal status {training_state['status']!r}"
                )
            if training_state["status"] != "interrupted":
                raise ValueError("resumable checkpoint status must be interrupted")
            if training_state["epochs_completed"] != checkpoint_epoch + 1:
                raise ValueError("checkpoint epochs_completed does not match metrics")
            if replayed["epoch_status"] != "running":
                raise ValueError(
                    "interrupted checkpoint must map from a running intrinsic epoch"
                )
            if existing_metric_rows[-1]["epoch"] != checkpoint_epoch:
                raise ValueError("metrics/checkpoint epoch mismatch")
            if payload["validation"] != existing_metric_rows[-1]["validation"]:
                raise ValueError("checkpoint validation does not match final metrics row")
            # Only mutate model/optimizer/scheduler after every checkpoint,
            # metrics, summary, and best-checkpoint cross-check has passed.
            self._restore_checkpoint_payload(payload)
            start_epoch = checkpoint_epoch + 1
            best_objective = tuple(training_state["best_objective"])
            best_epoch = training_state["best_epoch"]
            best_validation = dict(training_state["best_validation"])
            patience = training_state["patience"]
            optimizer_steps = training_state["optimizer_steps"]

        if stop_after_epoch is not None:
            if type(stop_after_epoch) is not int or not start_epoch <= stop_after_epoch:
                raise ValueError("stop_after_epoch must be an epoch in this fit call")
            if stop_after_epoch >= self.config.epochs:
                raise ValueError("stop_after_epoch must be below configured epochs")
        if start_epoch >= self.config.epochs:
            raise ValueError("checkpoint has no remaining configured epoch")

        # Audit every actual training layout across every configured epoch and
        # every validation batch before the first optimizer mutation.
        self._audit_all_batches(
            training,
            validation,
            train_batches,
            validation_batches,
            collator,
            training_neighbors,
            validation_neighbors,
        )

        if resume_from is None:
            self._create_output_layout()
            _atomic_write_text(self._metrics_path, "")

        epochs_completed = start_epoch
        status = "running"
        for epoch in range(start_epoch, self.config.epochs):
            seed_gate_training(self.config.seed + epoch)
            sampler = GroupConsistencyBatchSampler(
                training_dataset,
                batch_size=self.config.batch_size,
                seed=self.config.seed + epoch,
                shuffle=True,
            )
            loader = DataLoader(
                training_dataset,
                batch_sampler=sampler,
                collate_fn=collator,
                num_workers=0,
            )
            component_sums = {name: 0.0 for name in _LOSS_NAMES}
            total_sum = 0.0
            batch_count = 0
            evaluated_consistency = 0
            skipped_consistency = 0
            self.model.train()
            for raw_batch in loader:
                batch = _move_batch(raw_batch, self.device)
                self.optimizer.zero_grad(set_to_none=True)
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                _validate_model_output(output, batch["input_ids"].shape[0])
                pair = self._fixed_consistency_pair(batch, training_neighbors)
                if pair is None:
                    reference = _AlignedGateOutput(
                        output.close_logits, output.suitable_logits
                    )
                    alternate = reference
                    consistency_mask = torch.zeros_like(
                        batch["close_loss_mask"], dtype=torch.bool
                    )
                    skipped_consistency += 1
                else:
                    reference, alternate = pair
                    consistency_mask = torch.ones_like(
                        batch["close_loss_mask"], dtype=torch.bool
                    )
                    evaluated_consistency += 1
                cohort_ids = [
                    f"{group}\x00{window}\x00{budget}"
                    for group, window, budget in zip(
                        batch["group_ids"],
                        batch["window_start_unit_ids"],
                        batch["budgets"],
                        strict=True,
                    )
                ]
                total, components = self.loss_fn(
                    close_logits=output.close_logits,
                    suitable_logits=output.suitable_logits,
                    close_targets=batch["close_targets"],
                    suitable_targets=batch["suitable_targets"],
                    dangerous_negative=batch["dangerous_negative"],
                    group_ids=cohort_ids,
                    batch_reference_close_logits=reference.close_logits,
                    batch_reference_suitable_logits=reference.suitable_logits,
                    alternate_close_logits=alternate.close_logits,
                    alternate_suitable_logits=alternate.suitable_logits,
                    close_loss_mask=batch["close_loss_mask"],
                    suitable_loss_mask=batch["suitable_loss_mask"],
                    batch_consistency_mask=consistency_mask,
                )
                total.backward()
                self.optimizer.step()
                optimizer_steps += 1
                total_sum += float(total.detach().cpu())
                for name in _LOSS_NAMES:
                    component_sums[name] += float(components[name].detach().cpu())
                batch_count += 1
            self.scheduler.step()

            validation_metrics = self._validate(
                validation_dataset, collator, dry_validation
            )
            objective = (
                validation_metrics["suitable_false_positive_rate"],
                -validation_metrics["decision_consistency"],
            )
            improved = best_objective is None or objective < best_objective
            if improved:
                best_objective = objective
                best_epoch = epoch
                best_validation = validation_metrics
                patience = 0
            else:
                patience += 1
            epochs_completed = epoch + 1
            if patience >= self.config.early_stopping_patience:
                epoch_status = "early_stopped"
            elif epochs_completed >= self.config.epochs:
                epoch_status = "completed"
            else:
                epoch_status = "running"
            status = (
                "interrupted"
                if epoch_status == "running"
                and stop_after_epoch is not None
                and epoch >= stop_after_epoch
                else epoch_status
            )
            assert best_objective is not None
            assert best_epoch is not None
            assert best_validation is not None
            training_state = {
                "best_objective": list(best_objective),
                "best_epoch": best_epoch,
                "best_validation": best_validation,
                "patience": patience,
                "status": status,
                "epochs_completed": epochs_completed,
                "optimizer_steps": optimizer_steps,
            }
            row = {
                "contract_version": _METRICS_CONTRACT,
                "config_hash": self.config_hash,
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "epoch": epoch,
                "total_loss": total_sum / batch_count,
                "loss_components": {
                    name: component_sums[name] / batch_count for name in _LOSS_NAMES
                },
                "validation": validation_metrics,
                "best": improved,
                "batch_consistency_audit": {
                    "evaluated_batches": evaluated_consistency,
                    "skipped_batches": skipped_consistency,
                    "skip_reason": (
                        "no_valid_real_neighbor" if skipped_consistency else None
                    ),
                },
                "optimizer_steps": optimizer_steps,
                "epoch_status": epoch_status,
            }
            _validate_metric_row(row, expected_epoch=epoch)
            _atomic_append_json_line(self._metrics_path, row)
            checkpoint = self._checkpoint_payload(
                epoch, validation_metrics, training_state
            )
            self._save_checkpoint(self._last_checkpoint, checkpoint)
            if improved:
                best_training_state = dict(training_state)
                # External controlled interruption is a run-level condition,
                # not an intrinsic property of this epoch's best snapshot.
                best_training_state["status"] = epoch_status
                best_checkpoint = self._checkpoint_payload(
                    epoch, validation_metrics, best_training_state
                )
                self._save_checkpoint(self._best_checkpoint, best_checkpoint)
            if status in {"early_stopped", "interrupted", "completed"}:
                break

        assert best_epoch is not None and best_validation is not None
        summary: dict[str, Any] = {
            "contract_version": _SUMMARY_CONTRACT,
            "config_hash": self.config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "epochs_completed": epochs_completed,
            "best_epoch": best_epoch,
            "best_validation": best_validation,
            "early_stopped": status == "early_stopped",
            "status": status,
        }
        _validate_summary(summary)
        _atomic_write_text(
            self._summary_path,
            json.dumps(summary, allow_nan=False, sort_keys=True) + "\n",
        )
        return summary

    def load_checkpoint(self, path: Path) -> tuple[int, dict[str, Any]]:
        """Transactionally restore a strictly validated safe checkpoint."""

        payload = self._read_checkpoint_payload(path)
        self._restore_checkpoint_payload(payload)
        return payload["epoch"], dict(payload["training_state"])

    def _read_checkpoint_payload(self, path: Path) -> Mapping[str, Any]:
        self._validate_checkpoint_path(path)
        try:
            payload = torch.load(path, map_location=self.device, weights_only=True)
        except TypeError as exc:
            raise RuntimeError(
                "safe checkpoint loading requires torch.load(weights_only=True); "
                "upgrade PyTorch rather than using unsafe pickle loading"
            ) from exc
        except Exception as exc:
            raise ValueError("checkpoint could not be loaded safely") from exc
        self._validate_checkpoint_payload(payload)
        return payload

    def _restore_checkpoint_payload(self, payload: Mapping[str, Any]) -> None:
        model_before = copy.deepcopy(self.model.state_dict())
        optimizer_before = copy.deepcopy(self.optimizer.state_dict())
        scheduler_before = copy.deepcopy(self.scheduler.state_dict())
        try:
            self.model.load_state_dict(payload["model_state"], strict=True)
            self.optimizer.load_state_dict(payload["optimizer_state"])
            self.scheduler.load_state_dict(payload["scheduler_state"])
            if self.scheduler.last_epoch != payload["epoch"] + 1:
                raise ValueError("checkpoint scheduler epoch is inconsistent")
        except Exception as exc:
            self.model.load_state_dict(model_before, strict=True)
            self.optimizer.load_state_dict(optimizer_before)
            self.scheduler.load_state_dict(scheduler_before)
            raise ValueError("checkpoint state could not be restored transactionally") from exc

    @property
    def _checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def _last_checkpoint(self) -> Path:
        return self._checkpoint_dir / "last.pt"

    @property
    def _best_checkpoint(self) -> Path:
        return self._checkpoint_dir / "best.pt"

    @property
    def _metrics_path(self) -> Path:
        return self.output_dir / "training_metrics.jsonl"

    @property
    def _summary_path(self) -> Path:
        return self.output_dir / "development_summary.json"

    def _all_training_batch_indices(
        self, examples: tuple[GateExample, ...]
    ) -> list[list[int]]:
        batches: list[list[int]] = []
        seen: set[tuple[int, ...]] = set()
        for epoch in range(self.config.epochs):
            sampler = GroupConsistencyBatchSampler(
                examples,
                batch_size=self.config.batch_size,
                seed=self.config.seed + epoch,
                shuffle=True,
            )
            for batch in sampler:
                identity = tuple(batch)
                if identity not in seen:
                    seen.add(identity)
                    batches.append(batch)
        return batches

    def _dry_run_validation(
        self,
        examples: tuple[GateExample, ...],
        batches: Sequence[Sequence[int]],
        collator: GateCollator,
    ) -> dict[str, int | float]:
        total = evaluable = overflow = negatives = 0
        group_counts: dict[tuple[str, str, int], int] = {}
        for indices in batches:
            batch = collator([examples[index] for index in indices])
            total += len(indices)
            valid = batch["suitable_loss_mask"]
            evaluable += int(valid.sum())
            overflow += int(batch["overflow"].sum())
            negatives += int(((batch["suitable_targets"] == 0) & valid).sum())
            for group, window, budget, is_valid in zip(
                batch["group_ids"],
                batch["window_start_unit_ids"],
                batch["budgets"],
                valid.tolist(),
                strict=True,
            ):
                if is_valid:
                    identity = (str(group), str(window), int(budget))
                    group_counts[identity] = group_counts.get(identity, 0) + 1
        if evaluable == 0:
            raise ValueError("validation has zero evaluable non-overflow rows")
        if negatives == 0:
            raise ValueError("validation has zero evaluable suitable-negative rows")
        comparable_groups = sum(count >= 2 for count in group_counts.values())
        if comparable_groups == 0:
            raise ValueError("validation has zero comparable non-overflow cohorts")
        return {
            "total_count": total,
            "evaluable_count": evaluable,
            "overflow_count": overflow,
            "negative_count": negatives,
            "comparable_group_count": comparable_groups,
            "coverage": evaluable / total,
        }

    def _collect_valid_neighbors(
        self, examples: Sequence[GateExample], collator: GateCollator
    ) -> tuple[_Neighbor, ...]:
        neighbors: list[_Neighbor] = []
        seen: set[str] = set()
        for example in examples:
            if example.serialized_gate_input in seen:
                continue
            batch = collator([example])
            if bool(batch["overflow"][0]):
                continue
            seen.add(example.serialized_gate_input)
            neighbors.append(
                _Neighbor(batch["input_ids"][0], batch["attention_mask"][0])
            )
            if len(neighbors) == 2:
                break
        return tuple(neighbors)

    def _audit_all_batches(
        self,
        training: tuple[GateExample, ...],
        validation: tuple[GateExample, ...],
        training_batches: Sequence[Sequence[int]],
        validation_batches: Sequence[Sequence[int]],
        collator: GateCollator,
        training_neighbors: tuple[_Neighbor, ...],
        validation_neighbors: tuple[_Neighbor, ...],
    ) -> None:
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                for examples, batches, neighbors in (
                    (training, training_batches, training_neighbors),
                    (validation, validation_batches, validation_neighbors),
                ):
                    for indices in batches:
                        raw = collator([examples[index] for index in indices])
                        batch = _move_batch(raw, self.device)
                        output = self.model(
                            input_ids=batch["input_ids"],
                            attention_mask=batch["attention_mask"],
                        )
                        _validate_model_output(output, len(indices))
                        pair = self._fixed_consistency_pair(batch, neighbors)
                        if pair is not None:
                            _assert_invariant(pair)
        finally:
            self.model.train(was_training)

    def _fixed_consistency_pair(
        self,
        batch: Mapping[str, Any],
        neighbors: tuple[_Neighbor, ...],
    ) -> tuple[_AlignedGateOutput, _AlignedGateOutput] | None:
        if not neighbors:
            return None
        was_training = self.model.training
        self.model.eval()
        try:
            target_ids = batch["input_ids"]
            target_mask = batch["attention_mask"]
            target_count = target_ids.shape[0]
            neighbor_a = _neighbor_to_device(neighbors[0], self.device)
            neighbor_b = _neighbor_to_device(
                neighbors[1] if len(neighbors) > 1 else neighbors[0], self.device
            )
            layout_a_ids, layout_a_mask = _combine_rows(
                target_ids, target_mask, (neighbor_a,), extra_padding=0
            )
            reverse = torch.arange(target_count - 1, -1, -1, device=self.device)
            reversed_ids = target_ids.index_select(0, reverse)
            reversed_mask = target_mask.index_select(0, reverse)
            b_neighbors = (
                (neighbor_b,)
                if len(neighbors) > 1
                else (neighbor_b, neighbor_b)
            )
            layout_b_ids, layout_b_mask = _combine_rows(
                reversed_ids,
                reversed_mask,
                b_neighbors,
                extra_padding=_CONSISTENCY_EXTRA_PADDING,
            )
            reference_all = self.model(
                input_ids=layout_a_ids, attention_mask=layout_a_mask
            )
            alternate_all = self.model(
                input_ids=layout_b_ids, attention_mask=layout_b_mask
            )
            _validate_model_output(reference_all, layout_a_ids.shape[0])
            _validate_model_output(alternate_all, layout_b_ids.shape[0])
            return (
                _AlignedGateOutput(
                    reference_all.close_logits[:target_count],
                    reference_all.suitable_logits[:target_count],
                ),
                _AlignedGateOutput(
                    alternate_all.close_logits[:target_count].index_select(0, reverse),
                    alternate_all.suitable_logits[:target_count].index_select(0, reverse),
                ),
            )
        finally:
            self.model.train(was_training)

    def _validate(
        self,
        examples: GateDataset,
        collator: GateCollator,
        dry_counts: Mapping[str, int | float],
    ) -> dict[str, float | int]:
        loader = DataLoader(
            examples,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
        )
        rows: list[tuple[str, str, int, bool, bool, bool]] = []
        self.model.eval()
        with torch.no_grad():
            for raw_batch in loader:
                batch = _move_batch(raw_batch, self.device)
                output = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                _validate_model_output(output, batch["input_ids"].shape[0])
                close = (
                    torch.sigmoid(output.close_logits) >= self.config.decision_threshold
                ).cpu().tolist()
                suitable = (
                    torch.sigmoid(output.suitable_logits)
                    >= self.config.decision_threshold
                ).cpu().tolist()
                targets = batch["suitable_targets"].cpu().tolist()
                valid = batch["suitable_loss_mask"].cpu().tolist()
                for values in zip(
                    batch["group_ids"],
                    batch["window_start_unit_ids"],
                    batch["budgets"],
                    close,
                    suitable,
                    targets,
                    valid,
                    strict=True,
                ):
                    if not bool(values[6]):
                        continue
                    rows.append(
                        (
                            str(values[0]),
                            str(values[1]),
                            int(values[2]),
                            bool(values[3]),
                            bool(values[4]),
                            bool(values[5]),
                        )
                    )
        negatives = [row for row in rows if not row[5]]
        if not rows or not negatives:
            raise ValueError("validation quality metrics are not computable")
        metrics: dict[str, float | int] = dict(dry_counts)
        metrics.update(
            {
                "suitable_false_positive_rate": (
                    sum(row[4] for row in negatives) / len(negatives)
                ),
                "decision_consistency": _decision_consistency(rows),
            }
        )
        _validate_validation(metrics)
        return metrics

    def _checkpoint_payload(
        self,
        epoch: int,
        validation: Mapping[str, float | int],
        training_state: Mapping[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "contract_version": _CHECKPOINT_CONTRACT,
            "config_hash": self.config_hash,
            "dataset_manifest_hash": self.dataset_manifest_hash,
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "validation": dict(validation),
            "training_state": dict(training_state),
        }
        self._validate_checkpoint_payload(payload)
        return payload

    def _save_checkpoint(self, path: Path, payload: Mapping[str, Any]) -> None:
        if path.is_symlink():
            raise ValueError("checkpoint destination symlink is forbidden")
        self._validate_checkpoint_payload(payload)
        _atomic_write_binary(path, lambda handle: torch.save(dict(payload), handle))

    def _validate_checkpoint_payload(self, payload: object) -> None:
        if not isinstance(payload, Mapping) or set(payload) != _CHECKPOINT_KEYS:
            raise ValueError("checkpoint top-level schema mismatch")
        if payload["contract_version"] != _CHECKPOINT_CONTRACT:
            raise ValueError("checkpoint contract mismatch")
        if payload["config_hash"] != self.config_hash:
            raise ValueError("checkpoint config hash mismatch")
        if payload["dataset_manifest_hash"] != self.dataset_manifest_hash:
            raise ValueError("checkpoint dataset manifest hash mismatch")
        epoch = payload["epoch"]
        if type(epoch) is not int or not 0 <= epoch < self.config.epochs:
            raise ValueError("checkpoint epoch is invalid")
        expected_model = self.model.state_dict()
        _validate_model_state(payload["model_state"], expected=expected_model)
        _validate_training_state(
            payload["training_state"],
            epoch=epoch,
            configured_epochs=self.config.epochs,
            early_stopping_patience=self.config.early_stopping_patience,
        )
        _validate_optimizer_state(
            payload["optimizer_state"],
            optimizer=self.optimizer,
            expected_steps=payload["training_state"]["optimizer_steps"],
        )
        _validate_scheduler_state(
            payload["scheduler_state"],
            expected=self.scheduler.state_dict(),
            expected_last_epoch=epoch + 1,
        )
        scheduler_lrs = payload["scheduler_state"]["_last_lr"]
        optimizer_lrs = [
            group["lr"] for group in payload["optimizer_state"]["param_groups"]
        ]
        if scheduler_lrs != optimizer_lrs:
            raise ValueError("checkpoint optimizer/scheduler learning rates mismatch")
        _validate_validation(payload["validation"])
        expected_lr = self.config.learning_rate * max(
            0.0, 1.0 - (epoch + 1) / self.config.epochs
        )
        if any(
            not math.isclose(lr, expected_lr, rel_tol=0.0, abs_tol=1e-15)
            for lr in optimizer_lrs
        ):
            raise ValueError("checkpoint learning rate does not match config/epoch")

    def _validate_fresh_output(self) -> None:
        if _has_symlink_component(self.output_dir):
            raise ValueError("output_dir symlink traversal is forbidden")
        if self.output_dir.exists():
            if not self.output_dir.is_dir():
                raise ValueError("output_dir must be a directory")
            if any(self.output_dir.iterdir()):
                raise ValueError("fresh output_dir must be strictly empty")

    def _create_output_layout(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_dir.mkdir()

    def _validate_resume_layout(self, checkpoint: Path) -> None:
        self._validate_checkpoint_path(checkpoint)
        if checkpoint != self._last_checkpoint:
            raise ValueError("resume must use this output_dir's checkpoints/last.pt")
        actual = {
            path.relative_to(self.output_dir).as_posix()
            for path in self.output_dir.rglob("*")
        }
        if actual != (_ALLOWED_FILES | {"checkpoints"}):
            raise ValueError("resume output directory artifact allowlist mismatch")
        for path in self.output_dir.rglob("*"):
            if path.is_symlink():
                raise ValueError("resume artifacts and directories cannot be symlinks")

    def _validate_checkpoint_path(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise ValueError("checkpoint path must be a pathlib.Path")
        if path.is_symlink() or not path.is_file():
            raise ValueError("checkpoint must be a non-symlink file")
        if _has_symlink_component(path):
            raise ValueError("checkpoint path cannot traverse symlink directories")

    def _read_metrics(self) -> list[dict[str, Any]]:
        if self._metrics_path.is_symlink() or not self._metrics_path.is_file():
            raise ValueError("training_metrics.jsonl must be a non-symlink file")
        rows: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self._metrics_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid metrics JSON at line {line_number}") from exc
            _validate_metric_row(row, expected_epoch=line_number - 1)
            if row["config_hash"] != self.config_hash:
                raise ValueError("metrics config hash mismatch")
            if row["dataset_manifest_hash"] != self.dataset_manifest_hash:
                raise ValueError("metrics dataset manifest hash mismatch")
            rows.append(row)
        if not rows:
            raise ValueError("resume metrics must not be empty")
        return rows

    def _read_summary(self) -> dict[str, Any]:
        if self._summary_path.is_symlink() or not self._summary_path.is_file():
            raise ValueError("development_summary.json must be a non-symlink file")
        try:
            value = json.loads(self._summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError("development summary must be valid JSON") from exc
        _validate_summary(value)
        if value["config_hash"] != self.config_hash:
            raise ValueError("development summary config hash mismatch")
        if value["dataset_manifest_hash"] != self.dataset_manifest_hash:
            raise ValueError("development summary dataset manifest hash mismatch")
        return dict(value)


def _snapshot_examples(
    name: str, examples: Sequence[GateExample] | GateDataset
) -> tuple[GateExample, ...]:
    if not isinstance(examples, (Sequence, GateDataset)) or isinstance(
        examples, (str, bytes)
    ):
        raise ValueError(f"{name} must be a sequence or GateDataset")
    snapshot = tuple(examples[index] for index in range(len(examples)))
    if not snapshot or any(not isinstance(item, GateExample) for item in snapshot):
        raise ValueError(f"{name} must contain GateExample instances")
    return snapshot


def _sequential_batches(length: int, batch_size: int) -> list[list[int]]:
    return [list(range(start, min(start + batch_size, length))) for start in range(0, length, batch_size)]


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        name: value.to(device) if isinstance(value, torch.Tensor) else value
        for name, value in batch.items()
    }


def _neighbor_to_device(neighbor: _Neighbor, device: torch.device) -> _Neighbor:
    return _Neighbor(neighbor.input_ids.to(device), neighbor.attention_mask.to(device))


def _combine_rows(
    target_ids: torch.Tensor,
    target_mask: torch.Tensor,
    neighbors: Sequence[_Neighbor],
    *,
    extra_padding: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    length = max(
        target_ids.shape[1],
        *(neighbor.input_ids.numel() for neighbor in neighbors),
    ) + extra_padding
    ids = torch.nn.functional.pad(target_ids, (0, length - target_ids.shape[1]), value=0)
    mask = torch.nn.functional.pad(target_mask, (0, length - target_mask.shape[1]), value=0)
    neighbor_ids = [
        torch.nn.functional.pad(neighbor.input_ids, (0, length - neighbor.input_ids.numel()), value=0)
        for neighbor in neighbors
    ]
    neighbor_masks = [
        torch.nn.functional.pad(neighbor.attention_mask, (0, length - neighbor.attention_mask.numel()), value=0)
        for neighbor in neighbors
    ]
    return torch.cat((ids, torch.stack(neighbor_ids)), dim=0), torch.cat(
        (mask, torch.stack(neighbor_masks)), dim=0
    )


def _validate_model_output(output: object, expected_batch: int) -> None:
    values: list[torch.Tensor] = []
    for name in ("close_logits", "suitable_logits"):
        value = getattr(output, name, None)
        if not isinstance(value, torch.Tensor) or value.ndim != 1:
            raise ValueError(f"model {name} must be a 1-D tensor")
        if value.shape != (expected_batch,):
            raise ValueError(f"model {name} length must match input batch")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"model {name} must be finite floating point")
        values.append(value)
    if (
        values[0].shape != values[1].shape
        or values[0].device != values[1].device
        or values[0].dtype != values[1].dtype
    ):
        raise ValueError("model heads must have identical shape, device, and dtype")


def _assert_invariant(pair: tuple[_AlignedGateOutput, _AlignedGateOutput]) -> None:
    reference, alternate = pair
    differences = torch.cat(
        (
            (reference.close_logits - alternate.close_logits).abs(),
            (reference.suitable_logits - alternate.suitable_logits).abs(),
        )
    )
    maximum = float(differences.max().cpu()) if differences.numel() else 0.0
    if maximum > _INVARIANCE_ATOL:
        raise RuntimeError(
            "batch/padding invariance failed before training; inspect masking, "
            "pooling, model mode, initialization, or batch-member dependence"
        )


def _decision_consistency(
    rows: Sequence[tuple[str, str, int, bool, bool, bool]],
) -> float:
    groups: dict[tuple[str, str, int], list[tuple[bool, bool]]] = {}
    for group, window, budget, close, suitable, _ in rows:
        groups.setdefault((group, window, budget), []).append((close, suitable))
    scores: list[float] = []
    for values in groups.values():
        if len(values) < 2:
            continue
        scores.extend(
            (
                float(len({value[0] for value in values}) == 1),
                float(len({value[1] for value in values}) == 1),
            )
        )
    return sum(scores) / len(scores) if scores else 1.0


def _validate_digest(name: str, value: object) -> None:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_validation(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _VALIDATION_KEYS:
        raise ValueError("validation metrics schema mismatch")
    for name in (
        "total_count",
        "evaluable_count",
        "overflow_count",
        "negative_count",
        "comparable_group_count",
    ):
        if type(value[name]) is not int or value[name] < 0:
            raise ValueError(f"validation {name} must be a non-negative integer")
    if value["total_count"] <= 0 or value["evaluable_count"] <= 0 or value["negative_count"] <= 0:
        raise ValueError("validation metric denominators must be positive")
    if value["evaluable_count"] + value["overflow_count"] != value["total_count"]:
        raise ValueError("validation counts are inconsistent")
    if value["negative_count"] > value["evaluable_count"]:
        raise ValueError("validation negative count is inconsistent")
    if (
        value["comparable_group_count"] <= 0
        or value["comparable_group_count"] > value["evaluable_count"] // 2
    ):
        raise ValueError("validation comparable group count is inconsistent")
    for name in ("coverage", "suitable_false_positive_rate", "decision_consistency"):
        metric = value[name]
        if type(metric) not in (int, float) or not math.isfinite(metric) or not 0 <= metric <= 1:
            raise ValueError(f"validation {name} must be finite and in [0, 1]")
    if not math.isclose(
        value["coverage"],
        value["evaluable_count"] / value["total_count"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("validation coverage/counts mismatch")


def _validate_training_state(
    value: object,
    *,
    epoch: int,
    configured_epochs: int,
    early_stopping_patience: int,
) -> None:
    if not isinstance(value, Mapping) or set(value) != _TRAINING_STATE_KEYS:
        raise ValueError("checkpoint training_state schema mismatch")
    objective = value["best_objective"]
    if not isinstance(objective, list) or len(objective) != 2 or any(
        type(item) not in (int, float) or not math.isfinite(item) for item in objective
    ):
        raise ValueError("training best_objective is invalid")
    if type(value["best_epoch"]) is not int or not 0 <= value["best_epoch"] <= epoch:
        raise ValueError("training best_epoch is invalid")
    _validate_validation(value["best_validation"])
    if objective != [
        value["best_validation"]["suitable_false_positive_rate"],
        -value["best_validation"]["decision_consistency"],
    ]:
        raise ValueError("training best objective/validation mismatch")
    if type(value["patience"]) is not int or value["patience"] < 0:
        raise ValueError("training patience is invalid")
    if type(value["optimizer_steps"]) is not int or value["optimizer_steps"] <= 0:
        raise ValueError("training optimizer_steps is invalid")
    if value["status"] not in {"running", "interrupted", "early_stopped", "completed"}:
        raise ValueError("training status is invalid")
    if value["epochs_completed"] != epoch + 1:
        raise ValueError("training epochs_completed is inconsistent")
    expected_epoch_status = (
        "early_stopped"
        if value["patience"] >= early_stopping_patience
        else (
            "completed"
            if value["epochs_completed"] >= configured_epochs
            else "running"
        )
    )
    allowed_statuses = (
        {"running", "interrupted"}
        if expected_epoch_status == "running"
        else {expected_epoch_status}
    )
    if value["status"] not in allowed_statuses:
        raise ValueError("training status is inconsistent with patience/epoch")


def _validate_metric_row(row: object, *, expected_epoch: int) -> None:
    if not isinstance(row, Mapping) or set(row) != _METRIC_ROW_KEYS:
        raise ValueError("training metric row schema mismatch")
    if row["contract_version"] != _METRICS_CONTRACT or row["epoch"] != expected_epoch:
        raise ValueError("training metric row contract/epoch mismatch")
    _validate_digest("metrics config_hash", row["config_hash"])
    _validate_digest("metrics dataset_manifest_hash", row["dataset_manifest_hash"])
    if type(row["total_loss"]) not in (int, float) or not math.isfinite(row["total_loss"]):
        raise ValueError("training total_loss must be finite")
    components = row["loss_components"]
    if not isinstance(components, Mapping) or set(components) != set(_LOSS_NAMES):
        raise ValueError("training loss_components schema mismatch")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in components.values()):
        raise ValueError("training loss components must be finite")
    if not math.isclose(
        row["total_loss"], sum(components.values()), rel_tol=1e-6, abs_tol=1e-8
    ):
        raise ValueError("training total/component losses mismatch")
    _validate_validation(row["validation"])
    if type(row["best"]) is not bool:
        raise ValueError("training best must be bool")
    audit = row["batch_consistency_audit"]
    if not isinstance(audit, Mapping) or set(audit) != {"evaluated_batches", "skipped_batches", "skip_reason"}:
        raise ValueError("batch consistency audit schema mismatch")
    if any(type(audit[name]) is not int or audit[name] < 0 for name in ("evaluated_batches", "skipped_batches")):
        raise ValueError("batch consistency audit counts are invalid")
    expected_reason = "no_valid_real_neighbor" if audit["skipped_batches"] else None
    if audit["skip_reason"] != expected_reason:
        raise ValueError("batch consistency skip reason is invalid")
    if type(row["optimizer_steps"]) is not int or row["optimizer_steps"] <= 0:
        raise ValueError("training metric optimizer_steps is invalid")
    if row["epoch_status"] not in {"running", "early_stopped", "completed"}:
        raise ValueError("training metric epoch_status is invalid")


def _replay_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    configured_epochs: int,
    early_stopping_patience: int,
) -> dict[str, Any]:
    best_objective: tuple[float, float] | None = None
    best_epoch: int | None = None
    best_validation: Mapping[str, Any] | None = None
    patience = 0
    optimizer_steps = 0
    for index, row in enumerate(rows):
        if index < len(rows) - 1 and row["epoch_status"] in {
            "early_stopped",
            "completed",
        }:
            raise ValueError("metrics continue after intrinsic terminal epoch")
        audit = row["batch_consistency_audit"]
        optimizer_steps += audit["evaluated_batches"] + audit["skipped_batches"]
        if row["optimizer_steps"] != optimizer_steps:
            raise ValueError("metrics optimizer step history is inconsistent")
        objective = (
            row["validation"]["suitable_false_positive_rate"],
            -row["validation"]["decision_consistency"],
        )
        improved = best_objective is None or objective < best_objective
        if row["best"] != improved:
            raise ValueError("metrics best flags contradict validation objectives")
        if improved:
            best_objective = objective
            best_epoch = row["epoch"]
            best_validation = row["validation"]
            patience = 0
        else:
            patience += 1
        expected_epoch_status = (
            "early_stopped"
            if patience >= early_stopping_patience
            else (
                "completed"
                if row["epoch"] + 1 >= configured_epochs
                else "running"
            )
        )
        if row["epoch_status"] != expected_epoch_status:
            raise ValueError(
                "metrics epoch_status contradicts patience/configured epochs"
            )
    assert best_objective is not None
    assert best_epoch is not None
    assert best_validation is not None
    return {
        "best_objective": list(best_objective),
        "best_epoch": best_epoch,
        "best_validation": dict(best_validation),
        "patience": patience,
        "optimizer_steps": optimizer_steps,
        "epoch_status": rows[-1]["epoch_status"],
        "epochs_completed": rows[-1]["epoch"] + 1,
    }


def _validate_summary(value: object) -> None:
    if not isinstance(value, Mapping) or set(value) != _SUMMARY_KEYS:
        raise ValueError("development summary schema mismatch")
    if value["contract_version"] != _SUMMARY_CONTRACT:
        raise ValueError("development summary contract mismatch")
    _validate_digest("summary config_hash", value["config_hash"])
    _validate_digest("summary dataset_manifest_hash", value["dataset_manifest_hash"])
    if type(value["epochs_completed"]) is not int or value["epochs_completed"] <= 0:
        raise ValueError("summary epochs_completed is invalid")
    if type(value["best_epoch"]) is not int or value["best_epoch"] < 0:
        raise ValueError("summary best_epoch is invalid")
    if value["best_epoch"] >= value["epochs_completed"]:
        raise ValueError("summary best_epoch/epochs_completed mismatch")
    _validate_validation(value["best_validation"])
    if type(value["early_stopped"]) is not bool:
        raise ValueError("summary early_stopped must be bool")
    if value["status"] not in {"interrupted", "early_stopped", "completed"}:
        raise ValueError("summary status is invalid")
    if value["early_stopped"] != (value["status"] == "early_stopped"):
        raise ValueError("summary early-stopped flag/status mismatch")


def _validate_model_state(value: object, *, expected: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("checkpoint model_state keys mismatch")
    for name, tensor in value.items():
        if not isinstance(name, str) or name.endswith("_extra_state"):
            raise ValueError("checkpoint model_state extra state is forbidden")
        tokens = set(filter(None, re.split(r"[^a-z0-9]+", name.casefold())))
        if tokens & _SENSITIVE_TOKENS:
            raise ValueError(f"sensitive checkpoint model_state key {name!r}")
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("checkpoint model_state values must be tensors")
        expected_tensor = expected[name]
        if not isinstance(expected_tensor, torch.Tensor):
            raise ValueError("model extra state is forbidden")
        if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
            raise ValueError("checkpoint model_state tensor contract mismatch")
        if not torch.isfinite(tensor).all():
            raise ValueError("checkpoint model_state tensors must be finite")


def _validate_optimizer_state(
    value: object, *, optimizer: AdamW, expected_steps: int
) -> None:
    if not isinstance(value, Mapping) or set(value) != {"state", "param_groups"}:
        raise ValueError("checkpoint optimizer_state schema mismatch")
    if not isinstance(value["state"], Mapping) or not isinstance(value["param_groups"], list):
        raise ValueError("checkpoint optimizer_state shape mismatch")
    expected = optimizer.state_dict()
    expected_groups = expected["param_groups"]
    if len(value["param_groups"]) != len(expected_groups):
        raise ValueError("checkpoint optimizer param group count mismatch")
    valid_ids: set[int] = set()
    parameters_by_id: dict[int, torch.Tensor] = {}
    for candidate, reference, live in zip(
        value["param_groups"],
        expected_groups,
        optimizer.param_groups,
        strict=True,
    ):
        if not isinstance(candidate, Mapping) or set(candidate) != set(reference):
            raise ValueError("checkpoint optimizer param group schema mismatch")
        if candidate["params"] != reference["params"]:
            raise ValueError("checkpoint optimizer parameter identities mismatch")
        valid_ids.update(candidate["params"])
        parameters_by_id.update(
            zip(reference["params"], live["params"], strict=True)
        )
        for key in set(reference) - {"params", "lr"}:
            if candidate[key] != reference[key]:
                raise ValueError(
                    f"checkpoint optimizer static option {key!r} mismatch"
                )
    if any(type(key) is not int or key not in valid_ids for key in value["state"]):
        raise ValueError("checkpoint optimizer state parameter id mismatch")
    for parameter_id, state in value["state"].items():
        if not isinstance(state, Mapping):
            raise ValueError("checkpoint optimizer per-parameter state must be a mapping")
        expected_keys = {"step", "exp_avg", "exp_avg_sq"}
        group = next(
            group
            for group in value["param_groups"]
            if parameter_id in group["params"]
        )
        if group["amsgrad"]:
            expected_keys.add("max_exp_avg_sq")
        if set(state) != expected_keys:
            raise ValueError("checkpoint optimizer per-parameter schema mismatch")
        parameter = parameters_by_id[parameter_id]
        step = state["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1:
            raise ValueError("checkpoint optimizer step must be a scalar tensor")
        if float(step.detach().cpu()) != float(expected_steps):
            raise ValueError("checkpoint optimizer step count is inconsistent")
        for name in expected_keys - {"step"}:
            tensor = state[name]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.shape != parameter.shape
                or tensor.dtype != parameter.dtype
            ):
                raise ValueError(
                    f"checkpoint optimizer {name} tensor contract mismatch"
                )
    _validate_primitive_tree(value, path=("optimizer_state",))


def _validate_scheduler_state(
    value: object, *, expected: Mapping[str, Any], expected_last_epoch: int
) -> None:
    if not isinstance(value, Mapping) or set(value) != set(expected):
        raise ValueError("checkpoint scheduler_state schema mismatch")
    if value.get("last_epoch") != expected_last_epoch:
        raise ValueError("checkpoint scheduler epoch is inconsistent")
    if value.get("_step_count") != expected_last_epoch + 1:
        raise ValueError("checkpoint scheduler step count is inconsistent")
    if value.get("base_lrs") != expected.get("base_lrs"):
        raise ValueError("checkpoint scheduler base learning rates mismatch")
    if value.get("lr_lambdas") != expected.get("lr_lambdas"):
        raise ValueError("checkpoint scheduler lambda schema mismatch")
    last_lrs = value.get("_last_lr")
    if not isinstance(last_lrs, list) or len(last_lrs) != len(value["base_lrs"]):
        raise ValueError("checkpoint scheduler last learning rates are invalid")
    _validate_primitive_tree(value, path=("scheduler_state",))


def _validate_primitive_tree(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError("checkpoint state keys must be strings or integers")
            if isinstance(key, str):
                tokens = set(filter(None, re.split(r"[^a-z0-9]+", key.casefold())))
                if tokens & _SENSITIVE_TOKENS:
                    raise ValueError(f"sensitive checkpoint state key {key!r}")
            _validate_primitive_tree(child, path=(*path, str(key)))
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _validate_primitive_tree(child, path=(*path, str(index)))
        return
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("checkpoint state tensors must be finite")
        return
    if value is None or isinstance(value, (bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("checkpoint state floats must be finite")
        return
    if isinstance(value, str):
        raise ValueError("arbitrary strings are forbidden in checkpoint state")
    raise ValueError(
        f"unsupported checkpoint state type {type(value).__name__} at {'/'.join(path)}"
    )


def _atomic_append_json_line(path: Path, row: Mapping[str, Any]) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    text = existing + json.dumps(dict(row), allow_nan=False, sort_keys=True) + "\n"
    _atomic_write_text(path, text)


def _atomic_write_text(path: Path, text: str) -> None:
    encoded = text.encode("utf-8")
    _atomic_write_binary(path, lambda handle: handle.write(encoded))


def _atomic_write_binary(path: Path, writer: Callable[[BinaryIO], Any]) -> None:
    if _has_symlink_component(path.parent):
        raise ValueError("atomic artifact path cannot traverse symlinks")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise ValueError("atomic artifact destination symlink is forbidden")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _has_symlink_component(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))
