"""Auditable first-version losses for the semantic window gate."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

_INT8_ABS_MAX = 8.0
_INT8_LEVEL_MAX = 127.0
_INT8_SCALE = _INT8_ABS_MAX / _INT8_LEVEL_MAX


@dataclass(frozen=True)
class GateLossWeights:
    """Fixed-shape scalar weights for all v1 loss components."""

    close_bce: float = 1.0
    suitable_bce: float = 1.0
    close_positive: float = 1.0
    suitable_positive: float = 1.0
    suitable_false_positive: float = 4.0
    context_consistency: float = 1.0
    batch_consistency: float = 1.0
    quantization_consistency: float = 0.1


def fake_quantize_int8_ste(logits: torch.Tensor) -> torch.Tensor:
    """Apply fixed symmetric signed-int8 fake quantization with an STE.

    This is deliberately only a training approximation.  Formal bundle
    publication still requires the independent float/quantized validation
    matrix.
    """

    _validate_vector("logits", logits)
    clamped = logits.clamp(-_INT8_ABS_MAX, _INT8_ABS_MAX)
    dequantized = torch.round(clamped / _INT8_SCALE) * _INT8_SCALE
    # Forward value is quantized; backward derivative through the clamp's
    # interior is the identity straight-through estimator.
    return clamped + (dequantized - clamped).detach()


class GateLoss(nn.Module):
    """Compute the six directly supervised v1 training objectives."""

    def __init__(self, weights: GateLossWeights | None = None) -> None:
        super().__init__()
        self.weights = weights or GateLossWeights()
        if not isinstance(self.weights, GateLossWeights):
            raise ValueError("weights must be GateLossWeights")
        _validate_weights(self.weights)

    def close(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return positive-class-weighted close BCE."""

        logits, targets, selected = _validated_binary_inputs(
            logits, targets, mask=mask
        )
        if not bool(selected.any()):
            return logits.sum() * 0.0
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        example_weights = torch.where(
            targets > 0.5,
            torch.as_tensor(
                self.weights.close_positive, device=logits.device, dtype=logits.dtype
            ),
            torch.ones((), device=logits.device, dtype=logits.dtype),
        )
        return (losses * example_weights)[selected].mean()

    def suitable(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dangerous_negative: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return weighted suitable BCE including dangerous-negative FP cost."""

        base, dangerous = self._suitable_parts(
            logits, targets, dangerous_negative, mask=mask
        )
        return base + dangerous

    def consistency(
        self,
        probabilities: torch.Tensor,
        group_ids: Sequence[str],
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Penalize probability disagreement only within repeated groups."""

        _validate_vector("probabilities", probabilities)
        if len(group_ids) != probabilities.numel():
            raise ValueError("group_ids must match probabilities length")
        if any(not isinstance(group_id, str) or not group_id for group_id in group_ids):
            raise ValueError("group_ids must contain non-empty strings")
        if probabilities.numel() and (
            torch.any(probabilities < 0) or torch.any(probabilities > 1)
        ):
            raise ValueError("probabilities must be in [0, 1]")
        selected = _validated_mask("mask", mask, probabilities)
        if not bool(selected.any()):
            return probabilities.sum() * 0.0
        selected_indices = selected.nonzero(as_tuple=False).squeeze(-1).tolist()
        probabilities = probabilities[selected]
        group_ids = [group_ids[index] for index in selected_indices]

        terms: list[torch.Tensor] = []
        for group_id in dict.fromkeys(group_ids):
            indices = [
                index for index, candidate in enumerate(group_ids)
                if candidate == group_id
            ]
            if len(indices) < 2:
                continue
            values = probabilities[
                torch.tensor(indices, device=probabilities.device, dtype=torch.long)
            ]
            terms.append(torch.mean((values - values.mean()) ** 2))
        if not terms:
            return probabilities.sum() * 0.0
        return torch.stack(terms).mean()

    def batch_consistency(
        self,
        logits: torch.Tensor,
        alternate_logits: torch.Tensor,
        *,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compare probabilities for identical rows in two batch/padding layouts."""

        _validate_vector("logits", logits)
        _validate_vector("alternate_logits", alternate_logits)
        if logits.shape != alternate_logits.shape:
            raise ValueError("alternate_logits must match logits shape")
        if logits.device != alternate_logits.device:
            raise ValueError("alternate_logits must be on the same device as logits")
        if logits.dtype != alternate_logits.dtype:
            raise ValueError("alternate_logits must use the same dtype as logits")
        selected = _validated_mask("mask", mask, logits)
        if not bool(selected.any()):
            return (logits.sum() + alternate_logits.sum()) * 0.0
        logits = logits[selected]
        alternate_logits = alternate_logits[selected]
        return F.mse_loss(torch.sigmoid(logits), torch.sigmoid(alternate_logits))

    def quantization_consistency(
        self, logits: torch.Tensor, *, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Keep float logits near their fixed fake-int8 forward values."""

        _validate_vector("logits", logits)
        selected = _validated_mask("mask", mask, logits)
        if not bool(selected.any()):
            return logits.sum() * 0.0
        logits = logits[selected]
        quantized = fake_quantize_int8_ste(logits)
        # Detaching the quantized target avoids the zero-gradient cancellation
        # that would result from subtracting two identity-STE paths.
        return F.mse_loss(logits, quantized.detach())

    def forward(
        self,
        *,
        close_logits: torch.Tensor,
        suitable_logits: torch.Tensor,
        close_targets: torch.Tensor,
        suitable_targets: torch.Tensor,
        dangerous_negative: torch.Tensor,
        group_ids: Sequence[str],
        alternate_close_logits: torch.Tensor,
        alternate_suitable_logits: torch.Tensor,
        batch_reference_close_logits: torch.Tensor | None = None,
        batch_reference_suitable_logits: torch.Tensor | None = None,
        close_loss_mask: torch.Tensor | None = None,
        suitable_loss_mask: torch.Tensor | None = None,
        batch_consistency_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return total loss and all six weighted, named components."""

        _validate_matching_logits("suitable_logits", close_logits, suitable_logits)
        _validate_matching_logits(
            "alternate_close_logits", close_logits, alternate_close_logits
        )
        _validate_matching_logits(
            "alternate_suitable_logits", suitable_logits, alternate_suitable_logits
        )
        if batch_reference_close_logits is not None:
            _validate_matching_logits(
                "batch_reference_close_logits",
                close_logits,
                batch_reference_close_logits,
            )
        if batch_reference_suitable_logits is not None:
            _validate_matching_logits(
                "batch_reference_suitable_logits",
                suitable_logits,
                batch_reference_suitable_logits,
            )
        close_value = self.close(
            close_logits, close_targets, mask=close_loss_mask
        )
        suitable_base, dangerous_fp = self._suitable_parts(
            suitable_logits,
            suitable_targets,
            dangerous_negative,
            mask=suitable_loss_mask,
        )
        context = 0.5 * (
            self.consistency(
                torch.sigmoid(close_logits), group_ids, mask=close_loss_mask
            )
            + self.consistency(
                torch.sigmoid(suitable_logits), group_ids, mask=suitable_loss_mask
            )
        )
        batch_close = (
            close_logits
            if batch_reference_close_logits is None
            else batch_reference_close_logits
        )
        batch_suitable = (
            suitable_logits
            if batch_reference_suitable_logits is None
            else batch_reference_suitable_logits
        )
        close_batch_mask = _combined_masks(
            close_loss_mask, batch_consistency_mask, close_logits
        )
        suitable_batch_mask = _combined_masks(
            suitable_loss_mask, batch_consistency_mask, suitable_logits
        )
        batch = 0.5 * (
            self.batch_consistency(
                batch_close, alternate_close_logits, mask=close_batch_mask
            )
            + self.batch_consistency(
                batch_suitable,
                alternate_suitable_logits,
                mask=suitable_batch_mask,
            )
        )
        quantization = 0.5 * (
            self.quantization_consistency(close_logits, mask=close_loss_mask)
            + self.quantization_consistency(
                suitable_logits, mask=suitable_loss_mask
            )
        )
        components = {
            "close_bce": close_value * self.weights.close_bce,
            "suitable_bce": suitable_base * self.weights.suitable_bce,
            "dangerous_negative_fp": dangerous_fp * self.weights.suitable_bce,
            "context_consistency": context * self.weights.context_consistency,
            "batch_consistency": batch * self.weights.batch_consistency,
            "quantization_consistency": (
                quantization * self.weights.quantization_consistency
            ),
        }
        total = torch.stack(tuple(components.values())).sum()
        if not torch.isfinite(total):
            raise ValueError("gate loss must be finite")
        return total, components

    def _suitable_parts(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        dangerous_negative: torch.Tensor,
        *,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, targets, selected = _validated_binary_inputs(
            logits, targets, mask=mask
        )
        if not isinstance(dangerous_negative, torch.Tensor):
            raise ValueError("dangerous_negative must be a tensor")
        if dangerous_negative.shape != logits.shape or dangerous_negative.dtype != torch.bool:
            raise ValueError("dangerous_negative must be a matching bool tensor")
        if dangerous_negative.device != logits.device:
            raise ValueError("dangerous_negative must be on the same device as logits")
        losses = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        positive_weights = torch.where(
            targets > 0.5,
            torch.as_tensor(
                self.weights.suitable_positive,
                device=logits.device,
                dtype=logits.dtype,
            ),
            torch.ones((), device=logits.device, dtype=logits.dtype),
        )
        if not bool(selected.any()):
            zero = logits.sum() * 0.0
            return zero, zero
        base = (losses * positive_weights)[selected].mean()
        dangerous_mask = selected & dangerous_negative & (targets <= 0.5)
        extra_weights = dangerous_mask.to(logits.dtype) * (
            self.weights.suitable_false_positive - 1.0
        )
        # Keep the same selected-row denominator so changing dangerous labels
        # cannot change the reduction's population.
        dangerous = (losses * extra_weights)[selected].mean()
        return base, dangerous


def _validated_binary_inputs(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _validate_vector("logits", logits)
    _validate_vector("targets", targets)
    if logits.shape != targets.shape:
        raise ValueError("targets must match logits shape")
    if logits.device != targets.device:
        raise ValueError("targets must be on the same device as logits")
    if torch.any((targets != 0) & (targets != 1)):
        raise ValueError("targets must be binary")
    selected = _validated_mask("mask", mask, logits)
    return logits, targets.to(dtype=logits.dtype), selected


def _validated_mask(
    name: str, mask: torch.Tensor | None, reference: torch.Tensor
) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if not isinstance(mask, torch.Tensor) or mask.shape != reference.shape:
        raise ValueError(f"{name} must be a tensor matching logits shape")
    if mask.dtype != torch.bool:
        raise ValueError(f"{name} must use bool dtype")
    if mask.device != reference.device:
        raise ValueError(f"{name} must be on the same device as logits")
    return mask


def _combined_masks(
    first: torch.Tensor | None,
    second: torch.Tensor | None,
    reference: torch.Tensor,
) -> torch.Tensor:
    return _validated_mask("first mask", first, reference) & _validated_mask(
        "second mask", second, reference
    )


def _validate_matching_logits(
    name: str, reference: torch.Tensor, candidate: torch.Tensor
) -> None:
    _validate_vector("reference logits", reference)
    _validate_vector(name, candidate)
    if candidate.shape != reference.shape:
        raise ValueError(f"{name} must match primary logits shape")
    if candidate.device != reference.device:
        raise ValueError(f"{name} must be on the same device as primary logits")
    if candidate.dtype != reference.dtype:
        raise ValueError(f"{name} must use the same dtype as primary logits")


def _validate_vector(name: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor) or value.ndim != 1:
        raise ValueError(f"{name} must be a 1-D tensor")
    if value.numel() == 0:
        raise ValueError(f"{name} must not be empty")
    if not value.is_floating_point():
        raise ValueError(f"{name} must use a floating dtype")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} must be finite")


def _validate_weights(weights: GateLossWeights) -> None:
    for name in (
        "close_bce",
        "suitable_bce",
        "close_positive",
        "suitable_positive",
    ):
        value = getattr(weights, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be a finite positive number")
    if (
        type(weights.suitable_false_positive) not in (int, float)
        or not math.isfinite(weights.suitable_false_positive)
        or weights.suitable_false_positive < 1
    ):
        raise ValueError("suitable_false_positive must be finite and at least 1")
    for name in (
        "context_consistency",
        "batch_consistency",
        "quantization_consistency",
    ):
        value = getattr(weights, name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite non-negative number")
