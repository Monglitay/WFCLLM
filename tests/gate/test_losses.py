from __future__ import annotations

import pytest
import torch

from wfcllm.gate.losses import GateLoss, GateLossWeights, fake_quantize_int8_ste


def test_false_positive_penalty_is_larger_for_dangerous_negative() -> None:
    loss_fn = GateLoss(GateLossWeights(suitable_false_positive=4.0))
    logits = torch.tensor([2.0, 2.0])
    targets = torch.tensor([0.0, 0.0])
    ordinary = loss_fn.suitable(logits[:1], targets[:1], torch.tensor([False]))
    dangerous = loss_fn.suitable(logits[1:], targets[1:], torch.tensor([True]))
    assert dangerous > ordinary


def test_context_consistency_penalizes_same_group_disagreement() -> None:
    loss_fn = GateLoss(GateLossWeights(context_consistency=1.0))
    same = loss_fn.consistency(torch.tensor([0.1, 0.1]), ["g", "g"])
    different = loss_fn.consistency(torch.tensor([0.1, 0.9]), ["g", "g"])
    assert same.item() == pytest.approx(0.0)
    assert different > same


def test_consistency_does_not_mix_different_groups() -> None:
    loss_fn = GateLoss()
    value = loss_fn.consistency(torch.tensor([0.1, 0.9]), ["a", "b"])
    assert value.item() == pytest.approx(0.0)


def test_fake_quant_consistency_is_finite_and_uses_fixed_int8_grid() -> None:
    logits = torch.tensor([-20.0, -2.3, 0.2, 4.7, 20.0], requires_grad=True)
    quantized = fake_quantize_int8_ste(logits)
    assert torch.isfinite(quantized).all()
    assert quantized.min() >= -8.0
    assert quantized.max() <= 8.0

    value = GateLoss().quantization_consistency(logits)
    assert torch.isfinite(value)
    value.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_forward_returns_total_and_all_six_named_components() -> None:
    loss_fn = GateLoss()
    close_logits = torch.tensor([0.1, -0.4, 0.7], requires_grad=True)
    suitable_logits = torch.tensor([-0.2, 0.8, -0.6], requires_grad=True)
    total, parts = loss_fn(
        close_logits=close_logits,
        suitable_logits=suitable_logits,
        close_targets=torch.tensor([0.0, 1.0, 0.0]),
        suitable_targets=torch.tensor([0.0, 1.0, 0.0]),
        dangerous_negative=torch.tensor([True, False, False]),
        group_ids=["g", "g", "g"],
        alternate_close_logits=close_logits + torch.tensor([0.0, 0.1, 0.0]),
        alternate_suitable_logits=suitable_logits + torch.tensor([0.0, 0.0, 0.1]),
    )
    assert set(parts) == {
        "close_bce",
        "suitable_bce",
        "dangerous_negative_fp",
        "context_consistency",
        "batch_consistency",
        "quantization_consistency",
    }
    assert total.item() == pytest.approx(sum(part.item() for part in parts.values()))
    total.backward()
    assert close_logits.grad is not None
    assert suitable_logits.grad is not None


def test_loss_masks_exclude_overflow_suitable_row() -> None:
    loss_fn = GateLoss()
    ordinary = loss_fn.suitable(
        torch.tensor([0.0]),
        torch.tensor([0.0]),
        torch.tensor([False]),
        mask=torch.tensor([True]),
    )
    masked = loss_fn.suitable(
        torch.tensor([100.0, 0.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor([True, False]),
        mask=torch.tensor([False, True]),
    )
    assert masked.item() == pytest.approx(ordinary.item())


def test_overflow_mask_excludes_suitable_from_every_loss_branch_but_not_close() -> None:
    loss_fn = GateLoss()

    def compute(suitable_value: float, close_value: float):
        close_logits = torch.tensor([close_value], requires_grad=True)
        suitable_logits = torch.tensor([suitable_value], requires_grad=True)
        total, parts = loss_fn(
            close_logits=close_logits,
            suitable_logits=suitable_logits,
            close_targets=torch.tensor([1.0]),
            suitable_targets=torch.tensor([0.0]),
            dangerous_negative=torch.tensor([True]),
            group_ids=["overflow"],
            alternate_close_logits=close_logits + 0.25,
            alternate_suitable_logits=suitable_logits + 1000.0,
            batch_reference_close_logits=close_logits,
            batch_reference_suitable_logits=suitable_logits,
            close_loss_mask=torch.tensor([True]),
            suitable_loss_mask=torch.tensor([False]),
        )
        return total, parts

    baseline_total, baseline_parts = compute(0.0, 0.0)
    changed_suitable_total, changed_suitable_parts = compute(100.0, 0.0)
    assert changed_suitable_total.item() == pytest.approx(baseline_total.item())
    assert {
        name: value.item() for name, value in changed_suitable_parts.items()
    } == pytest.approx({name: value.item() for name, value in baseline_parts.items()})
    assert baseline_parts["suitable_bce"].item() == pytest.approx(0.0)
    assert baseline_parts["dangerous_negative_fp"].item() == pytest.approx(0.0)

    changed_close_total, _ = compute(0.0, 2.0)
    assert changed_close_total.item() != pytest.approx(baseline_total.item())


def test_forward_rejects_misaligned_reference_and_alternate_contracts() -> None:
    loss_fn = GateLoss()
    common = {
        "close_logits": torch.tensor([0.0, 0.0]),
        "suitable_logits": torch.tensor([0.0, 0.0]),
        "close_targets": torch.tensor([0.0, 1.0]),
        "suitable_targets": torch.tensor([0.0, 1.0]),
        "dangerous_negative": torch.tensor([True, False]),
        "group_ids": ["g", "g"],
        "alternate_suitable_logits": torch.tensor([0.0, 0.0]),
    }
    with pytest.raises(ValueError, match="alternate_close_logits"):
        loss_fn(
            **common,
            alternate_close_logits=torch.tensor([0.0]),
        )
    with pytest.raises(ValueError, match="batch_reference_close_logits"):
        loss_fn(
            **common,
            alternate_close_logits=torch.tensor([0.0, 0.0]),
            batch_reference_close_logits=torch.tensor([0.0], dtype=torch.float64),
        )


@pytest.mark.parametrize(
    "bad",
    [
        GateLossWeights(close_bce=0.0),
        GateLossWeights(suitable_false_positive=0.5),
        GateLossWeights(context_consistency=-1.0),
    ],
)
def test_loss_rejects_invalid_weights(bad: GateLossWeights) -> None:
    with pytest.raises(ValueError):
        GateLoss(bad)
