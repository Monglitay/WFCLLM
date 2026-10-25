"""The projection objective must constrain the declared public signature bits."""
from __future__ import annotations

import pytest
import torch

from wfcllm.encoder.projection_training import PUBLIC_PLANE_ID, _projection_public_planes
from wfcllm.encoder.region_training import semantic_region_diversity_loss
from wfcllm.semantic.lsh_space import LSHSpace


@pytest.mark.parametrize('language,count', [('cpp', 12), ('java', 4), ('js', 4), ('python', 4)])
def test_public_training_planes_match_runtime_prefix(language: str, count: int) -> None:
    training = _projection_public_planes(language, 128)
    runtime = LSHSpace(secret_key=PUBLIC_PLANE_ID, embed_dim=128, d=12).planes
    assert training.shape == (count, 128)
    assert torch.equal(training, runtime[:count])


def test_four_view_objective_has_finite_gradients_on_all_twelve_planes() -> None:
    generator = torch.Generator().manual_seed(173)
    groups = torch.randn(2, 4, 128, generator=generator, requires_grad=True)
    negatives = torch.randn(2, 128, generator=generator, requires_grad=True)
    planes = _projection_public_planes('cpp', 128).clone().requires_grad_()
    loss, _ = semantic_region_diversity_loss(groups, negatives, planes)
    loss.backward()
    assert torch.isfinite(loss)
    assert groups.grad is not None and torch.isfinite(groups.grad).all()
    assert planes.grad is not None and torch.isfinite(planes.grad).all()
    assert torch.all(planes.grad[4:].abs().sum(dim=1) > 0)
