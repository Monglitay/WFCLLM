from __future__ import annotations

import pytest
import torch

from wfcllm.watermark.anchor_lsh import (
    anchored_signature,
    hamming_distance,
    min_margin_with_planes,
    pairwise_hamming_diversity,
    project_planes_orthogonal,
    random_anchor,
    residual_signature,
    sign_with_planes,
)
from wfcllm.watermark.lsh_space import LSHSpace


def test_project_planes_orthogonal_removes_anchor_direction():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchor = torch.tensor([1.0, 0.0])

    projected = project_planes_orthogonal(planes, anchor)

    assert torch.allclose(projected[0], torch.zeros(2), atol=1e-6)
    assert torch.allclose(projected[1], torch.tensor([0.0, 1.0]), atol=1e-6)


def test_anchored_signature_is_invariant_to_removed_direction():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    anchor = torch.tensor([1.0, 0.0])

    sig_a = anchored_signature(torch.tensor([10.0, 1.0]), planes, anchor)
    sig_b = anchored_signature(torch.tensor([-10.0, 1.0]), planes, anchor)

    assert sig_a == sig_b


def test_sign_with_planes_accepts_planes_on_different_device():
    u = torch.tensor([1.0, 0.0])
    planes = torch.tensor([[1.0, 0.0]], dtype=torch.float64)

    assert sign_with_planes(u, planes) == (1,)


def test_min_margin_with_planes_normalizes_plane_rows():
    planes = torch.tensor([[2.0, 0.0]])

    assert min_margin_with_planes(torch.tensor([1.0, 0.0]), planes) == pytest.approx(1.0)


def test_pairwise_hamming_diversity_normalizes_by_signature_width():
    signatures = [(0, 0), (0, 1), (1, 1)]
    assert pairwise_hamming_diversity(signatures) == pytest.approx(2 / 3)


def test_residual_signature_subtracts_seqmark_center():
    planes = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    center = torch.tensor([10.0, 0.0])

    signature = residual_signature(
        torch.tensor([9.0, 1.0]),
        center=center,
        planes=planes,
    )

    assert signature == (0, 1)


def test_random_anchor_is_deterministic_and_unit_norm():
    a = random_anchor(secret_key="k", context_id="ctx-1", method="random", embed_dim=4)
    b = random_anchor(secret_key="k", context_id="ctx-1", method="random", embed_dim=4)
    c = random_anchor(secret_key="k", context_id="ctx-2", method="random", embed_dim=4)

    assert torch.allclose(a, b)
    assert not torch.allclose(a, c)
    assert torch.linalg.vector_norm(a).item() == pytest.approx(1.0)


def test_lsh_space_exposes_planes_as_defensive_copy():
    space = LSHSpace(secret_key="k", embed_dim=4, d=2)

    planes = space.planes
    planes[0, 0] = planes[0, 0] + 10.0

    assert not torch.allclose(planes, space.planes)
