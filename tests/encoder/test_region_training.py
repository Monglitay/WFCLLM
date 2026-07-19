from __future__ import annotations

import torch

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.region_training import (
    build_region_training_groups_from_blocks,
    semantic_region_diversity_loss,
    split_source_ids,
)
from wfcllm.semantic.lsh import resolve_checkpoint_encoder_config


def test_region_diversity_loss_prefers_semantically_close_orthogonal_codes() -> None:
    planes = torch.zeros(4, 128)
    planes[:, :4] = torch.eye(4)
    common = torch.zeros(128)
    common[4:] = 1.0
    hadamard = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, -1.0, 1.0, -1.0],
            [1.0, 1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0, 1.0],
        ]
    )
    diverse = torch.stack([common + 0.25 * row.tolist()[0] * planes[0]
                           + 0.25 * row.tolist()[1] * planes[1]
                           + 0.25 * row.tolist()[2] * planes[2]
                           + 0.25 * row.tolist()[3] * planes[3]
                           for row in hadamard]).unsqueeze(0)
    diverse = torch.nn.functional.normalize(diverse, dim=-1)
    collapsed = diverse[:, :1, :].expand(-1, 4, -1).clone()
    negative = torch.nn.functional.normalize((-common).unsqueeze(0), dim=-1)

    diverse_loss, diverse_metrics = semantic_region_diversity_loss(
        diverse,
        negative,
        planes,
    )
    collapsed_loss, collapsed_metrics = semantic_region_diversity_loss(
        collapsed,
        negative,
        planes,
    )

    assert diverse_loss < collapsed_loss
    assert diverse_metrics["region_orthogonality"] < collapsed_metrics["region_orthogonality"]
    assert diverse_metrics["prototype_sign_accuracy"] > collapsed_metrics["prototype_sign_accuracy"]
    assert diverse_metrics["mean_positive_cosine"] > 0.95


def test_region_training_group_selection_is_deterministic_and_cross_group_negative() -> None:
    blocks = [
        {
            "source": f"anchor-{index}",
            "positive_variants": [f"anchor-{index}-variant-{j}" for j in range(4)],
            "negative_variants": [],
            "source_id": f"source-{index // 2}",
        }
        for index in range(8)
    ]

    first = build_region_training_groups_from_blocks(blocks, max_groups=5, seed=17)
    repeated = build_region_training_groups_from_blocks(blocks, max_groups=5, seed=17)

    assert first == repeated
    assert len(first) == 5
    assert all(len(group["positives"]) == 3 for group in first)
    assert all(group["negative"] != group["anchor"] for group in first)
    assert all(group["negative_source_id"] != group["source_id"] for group in first)


def test_checkpoint_metadata_controls_pooling_and_lora_without_changing_model_path() -> None:
    base = EncoderConfig(
        model_name="/models/codet5-base",
        embed_dim=128,
        use_lora=False,
        use_bf16=False,
    )
    resolved = resolve_checkpoint_encoder_config(
        base,
        {
            "config": {
                "model_name": "stale/relative/model",
                "embed_dim": 128,
                "pooling": "masked_mean",
                "use_lora": True,
                "use_bf16": True,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.05,
                "lora_target_modules": ["q", "v"],
            }
        },
    )

    assert resolved.model_name == "/models/codet5-base"
    assert resolved.pooling == "masked_mean"
    assert resolved.use_lora is True
    assert resolved.use_bf16 is True
    assert resolved.lora_r == 8


def test_source_split_is_deterministic_disjoint_and_complete() -> None:
    source_ids = [f"source-{index}" for index in range(40)]

    first = split_source_ids(source_ids, seed=23)
    repeated = split_source_ids(reversed(source_ids), seed=23)

    assert first == repeated
    assert all(first[name] for name in ("train", "validation", "test"))
    assert set(first["train"]).isdisjoint(first["validation"])
    assert set(first["train"]).isdisjoint(first["test"])
    assert set(first["validation"]).isdisjoint(first["test"])
    assert set().union(*map(set, first.values())) == set(source_ids)
