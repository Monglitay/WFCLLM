from __future__ import annotations

import hashlib
import json
import random
import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

from wfcllm.gate.dataset import GateExample
from wfcllm.gate.input import GateInput
from wfcllm.gate.model import GateModel
from wfcllm.gate.trainer import GateTrainer, GateTrainerConfig, seed_gate_training
from wfcllm.windowing.normalization import WINDOW_NORMALIZATION_VERSION


class TinyTokenizer:
    pad_token_id = 0

    def __call__(self, text: str, **kwargs: object) -> dict[str, list[int]]:
        assert kwargs["truncation"] is False
        ids = [1 + (ord(char) % 29) for char in text]
        return {"input_ids": ids, "attention_mask": [1] * len(ids)}


class TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 6)

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        return SimpleNamespace(last_hidden_state=self.embedding(input_ids))


def _example(
    group: str,
    context: int,
    *,
    suitable: bool,
    code: str = "x = 1",
) -> GateExample:
    gate_input = GateInput(
        normalization_version=WINDOW_NORMALIZATION_VERSION,
        parent_descriptor="v1|module|parent=block|ordinal=0|role=body",
        depth=0,
        previous_units=(),
        previous_unit_types=(),
        current_units=(code,),
        current_unit_types=("expression_statement",),
        current_unit_count=1,
        current_token_count=3,
    )
    return GateExample.from_gate_input(
        group_id=group,
        window_start_unit_id=f"{group}-start",
        context_length=context,
        budget=1,
        gate_input=gate_input,
        close_target=not suitable,
        suitable_target=suitable,
        dangerous_negative=not suitable,
    )


def _examples() -> tuple[list[GateExample], list[GateExample]]:
    secret = "DEPLOYMENT_SECRET_DO_NOT_WRITE"
    train = [
        _example("positive", context, suitable=True)
        for context in (1, 2, 3)
    ] + [
        _example("negative", context, suitable=False, code=f"x = '{secret}'")
        for context in (1, 2, 3)
    ]
    validation = [
        _example("validation", 1, suitable=False),
        _example("validation", 2, suitable=True),
    ]
    assert len(train) + len(validation) == 8
    return train, validation


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _walk_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [
            item
            for key, child in value.items()
            for item in (_walk_strings(key) + _walk_strings(child))
        ]
    if isinstance(value, (list, tuple)):
        return [item for child in value for item in _walk_strings(child)]
    return []


def test_fit_one_fake_epoch_writes_only_safe_formal_artifacts(tmp_path: Path) -> None:
    train, validation = _examples()
    output_dir = tmp_path / "run"
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=output_dir,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6, seed=17),
    )
    summary = trainer.fit(train, validation)

    written = {
        path.relative_to(output_dir).as_posix()
        for path in output_dir.rglob("*")
        if path.is_file()
    }
    assert written == {
        "checkpoints/last.pt",
        "checkpoints/best.pt",
        "training_metrics.jsonl",
        "development_summary.json",
    }
    assert summary["epochs_completed"] == 1

    forbidden = ("DEPLOYMENT_SECRET_DO_NOT_WRITE", "[CURRENT]", "key-bank-raw")
    for path in output_dir.rglob("*"):
        if not path.is_file():
            continue
        assert all(token.encode("utf-8") not in path.read_bytes() for token in forbidden)

    checkpoint = torch.load(
        output_dir / "checkpoints" / "last.pt",
        map_location="cpu",
        weights_only=True,
    )
    strings = "\n".join(_walk_strings(checkpoint))
    assert "serialized_gate_input" not in strings
    assert "key_bank" not in strings
    assert "deployment_key" not in strings
    assert "optimizer_state" in checkpoint
    assert "scheduler_state" in checkpoint

    metrics = json.loads((output_dir / "training_metrics.jsonl").read_text("utf-8"))
    assert set(metrics["loss_components"]) == {
        "close_bce",
        "suitable_bce",
        "dangerous_negative_fp",
        "context_consistency",
        "batch_consistency",
        "quantization_consistency",
    }


def test_fit_sets_python_numpy_and_torch_seeds(tmp_path: Path) -> None:
    train, validation = _examples()
    config = GateTrainerConfig(epochs=1, batch_size=6, seed=73)
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path / "seeded",
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=config,
    )
    trainer.fit(train, validation)
    observed = (random.random(), float(np.random.random()), float(torch.rand(())))

    random.seed(73)
    np.random.seed(73)
    torch.manual_seed(73)
    # Training advances Torch's RNG, but Python/NumPy are not consumed after seeding.
    assert observed[:2] == pytest.approx((random.random(), float(np.random.random())))
    assert 0.0 <= observed[2] < 1.0


class PaddingSensitiveModel(GateModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[no-untyped-def]
        output = super().forward(input_ids, attention_mask)
        offset = input_ids.shape[1] * 0.01
        return type(output)(
            close_logits=output.close_logits + offset,
            suitable_logits=output.suitable_logits + offset,
        )


class BatchMembershipSensitiveModel(GateModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[no-untyped-def]
        output = super().forward(input_ids, attention_mask)
        # Permutation invariant and padding-mask aware, but incorrectly lets
        # the set of batch neighbors influence every target decision.
        offset = attention_mask.sum().to(output.close_logits.dtype) * 0.001
        return type(output)(
            close_logits=output.close_logits + offset,
            suitable_logits=output.suitable_logits + offset,
        )


def test_fit_exposes_padding_or_batch_invariance_defect_before_training(
    tmp_path: Path,
) -> None:
    train, validation = _examples()
    trainer = GateTrainer(
        model=PaddingSensitiveModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path / "bad",
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    with pytest.raises(RuntimeError, match="batch/padding invariance"):
        trainer.fit(train, validation)


def test_fit_exposes_order_invariant_batch_membership_defect_before_training(
    tmp_path: Path,
) -> None:
    train, validation = _examples()
    trainer = GateTrainer(
        model=BatchMembershipSensitiveModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path / "bad-members",
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    with pytest.raises(RuntimeError, match="batch/padding invariance"):
        trainer.fit(train, validation)


def test_validation_reports_overflow_coverage_and_excludes_it_from_fpr(
    tmp_path: Path,
) -> None:
    train, _ = _examples()
    validation = [
        _example("valid-pair", 1, suitable=False),
        _example("valid-pair", 2, suitable=True),
        _example("overflow-negative", 1, suitable=False, code="x" * 700),
    ]
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path / "coverage",
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    metrics = trainer.fit(train, validation)["best_validation"]
    assert metrics["total_count"] == 3
    assert metrics["evaluable_count"] == 2
    assert metrics["overflow_count"] == 1
    assert metrics["negative_count"] == 1
    assert metrics["comparable_group_count"] == 1
    assert metrics["coverage"] == pytest.approx(2 / 3)


@pytest.mark.parametrize("kind", ["all-overflow", "no-negative"])
def test_validation_fails_closed_before_output_or_training(
    tmp_path: Path, kind: str
) -> None:
    train, _ = _examples()
    if kind == "all-overflow":
        validation = [
            _example(f"overflow-{index}", 1, suitable=False, code="x" * 700)
            for index in range(2)
        ]
        message = "zero evaluable"
    else:
        validation = [
            _example(f"positive-{index}", 1, suitable=True)
            for index in range(2)
        ]
        message = "zero evaluable suitable-negative"
    output = tmp_path / kind
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=output,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    before = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(ValueError, match=message):
        trainer.fit(train, validation)
    assert not output.exists()
    for name, tensor in before.items():
        assert torch.equal(tensor, trainer.model.state_dict()[name])


def test_validation_rejects_zero_comparable_cohorts(tmp_path: Path) -> None:
    train, _ = _examples()
    validation = [
        _example("singleton-negative", 1, suitable=False),
        _example("singleton-positive", 1, suitable=True),
    ]
    output = tmp_path / "singletons"
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=output,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    with pytest.raises(ValueError, match="zero comparable"):
        trainer.fit(train, validation)
    assert not output.exists()


class SecretExtraStateModel(GateModel):
    def get_extra_state(self) -> object:
        return {"deployment_key": "DO_NOT_SERIALIZE"}

    def set_extra_state(self, state: object) -> None:
        self._loaded_extra = state


def test_model_extra_state_secret_is_rejected_before_artifact_write(
    tmp_path: Path,
) -> None:
    train, validation = _examples()
    output = tmp_path / "extra"
    trainer = GateTrainer(
        model=SecretExtraStateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=output,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    with pytest.raises(ValueError, match="extra state"):
        trainer.fit(train, validation)
    assert not output.exists()


@pytest.mark.parametrize("kind", ["polluted", "symlink"])
def test_fresh_output_rejects_pollution_and_symlink(
    tmp_path: Path, kind: str
) -> None:
    train, validation = _examples()
    output = tmp_path / "output"
    if kind == "polluted":
        output.mkdir()
        (output / "foreign.txt").write_text("do not overwrite", encoding="utf-8")
    else:
        target = tmp_path / "target"
        target.mkdir()
        output.symlink_to(target, target_is_directory=True)
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=output,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=6),
    )
    with pytest.raises(ValueError, match="empty|symlink"):
        trainer.fit(train, validation)


def test_all_overflow_training_skips_batch_consistency_but_trains_close(
    tmp_path: Path,
) -> None:
    train = [
        _example("overflow-train", context, suitable=False, code="x" * 700)
        for context in (1, 2, 3)
    ]
    _, validation = _examples()
    trainer = GateTrainer(
        model=GateModel(encoder=TinyEncoder(), hidden_size=6),
        tokenizer=TinyTokenizer(),
        output_dir=tmp_path / "all-overflow",
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=3),
    )
    trainer.fit(train, validation)
    row = json.loads((tmp_path / "all-overflow" / "training_metrics.jsonl").read_text("utf-8"))
    assert row["batch_consistency_audit"] == {
        "evaluated_batches": 0,
        "skipped_batches": 1,
        "skip_reason": "no_valid_real_neighbor",
    }
    assert row["loss_components"]["batch_consistency"] == pytest.approx(0.0)
    assert row["loss_components"]["close_bce"] > 0


class LaterBatchMembershipSensitiveModel(GateModel):
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):  # type: ignore[no-untyped-def]
        output = super().forward(input_ids, attention_mask)
        active = (attention_mask[0].sum() > 400).to(output.close_logits.dtype)
        offset = active * attention_mask.sum() * 0.001
        return type(output)(
            close_logits=output.close_logits + offset,
            suitable_logits=output.suitable_logits + offset,
        )


def test_preflight_audits_every_training_batch_before_any_optimizer_step(
    tmp_path: Path,
) -> None:
    train = [
        _example("short", context, suitable=False)
        for context in (1, 2, 3)
    ] + [
        _example("long", context, suitable=False, code="x" * 200)
        for context in (1, 2, 3)
    ]
    _, validation = _examples()
    output = tmp_path / "late-defect"
    trainer = GateTrainer(
        model=LaterBatchMembershipSensitiveModel(
            encoder=TinyEncoder(), hidden_size=6
        ),
        tokenizer=TinyTokenizer(),
        output_dir=output,
        config_hash=_sha("config"),
        dataset_manifest_hash=_sha("dataset"),
        config=GateTrainerConfig(epochs=1, batch_size=3),
    )
    before = copy.deepcopy(trainer.model.state_dict())
    with pytest.raises(RuntimeError, match="batch/padding invariance"):
        trainer.fit(train, validation)
    assert not output.exists()
    for name, tensor in before.items():
        assert torch.equal(tensor, trainer.model.state_dict()[name])


def test_public_seed_entry_point_reproduces_model_initialization() -> None:
    seed_gate_training(12345)
    first = GateModel(encoder=TinyEncoder(), hidden_size=6)
    seed_gate_training(12345)
    second = GateModel(encoder=TinyEncoder(), hidden_size=6)
    for name, tensor in first.state_dict().items():
        assert torch.equal(tensor, second.state_dict()[name])
