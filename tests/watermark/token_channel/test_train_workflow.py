"""Tests for token-channel training workflow configuration helpers."""

from __future__ import annotations

import math
from pathlib import Path
import random
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from wfcllm.watermark.token_channel.train import TokenChannelEpochMetrics
from wfcllm.watermark.token_channel import train_workflow
from wfcllm.watermark.token_channel.model import TokenChannelArtifactMetadata
from wfcllm.watermark.token_channel.train_workflow import (
    TokenChannelTrainWorkflowConfig,
    TokenChannelTrainWorkflowSummary,
    format_token_channel_train_workflow_summary,
    normalize_reference_solution_rows,
    run_token_channel_train_workflow,
    split_training_rows,
)


def _training_rows(*rows: dict[str, object]) -> list[dict[str, object]]:
    return list(rows)


def _default_cached_rows() -> list[dict[str, object]]:
    return _training_rows(
        {"switch_target": 1, "prefix_tokens": [1], "next_token": 2, "teacher_logits": [0.1, 0.9]},
        {"switch_target": 0, "prefix_tokens": [2], "next_token": 1, "teacher_logits": [0.8, 0.2]},
    )


def _setup_minimal_workflow_mocks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    config: TokenChannelTrainWorkflowConfig,
    build_rows: list[dict[str, object]] | None = None,
    cached_rows: list[dict[str, object]] | None = None,
    epoch: TokenChannelEpochMetrics | None = None,
) -> None:
    epoch_metrics = epoch or TokenChannelEpochMetrics(
        epoch=1,
        train_loss=0.8,
        validation_loss=0.6,
        switch_loss=0.2,
    )
    built_rows = build_rows if build_rows is not None else _default_cached_rows()
    persisted_rows = cached_rows if cached_rows is not None else list(built_rows)

    monkeypatch.setattr(
        train_workflow,
        "load_reference_solutions",
        lambda dataset, dataset_path: [
            {"generated_code": "print('a')"},
            {"generated_code": "print('b')"},
        ],
    )
    monkeypatch.setattr(
        train_workflow,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda path: SimpleNamespace(
                name_or_path="offline-tokenizer",
                vocab_size=17,
            )
        ),
    )
    monkeypatch.setattr(train_workflow, "_load_teacher_model", lambda lm_model_path: MagicMock())
    monkeypatch.setattr(train_workflow, "build_training_rows", lambda **kwargs: built_rows)
    monkeypatch.setattr(train_workflow, "save_training_cache", lambda path, rows: None)
    monkeypatch.setattr(train_workflow, "load_training_cache", lambda path: persisted_rows)
    monkeypatch.setattr(
        train_workflow,
        "build_token_channel_batch",
        lambda rows, *, context_width: {"rows": rows},
    )
    monkeypatch.setattr(train_workflow, "TokenChannelModel", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train_workflow.torch.optim, "AdamW", lambda params, lr: MagicMock())
    monkeypatch.setattr(train_workflow, "train_one_epoch", lambda **kwargs: epoch_metrics)
    monkeypatch.setattr(
        train_workflow,
        "build_training_evidence",
        lambda **kwargs: SimpleNamespace(
            switch_target_positive_count=1,
            switch_target_negative_count=1,
            train_loss=epoch_metrics.train_loss,
            validation_loss=epoch_metrics.validation_loss,
            epochs=(epoch_metrics,),
        ),
    )


def _build_config(tmp_path: Path, **overrides: object) -> TokenChannelTrainWorkflowConfig:
    lm_model_path = tmp_path / "lm-model"
    lm_model_path.mkdir()
    values: dict[str, object] = {
        "dataset": "humaneval",
        "dataset_path": tmp_path / "dataset.jsonl",
        "lm_model_path": lm_model_path,
        "model_path": tmp_path / "artifact",
        "cache_path": tmp_path / "cache.jsonl",
        "context_width": 128,
        "hidden_size": 64,
        "batch_size": 16,
        "epochs": 3,
        "lr": 0.001,
        "entropy_threshold": 0.0,
        "diversity_threshold": 1,
        "split_ratio": 0.8,
        "seed": 7,
    }
    values.update(overrides)
    return TokenChannelTrainWorkflowConfig(**values)


def test_workflow_config_accepts_valid_values(tmp_path: Path) -> None:
    config = _build_config(tmp_path, dataset="mbpp")

    assert config.dataset == "mbpp"
    assert config.context_width == 128
    assert config.lr == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("dataset", "invalid", "dataset"),
        ("context_width", 0, "context_width"),
        ("hidden_size", 0, "hidden_size"),
        ("batch_size", 0, "batch_size"),
        ("epochs", 0, "epochs"),
        ("lr", 0.0, "lr"),
        ("entropy_threshold", -0.1, "entropy_threshold"),
        ("diversity_threshold", 0, "diversity_threshold"),
        ("split_ratio", 0.0, "split_ratio"),
        ("split_ratio", 1.0, "split_ratio"),
    ],
)
def test_workflow_config_rejects_invalid_values(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_config(tmp_path, **{field_name: field_value})


def test_workflow_config_requires_existing_lm_model_path(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing-model"

    with pytest.raises(ValueError, match="lm_model_path"):
        _build_config(tmp_path, lm_model_path=missing_path)


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("context_width", True, "context_width"),
        ("hidden_size", True, "hidden_size"),
        ("batch_size", True, "batch_size"),
        ("epochs", True, "epochs"),
        ("diversity_threshold", True, "diversity_threshold"),
        ("seed", True, "seed"),
    ],
)
def test_workflow_config_rejects_bool_integer_fields(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_config(tmp_path, **{field_name: field_value})


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("lr", math.inf, "lr"),
        ("lr", math.nan, "lr"),
        ("entropy_threshold", math.inf, "entropy_threshold"),
        ("entropy_threshold", math.nan, "entropy_threshold"),
        ("split_ratio", math.inf, "split_ratio"),
        ("split_ratio", math.nan, "split_ratio"),
    ],
)
def test_workflow_config_rejects_non_finite_float_fields(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_config(tmp_path, **{field_name: field_value})


@pytest.mark.parametrize(
    ("field_name", "field_value", "message"),
    [
        ("lm_model_path", 123, "lm_model_path"),
        ("dataset_path", 123, "dataset_path"),
        ("model_path", 123, "model_path"),
        ("cache_path", 123, "cache_path"),
    ],
)
def test_workflow_config_requires_path_like_values(
    tmp_path: Path,
    field_name: str,
    field_value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_config(tmp_path, **{field_name: field_value})


def test_summary_formatting_returns_full_ordered_output(tmp_path: Path) -> None:
    summary = TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=20,
        train_rows=16,
        validation_rows=4,
        artifact_dir=tmp_path / "artifact",
        cache_path=tmp_path / "cache.jsonl",
        compatibility_ok=True,
        epochs=(
            TokenChannelEpochMetrics(
                epoch=1,
                train_loss=0.75,
                validation_loss=0.55,
                switch_loss=0.25,
            ),
            TokenChannelEpochMetrics(
                epoch=2,
                train_loss=0.50,
                validation_loss=0.40,
                switch_loss=0.10,
            ),
        ),
        switch_target_positive_count=11,
        switch_target_negative_count=9,
    )

    lines = format_token_channel_train_workflow_summary(summary)

    assert lines == [
        "dataset: humaneval",
        "training_rows: 20",
        "train_rows: 16",
        "validation_rows: 4",
        f"artifact_dir: {summary.artifact_dir}",
        f"cache_path: {summary.cache_path}",
        "compatibility_ok: yes",
        "switch_target_positive_count: 11",
        "switch_target_negative_count: 9",
        "epoch 1: train_loss=0.7500 validation_loss=0.5500 switch_loss=0.2500",
        "epoch 2: train_loss=0.5000 validation_loss=0.4000 switch_loss=0.1000",
    ]


@pytest.mark.parametrize(
    "row",
    [
        {},
        {"generated_code": None},
        {"generated_code": 123},
        {"generated_code": ""},
    ],
)
def test_normalize_reference_solution_rows_rejects_invalid_generated_code(
    row: dict[str, object],
) -> None:
    with pytest.raises(ValueError, match="generated_code"):
        normalize_reference_solution_rows([row])


def test_normalize_reference_solution_rows_returns_source_code_samples() -> None:
    assert normalize_reference_solution_rows(
        [
            {"generated_code": "print('a')"},
            {"generated_code": "print('b')"},
        ]
    ) == [
        {"source_code": "print('a')"},
        {"source_code": "print('b')"},
    ]


def test_normalize_reference_solution_rows_rejects_non_mapping_rows() -> None:
    with pytest.raises(ValueError, match="dataset row must be a mapping"):
        normalize_reference_solution_rows(["not-a-row"])  # type: ignore[list-item]


def test_split_training_rows_rejects_single_row() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        split_training_rows([{"source_code": "print('x')"}], split_ratio=0.8, seed=7)


@pytest.mark.parametrize("split_ratio", [0.0, 1.0, -0.1, 1.1])
def test_split_training_rows_rejects_invalid_split_ratio(split_ratio: float) -> None:
    with pytest.raises(ValueError, match="split_ratio"):
        split_training_rows(
            [{"source_code": "a"}, {"source_code": "b"}],
            split_ratio=split_ratio,
            seed=7,
        )


def test_split_training_rows_is_deterministic_for_seed() -> None:
    rows = [
        {"source_code": "a"},
        {"source_code": "b"},
        {"source_code": "c"},
        {"source_code": "d"},
    ]

    first_train, first_validation = split_training_rows(rows, split_ratio=0.5, seed=11)
    second_train, second_validation = split_training_rows(rows, split_ratio=0.5, seed=11)

    assert first_train == second_train
    assert first_validation == second_validation
    assert len(first_train) == 2
    assert len(first_validation) == 2
    assert {row["source_code"] for row in first_train}.isdisjoint(
        {row["source_code"] for row in first_validation}
    )
    assert sorted(row["source_code"] for row in first_train + first_validation) == [
        "a",
        "b",
        "c",
        "d",
    ]


def test_run_token_channel_train_workflow_orchestrates_training_and_export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, batch_size=2, epochs=2)
    dataset_rows = [
        {"generated_code": "print('a')"},
        {"generated_code": "print('b')"},
        {"generated_code": "print('c')"},
    ]
    training_rows = [
        {"switch_target": 1, "prefix_tokens": [1], "next_token": 2, "teacher_logits": [0.1, 0.9]},
        {"switch_target": 0, "prefix_tokens": [2], "next_token": 1, "teacher_logits": [0.8, 0.2]},
        {"switch_target": 1, "prefix_tokens": [3], "next_token": 0, "teacher_logits": [0.3, 0.7]},
    ]
    cached_rows = [
        {"switch_target": 0, "prefix_tokens": [9], "next_token": 8, "teacher_logits": [0.6, 0.4]},
        {"switch_target": 1, "prefix_tokens": [8], "next_token": 7, "teacher_logits": [0.3, 0.7]},
        {"switch_target": 1, "prefix_tokens": [7], "next_token": 6, "teacher_logits": [0.2, 0.8]},
    ]
    built_batches: list[dict[str, object]] = []
    epochs = [
        TokenChannelEpochMetrics(epoch=1, train_loss=0.8, validation_loss=0.6, switch_loss=0.2),
        TokenChannelEpochMetrics(epoch=2, train_loss=0.5, validation_loss=0.4, switch_loss=0.1),
    ]
    evidence = SimpleNamespace(
        switch_target_positive_count=2,
        switch_target_negative_count=1,
        train_loss=0.5,
        validation_loss=0.4,
        epochs=tuple(epochs),
    )
    artifact_metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 11,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
            "training_config": {"seed": config.seed},
        }
    )
    artifact = SimpleNamespace(metadata=artifact_metadata)

    teacher_model = MagicMock(name="teacher_model")
    token_model = MagicMock(name="token_channel_model")
    optimizer = MagicMock(name="optimizer")
    batch_calls: list[list[dict[str, object]]] = []
    saved_artifacts: dict[str, object] = {}

    monkeypatch.setattr(train_workflow, "load_reference_solutions", lambda dataset, dataset_path: dataset_rows)
    monkeypatch.setattr(train_workflow, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(name_or_path="offline-tokenizer", vocab_size=11)))
    monkeypatch.setattr(train_workflow, "_load_teacher_model", lambda lm_model_path: teacher_model)
    monkeypatch.setattr(train_workflow, "build_training_rows", lambda **kwargs: training_rows)
    monkeypatch.setattr(train_workflow, "save_training_cache", lambda path, rows: Path(path).write_text("cache", encoding="utf-8"))
    monkeypatch.setattr(train_workflow, "load_training_cache", lambda path: cached_rows)

    def fake_build_batch(rows: list[dict[str, object]], *, context_width: int) -> dict[str, object]:
        batch_calls.append(rows)
        batch = {"rows": list(rows), "context_width": context_width}
        built_batches.append(batch)
        return batch

    monkeypatch.setattr(train_workflow, "build_token_channel_batch", fake_build_batch)
    monkeypatch.setattr(train_workflow, "TokenChannelModel", lambda **kwargs: token_model)
    monkeypatch.setattr(train_workflow.torch.optim, "AdamW", lambda params, lr: optimizer)
    epoch_calls: list[dict[str, object]] = []

    def fake_train_one_epoch(**kwargs):
        epoch_calls.append(kwargs)
        return epochs[kwargs["epoch"] - 1]

    monkeypatch.setattr(train_workflow, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(train_workflow, "build_training_evidence", lambda **kwargs: evidence)

    def fake_save_artifacts(*, checkpoint_dir, model, metadata, evidence):
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "model.pt"
        metadata_path = checkpoint_dir / "metadata.json"
        evidence_path = checkpoint_dir / "training_evidence.json"
        checkpoint_path.write_text("model", encoding="utf-8")
        metadata_path.write_text("{}", encoding="utf-8")
        evidence_path.write_text("{}", encoding="utf-8")
        saved_artifacts.update(
            checkpoint_dir=checkpoint_dir,
            model=model,
            metadata=metadata,
            evidence=evidence,
        )
        return {
            "checkpoint_path": checkpoint_path,
            "metadata_path": metadata_path,
            "evidence_path": evidence_path,
        }

    monkeypatch.setattr(train_workflow, "save_token_channel_training_artifacts", fake_save_artifacts)
    monkeypatch.setattr(train_workflow, "load_token_channel_artifact", lambda path: artifact)
    compatibility_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        train_workflow,
        "require_token_channel_compatibility",
        lambda metadata, **kwargs: compatibility_calls.append({"metadata": metadata, **kwargs}),
    )

    summary = run_token_channel_train_workflow(config)

    assert summary == TokenChannelTrainWorkflowSummary(
        dataset="humaneval",
        training_rows=3,
        train_rows=2,
        validation_rows=1,
        artifact_dir=config.model_path,
        cache_path=config.cache_path,
        compatibility_ok=True,
        epochs=tuple(epochs),
        switch_target_positive_count=2,
        switch_target_negative_count=1,
    )
    assert config.cache_path.exists()
    shuffled_rows = list(cached_rows)
    random.Random(config.seed).shuffle(shuffled_rows)
    expected_train_rows = shuffled_rows[:2]
    expected_validation_rows = shuffled_rows[2:]
    assert batch_calls == [expected_train_rows, expected_validation_rows]
    assert [call["train_batches"] for call in epoch_calls] == [[built_batches[0]], [built_batches[0]]]
    assert [call["validation_batches"] for call in epoch_calls] == [[built_batches[1]], [built_batches[1]]]
    assert saved_artifacts["checkpoint_dir"] == config.model_path
    assert saved_artifacts["model"] is token_model
    assert saved_artifacts["evidence"] is evidence
    assert saved_artifacts["metadata"]
    assert (config.model_path / "model.pt").exists()
    assert (config.model_path / "metadata.json").exists()
    assert (config.model_path / "training_evidence.json").exists()
    assert compatibility_calls == [
        {
            "metadata": artifact_metadata,
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 11,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
        }
    ]


def test_run_token_channel_train_workflow_fails_fast_on_empty_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path)
    tokenizer_calls: list[object] = []
    teacher_calls: list[object] = []
    corpus_calls: list[object] = []

    monkeypatch.setattr(train_workflow, "load_reference_solutions", lambda dataset, dataset_path: [])
    monkeypatch.setattr(
        train_workflow,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path: tokenizer_calls.append(path)),
    )
    monkeypatch.setattr(
        train_workflow,
        "_load_teacher_model",
        lambda lm_model_path: teacher_calls.append(lm_model_path),
    )
    monkeypatch.setattr(
        train_workflow,
        "build_training_rows",
        lambda **kwargs: corpus_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="reference solution rows"):
        run_token_channel_train_workflow(config)

    assert tokenizer_calls == []
    assert teacher_calls == []
    assert corpus_calls == []


def test_run_token_channel_train_workflow_rejects_missing_export_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, epochs=1)
    _setup_minimal_workflow_mocks(monkeypatch, config=config)
    monkeypatch.setattr(
        train_workflow,
        "save_token_channel_training_artifacts",
        lambda **kwargs: {"metadata_path": config.model_path / "metadata.json"},
    )

    with pytest.raises(ValueError, match="checkpoint_path"):
        run_token_channel_train_workflow(config)


def test_run_token_channel_train_workflow_rejects_empty_training_corpus(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path)
    tokenizer_calls: list[object] = []

    monkeypatch.setattr(
        train_workflow,
        "load_reference_solutions",
        lambda dataset, dataset_path: [
            {"generated_code": "print('a')"},
            {"generated_code": "print('b')"},
        ],
    )
    monkeypatch.setattr(
        train_workflow,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda path: tokenizer_calls.append(path)
            or SimpleNamespace(name_or_path="offline-tokenizer", vocab_size=17)
        ),
    )
    monkeypatch.setattr(train_workflow, "_load_teacher_model", lambda lm_model_path: MagicMock())
    monkeypatch.setattr(train_workflow, "build_training_rows", lambda **kwargs: [])

    with pytest.raises(ValueError, match="training corpus rows must not be empty"):
        run_token_channel_train_workflow(config)

    assert tokenizer_calls == [config.lm_model_path]


def test_run_token_channel_train_workflow_batches_by_configured_batch_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, batch_size=2, epochs=1, seed=3)
    training_rows = [
        {"switch_target": 1, "prefix_tokens": [1], "next_token": 2, "teacher_logits": [0.1, 0.9]},
        {"switch_target": 0, "prefix_tokens": [2], "next_token": 1, "teacher_logits": [0.8, 0.2]},
        {"switch_target": 1, "prefix_tokens": [3], "next_token": 0, "teacher_logits": [0.3, 0.7]},
        {"switch_target": 0, "prefix_tokens": [4], "next_token": 3, "teacher_logits": [0.4, 0.6]},
        {"switch_target": 1, "prefix_tokens": [5], "next_token": 4, "teacher_logits": [0.5, 0.5]},
    ]
    epoch = TokenChannelEpochMetrics(epoch=1, train_loss=0.8, validation_loss=0.6, switch_loss=0.2)

    monkeypatch.setattr(
        train_workflow,
        "load_reference_solutions",
        lambda dataset, dataset_path: [
            {"generated_code": "print('a')"},
            {"generated_code": "print('b')"},
        ],
    )
    monkeypatch.setattr(
        train_workflow,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(name_or_path="offline-tokenizer", vocab_size=17)),
    )
    monkeypatch.setattr(train_workflow, "_load_teacher_model", lambda lm_model_path: MagicMock())
    monkeypatch.setattr(train_workflow, "build_training_rows", lambda **kwargs: training_rows)
    monkeypatch.setattr(train_workflow, "save_training_cache", lambda path, rows: None)
    monkeypatch.setattr(train_workflow, "load_training_cache", lambda path: training_rows)

    batch_calls: list[list[dict[str, object]]] = []

    def fake_build_batch(rows: list[dict[str, object]], *, context_width: int) -> dict[str, object]:
        batch_calls.append(list(rows))
        return {"rows": list(rows)}

    monkeypatch.setattr(train_workflow, "build_token_channel_batch", fake_build_batch)
    monkeypatch.setattr(train_workflow, "TokenChannelModel", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train_workflow.torch.optim, "AdamW", lambda params, lr: MagicMock())
    epoch_calls: list[dict[str, object]] = []

    def fake_train_one_epoch(**kwargs):
        epoch_calls.append(kwargs)
        return epoch

    monkeypatch.setattr(train_workflow, "train_one_epoch", fake_train_one_epoch)
    monkeypatch.setattr(
        train_workflow,
        "build_training_evidence",
        lambda **kwargs: SimpleNamespace(
            switch_target_positive_count=3,
            switch_target_negative_count=2,
            train_loss=epoch.train_loss,
            validation_loss=epoch.validation_loss,
            epochs=(epoch,),
        ),
    )
    metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 17,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
            "training_config": {},
        }
    )
    monkeypatch.setattr(
        train_workflow,
        "save_token_channel_training_artifacts",
        lambda **kwargs: _write_fake_artifact_outputs(config.model_path),
    )
    monkeypatch.setattr(train_workflow, "load_token_channel_artifact", lambda path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(train_workflow, "require_token_channel_compatibility", lambda metadata, **kwargs: None)

    run_token_channel_train_workflow(config)

    shuffled_rows = list(training_rows)
    random.Random(config.seed).shuffle(shuffled_rows)
    split_index = min(len(shuffled_rows) - 1, max(1, int(len(shuffled_rows) * config.split_ratio)))
    expected_train_rows = shuffled_rows[:split_index]
    expected_validation_rows = shuffled_rows[split_index:]
    assert batch_calls == [expected_train_rows[:2], expected_train_rows[2:], expected_validation_rows]
    assert len(epoch_calls[0]["train_batches"]) == 2
    assert len(epoch_calls[0]["validation_batches"]) == 1


def test_run_token_channel_train_workflow_seeds_python_and_torch_before_training(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, epochs=1, seed=123)
    _setup_minimal_workflow_mocks(monkeypatch, config=config)

    seed_calls: list[tuple[str, int]] = []
    monkeypatch.setattr(train_workflow.random, "seed", lambda value: seed_calls.append(("random", value)))
    monkeypatch.setattr(train_workflow.torch, "manual_seed", lambda value: seed_calls.append(("torch", value)))
    monkeypatch.setattr(train_workflow.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        train_workflow.torch.cuda,
        "manual_seed_all",
        lambda value: seed_calls.append(("cuda", value)),
    )
    monkeypatch.setattr(
        train_workflow,
        "save_token_channel_training_artifacts",
        lambda **kwargs: _write_fake_artifact_outputs(config.model_path),
    )
    metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 17,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
            "training_config": {},
        }
    )
    monkeypatch.setattr(train_workflow, "load_token_channel_artifact", lambda path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(train_workflow, "require_token_channel_compatibility", lambda metadata, **kwargs: None)

    run_token_channel_train_workflow(config)

    assert seed_calls == [
        ("random", config.seed),
        ("torch", config.seed),
        ("cuda", config.seed),
    ]


def test_run_token_channel_train_workflow_saves_required_metadata_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, epochs=1)
    captured_metadata: dict[str, object] = {}
    _setup_minimal_workflow_mocks(monkeypatch, config=config)

    def fake_save_artifacts(*, checkpoint_dir, model, metadata, evidence):
        captured_metadata.update(metadata)
        return _write_fake_artifact_outputs(Path(checkpoint_dir))

    metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 17,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
            "training_config": {},
        }
    )
    monkeypatch.setattr(train_workflow, "save_token_channel_training_artifacts", fake_save_artifacts)
    monkeypatch.setattr(train_workflow, "load_token_channel_artifact", lambda path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(train_workflow, "require_token_channel_compatibility", lambda metadata, **kwargs: None)

    run_token_channel_train_workflow(config)

    assert captured_metadata["schema_version"] == "token-channel/v1"
    assert captured_metadata["tokenizer_name"] == "offline-tokenizer"
    assert captured_metadata["tokenizer_vocab_size"] == 17
    assert captured_metadata["context_width"] == config.context_width
    assert captured_metadata["feature_version"] == "token-channel-features/v1"
    assert captured_metadata["training_config"] == {
        "dataset": config.dataset,
        "dataset_path": str(config.dataset_path),
        "cache_path": str(config.cache_path),
        "hidden_size": config.hidden_size,
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "lr": config.lr,
        "entropy_threshold": config.entropy_threshold,
        "diversity_threshold": config.diversity_threshold,
        "split_ratio": config.split_ratio,
        "seed": config.seed,
        "model_path": str(config.model_path),
    }


def test_run_token_channel_train_workflow_propagates_artifact_load_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, epochs=1)
    _setup_minimal_workflow_mocks(monkeypatch, config=config)
    monkeypatch.setattr(
        train_workflow,
        "save_token_channel_training_artifacts",
        lambda **kwargs: _write_fake_artifact_outputs(config.model_path),
    )
    monkeypatch.setattr(
        train_workflow,
        "load_token_channel_artifact",
        lambda path: (_ for _ in ()).throw(ValueError("artifact bundle is invalid")),
    )

    with pytest.raises(ValueError, match="artifact bundle is invalid"):
        run_token_channel_train_workflow(config)


def test_run_token_channel_train_workflow_propagates_compatibility_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _build_config(tmp_path, epochs=1)
    epoch = TokenChannelEpochMetrics(epoch=1, train_loss=0.8, validation_loss=0.6, switch_loss=0.2)

    monkeypatch.setattr(
        train_workflow,
        "load_reference_solutions",
        lambda dataset, dataset_path: [
            {"generated_code": "print('a')"},
            {"generated_code": "print('b')"},
        ],
    )
    monkeypatch.setattr(train_workflow, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda path: SimpleNamespace(name_or_path="offline-tokenizer", vocab_size=17)))
    monkeypatch.setattr(train_workflow, "_load_teacher_model", lambda lm_model_path: MagicMock())
    monkeypatch.setattr(
        train_workflow,
        "build_training_rows",
        lambda **kwargs: [
            {"switch_target": 1, "prefix_tokens": [1], "next_token": 2, "teacher_logits": [0.1, 0.9]},
            {"switch_target": 0, "prefix_tokens": [2], "next_token": 1, "teacher_logits": [0.8, 0.2]},
        ],
    )
    monkeypatch.setattr(train_workflow, "save_training_cache", lambda path, rows: None)
    monkeypatch.setattr(
        train_workflow,
        "load_training_cache",
        lambda path: [
            {"switch_target": 1, "prefix_tokens": [1], "next_token": 2, "teacher_logits": [0.1, 0.9]},
            {"switch_target": 0, "prefix_tokens": [2], "next_token": 1, "teacher_logits": [0.8, 0.2]},
        ],
    )
    monkeypatch.setattr(train_workflow, "build_token_channel_batch", lambda rows, *, context_width: {"rows": rows})
    monkeypatch.setattr(train_workflow, "TokenChannelModel", lambda **kwargs: MagicMock())
    monkeypatch.setattr(train_workflow.torch.optim, "AdamW", lambda params, lr: MagicMock())
    monkeypatch.setattr(train_workflow, "train_one_epoch", lambda **kwargs: epoch)
    monkeypatch.setattr(
        train_workflow,
        "build_training_evidence",
        lambda **kwargs: SimpleNamespace(
            switch_target_positive_count=1,
            switch_target_negative_count=1,
            train_loss=epoch.train_loss,
            validation_loss=epoch.validation_loss,
            epochs=(epoch,),
        ),
    )
    metadata = TokenChannelArtifactMetadata.from_mapping(
        {
            "schema_version": "token-channel/v1",
            "tokenizer_name": "offline-tokenizer",
            "tokenizer_vocab_size": 17,
            "context_width": config.context_width,
            "feature_version": "token-channel-features/v1",
            "training_config": {},
        }
    )
    monkeypatch.setattr(
        train_workflow,
        "save_token_channel_training_artifacts",
        lambda **kwargs: _write_fake_artifact_outputs(config.model_path),
    )
    monkeypatch.setattr(train_workflow, "load_token_channel_artifact", lambda path: SimpleNamespace(metadata=metadata))
    monkeypatch.setattr(
        train_workflow,
        "require_token_channel_compatibility",
        lambda metadata, **kwargs: (_ for _ in ()).throw(ValueError("Incompatible token-channel artifact: tokenizer_name mismatch")),
    )

    with pytest.raises(ValueError, match="Incompatible token-channel artifact"):
        run_token_channel_train_workflow(config)


def _write_fake_artifact_outputs(artifact_dir: Path) -> dict[str, Path]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "model.pt"
    metadata_path = artifact_dir / "metadata.json"
    evidence_path = artifact_dir / "training_evidence.json"
    checkpoint_path.write_text("model", encoding="utf-8")
    metadata_path.write_text("{}", encoding="utf-8")
    evidence_path.write_text("{}", encoding="utf-8")
    return {
        "checkpoint_path": checkpoint_path,
        "metadata_path": metadata_path,
        "evidence_path": evidence_path,
    }
