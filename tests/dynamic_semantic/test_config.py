from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wfcllm.dynamic_semantic.config import (
    CONTEXT_SCHEMA_VERSION,
    METHOD_SCHEMA_VERSION,
    ChannelConfig,
    ContextConfig,
    DynamicSemanticConfig,
    EncoderConfig,
    SchedulerConfig,
    load_public_config,
)


def _payload() -> dict[str, object]:
    return {
        "schema_version": METHOD_SCHEMA_VERSION,
        "context": {
            "schema_version": CONTEXT_SCHEMA_VERSION,
            "max_context_tokens": 256,
            "max_current_unit_tokens": 128,
            "global_ordinal_keying": False,
        },
        "encoder": {
            "model_path": "data/models/codet5-base",
            "checkpoint_path": "data/models/encoder/best_model.pt",
            "checkpoint_sha256": "a" * 64,
            "official_precision": "float32",
            "embedding_dimensions": 128,
        },
        "channel": {
            "whitening_dimensions": 64,
            "quantization_scale": 4096,
            "projection_rows": 7,
            "target_data_bits": 4,
            "minimum_independent_units": 3,
        },
        "scheduler": {
            "target_batch_contexts": 32,
            "max_batch_contexts": 128,
            "max_queue_completed_attempts": 4,
        },
    }


def test_load_public_config_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "public.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")

    config = load_public_config(path)

    assert config.schema_version == "wfcllm-dynamic-semantic-code-watermark/v3"
    assert config.context.schema_version == "wfcllm-dynamic-semantic-context/v3"
    assert config.encoder.official_precision == "float32"
    assert config.channel.projection_rows == 7
    assert config.scheduler.target_batch_contexts == 32
    assert config.to_dict() == _payload()


def test_config_records_are_frozen() -> None:
    config = DynamicSemanticConfig.from_dict(_payload())

    with pytest.raises(FrozenInstanceError):
        config.context.max_context_tokens = 1  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("schema_version", "v2", "schema_version"),
        ("private_key", "forbidden", "unknown"),
    ],
)
def test_public_config_rejects_wrong_schema_or_private_key(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _payload()
    payload[field] = value

    with pytest.raises(ValueError, match=message):
        DynamicSemanticConfig.from_dict(payload)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": "v2"}, "schema_version"),
        ({"max_context_tokens": 0}, "max_context_tokens"),
        ({"max_current_unit_tokens": 257}, "max_current_unit_tokens"),
        ({"global_ordinal_keying": True}, "global_ordinal_keying"),
    ],
)
def test_context_config_enforces_v3_contract(
    overrides: dict[str, object],
    message: str,
) -> None:
    values: dict[str, object] = {
        "schema_version": CONTEXT_SCHEMA_VERSION,
        "max_context_tokens": 256,
        "max_current_unit_tokens": 128,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        ContextConfig(**values)


@pytest.mark.parametrize("precision", ["bf16", "fp16", "auto"])
def test_encoder_config_rejects_nonofficial_precision(precision: str) -> None:
    with pytest.raises(ValueError, match="float32"):
        EncoderConfig(
            model_path="model",
            checkpoint_path="checkpoint.pt",
            checkpoint_sha256="b" * 64,
            official_precision=precision,
            embedding_dimensions=128,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"whitening_dimensions": 0}, "whitening_dimensions"),
        ({"quantization_scale": 0}, "quantization_scale"),
        ({"projection_rows": 8}, "projection_rows"),
        ({"target_data_bits": 5}, "target_data_bits"),
        ({"minimum_independent_units": 0}, "minimum_independent_units"),
    ],
)
def test_channel_config_enforces_hamming_7_4(
    overrides: dict[str, int],
    message: str,
) -> None:
    values = {
        "whitening_dimensions": 64,
        "quantization_scale": 4096,
        "projection_rows": 7,
        "target_data_bits": 4,
        "minimum_independent_units": 3,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        ChannelConfig(**values)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"target_batch_contexts": 0}, "target_batch_contexts"),
        ({"target_batch_contexts": 129}, "target_batch_contexts"),
        ({"max_queue_completed_attempts": 0}, "max_queue_completed_attempts"),
    ],
)
def test_scheduler_config_enforces_bounds(
    overrides: dict[str, int],
    message: str,
) -> None:
    values = {
        "target_batch_contexts": 32,
        "max_batch_contexts": 128,
        "max_queue_completed_attempts": 4,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        SchedulerConfig(**values)
