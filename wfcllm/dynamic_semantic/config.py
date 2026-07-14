from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar


METHOD_SCHEMA_VERSION = "wfcllm-dynamic-semantic-code-watermark/v3"
CONTEXT_SCHEMA_VERSION = "wfcllm-dynamic-semantic-context/v3"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_Record = TypeVar("_Record")


def _require_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _construct_strict(record_type: type[_Record], payload: object) -> _Record:
    if not isinstance(payload, dict):
        raise ValueError(f"{record_type.__name__} must be an object")
    allowed = {item.name for item in fields(record_type)}
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {record_type.__name__} fields: {unknown}"
        )
    try:
        return record_type(**payload)
    except TypeError as exc:
        raise ValueError(f"invalid {record_type.__name__}: {exc}") from exc


@dataclass(frozen=True)
class ContextConfig:
    schema_version: str = CONTEXT_SCHEMA_VERSION
    max_context_tokens: int = 256
    max_current_unit_tokens: int = 128
    global_ordinal_keying: bool = False

    def __post_init__(self) -> None:
        if self.schema_version != CONTEXT_SCHEMA_VERSION:
            raise ValueError(
                f"context schema_version must be {CONTEXT_SCHEMA_VERSION}"
            )
        context_budget = _require_positive_int(
            self.max_context_tokens,
            "max_context_tokens",
        )
        unit_budget = _require_positive_int(
            self.max_current_unit_tokens,
            "max_current_unit_tokens",
        )
        if unit_budget > context_budget:
            raise ValueError(
                "max_current_unit_tokens must not exceed max_context_tokens"
            )
        if self.global_ordinal_keying is not False:
            raise ValueError("global_ordinal_keying must be false in V3")


@dataclass(frozen=True)
class EncoderConfig:
    model_path: str
    checkpoint_path: str
    checkpoint_sha256: str
    official_precision: str = "float32"
    embedding_dimensions: int = 256

    def __post_init__(self) -> None:
        if not isinstance(self.model_path, str) or not self.model_path:
            raise ValueError("model_path must be a non-empty string")
        if not isinstance(self.checkpoint_path, str) or not self.checkpoint_path:
            raise ValueError("checkpoint_path must be a non-empty string")
        if not isinstance(self.checkpoint_sha256, str) or not _SHA256_PATTERN.fullmatch(
            self.checkpoint_sha256
        ):
            raise ValueError("checkpoint_sha256 must be 64 lowercase hex characters")
        if self.official_precision != "float32":
            raise ValueError("official_precision must be float32")
        _require_positive_int(self.embedding_dimensions, "embedding_dimensions")


@dataclass(frozen=True)
class ChannelConfig:
    whitening_dimensions: int = 64
    quantization_scale: int = 4096
    projection_rows: int = 7
    target_data_bits: int = 4
    minimum_independent_units: int = 3

    def __post_init__(self) -> None:
        _require_positive_int(self.whitening_dimensions, "whitening_dimensions")
        _require_positive_int(self.quantization_scale, "quantization_scale")
        if self.projection_rows != 7:
            raise ValueError("projection_rows must be 7 for Hamming(7,4)")
        if self.target_data_bits != 4:
            raise ValueError("target_data_bits must be 4 for Hamming(7,4)")
        _require_positive_int(
            self.minimum_independent_units,
            "minimum_independent_units",
        )


@dataclass(frozen=True)
class SchedulerConfig:
    target_batch_contexts: int = 32
    max_batch_contexts: int = 128
    max_queue_completed_attempts: int = 4

    def __post_init__(self) -> None:
        target = _require_positive_int(
            self.target_batch_contexts,
            "target_batch_contexts",
        )
        maximum = _require_positive_int(
            self.max_batch_contexts,
            "max_batch_contexts",
        )
        if target > maximum:
            raise ValueError(
                "target_batch_contexts must not exceed max_batch_contexts"
            )
        _require_positive_int(
            self.max_queue_completed_attempts,
            "max_queue_completed_attempts",
        )


@dataclass(frozen=True)
class DynamicSemanticConfig:
    schema_version: str
    context: ContextConfig
    encoder: EncoderConfig
    channel: ChannelConfig
    scheduler: SchedulerConfig

    def __post_init__(self) -> None:
        if self.schema_version != METHOD_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {METHOD_SCHEMA_VERSION}")
        expected = (
            (self.context, ContextConfig, "context"),
            (self.encoder, EncoderConfig, "encoder"),
            (self.channel, ChannelConfig, "channel"),
            (self.scheduler, SchedulerConfig, "scheduler"),
        )
        for value, expected_type, name in expected:
            if not isinstance(value, expected_type):
                raise ValueError(f"{name} must be {expected_type.__name__}")

    @classmethod
    def from_dict(cls, payload: object) -> DynamicSemanticConfig:
        if not isinstance(payload, dict):
            raise ValueError("dynamic semantic public config must be an object")
        allowed = {"schema_version", "context", "encoder", "channel", "scheduler"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise ValueError(f"unknown DynamicSemanticConfig fields: {unknown}")
        missing = sorted(allowed - set(payload))
        if missing:
            raise ValueError(f"missing DynamicSemanticConfig fields: {missing}")
        return cls(
            schema_version=str(payload["schema_version"]),
            context=_construct_strict(ContextConfig, payload["context"]),
            encoder=_construct_strict(EncoderConfig, payload["encoder"]),
            channel=_construct_strict(ChannelConfig, payload["channel"]),
            scheduler=_construct_strict(SchedulerConfig, payload["scheduler"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_public_config(path: str | Path) -> DynamicSemanticConfig:
    config_path = Path(path)
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load dynamic semantic config: {config_path}") from exc
    return DynamicSemanticConfig.from_dict(payload)
