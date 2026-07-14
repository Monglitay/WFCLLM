from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "wfcllm-batch-invariant-semantic-watermark/v4"


@dataclass(frozen=True)
class CanonicalConfig:
    schema_version: str
    max_unit_bytes: int
    max_context_bytes: int
    global_ordinal_keying: bool
    no_truncation: bool


@dataclass(frozen=True)
class ChannelConfig:
    bit_count_per_unit: int
    minimum_independent_units: int
    quantization: str
    margin_rule: str
    ecc: str


@dataclass(frozen=True)
class RuntimeConfig:
    encoder_used: bool
    neural_auxiliary_used: bool
    device: str
    cuda_required: bool


@dataclass(frozen=True)
class PublicConfig:
    schema_version: str
    canonical_context: CanonicalConfig
    channel: ChannelConfig
    runtime: RuntimeConfig
    raw: dict[str, Any]


_TOP_LEVEL_FIELDS = {
    "artifact_type",
    "cache",
    "canonical_context",
    "channel",
    "decision",
    "detector",
    "runtime",
    "schema_version",
    "secret_metadata_included",
    "selection",
}
_FORBIDDEN = {"secret_key", "raw_key", "key_fingerprint", "key_sha256"}


def _reject_forbidden(value: Any) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if str(key).lower() in _FORBIDDEN:
                raise ValueError("public config contains forbidden secret metadata")
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def load_public_config(path: str | Path) -> PublicConfig:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to load V4 public config: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError("public config must be a JSON object")
    _reject_forbidden(payload)
    if set(payload) != _TOP_LEVEL_FIELDS:
        raise ValueError("public config has unknown or missing fields")
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("public config schema version mismatch")
    if payload["secret_metadata_included"] is not False:
        raise ValueError("public config secret metadata flag must be false")
    try:
        canonical = payload["canonical_context"]
        channel = payload["channel"]
        runtime = payload["runtime"]
        result = PublicConfig(
            schema_version=payload["schema_version"],
            canonical_context=CanonicalConfig(
                schema_version=canonical["schema_version"],
                max_unit_bytes=int(canonical["max_unit_bytes"]),
                max_context_bytes=int(canonical["max_context_bytes"]),
                global_ordinal_keying=canonical["global_ordinal_keying"],
                no_truncation=canonical["no_truncation"],
            ),
            channel=ChannelConfig(
                bit_count_per_unit=int(channel["bit_count_per_unit"]),
                minimum_independent_units=int(channel["minimum_independent_units"]),
                quantization=channel["quantization"],
                margin_rule=channel["margin_rule"],
                ecc=channel["ecc"],
            ),
            runtime=RuntimeConfig(
                encoder_used=runtime["encoder_used"],
                neural_auxiliary_used=runtime["neural_auxiliary_used"],
                device=runtime["device"],
                cuda_required=runtime["cuda_required"],
            ),
            raw=payload,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("invalid V4 public config") from exc
    if result.canonical_context.global_ordinal_keying is not False:
        raise ValueError("global ordinal keying must be false")
    if result.runtime.encoder_used or result.runtime.neural_auxiliary_used:
        raise ValueError("formal V4 public config must be neural-free")
    return result
