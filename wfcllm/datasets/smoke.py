from __future__ import annotations

import os


def _positive_int_from_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def smoke_sample_count_from_env() -> int | None:
    return _positive_int_from_env("WFCLLM_SMOKE_SAMPLE_COUNT")


def smoke_gate_train_epochs_from_env() -> int | None:
    return _positive_int_from_env("WFCLLM_SMOKE_GATE_TRAIN_EPOCHS")


def smoke_encoder_train_epochs_from_env() -> int | None:
    return _positive_int_from_env("WFCLLM_SMOKE_ENCODER_TRAIN_EPOCHS")


def smoke_gate_data_groups_from_env() -> int | None:
    return _positive_int_from_env("WFCLLM_SMOKE_GATE_DATA_GROUPS")
