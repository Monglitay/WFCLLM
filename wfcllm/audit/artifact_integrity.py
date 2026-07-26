from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from wfcllm.audit.forbidden import (
    POSTHOC_PASS_REQUIRED_FLAGS,
    artifact_field_tokens_and_compact,
    canonical_artifact_field_name,
    is_forbidden_formal_quality_proxy_field,
    is_forbidden_secret_field,
    is_unsafe_checkpoint_state_field,
)

_MAX_ARTIFACT_DEPTH = 64
_MAX_ARTIFACT_NODES = 100_000
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_SCALAR_BYTES = 1024 * 1024
_MAX_FIELD_NAME_BYTES = 256
_MAX_RENDERED_PATH_CHARS = 1024
_MAX_INTEGER_BITS = 4096
_MAX_TENSOR_RANK = 16
_MAX_TENSOR_DIMENSION = 2**31 - 1
_MAX_TENSOR_NUMEL = 2**31 - 1
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_Path = tuple[str | int, ...]
_FORMAL_ARTIFACT_TYPES = frozenset(
    {
        "gate-data-jsonl",
        "training-metrics",
        "checkpoint-metadata",
        "bundle-manifest",
        "validation-summary",
        "generation-window-audit",
        "gated-calibration",
        "gated-detection-details",
        # The currently shipped detector uses this spelling.
        "wfcllm_detection_calibration",
    }
)
_FORMAL_ARTIFACT_TYPE_PREFIXES = tuple(
    artifact_field_tokens_and_compact(value)[1]
    for value in _FORMAL_ARTIFACT_TYPES
)
_DIAGNOSTIC_ARTIFACT_TYPES = frozenset(
    {
        "diagnostic-artifact",
        "diagnostic-report",
        "diagnostic-selector",
    }
)
_DIAGNOSTIC_ARTIFACT_TYPE_COMPACTS = frozenset(
    artifact_field_tokens_and_compact(value)[1]
    for value in _DIAGNOSTIC_ARTIFACT_TYPES
)
_FORMAL_VERSION_IDENTITIES = frozenset(
    {
        # Task 9: trainer output contracts.
        "wfcllm-gate-training-checkpoint/v1",
        "wfcllm-gate-training-metrics/v1",
        "wfcllm-gate-development-summary/v1",
        # Task 10: immutable bundle and validation contracts.
        "wfcllm-gate-bundle/v1",
        "wfcllm-gate-model-state/v1",
        "wfcllm-gate-validation/v1",
        "wfcllm-gate-input/v1",
        "python-statement-window/v1",
        # Task 12: data, candidate, and publication contracts.
        "gate-data-feasibility/v1",
        "wfcllm-gate-data/v1",
        "wfcllm-gate-data-manifest/v1",
        "wfcllm-gate-source-manifest/v1",
        "wfcllm-gate-split/v1",
        "wfcllm-training-key-bank-manifest/v1",
        "wfcllm-gate-train-candidate/v1",
        "wfcllm-gate-candidate-attempts/v2",
        "wfcllm-gate-label/v1",
        "wfcllm-production-gate-adapter/v1",
        # Task 14 audits the existing detector calibration family too.
        "wfcllm-detect-calibration/v1",
    }
)
_FORMAL_VERSION_COMPACTS = frozenset(
    artifact_field_tokens_and_compact(value)[1]
    for value in _FORMAL_VERSION_IDENTITIES
)
_FORMAL_VERSION_PREFIXES = (
    "gatedatafeasibility",
    "pythonstatementwindow",
    "wfcllmdetect",
    "wfcllmdetection",
    "wfcllmgate",
    "wfcllmproductiongateadapter",
    "wfcllmtrainingkeybank",
    "wfcllmwindow",
)
_CHECKPOINT_RUNTIME_FIELD_NAMES = frozenset(
    {
        "buffer",
        "coefficient",
        "model",
        "model_payload",
        "moment",
        "optim",
        "optimizer",
        "parameter",
        "params",
        "payload",
        "scheduler",
        "state",
        "state_payload",
        "value",
        "weight",
    }
)
_CHECKPOINT_RUNTIME_COMPACTS = frozenset(
    artifact_field_tokens_and_compact(name)[1]
    for name in _CHECKPOINT_RUNTIME_FIELD_NAMES
)
_CHECKPOINT_RUNTIME_TOKENS = frozenset(
    {
        "buffer",
        "coefficient",
        "model",
        "moment",
        "optim",
        "optimizer",
        "parameter",
        "payload",
        "pickle",
        "rng",
        "scheduler",
        "state",
        "storage",
        "tensor",
        "value",
        "weight",
    }
)
_CHECKPOINT_SAFE_DESCRIPTOR_TOKENS = frozenset(
    {
        "count",
        "dtype",
        "epoch",
        "format",
        "hash",
        "id",
        "name",
        "numel",
        "present",
        "sha256",
        "shape",
        "status",
        "step",
        "steps",
        "version",
    }
)
_CHECKPOINT_ALLOWED_FIELDS = frozenset(
    {
        "artifact_type",
        "base_model_id",
        "best",
        "best_epoch",
        "checkpoint_manifest",
        "checkpoint_metadata",
        "comparable_group_count",
        "contract_version",
        "coverage",
        "decision_consistency",
        "diagnostic_only",
        "diagnostic_test_backend",
        "dtype",
        "early_stopped",
        "epoch",
        "epoch_status",
        "epochs_completed",
        "evaluable_count",
        "formal_eligible",
        "manifest",
        "metadata",
        "model_architecture",
        "model_format_version",
        "name",
        "negative_count",
        "not_official_method",
        "numel",
        "optimizer_steps",
        "overflow_count",
        "parameter_count",
        "patience",
        "schema_version",
        "shape",
        "status",
        "suitable_false_positive_rate",
        "tensor_count",
        "tensors",
        "torch_version",
        "total_count",
        "validated",
        "validation",
    }
)
_CHECKPOINT_ALLOWED_COMPACTS = frozenset(
    artifact_field_tokens_and_compact(field)[1]
    for field in _CHECKPOINT_ALLOWED_FIELDS
)
_TENSOR_METADATA_DTYPES = frozenset(
    {
        "bfloat16",
        "bool",
        "float16",
        "float32",
        "float64",
        "int8",
        "int16",
        "int32",
        "int64",
        "qint8",
        "quint8",
        "torch.bfloat16",
        "torch.bool",
        "torch.float16",
        "torch.float32",
        "torch.float64",
        "torch.int8",
        "torch.int16",
        "torch.int32",
        "torch.int64",
        "torch.qint8",
        "torch.quint8",
        "torch.uint8",
        "uint8",
    }
)


@dataclass
class _ArtifactBudget:
    nodes: int = 0
    bytes_seen: int = 0


def assert_audit_only_marker(payload: dict[str, Any]) -> None:
    if payload.get("audit_only") is not True:
        raise ValueError("audit_only must be true")
    if payload.get("detector_input_allowed") is not False:
        raise ValueError("detector_input_allowed must be false")


def assert_posthoc_pass_report_marker(payload: dict[str, Any]) -> None:
    for field in sorted(POSTHOC_PASS_REQUIRED_FLAGS):
        if payload.get(field) is not True:
            raise ValueError(f"{field} must be true")


def reject_secret_key_leak(payload: Any) -> None:
    if not isinstance(payload, (dict, list)):
        raise ValueError("public artifact root must be a JSON object or array")
    _walk_gate_artifact(
        payload,
        path=(),
        depth=0,
        active_containers=set(),
        budget=_ArtifactBudget(),
        reject_secrets=True,
        reject_quality=False,
        feasibility_metadata=False,
        checkpoint_metadata_only=False,
        enforce_checkpoint_schema=False,
    )


def audit_gate_artifact(payload: Any) -> None:
    """Audit one already-decoded public gate artifact without deserializing state.

    Only JSON-safe in-memory values are accepted.  In particular this function
    never opens a checkpoint and never invokes pickle; callers must extract and
    pass safe checkpoint metadata at the point where the checkpoint is written.
    """

    if not isinstance(payload, (dict, list)):
        raise ValueError("gate artifact root must be a JSON object or array")
    diagnostic = _diagnostic_quality_exemption(payload)
    _walk_gate_artifact(
        payload,
        path=(),
        depth=0,
        active_containers=set(),
        budget=_ArtifactBudget(),
        reject_secrets=True,
        reject_quality=not diagnostic,
        feasibility_metadata=_is_feasibility_metadata_artifact(payload),
        checkpoint_metadata_only=_is_checkpoint_metadata_artifact(payload),
        enforce_checkpoint_schema=True,
    )


def reject_formal_quality_proxy_fields(payload: Any) -> None:
    """Apply the gate artifact's normalized quality-proxy policy."""

    if not isinstance(payload, (dict, list)):
        raise ValueError("gate artifact root must be a JSON object or array")
    diagnostic = _diagnostic_quality_exemption(payload)
    _walk_gate_artifact(
        payload,
        path=(),
        depth=0,
        active_containers=set(),
        budget=_ArtifactBudget(),
        reject_secrets=False,
        reject_quality=not diagnostic,
        feasibility_metadata=_is_feasibility_metadata_artifact(payload),
        checkpoint_metadata_only=_is_checkpoint_metadata_artifact(payload),
        enforce_checkpoint_schema=True,
    )


def _walk_gate_artifact(
    value: Any,
    *,
    path: _Path,
    depth: int,
    active_containers: set[int],
    budget: _ArtifactBudget,
    reject_secrets: bool,
    reject_quality: bool,
    feasibility_metadata: bool,
    checkpoint_metadata_only: bool,
    enforce_checkpoint_schema: bool,
) -> None:
    location = _render_path(path)
    if depth > _MAX_ARTIFACT_DEPTH:
        raise ValueError(f"gate artifact nesting depth limit exceeded at {location}")
    budget.nodes += 1
    if budget.nodes > _MAX_ARTIFACT_NODES:
        raise ValueError(f"gate artifact node limit exceeded at {location}")

    if isinstance(value, dict):
        if enforce_checkpoint_schema and _is_checkpoint_metadata_artifact(value):
            checkpoint_metadata_only = True
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{location} contains a cyclic container")
        active_containers.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ValueError(f"{location} must use string field names")
                child_path = (*path, key)
                _consume_field_name(key, path=child_path, budget=budget)
                if reject_secrets and is_forbidden_secret_field(key):
                    raise ValueError(
                        "secret material is forbidden in public gate artifact: "
                        f"{_render_path(child_path)}"
                    )
                feasibility_status = feasibility_metadata and (
                    child_path == ("passed",)
                    or (
                        len(child_path) == 3
                        and child_path[0] == "admissions"
                        and child_path[2] == "passed"
                    )
                )
                if (
                    reject_quality
                    and not feasibility_status
                    and is_forbidden_formal_quality_proxy_field(key)
                ):
                    raise ValueError(
                        "formal quality proxy field is forbidden: "
                        f"{_render_path(child_path)}"
                    )
                checkpoint_named = (
                    enforce_checkpoint_schema and _opens_checkpoint_context(key)
                )
                opens_checkpoint_context = checkpoint_named and isinstance(child, dict)
                _, checkpoint_compact = artifact_field_tokens_and_compact(key)
                if (
                    checkpoint_named
                    and not isinstance(child, dict)
                    and checkpoint_compact
                    in {"checkpoint", "checkpointmanifest", "checkpointmetadata"}
                ):
                    raise ValueError(
                        f"{_render_path(child_path)} must be a checkpoint metadata object"
                    )
                if enforce_checkpoint_schema and checkpoint_metadata_only:
                    _audit_checkpoint_metadata_field(
                        key,
                        child,
                        path=child_path,
                    )
                _walk_gate_artifact(
                    child,
                    path=child_path,
                    depth=depth + 1,
                    active_containers=active_containers,
                    budget=budget,
                    reject_secrets=reject_secrets,
                    reject_quality=reject_quality,
                    feasibility_metadata=feasibility_metadata,
                    checkpoint_metadata_only=(
                        checkpoint_metadata_only or opens_checkpoint_context
                    ),
                    enforce_checkpoint_schema=enforce_checkpoint_schema,
                )
        finally:
            active_containers.remove(identity)
        return

    if isinstance(value, list):
        identity = id(value)
        if identity in active_containers:
            raise ValueError(f"{location} contains a cyclic container")
        active_containers.add(identity)
        try:
            for index, child in enumerate(value):
                child_path = (*path, index)
                _walk_gate_artifact(
                    child,
                    path=child_path,
                    depth=depth + 1,
                    active_containers=active_containers,
                    budget=budget,
                    reject_secrets=reject_secrets,
                    reject_quality=reject_quality,
                    feasibility_metadata=feasibility_metadata,
                    checkpoint_metadata_only=checkpoint_metadata_only,
                    enforce_checkpoint_schema=enforce_checkpoint_schema,
                )
        finally:
            active_containers.remove(identity)
        return

    if value is None or type(value) is bool:
        _consume_bytes(1, path=path, budget=budget)
        return
    if type(value) is int:
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise ValueError(f"integer size limit exceeded at {location}")
        _consume_bytes(
            max(1, (value.bit_length() + 7) // 8),
            path=path,
            budget=budget,
        )
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError(f"gate artifact contains non-finite number at {location}")
        _consume_bytes(8, path=path, budget=budget)
        return
    if isinstance(value, str):
        _consume_text(value, path=path, budget=budget)
        return
    raise ValueError(f"{location} must contain only JSON-safe metadata")


def _diagnostic_quality_exemption(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    backend_fields = {"diagnostic_test_backend", "formal_eligible"}
    diagnostic_fields = {"diagnostic_only", "not_official_method"}
    present_backend = backend_fields & set(payload)
    present_diagnostic = diagnostic_fields & set(payload)
    if not present_backend and not present_diagnostic:
        return False
    formal_values = {
        "diagnostic_test_backend": False,
        "formal_eligible": True,
        "diagnostic_only": False,
        "not_official_method": False,
    }
    present_identity = present_backend | present_diagnostic
    if all(payload[field] is formal_values[field] for field in present_identity):
        return False
    if present_backend and present_diagnostic:
        raise ValueError("diagnostic identity cannot mix marker families")
    if present_backend:
        if (
            present_backend == backend_fields
            and payload["diagnostic_test_backend"] is False
            and payload["formal_eligible"] is True
        ):
            return False
        if (
            present_backend != backend_fields
            or payload["diagnostic_test_backend"] is not True
            or payload["formal_eligible"] is not False
        ):
            raise ValueError("diagnostic test backend identity is inconsistent")
    else:
        if (
            present_diagnostic != diagnostic_fields
            or payload["diagnostic_only"] is not True
            or payload["not_official_method"] is not True
        ):
            raise ValueError("diagnostic-only identity is inconsistent")
    if (
        payload.get("validated") is True
        or payload.get("formal_bundle") is True
        or payload.get("official_method") is True
        or payload.get("detector_input_allowed") is True
    ):
        raise ValueError("diagnostic identity contradicts formal artifact markers")
    _reject_diagnostic_formal_identity(payload)
    return True


def _reject_diagnostic_formal_identity(payload: dict[str, Any]) -> None:
    artifact_type = payload.get("artifact_type")
    if artifact_type is not None:
        if not isinstance(artifact_type, str):
            raise ValueError("diagnostic identity has invalid artifact_type")
        tokens, compact = artifact_field_tokens_and_compact(artifact_type)
        if compact in _DIAGNOSTIC_ARTIFACT_TYPE_COMPACTS:
            pass
        elif (
            "formal" in tokens
            or "formal" in compact
            or compact.startswith("wfcllm")
            or any(
                compact.startswith(prefix)
                for prefix in _FORMAL_ARTIFACT_TYPE_PREFIXES
            )
        ):
            raise ValueError(
                "diagnostic identity contradicts formal artifact_type"
            )
        else:
            raise ValueError(
                "diagnostic identity has unsupported artifact_type; only explicit "
                "diagnostic artifact types are allowed"
            )

    for marker_name in ("schema_version", "contract_version"):
        marker = payload.get(marker_name)
        if marker is None:
            continue
        if not isinstance(marker, str):
            raise ValueError(f"diagnostic identity has invalid {marker_name}")
        tokens, compact = artifact_field_tokens_and_compact(marker)
        if (
            compact in _FORMAL_VERSION_COMPACTS
            or "formal" in tokens
            or "formal" in compact
            or any(compact.startswith(prefix) for prefix in _FORMAL_VERSION_PREFIXES)
        ):
            raise ValueError(
                f"diagnostic identity contradicts formal {marker_name}"
            )


def _is_feasibility_metadata_artifact(payload: Any) -> bool:
    return (
        isinstance(payload, dict)
        and payload.get("contract_version") == "gate-data-feasibility/v1"
    )


def _is_checkpoint_metadata_artifact(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    for marker_name in ("artifact_type", "contract_version"):
        marker = payload.get(marker_name)
        if not isinstance(marker, str):
            continue
        tokens, compact = artifact_field_tokens_and_compact(marker)
        if "checkpoint" in tokens or "checkpoint" in compact:
            return True
    return False


def _opens_checkpoint_context(field: str) -> bool:
    tokens, compact = artifact_field_tokens_and_compact(field)
    return "checkpoint" in tokens or "checkpoint" in compact


def _audit_checkpoint_metadata_field(
    field: str,
    value: Any,
    *,
    path: _Path,
) -> None:
    canonical = canonical_artifact_field_name(field)
    tokens, compact = artifact_field_tokens_and_compact(field)
    if field == "tensors":
        _validate_tensor_metadata_list(value, path=path)
        return
    if _is_checkpoint_runtime_payload_field(field, tokens=tokens, compact=compact):
        raise ValueError(
            f"{_render_path(path)} is forbidden; checkpoint audit accepts "
            "safe metadata only"
        )
    is_digest = canonical.endswith("sha256") or canonical.endswith("hash")
    if compact not in _CHECKPOINT_ALLOWED_COMPACTS and not is_digest:
        raise ValueError(
            f"{_render_path(path)} is not allowed by checkpoint metadata schema"
        )
    if tokens and tokens[-1] == "shape":
        _validate_shape(value, path=path)
        return
    if tokens and tokens[-1] == "dtype":
        if not isinstance(value, str) or value not in _TENSOR_METADATA_DTYPES:
            raise ValueError(f"{_render_path(path)} must be an allowlisted dtype")
        return
    if is_digest:
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ValueError(
                f"{_render_path(path)} must be a lowercase SHA-256 digest"
            )
        return
    if isinstance(value, list):
        if not value or all(type(item) in (int, float, bool) for item in value):
            raise ValueError(
                f"{_render_path(path)} contains a forbidden checkpoint numeric array"
            )
        raise ValueError(
            f"{_render_path(path)} is forbidden; checkpoint metadata lists "
            "must be shape or tensors"
        )


def _is_checkpoint_runtime_payload_field(
    field: str,
    *,
    tokens: tuple[str, ...],
    compact: str,
) -> bool:
    if is_unsafe_checkpoint_state_field(field) or (
        compact in _CHECKPOINT_RUNTIME_COMPACTS
    ):
        return True
    runtime_tokens = set(tokens) & _CHECKPOINT_RUNTIME_TOKENS
    if not runtime_tokens:
        return False
    if tokens and tokens[-1] in _CHECKPOINT_SAFE_DESCRIPTOR_TOKENS:
        return False
    return True


def _validate_tensor_metadata_list(value: Any, *, path: _Path) -> None:
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"{_render_path(path)} must be a non-empty tensor metadata list"
        )
    required = {"name", "shape", "dtype"}
    allowed = required | {"numel", "sha256"}
    for index, item in enumerate(value):
        item_path = (*path, index)
        if type(item) is not dict or not required <= set(item) or set(item) - allowed:
            raise ValueError(
                f"{_render_path(item_path)} tensor metadata schema mismatch"
            )
        name = item["name"]
        if not isinstance(name, str) or not name:
            raise ValueError(
                f"{_render_path((*item_path, 'name'))} must be a non-empty string"
            )
        expected_numel = _validate_shape(
            item["shape"], path=(*item_path, "shape")
        )
        dtype = item["dtype"]
        if not isinstance(dtype, str) or dtype not in _TENSOR_METADATA_DTYPES:
            raise ValueError(
                f"{_render_path((*item_path, 'dtype'))} must be an allowlisted dtype"
            )
        digest = item.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or _SHA256.fullmatch(digest) is None
        ):
            raise ValueError(
                f"{_render_path((*item_path, 'sha256'))} must be a lowercase "
                "SHA-256 digest"
            )
        numel = item.get("numel")
        if numel is not None and (
            type(numel) is not int or numel < 0 or numel != expected_numel
        ):
            raise ValueError(
                f"{_render_path((*item_path, 'numel'))} must match shape"
            )


def _validate_shape(value: Any, *, path: _Path) -> int:
    if not isinstance(value, list) or len(value) > _MAX_TENSOR_RANK:
        raise ValueError(
            f"{_render_path(path)} must be a bounded list of non-negative "
            "integer dimensions"
        )
    numel = 1
    for dimension in value:
        if (
            type(dimension) is not int
            or not 0 <= dimension <= _MAX_TENSOR_DIMENSION
        ):
            raise ValueError(
                f"{_render_path(path)} must be a bounded list of non-negative "
                "integer dimensions"
            )
        if dimension == 0:
            numel = 0
        elif numel and numel > _MAX_TENSOR_NUMEL // dimension:
            raise ValueError(
                f"{_render_path(path)} tensor numel exceeds {_MAX_TENSOR_NUMEL}"
            )
        else:
            numel *= dimension
    return numel


def _consume_text(
    value: str,
    *,
    path: _Path,
    budget: _ArtifactBudget,
) -> None:
    if len(value) > _MAX_SCALAR_BYTES:
        raise ValueError(f"{_render_path(path)} exceeds the scalar size limit")
    byte_count = len(value.encode("utf-8"))
    if byte_count > _MAX_SCALAR_BYTES:
        raise ValueError(f"{_render_path(path)} exceeds the scalar size limit")
    _consume_bytes(byte_count, path=path, budget=budget)


def _consume_field_name(
    value: str,
    *,
    path: _Path,
    budget: _ArtifactBudget,
) -> None:
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise ValueError(
            f"artifact field name at {_render_path(path)} is not valid UTF-8"
        ) from exc
    if len(encoded) > _MAX_FIELD_NAME_BYTES:
        raise ValueError(
            f"artifact field name at {_render_path(path)} exceeds the "
            f"{_MAX_FIELD_NAME_BYTES}-byte limit"
        )
    try:
        canonical_artifact_field_name(value)
    except ValueError as exc:
        raise ValueError(
            f"artifact field name at {_render_path(path)} must normalize to "
            "printable ASCII"
        ) from exc
    _consume_bytes(len(encoded), path=path, budget=budget)


def _consume_bytes(size: int, *, path: _Path, budget: _ArtifactBudget) -> None:
    budget.bytes_seen += size
    if budget.bytes_seen > _MAX_ARTIFACT_BYTES:
        raise ValueError(
            f"gate artifact total size limit exceeded at {_render_path(path)}"
        )


def _render_path(path: _Path) -> str:
    if not path:
        return "<root>"
    parts: list[str] = []
    length = 0
    for segment in path:
        if isinstance(segment, int):
            part = f"[{segment}]"
        else:
            safe = segment.encode("unicode_escape", errors="backslashreplace").decode(
                "ascii"
            )
            part = safe if not parts else f".{safe}"
        if length + len(part) > _MAX_RENDERED_PATH_CHARS:
            parts.append("...")
            break
        parts.append(part)
        length += len(part)
    return "".join(parts)
