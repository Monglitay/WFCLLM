"""Fail-closed formal gate bundles and precision-stable prediction."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
import math
from pathlib import Path
import re
import shutil
import tempfile
from typing import Any, Protocol

import torch
from torch import nn

from wfcllm.gate.input import GATE_INPUT_CONTRACT_VERSION
from wfcllm.windowing.contracts import GateScores, WINDOW_CONTRACT_VERSION

GATE_BUNDLE_CONTRACT_VERSION = "wfcllm-gate-bundle/v1"
GATE_MODEL_FORMAT_VERSION = "wfcllm-gate-model-state/v1"
_ROOT_ALLOWLIST = frozenset(
    {
        "manifest.json",
        "gate_float.pt",
        "gate_int8.pt",
        "tokenizer",
        "validation_summary.json",
    }
)
_QUANTIZATION = "torch-dynamic-qint8-linear"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SENSITIVE_TOKENS = frozenset(
    {"raw", "key", "secret", "code", "sample", "candidate", "deployment"}
)


@dataclass(frozen=True)
class TokenizerSnapshot:
    """Immutable, hash-bound tokenizer contents exposed to controlled adapters."""

    entries: tuple[tuple[str, str, bytes], ...]
    sha256: str

    @classmethod
    def from_directory(cls, root: Path) -> TokenizerSnapshot:
        if root.is_symlink() or not root.is_dir():
            raise ValueError("tokenizer snapshot source must be a non-symlink directory")
        entries: list[tuple[str, str, bytes]] = []
        for path in sorted(
            root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
        ):
            if path.is_symlink():
                raise ValueError("tokenizer snapshot cannot contain symlinks")
            relative = path.relative_to(root).as_posix()
            if path.is_file():
                entries.append((relative, "file", _read_file_snapshot(path)))
            elif path.is_dir():
                entries.append((relative, "directory", b""))
            else:
                raise ValueError("tokenizer snapshot contains unsupported entry")
        if not entries or not any(kind == "file" for _, kind, _ in entries):
            raise ValueError("tokenizer snapshot must contain at least one file")
        frozen = tuple(entries)
        return cls(entries=frozen, sha256=_sha256_tokenizer_entries(frozen))

    def file_bytes(self, relative_path: str) -> bytes:
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError("tokenizer relative path must be a non-empty string")
        for path, kind, content in self.entries:
            if path == relative_path and kind == "file":
                return content
        raise ValueError("tokenizer snapshot file is missing")


@dataclass(frozen=True)
class GateBundleManifest:
    """Strict, immutable public metadata for a formal gate bundle."""

    contract_version: str
    model_format_version: str
    window_contract_version: str
    gate_input_contract_version: str
    model_architecture: str
    base_model_id: str
    torch_version: str
    tokenizer_sha256: str
    float_model_sha256: str
    quantized_model_sha256: str
    validation_summary_sha256: str
    close_low_threshold: float
    close_high_threshold: float
    suitable_accept_threshold: float
    max_probability_delta: float
    max_tokens: int
    training_data_manifest_sha256: str
    training_key_bank_id: str
    holdout_key_bank_id: str
    quantization: str

    def __post_init__(self) -> None:
        if self.contract_version != GATE_BUNDLE_CONTRACT_VERSION:
            raise ValueError("bundle contract version mismatch")
        if self.model_format_version != GATE_MODEL_FORMAT_VERSION:
            raise ValueError("model format version mismatch")
        if self.window_contract_version != WINDOW_CONTRACT_VERSION:
            raise ValueError("window contract version mismatch")
        if self.gate_input_contract_version != GATE_INPUT_CONTRACT_VERSION:
            raise ValueError("gate input contract version mismatch")
        for name in ("model_architecture", "base_model_id", "torch_version"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value or value.strip() != value:
                raise ValueError(f"{name} must be a non-empty canonical string")
        first = self.base_model_id.split("/", 1)[0]
        if self.base_model_id in {".", ""} or ":" in first:
            raise ValueError("base_model_id must identify a local model")
        for name in (
            "tokenizer_sha256",
            "float_model_sha256",
            "quantized_model_sha256",
            "validation_summary_sha256",
            "training_data_manifest_sha256",
        ):
            _validate_digest(name, getattr(self, name))
        bank_patterns = {
            "training_key_bank_id": re.compile(
                r"^training-key-bank/v1:sha256:[0-9a-f]{64}$"
            ),
            "holdout_key_bank_id": re.compile(
                r"^holdout-key-bank/v1:sha256:[0-9a-f]{64}$"
            ),
        }
        for name, pattern in bank_patterns.items():
            value = getattr(self, name)
            if not isinstance(value, str) or pattern.fullmatch(value) is None:
                raise ValueError(f"{name} must be an irreversible bank identifier")
        if self.training_key_bank_id == self.holdout_key_bank_id:
            raise ValueError("training and holdout bank identifiers must differ")
        _validate_probability("close_low_threshold", self.close_low_threshold)
        _validate_probability("close_high_threshold", self.close_high_threshold)
        _validate_probability(
            "suitable_accept_threshold", self.suitable_accept_threshold
        )
        _validate_probability("max_probability_delta", self.max_probability_delta)
        if self.close_low_threshold >= self.close_high_threshold:
            raise ValueError("close thresholds must leave a non-empty uncertain region")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 512:
            raise ValueError("max_tokens must be an integer from 1 through 512")
        if self.quantization != _QUANTIZATION:
            raise ValueError("formal quantization contract mismatch")

    @classmethod
    def from_dict(cls, value: object) -> GateBundleManifest:
        fields = tuple(cls.__dataclass_fields__)
        if not isinstance(value, Mapping) or set(value) != set(fields):
            raise ValueError("bundle manifest schema mismatch")
        if any(not isinstance(key, str) for key in value):
            raise ValueError("bundle manifest schema mismatch")
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        ) + "\n"


@dataclass(frozen=True)
class GateBundle:
    """A verified five-entry bundle with safe eager state validation."""

    root: Path
    manifest: GateBundleManifest
    validation_summary: Mapping[str, Any]
    _float_state_snapshot: Mapping[str, Any] = field(repr=False, compare=False)
    _int8_state_snapshot: Mapping[str, Any] = field(repr=False, compare=False)
    _float_state_digest: str
    _int8_state_digest: str
    _tokenizer_snapshot: TokenizerSnapshot = field(repr=False, compare=False)

    @classmethod
    def load(cls, root: Path) -> GateBundle:
        if not isinstance(root, Path):
            raise ValueError("bundle root must be a pathlib.Path")
        if _has_symlink_component(root):
            raise ValueError("bundle root symlink traversal is forbidden")
        if not root.is_dir():
            raise ValueError("bundle directory is missing")
        names = {item.name for item in root.iterdir()}
        if names != _ROOT_ALLOWLIST:
            missing = sorted(_ROOT_ALLOWLIST - names)
            raise ValueError(
                "bundle root allowlist mismatch"
                + (f"; missing {missing!r}" if missing else "")
            )
        for item in root.rglob("*"):
            if item.is_symlink():
                raise ValueError("bundle symlink is forbidden")
        for name in _ROOT_ALLOWLIST - {"tokenizer"}:
            if not (root / name).is_file():
                raise ValueError(f"bundle file is missing: {name}")
        tokenizer = root / "tokenizer"
        if not tokenizer.is_dir() or not any(tokenizer.iterdir()):
            raise ValueError("tokenizer directory is missing or empty")
        manifest_value = _read_strict_json(root / "manifest.json", "manifest")
        manifest = GateBundleManifest.from_dict(manifest_value)
        if manifest.torch_version != torch.__version__:
            raise ValueError("bundle torch runtime version mismatch")
        canonical = manifest.to_json()
        if (root / "manifest.json").read_text(encoding="utf-8") != canonical:
            raise ValueError("bundle manifest must use canonical JSON")
        float_snapshot = _read_file_snapshot(root / "gate_float.pt")
        int8_snapshot = _read_file_snapshot(root / "gate_int8.pt")
        summary_snapshot = _read_file_snapshot(root / "validation_summary.json")
        expected = (
            ("gate_float.pt", float_snapshot, manifest.float_model_sha256),
            ("gate_int8.pt", int8_snapshot, manifest.quantized_model_sha256),
            (
                "validation_summary.json",
                summary_snapshot,
                manifest.validation_summary_sha256,
            ),
        )
        for name, snapshot, digest in expected:
            if hashlib.sha256(snapshot).hexdigest() != digest:
                raise ValueError(f"bundle hash mismatch for {name}")
        tokenizer_snapshot = TokenizerSnapshot.from_directory(tokenizer)
        if tokenizer_snapshot.sha256 != manifest.tokenizer_sha256:
            raise ValueError("bundle hash mismatch for tokenizer")
        summary = _read_strict_json_bytes(summary_snapshot, "validation summary")
        _validate_summary(summary)
        canonical_summary = json.dumps(
            dict(summary), allow_nan=False, sort_keys=True, separators=(",", ":")
        ) + "\n"
        if (root / "validation_summary.json").read_text(
            encoding="utf-8"
        ) != canonical_summary:
            raise ValueError("validation summary must use canonical JSON")
        if not summary["validated"]:
            raise ValueError("formal bundle requires validation summary validated=true")
        _bind_summary_to_manifest(summary, manifest)
        float_state = _load_model_state_snapshot(float_snapshot, precision="float")
        int8_state = _load_model_state_snapshot(int8_snapshot, precision="int8")
        bundle = cls(
            root=root,
            manifest=manifest,
            validation_summary=dict(summary),
            _float_state_snapshot=float_state,
            _int8_state_snapshot=int8_state,
            _float_state_digest=_state_content_digest(float_state),
            _int8_state_digest=_state_content_digest(int8_state),
            _tokenizer_snapshot=tokenizer_snapshot,
        )
        return bundle

    @classmethod
    def create(
        cls,
        *,
        root: Path,
        validated_float_artifact: Path,
        validated_int8_artifact: Path,
        tokenizer_source: Path,
        validation_summary: Mapping[str, Any],
        model_architecture: str,
        base_model_id: str,
        thresholds: Any,
        training_data_manifest_sha256: str,
        training_key_bank_id: str,
        holdout_key_bank_id: str,
    ) -> GateBundle:
        """Create and immediately re-verify a formal bundle.

        The exact already-validated artifact bytes are copied without
        re-serialization. The output directory must be absent or strictly
        empty, preventing artifact mixing and validation/publication TOCTOU.
        """

        if not isinstance(root, Path) or _has_symlink_component(root):
            raise ValueError("bundle root must be a non-symlink pathlib.Path")
        if root.exists() and (not root.is_dir() or any(root.iterdir())):
            raise ValueError("bundle output must be absent or strictly empty")
        for name, artifact in (
            ("validated_float_artifact", validated_float_artifact),
            ("validated_int8_artifact", validated_int8_artifact),
        ):
            if (
                not isinstance(artifact, Path)
                or _has_symlink_component(artifact)
                or not artifact.is_file()
            ):
                raise ValueError(f"{name} must be a non-symlink artifact file")
        if validated_float_artifact == validated_int8_artifact:
            raise ValueError("validated float and int8 artifacts must differ")
        if (
            not isinstance(tokenizer_source, Path)
            or _has_symlink_component(tokenizer_source)
            or not tokenizer_source.is_dir()
            or not any(tokenizer_source.iterdir())
        ):
            raise ValueError("tokenizer_source must be a non-empty local directory")
        for path in tokenizer_source.rglob("*"):
            if path.is_symlink():
                raise ValueError("tokenizer source symlink is forbidden")
        _validate_summary(validation_summary)
        if not validation_summary["validated"]:
            raise ValueError("formal bundle requires validated=true")
        from wfcllm.gate.validation import GateValidationThresholds

        if not isinstance(thresholds, GateValidationThresholds):
            raise ValueError("thresholds must be fitted GateValidationThresholds")
        thresholds.validate()
        if validation_summary["thresholds"] != asdict(thresholds):
            raise ValueError("validation summary thresholds do not match fitted thresholds")

        expected_float_hashes = {
            row["artifact_sha256"]
            for row in validation_summary["process_attestations"]
            if row["precision"] == "float"
        }
        expected_int8_hashes = {
            row["artifact_sha256"]
            for row in validation_summary["process_attestations"]
            if row["precision"] == "int8"
        }
        if expected_float_hashes != {sha256_file(validated_float_artifact)}:
            raise ValueError("validated float artifact hash does not match summary")
        if expected_int8_hashes != {sha256_file(validated_int8_artifact)}:
            raise ValueError("validated int8 artifact hash does not match summary")

        root.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent)
        )
        try:
            tokenizer_out = staging / "tokenizer"
            shutil.copytree(tokenizer_source, tokenizer_out)
            shutil.copy2(validated_float_artifact, staging / "gate_float.pt")
            shutil.copy2(validated_int8_artifact, staging / "gate_int8.pt")
            (staging / "validation_summary.json").write_text(
                json.dumps(
                    dict(validation_summary),
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n",
                encoding="utf-8",
            )
            manifest = GateBundleManifest(
                contract_version=GATE_BUNDLE_CONTRACT_VERSION,
                model_format_version=GATE_MODEL_FORMAT_VERSION,
                window_contract_version=WINDOW_CONTRACT_VERSION,
                gate_input_contract_version=GATE_INPUT_CONTRACT_VERSION,
                model_architecture=model_architecture,
                base_model_id=base_model_id,
                torch_version=torch.__version__,
                tokenizer_sha256=sha256_directory(tokenizer_out),
                float_model_sha256=sha256_file(staging / "gate_float.pt"),
                quantized_model_sha256=sha256_file(staging / "gate_int8.pt"),
                validation_summary_sha256=sha256_file(
                    staging / "validation_summary.json"
                ),
                close_low_threshold=thresholds.close_low,
                close_high_threshold=thresholds.close_high,
                suitable_accept_threshold=thresholds.suitable_accept,
                max_probability_delta=thresholds.max_probability_delta,
                max_tokens=thresholds.max_tokens,
                training_data_manifest_sha256=training_data_manifest_sha256,
                training_key_bank_id=training_key_bank_id,
                holdout_key_bank_id=holdout_key_bank_id,
                quantization=_QUANTIZATION,
            )
            _bind_summary_to_manifest(validation_summary, manifest)
            (staging / "manifest.json").write_text(
                manifest.to_json(), encoding="utf-8"
            )
            cls.load(staging)
            if root.exists():
                root.rmdir()
            staging.replace(root)
        except Exception:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        return cls.load(root)

    def stable_predictor(self, loader: BundlePredictorLoader) -> StableGatePredictor:
        """Construct the sole public inference path from both formal states."""

        if not callable(loader):
            raise ValueError("loader must be callable")
        quantized = self._load_verified_adapter(loader, "int8")
        float_reference = self._load_verified_adapter(loader, "float")
        return StableGatePredictor(
            quantized=quantized,
            float_reference=float_reference,
            manifest=self.manifest,
        )

    def _load_verified_adapter(
        self, loader: BundlePredictorLoader, precision: str
    ) -> _VerifiedPredictorAdapter:
        state = self._load_state(precision)
        expected_content_digest = (
            self._float_state_digest
            if precision == "float"
            else self._int8_state_digest
        )
        if _state_content_digest(state) != expected_content_digest:
            raise ValueError("cached bundle artifact state changed after validation")
        candidate = loader(
            precision=precision,
            state=state,
            tokenizer_snapshot=self._tokenizer_snapshot,
            manifest=self.manifest,
        )
        if _adapter_retains_artifact_state(candidate, state):
            raise ValueError("bundle adapter must not retain mutable artifact state")
        if _state_content_digest(state) != expected_content_digest:
            raise ValueError("bundle loader mutated the cached artifact state")
        return _verify_predictor_adapter(
            candidate,
            precision=precision,
            expected_content_digest=expected_content_digest,
            tokenizer_sha256=self._tokenizer_snapshot.sha256,
        )

    def _load_state(self, precision: str) -> Mapping[str, Any]:
        if precision not in {"float", "int8"}:
            raise ValueError("precision must be 'float' or 'int8'")
        return (
            self._float_state_snapshot
            if precision == "float"
            else self._int8_state_snapshot
        )


class BundlePredictorAdapter(Protocol):
    """Controlled adapter: bundle owns model execution and probability decoding."""

    model: nn.Module
    tokenizer_sha256: str

    def encode_input(self, serialized_input: str) -> Mapping[str, torch.Tensor]: ...


class BundlePredictorLoader(Protocol):
    def __call__(
        self,
        *,
        precision: str,
        state: Mapping[str, Any],
        tokenizer_snapshot: TokenizerSnapshot,
        manifest: GateBundleManifest,
    ) -> BundlePredictorAdapter: ...


@dataclass(frozen=True)
class _VerifiedPredictorAdapter:
    adapter: BundlePredictorAdapter
    precision: str
    inspected_modules: tuple[nn.Module, ...]
    runtime_seal: _ModelRuntimeSeal


@dataclass(frozen=True)
class _TensorRuntimeSeal:
    name: str
    kind: str
    object_id: int
    data_ptr: int
    dtype: str
    shape: tuple[int, ...]
    device: str
    layout: str
    requires_grad: bool
    is_quantized: bool
    quantization: tuple[Any, ...]
    version: int


@dataclass
class _MutationCounter:
    value: int = 0


@dataclass(frozen=True)
class _ModelRuntimeSeal:
    model_id: int
    modules: tuple[tuple[str, int, type[nn.Module], bool], ...]
    tensors: tuple[_TensorRuntimeSeal, ...]
    packed_objects: tuple[_PackedRuntimeSeal, ...]
    mutation_counter: _MutationCounter
    hook_handles: tuple[Any, ...]


@dataclass(frozen=True)
class _PackedRuntimeSeal:
    module_name: str
    module_id: int
    module_type: type[nn.Module]
    packed_module_id: int
    packed_object_id: int
    dtype: str
    scale: float
    zero_point: int


class StableGatePredictor:
    """The sole public precision path: qint8 first, then float reference."""

    def __init__(
        self,
        *,
        quantized: _VerifiedPredictorAdapter,
        float_reference: _VerifiedPredictorAdapter,
        manifest: GateBundleManifest,
    ) -> None:
        if (
            not isinstance(quantized, _VerifiedPredictorAdapter)
            or quantized.precision != "int8"
            or not isinstance(float_reference, _VerifiedPredictorAdapter)
            or float_reference.precision != "float"
        ):
            raise ValueError("StableGatePredictor requires verified int8 and float adapters")
        if not isinstance(manifest, GateBundleManifest):
            raise ValueError("manifest must be a GateBundleManifest")
        self._quantized = quantized
        self._float = float_reference
        self._manifest = manifest

    def predict(self, serialized_input: str) -> GateScores:
        if not isinstance(serialized_input, str) or not serialized_input:
            raise ValueError("serialized_input must be a non-empty string")
        int8 = _run_verified_adapter(self._quantized, serialized_input)
        reference = _run_verified_adapter(self._float, serialized_input)
        delta = max(abs(left - right) for left, right in zip(int8, reference))
        int8_decision = _decisions(int8, self._manifest)
        float_decision = _decisions(reference, self._manifest)
        agreement = int8_decision == float_decision
        return GateScores(
            close_probability=int8[0],
            suitable_probability=int8[1],
            stable=agreement and delta <= self._manifest.max_probability_delta,
            precision_delta=delta,
            decision_agreement=agreement,
        )


def _verify_predictor_adapter(
    value: object,
    *,
    precision: str,
    expected_content_digest: str,
    tokenizer_sha256: str,
) -> _VerifiedPredictorAdapter:
    model = getattr(value, "model", None)
    if not isinstance(model, nn.Module):
        raise ValueError("bundle adapter must expose its inspected torch model")
    if model.training:
        raise ValueError("bundle adapter model must be in eval mode")
    if not callable(getattr(value, "encode_input", None)):
        raise ValueError("bundle adapter must implement controlled encode_input")
    if getattr(value, "tokenizer_sha256", None) != tokenizer_sha256:
        raise ValueError("bundle adapter tokenizer snapshot attestation mismatch")
    from wfcllm.gate.validation import (
        _precision_modules,
        _validate_bound_model_structure,
    )

    _validate_bound_model_structure(model, precision=precision, device="cpu")
    if _state_content_digest(model.state_dict()) != expected_content_digest:
        raise ValueError("bundle adapter model state does not match artifact content")
    modules = _precision_modules(model, precision)
    runtime_seal = _build_runtime_seal(model)
    return _VerifiedPredictorAdapter(
        adapter=value,
        precision=precision,
        inspected_modules=modules,
        runtime_seal=runtime_seal,
    )


def _run_verified_adapter(
    verified: _VerifiedPredictorAdapter, serialized_input: str
) -> tuple[float, float]:
    from wfcllm.gate.validation import _walk_tensors

    _validate_runtime_seal(verified.adapter.model, verified.runtime_seal)
    inputs = verified.adapter.encode_input(serialized_input)
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("bundle adapter encoder must return named tensor inputs")
    if any(
        not isinstance(name, str)
        or not isinstance(value, torch.Tensor)
        or value.ndim == 0
        or value.shape[0] != 1
        for name, value in inputs.items()
    ):
        raise ValueError("bundle adapter inputs must be named single-item tensors")
    if any(tensor.device.type != "cpu" for tensor in inputs.values()):
        raise ValueError("formal bundle adapter inputs must execute on CPU")
    if any(
        (tensor.is_floating_point() or tensor.is_complex())
        and not torch.isfinite(tensor).all()
        for tensor in inputs.values()
    ):
        raise ValueError("formal bundle adapter inputs must be finite")
    observed_devices: list[str] = []

    def observe_execution(
        _module: nn.Module, module_inputs: tuple[Any, ...], output: Any
    ) -> None:
        tensors = tuple(_walk_tensors((module_inputs, output)))
        if not tensors:
            raise ValueError("inspected bundle model execution produced no tensors")
        observed_devices.extend(tensor.device.type for tensor in tensors)

    hooks = [module.register_forward_hook(observe_execution) for module in verified.inspected_modules]
    try:
        with torch.inference_mode():
            output = verified.adapter.model(**dict(inputs))
    finally:
        for hook in hooks:
            hook.remove()
    if not observed_devices or any(device != "cpu" for device in observed_devices):
        raise ValueError("bundle predictor model execution attestation failed")
    _validate_runtime_seal(verified.adapter.model, verified.runtime_seal)
    close_logits = getattr(output, "close_logits", None)
    suitable_logits = getattr(output, "suitable_logits", None)
    if (
        not isinstance(close_logits, torch.Tensor)
        or not isinstance(suitable_logits, torch.Tensor)
        or close_logits.shape != (1,)
        or suitable_logits.shape != (1,)
    ):
        raise ValueError("bundle model must return one close and suitable logit")
    probabilities = torch.stack(
        (torch.sigmoid(close_logits), torch.sigmoid(suitable_logits)), dim=1
    )
    if not torch.isfinite(probabilities).all():
        raise ValueError("bundle model probabilities must be finite")
    values = probabilities.detach().cpu().tolist()[0]
    return _probability_pair((float(values[0]), float(values[1])))


def _state_content_digest(value: object) -> str:
    """Canonical one-time digest for handing an artifact state to an adapter."""

    digest = hashlib.sha256()

    def update(item: object) -> None:
        if isinstance(item, Mapping):
            digest.update(b"mapping")
            keys = sorted(item, key=lambda key: (type(key).__name__, str(key)))
            digest.update(len(keys).to_bytes(8, "big"))
            for key in keys:
                update(key)
                update(item[key])
            return
        if isinstance(item, tuple):
            digest.update(b"tuple")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
            return
        if isinstance(item, list):
            digest.update(b"list")
            digest.update(len(item).to_bytes(8, "big"))
            for child in item:
                update(child)
            return
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu()
            if tensor.layout != torch.strided:
                raise ValueError("artifact state digest requires dense strided tensors")
            digest.update(b"tensor")
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tuple(tensor.stride())).encode("ascii"))
            digest.update(repr(_tensor_quantization_metadata(tensor)).encode("ascii"))
            raw = tensor.int_repr() if tensor.is_quantized else tensor
            raw_bytes = (
                raw.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
            )
            digest.update(len(raw_bytes).to_bytes(8, "big"))
            digest.update(raw_bytes)
            return
        if isinstance(item, str):
            encoded = item.encode("utf-8")
            digest.update(b"str")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
            return
        if item is None:
            digest.update(b"none")
            return
        if isinstance(item, bool):
            digest.update(b"bool1" if item else b"bool0")
            return
        if isinstance(item, int):
            digest.update(b"int")
            digest.update(str(item).encode("ascii"))
            return
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ValueError("artifact state digest values must be finite")
            digest.update(b"float")
            digest.update(item.hex().encode("ascii"))
            return
        if isinstance(item, torch.dtype):
            digest.update(b"dtype")
            digest.update(str(item).encode("ascii"))
            return
        raise ValueError("artifact state digest contains an unsupported value")

    update(value)
    return digest.hexdigest()


def _load_model_state_snapshot(
    raw_bytes: bytes, *, precision: str
) -> Mapping[str, Any]:
    try:
        value = torch.load(
            io.BytesIO(raw_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError("safe model load failed; pickle fallback is forbidden") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError("model state must be a non-empty mapping")
    _validate_safe_state(value, path=())
    tensors = tuple(_state_tensors(value))
    if precision == "float" and not any(item.is_floating_point() for item in tensors):
        raise ValueError("float model state must contain floating tensors")
    if precision == "int8" and not any(item.is_quantized for item in tensors):
        raise ValueError("quantized model state is not qint8; float fallback forbidden")
    return value


def _adapter_retains_artifact_state(adapter: object, state: object) -> bool:
    """Reject direct retention of the transient deserialized state graph."""

    state_ids: set[int] = set()

    def collect(item: object) -> None:
        if not isinstance(item, (Mapping, tuple, list, torch.Tensor)):
            return
        if id(item) in state_ids:
            return
        state_ids.add(id(item))
        if isinstance(item, Mapping):
            for child in item.values():
                collect(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                collect(child)

    collect(state)

    def contains(item: object, seen: set[int]) -> bool:
        if id(item) in state_ids:
            return True
        if id(item) in seen:
            return False
        seen.add(id(item))
        if item is None or isinstance(
            item,
            (
                str,
                bytes,
                bool,
                int,
                float,
                complex,
                Path,
                torch.dtype,
                torch.Tensor,
                type,
            ),
        ):
            return False
        if isinstance(item, nn.Module):
            framework_names = set(nn.Module().__dict__)
            try:
                attributes = getattr(item, "__dict__", {})
                registered_children = getattr(item, "_modules", {})
            except Exception:
                return True
            custom_values = (
                child
                for name, child in attributes.items()
                if name not in framework_names
            )
            children = registered_children.values()
            if any(contains(child, seen) for child in (*children, *custom_values)):
                return True
            return _slots_retain_state(item, seen, contains)
        if isinstance(item, Mapping):
            return any(
                contains(key, seen) or contains(child, seen)
                for key, child in item.items()
            )
        if isinstance(item, (tuple, list, set, frozenset)):
            return any(contains(child, seen) for child in item)
        try:
            attributes = getattr(item, "__dict__", None)
        except Exception:
            return True
        if isinstance(attributes, Mapping) and any(
            contains(child, seen) for child in attributes.values()
        ):
            return True
        return _slots_retain_state(item, seen, contains)

    return contains(adapter, set())


def _slots_retain_state(
    value: object,
    seen: set[int],
    contains: Any,
) -> bool:
    for owner in type(value).__mro__:
        slots = owner.__dict__.get("__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        for slot in slots:
            if slot in {"__dict__", "__weakref__"}:
                continue
            attribute_name = slot
            if slot.startswith("__") and not slot.endswith("__"):
                attribute_name = f"_{owner.__name__.lstrip('_')}{slot}"
            try:
                child = getattr(value, attribute_name)
            except Exception:
                return True
            if contains(child, seen):
                return True
    return False


def _build_runtime_seal(model: nn.Module) -> _ModelRuntimeSeal:
    counter = _MutationCounter()

    def observe_load(*_: Any, **__: Any) -> None:
        counter.value += 1

    handles = tuple(
        module.register_load_state_dict_pre_hook(observe_load)
        for module in model.modules()
    )
    return _ModelRuntimeSeal(
        model_id=id(model),
        modules=_module_runtime_metadata(model),
        tensors=_tensor_runtime_metadata(model),
        packed_objects=_packed_runtime_metadata(model),
        mutation_counter=counter,
        hook_handles=handles,
    )


def _validate_runtime_seal(model: nn.Module, seal: _ModelRuntimeSeal) -> None:
    if id(model) != seal.model_id:
        raise ValueError("bundle predictor model identity changed after verification")
    if seal.mutation_counter.value:
        raise ValueError("bundle predictor state was reloaded after verification")
    if _module_runtime_metadata(model) != seal.modules:
        raise ValueError("bundle predictor module structure or mode changed")
    if _tensor_runtime_metadata(model) != seal.tensors:
        raise ValueError("bundle predictor tensor metadata or version changed")
    if _packed_runtime_metadata(model) != seal.packed_objects:
        raise ValueError("bundle predictor quantized packed structure changed")


def _module_runtime_metadata(
    model: nn.Module,
) -> tuple[tuple[str, int, type[nn.Module], bool], ...]:
    return tuple(
        (name, id(module), type(module), module.training)
        for name, module in model.named_modules(remove_duplicate=False)
    )


def _tensor_runtime_metadata(model: nn.Module) -> tuple[_TensorRuntimeSeal, ...]:
    values: list[_TensorRuntimeSeal] = []
    for kind, named_values in (
        ("parameter", model.named_parameters(remove_duplicate=False)),
        ("buffer", model.named_buffers(remove_duplicate=False)),
    ):
        for name, tensor in named_values:
            values.append(
                _TensorRuntimeSeal(
                    name=name,
                    kind=kind,
                    object_id=id(tensor),
                    data_ptr=tensor.data_ptr(),
                    dtype=str(tensor.dtype),
                    shape=tuple(tensor.shape),
                    device=str(tensor.device),
                    layout=str(tensor.layout),
                    requires_grad=tensor.requires_grad,
                    is_quantized=tensor.is_quantized,
                    quantization=_tensor_quantization_metadata(tensor),
                    version=tensor._version,
                )
            )
    return tuple(values)


def _tensor_quantization_metadata(tensor: torch.Tensor) -> tuple[Any, ...]:
    if not tensor.is_quantized:
        return ()
    scheme = tensor.qscheme()
    if scheme in {torch.per_tensor_affine, torch.per_tensor_symmetric}:
        return (str(scheme), tensor.q_scale(), tensor.q_zero_point())
    scales = tensor.q_per_channel_scales()
    zero_points = tensor.q_per_channel_zero_points()
    return (
        str(scheme),
        tensor.q_per_channel_axis(),
        tuple(scales.shape),
        str(scales.dtype),
        tuple(zero_points.shape),
        str(zero_points.dtype),
    )


def _packed_runtime_metadata(model: nn.Module) -> tuple[_PackedRuntimeSeal, ...]:
    result: list[_PackedRuntimeSeal] = []
    for name, module in model.named_modules(remove_duplicate=False):
        if not (
            "quantized.dynamic" in type(module).__module__
            and type(module).__name__ == "Linear"
        ):
            continue
        packed_module = getattr(module, "_packed_params", None)
        packed_object = getattr(packed_module, "_packed_params", None)
        if not isinstance(packed_module, nn.Module) or packed_object is None:
            raise ValueError("dynamic qint8 Linear has no inspectable packed parameters")
        result.append(
            _PackedRuntimeSeal(
                module_name=name,
                module_id=id(module),
                module_type=type(module),
                packed_module_id=id(packed_module),
                packed_object_id=id(packed_object),
                dtype=str(getattr(packed_module, "dtype", "")),
                scale=float(getattr(module, "scale")),
                zero_point=int(getattr(module, "zero_point")),
            )
        )
    return tuple(result)


def quantize_gate_model_dynamic(model: nn.Module) -> nn.Module:
    """Apply actual qint8 dynamic quantization to every float ``nn.Linear``."""

    if not isinstance(model, nn.Module):
        raise ValueError("model must be a torch.nn.Module")
    expected = sum(1 for item in model.modules() if isinstance(item, nn.Linear))
    if expected == 0:
        raise RuntimeError("formal bundle requires at least one nn.Linear to quantize")
    try:
        candidate = copy.deepcopy(model).cpu().eval()
        quantized = torch.ao.quantization.quantize_dynamic(
            candidate, {nn.Linear}, dtype=torch.qint8, inplace=False
        )
    except Exception as exc:
        raise RuntimeError(
            "runtime does not support qint8 Linear; no formal bundle produced"
        ) from exc
    actual = sum(
        1
        for item in quantized.modules()
        if "quantized.dynamic" in type(item).__module__
        and type(item).__name__ == "Linear"
    )
    remaining = sum(1 for item in quantized.modules() if isinstance(item, nn.Linear))
    if actual != expected or remaining:
        raise RuntimeError(
            "runtime did not produce qint8 Linear modules; no formal bundle produced"
        )
    return quantized


def sha256_file(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError("hash input must be a non-symlink file")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(root: Path) -> str:
    return TokenizerSnapshot.from_directory(root).sha256


def _sha256_tokenizer_entries(
    entries: tuple[tuple[str, str, bytes], ...]
) -> str:
    digest = hashlib.sha256()
    for relative_path, entry_kind, content in entries:
        relative = relative_path.encode("utf-8")
        digest.update(b"F" if entry_kind == "file" else b"D")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        if entry_kind == "file":
            digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _read_file_snapshot(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise ValueError("snapshot input must be a non-symlink file")
    return path.read_bytes()


def _read_strict_json(path: Path, name: str) -> Mapping[str, Any]:
    return _read_strict_json_bytes(_read_file_snapshot(path), name)


def _read_strict_json_bytes(raw: bytes, name: str) -> Mapping[str, Any]:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate {name} JSON key")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} must be strict UTF-8 JSON") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _validate_summary(value: object) -> None:
    expected_keys = {
        "contract_version",
        "validated",
        "decision_agreement",
        "float_quantized_accepted_set_agreement",
        "formal_accepted_span_consensus",
        "suitable_false_positive_rate",
        "gpu_status",
        "validation_scope",
        "mode_names",
        "threshold_fit_group_digest",
        "agreement_group_digest",
        "thresholds",
        "forced_uncertain_count",
        "formal_accepted_spans",
        "process_attestations",
    }
    if not isinstance(value, Mapping) or set(value) != expected_keys:
        raise ValueError("validation summary schema mismatch")
    if value["contract_version"] != "wfcllm-gate-validation/v1":
        raise ValueError("validation summary contract mismatch")
    if type(value["validated"]) is not bool:
        raise ValueError("validation summary validated must be bool")
    metric_names = (
        "decision_agreement",
        "float_quantized_accepted_set_agreement",
        "formal_accepted_span_consensus",
        "suitable_false_positive_rate",
    )
    for name in metric_names:
        _validate_probability(f"validation summary {name}", value[name])
    scope_gpu = {
        "cpu": "not_available",
        "cpu+gpu-float": "validated",
    }
    if value["validation_scope"] not in scope_gpu:
        raise ValueError("validation summary scope is invalid")
    if value["gpu_status"] != scope_gpu[value["validation_scope"]]:
        raise ValueError("validation summary GPU status/scope mismatch")
    expected_modes = [
        f"cpu-{precision}-batch-{batch}-{order}"
        for precision in ("float", "int8")
        for batch in (1, 2, 4, 8)
        for order in ("original", "reverse", "random")
    ]
    if value["validation_scope"] == "cpu+gpu-float":
        expected_modes.extend(
            f"gpu-float-batch-{batch}-{order}"
            for batch in (1, 2, 4, 8)
            for order in ("original", "reverse", "random")
        )
    if value["mode_names"] != expected_modes:
        raise ValueError("validation summary mode matrix mismatch")
    attestations = value["process_attestations"]
    if not isinstance(attestations, list) or len(attestations) != len(expected_modes):
        raise ValueError("validation summary process attestation count mismatch")
    for expected_mode, attestation in zip(expected_modes, attestations, strict=True):
        if not isinstance(attestation, Mapping) or set(attestation) != {
            "mode_name",
            "pids",
            "device",
            "precision",
            "artifact_name",
            "artifact_sha256",
        }:
            raise ValueError("validation summary process attestation schema mismatch")
        expected_device, expected_precision = expected_mode.split("-", 2)[:2]
        expected_artifact_name = (
            "gate_float.pt" if expected_precision == "float" else "gate_int8.pt"
        )
        pids = attestation["pids"]
        if (
            attestation["mode_name"] != expected_mode
            or attestation["device"] != expected_device
            or attestation["precision"] != expected_precision
            or attestation["artifact_name"] != expected_artifact_name
            or not isinstance(attestation["artifact_sha256"], str)
            or _DIGEST_RE.fullmatch(attestation["artifact_sha256"]) is None
            or not isinstance(pids, list)
            or len(pids) != 2
            or any(type(pid) is not int or pid <= 0 for pid in pids)
            or pids[0] == pids[1]
        ):
            raise ValueError("validation summary process attestation mismatch")
    _validate_digest(
        "validation summary threshold_fit_group_digest",
        value["threshold_fit_group_digest"],
    )
    _validate_digest(
        "validation summary agreement_group_digest",
        value["agreement_group_digest"],
    )
    if value["threshold_fit_group_digest"] == value["agreement_group_digest"]:
        raise ValueError("validation summary grouped subsets must differ")
    from wfcllm.gate.validation import GateValidationThresholds

    thresholds = value["thresholds"]
    expected_threshold_keys = set(GateValidationThresholds.__dataclass_fields__)
    if not isinstance(thresholds, Mapping) or set(thresholds) != expected_threshold_keys:
        raise ValueError("validation summary threshold schema mismatch")
    fitted = GateValidationThresholds(**dict(thresholds))
    fitted.validate()
    if fitted.threshold_fit_group_digest != value["threshold_fit_group_digest"]:
        raise ValueError("validation summary threshold fit digest mismatch")
    if (
        type(value["forced_uncertain_count"]) is not int
        or value["forced_uncertain_count"] < 0
    ):
        raise ValueError("validation summary forced_uncertain_count is invalid")
    spans = value["formal_accepted_spans"]
    if not isinstance(spans, list):
        raise ValueError("validation summary formal spans must be a list")
    canonical_spans: list[tuple[str, str, int, int]] = []
    for span in spans:
        if (
            not isinstance(span, list)
            or len(span) != 4
            or any(
                not isinstance(item, str) or not item
                for item in span[:2]
            )
            or any(type(item) is not int for item in span[2:])
            or span[2] < 0
            or span[3] <= span[2]
        ):
            raise ValueError("validation summary formal window identity is invalid")
        canonical_spans.append((span[0], span[1], span[2], span[3]))
    if canonical_spans != sorted(set(canonical_spans)):
        raise ValueError("validation summary formal window identities must be canonical")
    hard_pass = (
        value["decision_agreement"] >= 0.999
        and value["float_quantized_accepted_set_agreement"] >= 0.999
        and value["formal_accepted_span_consensus"] == 1.0
        and bool(canonical_spans)
        and value["suitable_false_positive_rate"] <= 0.05
    )
    if value["validated"] != hard_pass:
        raise ValueError("validation summary validated flag contradicts hard metrics or coverage")


def _bind_summary_to_manifest(
    summary: Mapping[str, Any], manifest: GateBundleManifest
) -> None:
    thresholds = summary["thresholds"]
    bindings = {
        "close_low": manifest.close_low_threshold,
        "close_high": manifest.close_high_threshold,
        "suitable_accept": manifest.suitable_accept_threshold,
        "max_probability_delta": manifest.max_probability_delta,
        "max_tokens": manifest.max_tokens,
    }
    if any(thresholds[name] != expected for name, expected in bindings.items()):
        raise ValueError("validation summary thresholds do not match manifest")
    for attestation in summary["process_attestations"]:
        expected_hash = (
            manifest.float_model_sha256
            if attestation["precision"] == "float"
            else manifest.quantized_model_sha256
        )
        if attestation["artifact_sha256"] != expected_hash:
            raise ValueError(
                "validation process artifact hash does not match manifest"
            )


def _validate_digest(name: str, value: object) -> None:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 digest")


def _validate_probability(name: str, value: object) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0.0 <= value <= 1.0
    ):
        raise ValueError(f"{name} must be a probability in [0, 1]")


def _probability_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("predictor must return a two-probability tuple")
    for item in value:
        _validate_probability("predictor probability", item)
    return float(value[0]), float(value[1])


def _decisions(
    value: tuple[float, float], manifest: GateBundleManifest
) -> tuple[str, bool]:
    close = (
        "continue"
        if value[0] <= manifest.close_low_threshold
        else "close"
        if value[0] >= manifest.close_high_threshold
        else "uncertain"
    )
    return close, value[1] >= manifest.suitable_accept_threshold


def _validate_safe_state(value: object, *, path: tuple[str, ...]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, (str, int)):
                raise ValueError("model state keys must be strings or integers")
            if isinstance(key, str):
                if key.endswith("_extra_state"):
                    raise ValueError("model state extra state is forbidden")
                tokens = set(filter(None, re.split(r"[^a-z0-9]+", key.casefold())))
                if tokens & _SENSITIVE_TOKENS:
                    raise ValueError("sensitive material is forbidden in model state")
            _validate_safe_state(child, path=(*path, str(key)))
        return
    if isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            _validate_safe_state(child, path=(*path, str(index)))
        return
    if isinstance(value, torch.Tensor):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError("model state tensors must be finite")
        return
    if value is None or isinstance(value, (bool, int, float, torch.dtype)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("model state values must be finite")
        return
    raise ValueError(f"unsupported safe model state at {'/'.join(path)}")


def _state_tensors(value: object):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _state_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _state_tensors(child)
    elif isinstance(value, torch.Tensor):
        yield value


def _has_symlink_component(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))
