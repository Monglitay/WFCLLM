"""Offline-only production dependency boundary for gate artifact pipelines."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from wfcllm.common.secrets import load_secret
from wfcllm.gate.pipeline import GateDataPipelineConfig, KeyBankSnapshot

_MAX_PUBLIC_JSON_BYTES = 1024 * 1024
PRODUCTION_GATE_ADAPTER_CONTRACT = "wfcllm-production-gate-adapter/v1"
PRODUCTION_GATE_ADAPTER_CAPABILITIES = frozenset(
    {
        "parse_statement_units",
        "generate_candidate_trajectories",
        "multi_key_lsh_probe_with_private_material",
        "split_groups",
        "audit_gate_data",
        "train_candidate",
    }
)


def _local_hf_adapter_factory(options: object):
    from wfcllm.gate.production import LocalHFGateRuntimeOptions, LocalHFProductionAdapter

    if not isinstance(options, LocalHFGateRuntimeOptions):
        raise ValueError("local-hf-v1 requires LocalHFGateRuntimeOptions")
    return LocalHFProductionAdapter(options)


_PRODUCTION_ADAPTER_FACTORIES = MappingProxyType(
    {"local-hf-v1": _local_hf_adapter_factory}
)
_CONSTRUCTION_TOKEN = object()


@dataclass(frozen=True)
class SecretReference:
    """A runtime-only secret source; values never enter public config/state."""

    secret_file: str | Path | None = None
    env_name: str | None = None


@runtime_checkable
class ProductionGateAdapter(Protocol):
    """Attested adapter surface for a future approved formal experiment.

    A trusted adapter must consume private key views synchronously and must not
    copy or retain their material.  Release can only zero buffers and views
    owned by this boundary; Python cannot revoke a malicious adapter's copy.
    """

    diagnostic_test_backend: bool
    adapter_contract_version: str
    capabilities: frozenset[str]

    def parse_statement_units(self, source_manifest, config): ...
    def generate_candidate_trajectories(self, parsed_units, config): ...
    def run_multi_key_lsh_probe(
        self, groups, *, training_keys, holdout_keys, config
    ): ...
    def split_groups(self, groups, config): ...
    def audit_gate_data(self, staging_dir, manifest): ...
    def train_candidate(self, **kwargs): ...


class _PrivateKeyBank:
    __slots__ = ("key_ids", "bank_id", "_materials", "_released")

    def __init__(
        self,
        *,
        key_ids: tuple[str, ...],
        bank_id: str,
        materials: tuple[bytes, ...],
    ) -> None:
        self.key_ids = key_ids
        self.bank_id = bank_id
        self._materials = tuple(bytearray(value) for value in materials)
        self._released = False

    def material_for(self, key_id: str) -> memoryview:
        if self._released:
            raise ValueError("private key material has been released")
        try:
            index = self.key_ids.index(key_id)
        except ValueError as exc:
            raise KeyError(key_id) from exc
        return memoryview(self._materials[index]).toreadonly()

    def temporary_digest_set(self) -> set[bytes]:
        if self._released:
            raise ValueError("private key material has been released")
        return {hashlib.sha256(material).digest() for material in self._materials}

    def release(self) -> None:
        if self._released:
            return
        for material in self._materials:
            material[:] = b"\x00" * len(material)
        self._released = True


class PrivateKeyBankView:
    """Trusted synchronous view; never serialize, copy, or retain key material.

    ``release_private_keys`` best-effort zeroizes boundary-owned buffers and
    existing memoryviews.  It cannot erase copies made by malicious Python.
    """

    __slots__ = ("_bank",)

    def __init__(self, bank: _PrivateKeyBank) -> None:
        self._bank = bank

    @property
    def key_ids(self) -> tuple[str, ...]:
        return self._bank.key_ids

    @property
    def bank_id(self) -> str:
        return self._bank.bank_id

    def material_for(self, key_id: str) -> memoryview:
        return self._bank.material_for(key_id)

    def __repr__(self) -> str:
        return f"PrivateKeyBankView(key_count={len(self.key_ids)}, material=<redacted>)"

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("pickling private key views is forbidden")


class LocalGateDependencies:
    """Local-only dependencies with an explicit production/test identity.

    Heavy parser/rewriter/trainer/validator adapters are optional injected local
    components.  In their absence this boundary reports the exact unavailable
    resource instead of downloading or falling back to a diagnostic backend.
    """

    def __init__(
        self,
        *,
        source_manifest: Path | None,
        training_keys: SecretReference,
        holdout_keys: SecretReference,
        base_model_path: Path | None,
        adapter: ProductionGateAdapter,
        adapter_name: str,
        diagnostic_test_backend: bool,
        _token: object,
    ) -> None:
        if _token is not _CONSTRUCTION_TOKEN:
            raise ValueError("LocalGateDependencies must be built by an approved factory")
        _validate_production_adapter(adapter)
        if type(diagnostic_test_backend) is not bool:
            raise ValueError("diagnostic_test_backend must be boolean")
        self.diagnostic_test_backend = diagnostic_test_backend
        self._source_manifest = source_manifest
        self._key_sources = {"training": training_keys, "holdout": holdout_keys}
        self._base_model_path = base_model_path
        self._adapter = adapter
        self.adapter_name = adapter_name
        self._private_banks: dict[str, _PrivateKeyBank] = {}

    def load_source_manifest(self, config: GateDataPipelineConfig) -> dict[str, Any]:
        if self._source_manifest is None:
            raise ValueError("local gate source manifest is required; downloads are disabled")
        return _read_public_json(self._source_manifest, "source manifest")

    def load_key_bank(
        self,
        *,
        role: str,
        expected_count: int,
        config: GateDataPipelineConfig,
    ) -> KeyBankSnapshot:
        try:
            reference = self._key_sources[role]
        except KeyError as exc:
            raise ValueError("gate key role must be training or holdout") from exc
        content = load_secret(
            secret_file=reference.secret_file,
            env_name=reference.env_name,
        )
        try:
            records = json.loads(
                content.decode("utf-8"), object_pairs_hook=_reject_duplicate_pairs
            )
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{role} key source must be UTF-8 JSON") from exc
        materials = _parse_private_key_materials(records, expected_count, role)
        if len(materials) != expected_count:
            raise ValueError(
                f"{role} key source must contain exactly {expected_count} records"
            )
        prefix = "train" if role == "training" else "holdout"
        key_ids = tuple(f"{prefix}-key-{index:03d}" for index in range(expected_count))
        bank_id = f"{role}-key-bank/v1:sha256:{hashlib.sha256(content).hexdigest()}"
        candidate_digests = {hashlib.sha256(material).digest() for material in materials}
        try:
            for other_role, bank in self._private_banks.items():
                if other_role == role:
                    continue
                other_digests = bank.temporary_digest_set()
                try:
                    if not candidate_digests.isdisjoint(other_digests):
                        raise ValueError(
                            "training and holdout key material must be disjoint"
                        )
                finally:
                    other_digests.clear()
        finally:
            candidate_digests.clear()
        previous = self._private_banks.pop(role, None)
        if previous is not None:
            previous.release()
        self._private_banks[role] = _PrivateKeyBank(
            key_ids=key_ids, bank_id=bank_id, materials=materials
        )
        return KeyBankSnapshot(key_ids, bank_id)

    def parse_statement_units(self, source_manifest, config):
        return self._delegate(
            "parse_statement_units",
            "local parser/rewriter dependency adapter is required; downloads are disabled",
            source_manifest,
            config,
        )

    def generate_candidate_trajectories(self, parsed_units, config):
        return self._delegate(
            "generate_candidate_trajectories",
            "local parser/rewriter dependency adapter is required; downloads are disabled",
            parsed_units,
            config,
        )

    def gate_data_selection_summary(self):
        summary = getattr(self._adapter, "gate_data_selection_summary", None)
        if not callable(summary):
            return None
        return summary()

    def run_multi_key_lsh_probe(self, groups, **kwargs):
        try:
            training = self._private_banks["training"]
            holdout = self._private_banks["holdout"]
        except KeyError as exc:
            raise ValueError("private key banks must be loaded before LSH probing") from exc
        training_ids = kwargs.pop("training_key_ids", None)
        holdout_ids = kwargs.pop("holdout_key_ids", None)
        if training_ids != training.key_ids or holdout_ids != holdout.key_ids:
            raise ValueError("public key IDs do not match the loaded private key banks")
        return self._adapter.run_multi_key_lsh_probe(
            groups,
            training_keys=PrivateKeyBankView(training),
            holdout_keys=PrivateKeyBankView(holdout),
            **kwargs,
        )

    def split_groups(self, groups, config):
        return self._delegate(
            "split_groups",
            "local gate-data split adapter is required",
            groups,
            config,
        )

    def audit_gate_data(self, staging_dir, manifest):
        return self._delegate(
            "audit_gate_data",
            "local gate-data audit adapter is required",
            staging_dir,
            manifest,
        )

    def train_candidate(self, **kwargs):
        if (
            self._base_model_path is None
            or self._base_model_path.is_symlink()
            or not self._base_model_path.is_dir()
        ):
            raise ValueError("local gate base model is missing; downloads are disabled")
        return self._delegate(
            "train_candidate",
            "local gate trainer dependency adapter is required; downloads are disabled",
            **kwargs,
        )

    def _delegate(self, name: str, error: str, *args, **kwargs):
        adapter = getattr(self._adapter, name, None)
        if not callable(adapter):  # defensive after construction attestation
            raise ValueError(error)
        return adapter(*args, **kwargs)

    def release_private_keys(self) -> None:
        for bank in self._private_banks.values():
            bank.release()
        self._private_banks.clear()

    def __del__(self) -> None:
        try:
            self.release_private_keys()
        except Exception:
            pass


def build_local_gate_dependencies(
    *,
    source_manifest: Path | None,
    training_key_file: str | Path | None,
    training_key_env: str | None,
    holdout_key_file: str | Path | None,
    holdout_key_env: str | None,
    base_model_path: Path | None,
    adapter_name: str | None = None,
    adapter_options: object | None = None,
) -> LocalGateDependencies:
    """Construct from a static whitelist; arbitrary imports/objects are forbidden."""

    if adapter_name is None:
        raise ValueError(
            "an allowlisted production gate adapter is required"
        )
    factory = _PRODUCTION_ADAPTER_FACTORIES.get(adapter_name)
    if factory is None:
        raise ValueError("adapter_name must select an allowlisted production gate adapter")
    adapter = factory(adapter_options)
    return _build_dependencies(
        source_manifest=source_manifest,
        training_key_file=training_key_file,
        training_key_env=training_key_env,
        holdout_key_file=holdout_key_file,
        holdout_key_env=holdout_key_env,
        base_model_path=base_model_path,
        adapter=adapter,
        adapter_name=adapter_name,
        diagnostic_test_backend=False,
    )


def build_trusted_test_gate_dependencies(
    *,
    source_manifest: Path | None,
    training_key_file: str | Path | None,
    training_key_env: str | None,
    holdout_key_file: str | Path | None,
    holdout_key_env: str | None,
    base_model_path: Path | None,
    adapter: ProductionGateAdapter,
) -> LocalGateDependencies:
    """Explicit test-only seam with complete diagnostic artifact identity."""

    return _build_dependencies(
        source_manifest=source_manifest,
        training_key_file=training_key_file,
        training_key_env=training_key_env,
        holdout_key_file=holdout_key_file,
        holdout_key_env=holdout_key_env,
        base_model_path=base_model_path,
        adapter=adapter,
        adapter_name="trusted-test",
        diagnostic_test_backend=True,
    )


def _build_dependencies(
    *,
    source_manifest: Path | None,
    training_key_file: str | Path | None,
    training_key_env: str | None,
    holdout_key_file: str | Path | None,
    holdout_key_env: str | None,
    base_model_path: Path | None,
    adapter: ProductionGateAdapter,
    adapter_name: str,
    diagnostic_test_backend: bool,
) -> LocalGateDependencies:
    _validate_production_adapter(adapter)

    return LocalGateDependencies(
        source_manifest=source_manifest,
        training_keys=SecretReference(training_key_file, training_key_env),
        holdout_keys=SecretReference(holdout_key_file, holdout_key_env),
        base_model_path=base_model_path,
        adapter=adapter,
        adapter_name=adapter_name,
        diagnostic_test_backend=diagnostic_test_backend,
        _token=_CONSTRUCTION_TOKEN,
    )


def _validate_production_adapter(adapter: object) -> None:
    if not isinstance(adapter, ProductionGateAdapter):
        raise ValueError("production adapter attestation is incomplete")
    if (
        getattr(adapter, "diagnostic_test_backend", None) is not False
        or getattr(adapter, "adapter_contract_version", None)
        != PRODUCTION_GATE_ADAPTER_CONTRACT
        or getattr(adapter, "capabilities", None)
        != PRODUCTION_GATE_ADAPTER_CAPABILITIES
    ):
        raise ValueError("production adapter attestation is invalid")
    for capability in PRODUCTION_GATE_ADAPTER_CAPABILITIES - {
        "multi_key_lsh_probe_with_private_material"
    }:
        if not callable(getattr(adapter, capability, None)):
            raise ValueError("production adapter attestation is incomplete")
    if not callable(getattr(adapter, "run_multi_key_lsh_probe", None)):
        raise ValueError("production adapter attestation is incomplete")


def _parse_private_key_materials(
    records: object, expected_count: int, role: str
) -> tuple[bytes, ...]:
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{role} key source must contain exactly {expected_count} records")
    materials: list[bytes] = []
    for index, record in enumerate(records):
        material: object
        if isinstance(record, str):
            material = record
        elif isinstance(record, dict) and set(record) == {"id", "material"}:
            material = record["material"]
        else:
            raise ValueError(f"{role} key record {index} schema mismatch")
        if not isinstance(material, str) or not material:
            raise ValueError(f"{role} key record {index} material must be non-empty text")
        materials.append(material.encode("utf-8"))
    if len(set(materials)) != len(materials):
        raise ValueError(f"duplicate {role} key material is forbidden")
    return tuple(materials)


def _reject_duplicate_pairs(pairs):
    output = {}
    for key, value in pairs:
        if key in output:
            raise ValueError("duplicate private key JSON field")
        output[key] = value
    return output


def _read_public_json(path: Path, name: str) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise ValueError(f"local {name} path must be a pathlib.Path")
    absolute = path if path.is_absolute() else Path.cwd() / path
    for candidate in (absolute, *absolute.parents):
        if candidate.is_symlink():
            raise ValueError(f"local {name} path cannot traverse symlinks")
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_PUBLIC_JSON_BYTES:
            raise ValueError(f"local {name} must be a bounded regular file")
        chunks = bytearray()
        while chunk := os.read(
            descriptor,
            min(1024 * 1024, _MAX_PUBLIC_JSON_BYTES + 1 - len(chunks)),
        ):
            chunks.extend(chunk)
            if len(chunks) > _MAX_PUBLIC_JSON_BYTES:
                raise ValueError(f"local {name} exceeds the size limit")
        raw = bytes(chunks)
        after = os.fstat(descriptor)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"local {name} changed while reading")
    except OSError as exc:
        raise ValueError(f"local {name} is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    def reject_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key in local {name}")
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"local {name} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError(f"local {name} must be a JSON object")
    return value
