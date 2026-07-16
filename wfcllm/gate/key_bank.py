"""In-process training-key storage with a non-secret public manifest."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Mapping, Sequence

from wfcllm.method.contracts import reject_quality_proxy_fields

TRAINING_KEY_BANK_MANIFEST_VERSION = "wfcllm-training-key-bank-manifest/v1"
_TRAINING_SOURCE_ID_RE = re.compile(
    r"training-source-[A-Za-z0-9][A-Za-z0-9._-]*\Z", re.ASCII
)


class TrainingKeyBank:
    """Own raw training-key bytes without exposing them through public metadata."""

    __slots__ = ("_key_file_sha256", "_key_ids", "_materials", "_sealed")

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise ValueError(
            "direct construction is forbidden; use load() or from_records() factory"
        )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TrainingKeyBank is immutable")
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        raise AttributeError("TrainingKeyBank is immutable")

    @classmethod
    def load(cls, path: str | Path, *, expected_count: int) -> TrainingKeyBank:
        """Load the strict ``{"keys": [str, ...]}`` private file format."""

        key_path = Path(path)
        raw_file = key_path.read_bytes()
        try:
            payload = json.loads(
                raw_file.decode("utf-8"),
                object_pairs_hook=_reject_duplicate_object_pairs,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("training key file must be UTF-8 JSON") from exc
        if not isinstance(payload, dict) or set(payload) != {"keys"}:
            raise ValueError("training key file may contain only the keys field")
        keys = payload["keys"]
        if not isinstance(keys, list) or any(not isinstance(key, str) for key in keys):
            raise ValueError("training key file keys must be a list of strings")
        records = [
            {
                "id": f"training-source-private-file-{index:03d}",
                "material": material,
            }
            for index, material in enumerate(keys)
        ]
        materials = _validate_records(records, expected_count=expected_count)
        file_sha256 = hashlib.sha256(raw_file).hexdigest()
        bank = object.__new__(cls)
        object.__setattr__(bank, "_materials", materials)
        object.__setattr__(
            bank,
            "_key_ids",
            tuple(
                f"train-key-{index:03d}" for index in range(len(materials))
            ),
        )
        object.__setattr__(bank, "_key_file_sha256", file_sha256)
        object.__setattr__(bank, "_sealed", True)
        return bank

    @classmethod
    def from_records(
        cls,
        records: Sequence[Mapping[str, object]],
        *,
        expected_count: int,
    ) -> TrainingKeyBank:
        """Construct a private bank from already-loaded in-process records."""

        materials = _validate_records(records, expected_count=expected_count)
        canonical_private_bytes = _canonical_material_bytes(materials)
        file_sha256 = hashlib.sha256(canonical_private_bytes).hexdigest()
        bank = object.__new__(cls)
        object.__setattr__(bank, "_materials", materials)
        object.__setattr__(
            bank,
            "_key_ids",
            tuple(
                f"train-key-{index:03d}" for index in range(len(materials))
            ),
        )
        object.__setattr__(bank, "_key_file_sha256", file_sha256)
        object.__setattr__(bank, "_sealed", True)
        return bank

    @property
    def key_ids(self) -> tuple[str, ...]:
        return self._key_ids

    def material_for(self, key_id: str) -> bytes:
        """Return a private byte copy for an explicit in-process key lookup."""

        try:
            index = self._key_ids.index(key_id)
        except ValueError as exc:
            raise KeyError(key_id) from exc
        return bytes(self._materials[index])

    def public_manifest(self) -> dict[str, Any]:
        """Return the complete allowlisted public representation of this bank."""

        manifest: dict[str, Any] = {
            "schema_version": TRAINING_KEY_BANK_MANIFEST_VERSION,
            "bank_id": f"training-key-bank/v1:sha256:{self._key_file_sha256}",
            "key_count": len(self._key_ids),
            "key_ids": list(self._key_ids),
            "key_file_sha256": self._key_file_sha256,
        }
        reject_quality_proxy_fields(manifest)
        return manifest

    def __repr__(self) -> str:
        return f"TrainingKeyBank(key_count={len(self._key_ids)}, raw_keys=<redacted>)"

    def __reduce_ex__(self, protocol: int) -> object:
        raise TypeError("pickling TrainingKeyBank is forbidden")


def _reject_duplicate_object_pairs(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key}")
        output[key] = value
    return output


def _validate_records(
    records: Sequence[Mapping[str, object]], *, expected_count: int
) -> tuple[bytes, ...]:
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    if isinstance(records, (str, bytes)) or not isinstance(records, Sequence):
        raise ValueError("records must be a sequence of key records")
    if len(records) != expected_count:
        raise ValueError(
            f"expected_count is {expected_count}, but found {len(records)} records"
        )

    materials: list[bytes] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping) or set(record) != {"id", "material"}:
            raise ValueError(
                f"key record {index} must contain exactly id and material"
            )
        source_id = record["id"]
        if not isinstance(source_id, str) or not source_id:
            raise ValueError(f"key record {index} id must be a non-empty string")
        normalized_id = unicodedata.normalize("NFKC", source_id).casefold()
        collapsed_id = "".join(
            character for character in normalized_id if character.isalnum()
        )
        if "deployment" in collapsed_id:
            raise ValueError("deployment keys cannot be loaded as training keys")
        if _TRAINING_SOURCE_ID_RE.fullmatch(source_id) is None:
            raise ValueError(
                "key record id must match the ASCII training-source-* namespace"
            )
        material = record["material"]
        if isinstance(material, str):
            raw_material = material.encode("utf-8")
        elif isinstance(material, bytes):
            raw_material = bytes(material)
        else:
            raise ValueError(f"key record {index} material must be str or bytes")
        if not raw_material:
            raise ValueError(f"key record {index} has empty key material")
        materials.append(raw_material)

    if len(set(materials)) != len(materials):
        raise ValueError("duplicate training key material is forbidden")
    return tuple(materials)


def _canonical_material_bytes(materials: tuple[bytes, ...]) -> bytes:
    output = bytearray(b"wfcllm-training-key-bank/private/v1\0")
    for material in materials:
        output.extend(len(material).to_bytes(8, "big"))
        output.extend(material)
    return bytes(output)
