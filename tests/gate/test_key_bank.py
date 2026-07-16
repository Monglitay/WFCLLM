from __future__ import annotations

import hashlib
import json
import pickle

import pytest

import wfcllm.gate.key_bank as key_bank_module
from wfcllm.gate.key_bank import TrainingKeyBank


def test_direct_constructor_is_not_a_public_key_bank_factory() -> None:
    assert not hasattr(key_bank_module, "_FACTORY_TOKEN")
    assert not hasattr(TrainingKeyBank, "_from_records")

    with pytest.raises(ValueError, match="load.*from_records"):
        TrainingKeyBank()
    with pytest.raises(ValueError, match="factory"):
        TrainingKeyBank((b"direct-alpha",), file_sha256="a" * 64)
    with pytest.raises(ValueError, match="load.*from_records"):
        TrainingKeyBank(
            (b"direct-alpha",),
            digest="a" * 64,
        )


def test_public_api_cannot_forge_same_manifest_identity_for_different_keys() -> None:
    first = TrainingKeyBank.from_records(
        [{"id": "training-source-first", "material": b"first-secret"}],
        expected_count=1,
    )
    second = TrainingKeyBank.from_records(
        [{"id": "training-source-second", "material": b"second-secret"}],
        expected_count=1,
    )

    assert first.public_manifest()["bank_id"] != second.public_manifest()["bank_id"]
    forged_digest = first.public_manifest()["key_file_sha256"]
    with pytest.raises(ValueError, match="factory"):
        TrainingKeyBank((b"second-secret",), file_sha256=forged_digest)


def test_from_records_identity_is_sha256_of_canonical_private_bytes() -> None:
    material = b"canonical-secret"
    bank = TrainingKeyBank.from_records(
        [{"id": "training-source-one", "material": material}], expected_count=1
    )
    canonical = (
        b"wfcllm-training-key-bank/private/v1\0"
        + len(material).to_bytes(8, "big")
        + material
    )
    expected = hashlib.sha256(canonical).hexdigest()

    assert bank.public_manifest()["key_file_sha256"] == expected
    assert bank.public_manifest()["bank_id"].endswith(expected)


def test_key_bank_slots_cannot_be_assigned_or_deleted() -> None:
    bank = TrainingKeyBank.from_records(
        [{"id": "training-source-one", "material": "secret"}], expected_count=1
    )
    for name in ("_materials", "_key_ids", "_key_file_sha256", "_sealed"):
        with pytest.raises(AttributeError, match="immutable"):
            setattr(bank, name, object())
        with pytest.raises(AttributeError, match="immutable"):
            delattr(bank, name)


def test_key_bank_cannot_be_pickled() -> None:
    bank = TrainingKeyBank.from_records(
        [{"id": "training-source-one", "material": "secret"}], expected_count=1
    )
    with pytest.raises(TypeError, match="pickl"):
        pickle.dumps(bank)


def test_key_bank_never_serializes_raw_keys(tmp_path) -> None:
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"keys": ["alpha", "beta"]}), encoding="utf-8")
    bank = TrainingKeyBank.load(key_file, expected_count=2)

    manifest_dict = bank.public_manifest()
    manifest = json.dumps(manifest_dict, sort_keys=True)

    assert "alpha" not in manifest
    assert "beta" not in manifest
    assert bank.key_ids == ("train-key-000", "train-key-001")
    assert set(manifest_dict) == {
        "schema_version",
        "bank_id",
        "key_count",
        "key_ids",
        "key_file_sha256",
    }
    assert manifest_dict["key_file_sha256"] == hashlib.sha256(
        key_file.read_bytes()
    ).hexdigest()


def test_raw_key_material_is_available_only_via_in_process_lookup(tmp_path) -> None:
    key_file = tmp_path / "keys.json"
    key_file.write_text(json.dumps({"keys": ["alpha"]}), encoding="utf-8")
    bank = TrainingKeyBank.load(key_file, expected_count=1)

    assert bank.material_for("train-key-000") == b"alpha"
    assert "alpha" not in repr(bank)
    with pytest.raises(KeyError, match="train-key-999"):
        bank.material_for("train-key-999")


def test_deployment_key_cannot_be_loaded_as_training_key() -> None:
    with pytest.raises(ValueError, match="deployment"):
        TrainingKeyBank.from_records(
            [{"id": "deployment-key", "material": "forbidden"}], expected_count=1
        )


@pytest.mark.parametrize(
    ("records", "expected_count", "message"),
    [
        ([], 1, "expected_count"),
        ([{"id": "training-source-a", "material": ""}], 1, "empty"),
        (
            [
                {"id": "training-source-a", "material": "duplicate"},
                {"id": "training-source-b", "material": "duplicate"},
            ],
            2,
            "duplicate",
        ),
        ([{"id": "training-source-a", "material": "key"}], 0, "expected_count"),
    ],
)
def test_key_bank_rejects_invalid_records(
    records: list[dict[str, str]], expected_count: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        TrainingKeyBank.from_records(records, expected_count=expected_count)


def test_key_file_contract_is_strict(tmp_path) -> None:
    key_file = tmp_path / "keys.json"
    key_file.write_text(
        json.dumps({"keys": ["alpha"], "deployment_key": "forbidden"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only the keys field"):
        TrainingKeyBank.load(key_file, expected_count=1)


@pytest.mark.parametrize(
    "payload",
    [
        '{"keys":["alpha"],"keys":["beta"]}',
        '{"keys":["alpha"],"metadata":{"id":1,"id":2}}',
    ],
)
def test_key_file_rejects_duplicate_json_object_keys(tmp_path, payload: str) -> None:
    key_file = tmp_path / "keys.json"
    key_file.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        TrainingKeyBank.load(key_file, expected_count=1)


def test_different_key_files_have_different_public_bank_ids(tmp_path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(json.dumps({"keys": ["alpha"]}), encoding="utf-8")
    second.write_text(json.dumps({"keys": ["beta"]}), encoding="utf-8")

    first_bank = TrainingKeyBank.load(first, expected_count=1)
    second_bank = TrainingKeyBank.load(second, expected_count=1)

    assert first_bank.public_manifest()["bank_id"] != second_bank.public_manifest()[
        "bank_id"
    ]


def test_from_records_accepts_bytes_without_rendering_them_publicly() -> None:
    bank = TrainingKeyBank.from_records(
        [{"id": "training-source-source-key", "material": b"raw-secret"}],
        expected_count=1,
    )

    assert bank.material_for("train-key-000") == b"raw-secret"
    assert "raw-secret" not in json.dumps(bank.public_manifest(), sort_keys=True)


@pytest.mark.parametrize(
    "source_id",
    [
        "deployment-key",
        "deploy-ment-key",
        "deplo\u200byment-key",
        "ｄｅｐｌｏｙｍｅｎｔ-key",
    ],
)
def test_from_records_rejects_obfuscated_deployment_source_ids(
    source_id: str,
) -> None:
    with pytest.raises(ValueError, match="deployment"):
        TrainingKeyBank.from_records(
            [{"id": source_id, "material": "secret"}], expected_count=1
        )


@pytest.mark.parametrize("source_id", ["prod-key", "raw-key", "source-key"])
def test_from_records_rejects_non_training_source_namespace(source_id: str) -> None:
    with pytest.raises(ValueError, match="training-source"):
        TrainingKeyBank.from_records(
            [{"id": source_id, "material": "secret"}], expected_count=1
        )


def test_from_records_accepts_ascii_training_source_namespace() -> None:
    bank = TrainingKeyBank.from_records(
        [
            {
                "id": "training-source-Corpus_1.release-2",
                "material": "secret",
            }
        ],
        expected_count=1,
    )
    assert bank.key_ids == ("train-key-000",)
