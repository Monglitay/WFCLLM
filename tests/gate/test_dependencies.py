from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from wfcllm.gate.dependencies import (
    PRODUCTION_GATE_ADAPTER_CAPABILITIES,
    PRODUCTION_GATE_ADAPTER_CONTRACT,
    build_local_gate_dependencies,
    build_trusted_test_gate_dependencies,
)


class ControlledProductionTestAdapter:
    diagnostic_test_backend = False
    adapter_contract_version = PRODUCTION_GATE_ADAPTER_CONTRACT
    capabilities = PRODUCTION_GATE_ADAPTER_CAPABILITIES

    def __init__(self) -> None:
        self.training_observation_digests: tuple[str, ...] | None = None
        self.holdout_observation_digests: tuple[str, ...] | None = None
        self.training_view = None
        self.existing_material_view = None
        self.probe_read_count = 0

    def parse_statement_units(self, source_manifest, config):
        return (source_manifest,)

    def generate_candidate_trajectories(self, parsed_units, config):
        return ()

    def run_multi_key_lsh_probe(
        self, groups, *, training_keys, holdout_keys, config
    ):
        self.training_view = training_keys
        self.training_observation_digests = tuple(
            hashlib.sha256(training_keys.material_for(key_id)).hexdigest()
            for key_id in training_keys.key_ids
        )
        self.holdout_observation_digests = tuple(
            hashlib.sha256(holdout_keys.material_for(key_id)).hexdigest()
            for key_id in holdout_keys.key_ids
        )
        self.existing_material_view = training_keys.material_for(training_keys.key_ids[0])
        assert self.existing_material_view.readonly is True
        # A formal group has 3 windows x 7 candidates and probes all 32+8 keys.
        # The view must remain readable for the complete 21 x 40 operation.
        for _candidate in range(21):
            for key_id in training_keys.key_ids:
                training_keys.material_for(key_id)
                self.probe_read_count += 1
            for key_id in holdout_keys.key_ids:
                holdout_keys.material_for(key_id)
                self.probe_read_count += 1
        return groups

    def split_groups(self, groups, config):
        return groups

    def audit_gate_data(self, staging_dir, manifest):
        return None

    def train_candidate(self, **kwargs):
        return {}

    def validate_candidate(self, **kwargs):
        return None


class LaunderedFakeAdapter:
    diagnostic_test_backend = True


def _key_files(
    tmp_path: Path, *, holdout_materials: list[str] | None = None
) -> tuple[Path, Path]:
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    training.write_text(json.dumps([f"training-secret-{index}" for index in range(32)]))
    holdout.write_text(
        json.dumps(
            holdout_materials
            if holdout_materials is not None
            else [f"holdout-secret-{index}" for index in range(8)]
        )
    )
    return training, holdout


def _trusted_dependencies(tmp_path: Path, *, holdout_materials: list[str] | None = None):
    training, holdout = _key_files(tmp_path, holdout_materials=holdout_materials)
    return build_trusted_test_gate_dependencies(
        source_manifest=None,
        training_key_file=training,
        training_key_env=None,
        holdout_key_file=holdout,
        holdout_key_env=None,
        base_model_path=None,
        adapter=ControlledProductionTestAdapter(),
    )


def test_arbitrary_or_diagnostic_adapter_cannot_be_laundered_as_formal(tmp_path):
    training, holdout = _key_files(tmp_path)
    with pytest.raises(ValueError, match="production adapter attestation"):
        build_trusted_test_gate_dependencies(
            source_manifest=None,
            training_key_file=training,
            training_key_env=None,
            holdout_key_file=holdout,
            holdout_key_env=None,
            base_model_path=None,
            adapter=LaunderedFakeAdapter(),
        )


def test_production_factory_uses_only_static_adapter_names(tmp_path):
    training, holdout = _key_files(tmp_path)
    with pytest.raises(ValueError, match="allowlisted production gate adapter"):
        build_local_gate_dependencies(
            source_manifest=None,
            training_key_file=training,
            training_key_env=None,
            holdout_key_file=holdout,
            holdout_key_env=None,
            base_model_path=None,
            adapter_name="tests.dynamic_adapter",
        )


def test_trusted_probe_receives_private_material_and_release_zeroizes(tmp_path):
    training, holdout = _key_files(tmp_path)
    adapter = ControlledProductionTestAdapter()
    dependencies = build_trusted_test_gate_dependencies(
        source_manifest=None,
        training_key_file=training,
        training_key_env=None,
        holdout_key_file=holdout,
        holdout_key_env=None,
        base_model_path=None,
        adapter=adapter,
    )
    training_public = dependencies.load_key_bank(
        role="training", expected_count=32, config=object()
    )
    holdout_public = dependencies.load_key_bank(
        role="holdout", expected_count=8, config=object()
    )

    assert dependencies.run_multi_key_lsh_probe(
        ("group",),
        training_key_ids=training_public.key_ids,
        holdout_key_ids=holdout_public.key_ids,
        config=object(),
    ) == ("group",)
    assert adapter.training_observation_digests[0] == hashlib.sha256(
        b"training-secret-0"
    ).hexdigest()
    assert adapter.holdout_observation_digests[0] == hashlib.sha256(
        b"holdout-secret-0"
    ).hexdigest()
    assert adapter.probe_read_count == 21 * 40

    dependencies.release_private_keys()
    assert adapter.existing_material_view.readonly is True
    assert bytes(adapter.existing_material_view) == b"\x00" * len(b"training-secret-0")
    with pytest.raises(ValueError, match="released"):
        adapter.training_view.material_for("train-key-000")


@pytest.mark.parametrize(
    "holdout_materials",
    [
        [f"training-secret-{index}" for index in range(8)],
        ["training-secret-0", *[f"holdout-secret-{index}" for index in range(1, 8)]],
    ],
    ids=("all-eight-copied", "one-of-eight-copied"),
)
def test_cross_bank_private_material_overlap_is_rejected_without_disclosure(
    tmp_path, holdout_materials
):
    dependencies = _trusted_dependencies(
        tmp_path, holdout_materials=holdout_materials
    )
    dependencies.load_key_bank(role="training", expected_count=32, config=object())

    with pytest.raises(ValueError, match="key material must be disjoint") as exc_info:
        dependencies.load_key_bank(role="holdout", expected_count=8, config=object())

    error = str(exc_info.value)
    copied_material = holdout_materials[0]
    assert copied_material not in error
    assert hashlib.sha256(copied_material.encode()).hexdigest() not in error


def test_distinct_cross_bank_private_material_is_accepted(tmp_path):
    dependencies = _trusted_dependencies(tmp_path)

    training = dependencies.load_key_bank(
        role="training", expected_count=32, config=object()
    )
    holdout = dependencies.load_key_bank(
        role="holdout", expected_count=8, config=object()
    )

    assert set(training.key_ids).isdisjoint(holdout.key_ids)
