from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types
import weakref
from collections import OrderedDict
from functools import lru_cache
from dataclasses import FrozenInstanceError, replace

import pytest
import torch
from torch import nn

import wfcllm.gate.pipeline as gate_pipeline
from wfcllm.gate.feasibility import FEASIBILITY_THRESHOLD_ITEMS, FeasibilityGroup, evaluate_gate_data_feasibility
from wfcllm.gate.data import LshProbeResult
from wfcllm.gate.dependencies import (
    PRODUCTION_GATE_ADAPTER_CAPABILITIES,
    PRODUCTION_GATE_ADAPTER_CONTRACT,
    build_trusted_test_gate_dependencies,
)
from wfcllm.gate.schema import CandidateObservation
from wfcllm.gate.labels import build_gate_labels
from wfcllm.gate.pipeline import (
    GateDataPipelineConfig,
    CandidateTrajectoryGroup,
    GateGroupIdentity,
    LabeledGroup,
    ParsedWindowGroup,
    ProbedGroup,
    SplitGroup,
    GatePipelineGroup,
    GateTrainPipelineConfig,
    KeyBankSnapshot,
    run_gate_data,
    run_gate_train,
    _validate_formal_candidate_tree,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class TinyPipelineGate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 2)

    def forward(self, *, input_features: torch.Tensor) -> object:
        logits = self.projection(input_features)
        return types.SimpleNamespace(close_logits=logits[:, 0], suitable_logits=logits[:, 1])


def _group(
    index: int, *, scale: str = "pilot", max_units: int = 3
) -> GatePipelineGroup:
    suitable = index < (15 if scale == "pilot" else 150)
    split = "validation" if index % 10 == 0 else "test" if index % 10 == 1 else "train"
    lengths = tuple(range(1, max_units + 1))
    observations, probes = _evidence_templates(suitable)
    return GatePipelineGroup(
        group_id=f"group-{index:05d}",
        split_group_id=f"repository:repo-{index:05d}",
        split=split,
        suitable_target=suitable,
        close_target=True,
        window_lengths=lengths,
        statement_family=("assignment", "branch", "loop", "return")[index % 4],
        r1_success_rate=0.25 if suitable else 0.0,
        r3_success_rate=0.625 if suitable else 0.0,
        holdout_success_rate=0.625 if suitable else 0.0,
        repository_id=f"repo-{index:05d}",
        task_id=f"task-{index:05d}",
        generation_model_id=f"model-{index % 4}",
        structural_invalid_rate=0.0,
        numeric_instability_rate=0.0,
        first_hit_candidate_position=0 if suitable else None,
        candidate_indices_by_window_length={
            length: tuple(range(4)) for length in lengths
        },
        observed_training_key_ids=tuple(f"train-key-{i:03d}" for i in range(32)),
        observed_holdout_key_ids=tuple(f"holdout-key-{i:03d}" for i in range(8)),
        candidate_observations_by_length={
            str(length): observations[str(length)] for length in lengths
        },
        probe_results_by_length={
            str(length): probes[str(length)] for length in lengths
        },
        row={"schema_version": "wfcllm-gate-data/v1", "group_id": f"group-{index:05d}", "split": split},
    )


def test_pipeline_stage_types_accept_only_the_configured_max_units_prefix() -> None:
    group = _group(0, max_units=1)
    identity = GateGroupIdentity(
        group_id=group.group_id,
        split_group_id=group.split_group_id,
        repository_id=group.repository_id,
        task_id=group.task_id,
        generation_model_id=group.generation_model_id,
        statement_family=group.statement_family,
        window_lengths=(1,),
    )
    parsed = ParsedWindowGroup(identity, "python-statement-window/v1")
    trajectory = CandidateTrajectoryGroup(
        parsed, {1: tuple(range(4))}
    )
    probed = ProbedGroup(
        trajectory,
        group.candidate_observations_by_length,
        group.probe_results_by_length,
    )
    labels = build_gate_labels(
        group.candidate_observations_by_length["1"],
        training_key_count=32,
    )

    labeled = LabeledGroup(probed, {1: labels}, 0.5, 0)

    assert labeled.identity.window_lengths == (1,)


@lru_cache(maxsize=2)
def _evidence_templates(suitable: bool):
    training_ids = tuple(f"train-key-{index:03d}" for index in range(32))
    holdout_ids = tuple(f"holdout-key-{index:03d}" for index in range(8))
    observations_by_length = {}
    probes_by_length = {}
    for length in (1, 2, 3):
        observations = []
        probes = []
        for candidate_index in range(4):
            training_hits = set()
            holdout_hits = set()
            if suitable:
                if candidate_index == 0:
                    training_hits.update(training_ids[:8])
                    holdout_hits.update(holdout_ids[:2])
                elif candidate_index == 2:
                    training_hits.update(training_ids[8:20])
                    holdout_hits.update(holdout_ids[2:5])
            raw_results = {
                key_id: LshProbeResult(
                    signature=(1, 0), margin=1.0, hit=key_id in training_hits | holdout_hits,
                    stable=True, stable_across_precision_modes=True,
                    stable_across_batch_modes=True,
                )
                for key_id in (*training_ids, *holdout_ids)
            }
            observations.append(CandidateObservation(
                candidate_index=candidate_index,
                code=f"candidate_{candidate_index} = {candidate_index}",
                parse_status="ok", unit_count=length, same_parent_scope=True,
                boundary_span=(0, 1), stable_across_precision_modes=True,
                stable_across_batch_modes=True,
                lsh_by_key_id={
                    key_id: {"hit": raw_results[key_id].hit, "stable": True, "margin": 1.0}
                    for key_id in training_ids
                },
                generation_seed_id=f"seed-{candidate_index}",
                rewrite_config_id="rewrite-v1", lsh_signature=(1, 0),
                semantic_reference_cosine=(
                    1.0 if candidate_index == 0 else 0.95
                ),
                semantic_preservation_passed=True,
            ))
            probes.append(types.MappingProxyType(raw_results))
        observations_by_length[str(length)] = tuple(observations)
        probes_by_length[str(length)] = tuple(probes)
    return observations_by_length, probes_by_length


class FakeDependencies:
    def __init__(
        self,
        groups: tuple[GatePipelineGroup, ...],
        *,
        diagnostic: bool = True,
    ) -> None:
        self.groups = groups
        self.diagnostic_test_backend = diagnostic
        self.calls: list[str] = []
        self.train_called = False
        self.generated_count = 0

    def load_source_manifest(self, config):
        self.calls.append("source")
        return {"schema_version": "wfcllm-gate-source-manifest/v1", "sources": [{"source_id": "safe"}]}

    def load_key_bank(self, *, role, expected_count, config):
        self.calls.append(role)
        prefix = "training" if role == "training" else "holdout"
        return KeyBankSnapshot(
            key_ids=tuple(f"{'train' if role == 'training' else 'holdout'}-key-{i:03d}" for i in range(expected_count)),
            bank_id=f"{prefix}-key-bank/v1:sha256:{_sha(role)}",
        )

    def parse_statement_units(self, source_manifest, config):
        self.calls.append("parse")
        return ("parsed",)

    def generate_candidate_trajectories(self, parsed_sources, config):
        self.calls.append("generate")
        def counted_groups():
            for group in self.groups:
                self.generated_count += 1
                yield group

        return counted_groups()

    def gate_data_selection_summary(self):
        return {
            "algorithm_version": "test-source-selection/v1",
            "candidate_window_count": self.generated_count,
            "selected_group_count": self.generated_count,
            "selection_sha256": _sha("selection"),
        }

    def run_multi_key_lsh_probe(self, groups, *, training_key_ids, holdout_key_ids, config):
        self.calls.append("probe")
        assert len(training_key_ids) == 32 and len(holdout_key_ids) == 8
        return groups

    def split_groups(self, groups, config):
        self.calls.append("split")
        return groups

    def audit_gate_data(self, staging_dir, manifest):
        self.calls.append("audit")

    def train_candidate(self, *, config, data_manifest, data_jsonl, output_dir, learning_curve_plan):
        self.train_called = True
        output_dir.mkdir(parents=True)
        model = TinyPipelineGate().eval()
        torch.save(model.state_dict(), output_dir / "gate_float.pt")
        tokenizer = output_dir / "tokenizer"
        tokenizer.mkdir()
        (tokenizer / "tokenizer.json").write_text('{"version":1}\n', encoding="utf-8")
        return {"backend": "fake", "candidate_sha256": _sha("candidate")}



class ControlledProductionPipelineAdapter:
    diagnostic_test_backend = False
    adapter_contract_version = PRODUCTION_GATE_ADAPTER_CONTRACT
    capabilities = PRODUCTION_GATE_ADAPTER_CAPABILITIES

    def __init__(self, backend: FakeDependencies) -> None:
        self.backend = backend
        self.training_view = None
        self.training_observation_digest = None
        self.probe_count = 0

    def parse_statement_units(self, source_manifest, config):
        return self.backend.parse_statement_units(source_manifest, config)

    def generate_candidate_trajectories(self, parsed_units, config):
        return self.backend.generate_candidate_trajectories(parsed_units, config)

    def run_multi_key_lsh_probe(
        self, groups, *, training_keys, holdout_keys, config
    ):
        self.training_view = training_keys
        self.probe_count += 1
        training_material = training_keys.material_for(training_keys.key_ids[0])
        holdout_material = holdout_keys.material_for(holdout_keys.key_ids[0])
        assert training_material.readonly is True
        assert holdout_material.readonly is True
        self.training_observation_digest = hashlib.sha256(training_material).hexdigest()
        assert self.training_observation_digest == hashlib.sha256(b"training-0").hexdigest()
        assert hashlib.sha256(holdout_material).hexdigest() == hashlib.sha256(
            b"holdout-0"
        ).hexdigest()
        return self.backend.run_multi_key_lsh_probe(
            groups,
            training_key_ids=training_keys.key_ids,
            holdout_key_ids=holdout_keys.key_ids,
            config=config,
        )

    def split_groups(self, groups, config):
        return self.backend.split_groups(groups, config)

    def audit_gate_data(self, staging_dir, manifest):
        return self.backend.audit_gate_data(staging_dir, manifest)

    def train_candidate(self, **kwargs):
        return self.backend.train_candidate(**kwargs)


def _data_config(tmp_path: Path, *, scale: str = "pilot", pilot: Path | None = None) -> GateDataPipelineConfig:
    return GateDataPipelineConfig(
        output_root=tmp_path,
        scale=scale,
        config_hash=_sha("config"),
        parser_contract="wfcllm-window/v1",
        rewriter_config_hash=_sha("rewriter"),
        semantic_encoder_hash=_sha("encoder"),
        lsh_config_hash=_sha("lsh"),
        feasibility_contract="gate-data-feasibility/v1",
        feasibility_thresholds=FEASIBILITY_THRESHOLD_ITEMS,
        pilot_feasibility_path=pilot,
    )


def test_gate_data_writes_manifest_and_grouped_jsonl_in_fixed_order(tmp_path: Path) -> None:
    deps = FakeDependencies(tuple(_group(i) for i in range(100)))
    result = run_gate_data(_data_config(tmp_path), deps)
    assert result.group_count == 100
    assert result.manifest_path.exists()
    assert result.manifest["human_eval_included"] is False
    assert result.manifest["rewrite_count"] == 3
    assert result.manifest["selection_summary"] == {
        "algorithm_version": "test-source-selection/v1",
        "candidate_window_count": 100,
        "selected_group_count": 100,
        "selection_sha256": _sha("selection"),
    }
    assert result.manifest["collection_statistics"] == {
        "close_suitable_counts": {
            "close_false_suitable_false": 0,
            "close_true_suitable_false": 85,
            "close_true_suitable_true": 15,
        },
        "statement_family_counts": {
            "assignment": 25,
            "branch": 25,
            "loop": 25,
            "return": 25,
        },
        "rewrite_parse_status_counts": {"ok": 900},
        "rewrite_structurally_valid_count": 900,
        "rewrite_structurally_invalid_count": 0,
        "rewrite_semantic_signature_stable_count": 900,
        "rewrite_semantic_signature_unstable_count": 0,
        "unique_repository_id_count": 100,
        "unique_task_id_count": 100,
    }
    assert deps.calls[:5] == ["source", "training", "holdout", "parse", "generate"]
    assert deps.calls[5:-1] == [value for _ in range(100) for value in ("probe", "split")]
    assert deps.calls[-1] == "audit"
    rows = result.data_path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 100
    assert set(result.manifest["artifacts"]) == {
        "window_groups.jsonl", "candidate_attempts.jsonl", "labels.jsonl",
        "split_manifest.json", "training_key_bank_manifest.json", "group_index.jsonl",
        "feasibility_summary.json",
    }
    evidence_count = 0
    attempt_count = 0
    first_evidence = None
    with (result.output_dir / "candidate_attempts.jsonl").open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row["record_type"] == "probe_evidence":
                evidence_count += 1
                first_evidence = first_evidence or row
            else:
                attempt_count += 1
    assert attempt_count == 100 * 3 * 4
    assert evidence_count == 18  # canonical content dedupes identical candidates
    assert len(first_evidence["candidate_observation"]["lsh_by_key_id"]) == 32
    assert len(first_evidence["lsh_probe_results"]) == 40


def test_production_local_dependencies_run_formal_pipeline_with_local_adapters(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.json"
    source.write_text(json.dumps({
        "schema_version": "wfcllm-gate-source-manifest/v1",
        "sources": [{"source_id": "local-fixture"}],
    }))
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    training.write_text(json.dumps([f"training-{index}" for index in range(32)]))
    holdout.write_text(json.dumps([f"holdout-{index}" for index in range(8)]))
    adapters = FakeDependencies(tuple(_group(index) for index in range(100)))
    adapter = ControlledProductionPipelineAdapter(adapters)
    dependencies = build_trusted_test_gate_dependencies(
        source_manifest=source,
        training_key_file=training,
        training_key_env=None,
        holdout_key_file=holdout,
        holdout_key_env=None,
        base_model_path=None,
        adapter=adapter,
    )

    result = run_gate_data(_data_config(tmp_path / "run"), dependencies)

    assert dependencies.diagnostic_test_backend is True
    assert result.group_count == 100
    assert result.manifest["diagnostic_test_backend"] is True
    assert result.manifest["formal_eligible"] is False
    assert adapter.probe_count == 100
    with pytest.raises(ValueError, match="released"):
        adapter.training_view.material_for("train-key-000")
    published = b"".join(
        path.read_bytes() for path in result.output_dir.rglob("*") if path.is_file()
    )
    assert b"training-0" not in published
    assert b"holdout-0" not in published


def test_gate_data_releases_private_keys_when_validation_fails() -> None:
    class ReleasingDependencies:
        released = False

        def release_private_keys(self):
            self.released = True

    dependencies = ReleasingDependencies()
    with pytest.raises(ValueError, match="GateDataPipelineConfig"):
        run_gate_data(object(), dependencies)
    assert dependencies.released is True


def _probed_for_writer(group: GatePipelineGroup) -> ProbedGroup:
    identity = GateGroupIdentity(
        group.group_id, group.split_group_id, group.repository_id, group.task_id,
        group.generation_model_id, group.statement_family, group.window_lengths,
    )
    trajectory = CandidateTrajectoryGroup(
        ParsedWindowGroup(identity, "wfcllm-window/v1"),
        group.candidate_indices_by_window_length,
    )
    return ProbedGroup(
        trajectory,
        group.candidate_observations_by_length,
        group.probe_results_by_length,
    )


def _distinct_evidence_group(index: int) -> GatePipelineGroup:
    base = _group(0)
    observations = {
        length: tuple(
            replace(observation, code=f"{observation.code} # evidence-{index}")
            for observation in rows
        )
        for length, rows in base.candidate_observations_by_length.items()
    }
    probes = {
        length: tuple(types.MappingProxyType(dict(row)) for row in rows)
        for length, rows in base.probe_results_by_length.items()
    }
    return replace(
        base,
        group_id=f"evidence-group-{index:03d}",
        split_group_id=f"repository:evidence-{index:03d}",
        repository_id=f"evidence-repo-{index:03d}",
        task_id=f"evidence-task-{index:03d}",
        candidate_observations_by_length=observations,
        probe_results_by_length=probes,
        row={
            "schema_version": "wfcllm-gate-data/v1",
            "group_id": f"evidence-group-{index:03d}",
            "split": base.split,
        },
    )


def test_candidate_attempt_writer_deduplicates_equal_content_from_distinct_objects(
    tmp_path: Path,
) -> None:
    first = _distinct_evidence_group(0)
    second = replace(
        first,
        candidate_observations_by_length={
            length: tuple(replace(observation) for observation in rows)
            for length, rows in first.candidate_observations_by_length.items()
        },
        probe_results_by_length={
            length: tuple(types.MappingProxyType(dict(row)) for row in rows)
            for length, rows in first.probe_results_by_length.items()
        },
    )
    path = tmp_path / "attempts.jsonl"
    writer = gate_pipeline._BoundedJsonlWriter.open(path)
    cache: OrderedDict[str, None] = OrderedDict()
    try:
        gate_pipeline._write_group_attempts(
            writer,
            first,
            _probed_for_writer(first),
            cache,
            diagnostic=True,
        )
        gate_pipeline._write_group_attempts(
            writer,
            second,
            _probed_for_writer(second),
            cache,
            diagnostic=True,
        )
    finally:
        writer.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert sum(row["record_type"] == "probe_evidence" for row in rows) == 12
    assert sum(row["record_type"] == "candidate_attempt" for row in rows) == 24


def test_candidate_attempt_writer_reemits_content_after_lru_eviction(tmp_path: Path) -> None:
    groups = tuple(_distinct_evidence_group(index) for index in range(44))
    path = tmp_path / "attempts.jsonl"
    writer = gate_pipeline._BoundedJsonlWriter.open(path)
    cache: OrderedDict[str, None] = OrderedDict()
    try:
        for group in groups:
            gate_pipeline._write_group_attempts(
                writer,
                group,
                _probed_for_writer(group),
                cache,
                diagnostic=True,
            )
        gate_pipeline._write_group_attempts(
            writer,
            groups[0],
            _probed_for_writer(groups[0]),
            cache,
            diagnostic=True,
        )
    finally:
        writer.close()
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(cache) == 512
    assert sum(row["record_type"] == "probe_evidence" for row in rows) == 45 * 12
    assert sum(row["record_type"] == "candidate_attempt" for row in rows) == 45 * 12


def test_gate_data_streams_generator_with_bounded_live_groups_and_no_read_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    live = 0
    maximum_live = 0

    def groups():
        nonlocal live, maximum_live
        for index in range(100):
            group = _group(index)
            live += 1
            maximum_live = max(maximum_live, live)
            weakref.finalize(group, lambda: _decrement())
            yield group

    def _decrement() -> None:
        nonlocal live
        live -= 1

    deps = FakeDependencies(groups())
    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("read_bytes is forbidden")),
    )
    result = run_gate_data(_data_config(tmp_path), deps)
    assert result.group_count == 100
    assert maximum_live <= 2


def test_progressive_stage_records_are_frozen_and_keep_one_identity_digest() -> None:
    group = _group(0)
    identity = GateGroupIdentity(
        group.group_id, group.split_group_id, group.repository_id, group.task_id,
        group.generation_model_id, group.statement_family, group.window_lengths,
    )
    parsed = ParsedWindowGroup(identity, "wfcllm-window/v1")
    trajectory = CandidateTrajectoryGroup(parsed, group.candidate_indices_by_window_length)
    probed = ProbedGroup(
        trajectory, group.candidate_observations_by_length, group.probe_results_by_length,
    )
    labels = {
        length: build_gate_labels(
            group.candidate_observations_by_length[str(length)], training_key_count=32,
        )
        for length in (1, 2, 3)
    }
    labeled = LabeledGroup(probed, labels, group.holdout_success_rate, 0)
    split = SplitGroup(labeled, group.split, group.row)
    assert parsed.identity.digest == trajectory.identity.digest == probed.identity.digest
    assert labeled.identity.digest == split.identity.digest == identity.digest
    assert not hasattr(parsed, "candidate_observations_by_length")
    assert not hasattr(trajectory, "probe_results_by_length")
    assert not hasattr(probed, "labels_by_window_length")
    assert not hasattr(labeled, "split")
    with pytest.raises(FrozenInstanceError):
        identity.group_id = "changed"


def test_audit_label_recompute_treats_invalid_candidate_empty_probe_as_no_holdout_hit() -> None:
    observations_template, probes_template = _evidence_templates(True)
    observations = {
        length: list(observations_template[str(length)])
        for length in (1, 2, 3)
    }
    results = {
        length: list(probes_template[str(length)])
        for length in (1, 2, 3)
    }
    observations[3][1] = replace(
        observations[3][1],
        parse_status="parse_error",
        unit_count=0,
        same_parent_scope=False,
        stable_across_precision_modes=False,
        stable_across_batch_modes=False,
        lsh_by_key_id={},
        lsh_signature=None,
        semantic_reference_cosine=None,
        semantic_preservation_passed=None,
    )
    results[3][1] = types.MappingProxyType({})

    payload = gate_pipeline._derived_label_payload(
        observations,
        results,
        holdout_ids=tuple(f"holdout-key-{index:03d}" for index in range(8)),
    )

    assert payload["holdout_success_rate"] == pytest.approx(5 / 8)


def test_formal_candidate_tree_rejects_excess_entries_depth_and_path_bytes(tmp_path: Path) -> None:
    def candidate(name: str) -> Path:
        root = tmp_path / name
        (root / "tokenizer").mkdir(parents=True)
        (root / "gate_float.pt").write_bytes(b"model")
        return root

    too_many = candidate("too-many")
    for index in range(127):
        (too_many / "tokenizer" / f"dir-{index:03d}").mkdir()
    with pytest.raises(ValueError, match="entry/file-count"):
        _validate_formal_candidate_tree(too_many)

    too_deep = candidate("too-deep")
    deep = too_deep / "tokenizer"
    for index in range(9):
        deep = deep / f"d{index}"
    deep.mkdir(parents=True)
    with pytest.raises(ValueError, match="depth"):
        _validate_formal_candidate_tree(too_deep)

    too_long = candidate("too-long")
    long_path = too_long / "tokenizer"
    for index in range(5):
        long_path = long_path / (str(index) + "x" * 219)
    long_path.mkdir(parents=True)
    with pytest.raises(ValueError, match="path length"):
        _validate_formal_candidate_tree(too_long)


def test_gate_data_rejects_humaneval_before_key_loading(tmp_path: Path) -> None:
    deps = FakeDependencies(tuple(_group(i) for i in range(100)))
    deps.load_source_manifest = lambda config: {"sources": [{"source_id": "HumanEval/0"}]}
    with pytest.raises(ValueError, match="HumanEval"):
        run_gate_data(_data_config(tmp_path), deps)
    assert deps.calls == []
    assert not (tmp_path / "gate-data").exists()


def test_formal_gate_data_rejects_diagnostic_window_row_and_cleans_staging(tmp_path: Path) -> None:
    group = _group(0)
    corrupted = replace(
        group,
        row={
            **dict(group.row),
            "diagnostic_test_backend": True,
            "formal_eligible": False,
            "diagnostic_only": True,
            "not_official_method": True,
        },
    )
    deps = FakeDependencies((corrupted,), diagnostic=False)
    with pytest.raises(ValueError, match="window group public row schema"):
        run_gate_data(_data_config(tmp_path), deps)
    assert not (tmp_path / "gate-data").exists()
    assert not list(tmp_path.glob(".gate-data-*"))
    assert "audit" not in deps.calls


def test_diagnostic_gate_data_marks_every_public_row_non_formal(
    tmp_path: Path,
) -> None:
    deps = FakeDependencies((_group(0),))
    deps.diagnostic_test_backend = True
    result = run_gate_data(_data_config(tmp_path), deps)

    for name in (
        "window_groups.jsonl",
        "candidate_attempts.jsonl",
        "labels.jsonl",
        "group_index.jsonl",
    ):
        rows = [
            json.loads(line)
            for line in (result.output_dir / name).read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        assert rows
        for row in rows:
            assert row["diagnostic_test_backend"] is True
            assert row["formal_eligible"] is False
            assert row["diagnostic_only"] is True
            assert row["not_official_method"] is True


def test_gate_data_audit_failure_removes_staging_and_publishes_nothing(tmp_path: Path) -> None:
    deps = FakeDependencies(tuple(_group(i) for i in range(100)))
    deps.audit_gate_data = lambda staging, manifest: (_ for _ in ()).throw(ValueError("audit failed"))
    with pytest.raises(ValueError, match="audit failed"):
        run_gate_data(_data_config(tmp_path), deps)
    assert not (tmp_path / "gate-data").exists()
    assert not list(tmp_path.glob(".gate-data-*"))


def test_gate_data_rejects_prefilled_probe_facts_that_disagree_with_raw_evidence(tmp_path: Path) -> None:
    original = _group(0)
    probes = {key: tuple(rows) for key, rows in original.probe_results_by_length.items()}
    first = dict(probes["1"][0])
    key_id = original.observed_training_key_ids[0]
    prior = first[key_id]
    first[key_id] = LshProbeResult(prior.signature, prior.margin, not prior.hit, prior.stable, True, True)
    probes["1"] = (types.MappingProxyType(first), *probes["1"][1:])
    corrupted = replace(original, probe_results_by_length=probes)
    deps = FakeDependencies((corrupted, *tuple(_group(i) for i in range(1, 100))))
    with pytest.raises(ValueError, match="contradicts LshProbeResult"):
        run_gate_data(_data_config(tmp_path), deps)
    assert "audit" not in deps.calls


def test_probe_contract_accepts_explicit_semantic_rejection_without_keyed_results() -> None:
    original = _group(0)
    observations = {
        length: list(rows)
        for length, rows in original.candidate_observations_by_length.items()
    }
    probes = {
        length: list(rows)
        for length, rows in original.probe_results_by_length.items()
    }
    for length in ("1", "2", "3"):
        observations[length][1] = replace(
            observations[length][1],
            stable_across_precision_modes=False,
            stable_across_batch_modes=False,
            lsh_by_key_id={},
            lsh_signature=None,
            semantic_reference_cosine=0.75,
            semantic_preservation_passed=False,
        )
        probes[length][1] = types.MappingProxyType({})
    rejected = replace(
        original,
        candidate_observations_by_length={
            length: tuple(rows) for length, rows in observations.items()
        },
        probe_results_by_length={
            length: tuple(rows) for length, rows in probes.items()
        },
    )
    deps = FakeDependencies((rejected,))

    gate_pipeline._validate_probe_contract(
        (rejected,),
        deps.load_key_bank(role="training", expected_count=32, config=None),
        deps.load_key_bank(role="holdout", expected_count=8, config=None),
    )


def test_artifact_validator_accepts_only_explicit_semantic_rejection_without_results() -> None:
    original = _group(0).candidate_observations_by_length["1"][0]
    rejected = replace(
        original,
        stable_across_precision_modes=False,
        stable_across_batch_modes=False,
        lsh_by_key_id={},
        lsh_signature=None,
        semantic_reference_cosine=0.75,
        semantic_preservation_passed=False,
    )

    gate_pipeline._validate_observation_against_results(
        rejected,
        {},
        _group(0).observed_training_key_ids,
    )

    with pytest.raises(ValueError, match="missing semantic evidence"):
        gate_pipeline._validate_observation_against_results(
            replace(
                rejected,
                semantic_reference_cosine=None,
                semantic_preservation_passed=None,
                semantic_probe_pending=True,
            ),
            {},
            _group(0).observed_training_key_ids,
        )


def test_gate_data_rejects_injected_label_that_disagrees_with_causal_recompute(tmp_path: Path) -> None:
    corrupted = replace(_group(0), suitable_target=False)
    deps = FakeDependencies((corrupted, *tuple(_group(i) for i in range(1, 100))))
    with pytest.raises(ValueError, match="causal labels"):
        run_gate_data(_data_config(tmp_path), deps)
    assert "split" not in deps.calls and "audit" not in deps.calls


def test_gate_data_rejects_split_stage_base_identity_mutation(tmp_path: Path) -> None:
    deps = FakeDependencies(tuple(_group(i) for i in range(100)))
    def mutate_identity(groups, config):
        iterator = iter(groups)
        yield replace(next(iterator), repository_id="tampered-repository")
        yield from iterator
    deps.split_groups = mutate_identity
    with pytest.raises(ValueError, match="immutable base identity"):
        run_gate_data(_data_config(tmp_path), deps)
    assert "audit" not in deps.calls


def test_gate_train_requires_full_data_manifest_and_passed_pilot(tmp_path: Path) -> None:
    deps = FakeDependencies(())
    config = GateTrainPipelineConfig(
        output_root=tmp_path,
        data_dir=tmp_path / "missing",
        pilot_feasibility_path=tmp_path / "missing-pilot.json",
        config_hash=_sha("config"),
    )
    with pytest.raises(ValueError, match="data manifest"):
        run_gate_train(config, deps)
    assert deps.train_called is False


def test_gate_train_enforces_independent_group_minima_before_trainer(tmp_path: Path) -> None:
    pilot_dir = tmp_path / "pilot"
    pilot_deps = FakeDependencies(
        tuple(_group(i) for i in range(100)),
        diagnostic=False,
    )
    pilot = run_gate_data(_data_config(pilot_dir), pilot_deps)
    full_dir = tmp_path / "full"
    full_deps = FakeDependencies(
        tuple(_group(i, scale="full") for i in range(300)),
        diagnostic=False,
    )
    full = run_gate_data(_data_config(full_dir, scale="full", pilot=pilot.feasibility_path), full_deps)
    for index, artifact_name in enumerate(full.manifest["artifacts"]):
        artifact = full.output_dir / artifact_name
        original = artifact.read_bytes()
        for mutation in ("delete", "tamper"):
            if mutation == "delete":
                artifact.unlink()
            else:
                artifact.write_bytes(original + b"tamper")
            blocked = FakeDependencies((), diagnostic=False)
            with pytest.raises(ValueError, match="artifact|hash|missing"):
                run_gate_train(
                    GateTrainPipelineConfig(
                        tmp_path / f"blocked-{index}-{mutation}",
                        full.output_dir,
                        _sha("config"),
                        pilot.feasibility_path,
                    ),
                    blocked,
                )
            assert blocked.train_called is False
            artifact.write_bytes(original)
    labels_path = full.output_dir / "labels.jsonl"
    manifest_path = full.output_dir / "manifest.json"
    original_labels = labels_path.read_bytes()
    original_manifest = manifest_path.read_bytes()
    first, separator, remainder = original_labels.partition(b"\n")
    tampered_label = json.loads(first)
    tampered_label["r3_success_rate"] = 0.0 if tampered_label["r3_success_rate"] else 1.0
    labels_path.write_bytes(
        json.dumps(tampered_label, sort_keys=True, separators=(",", ":")).encode()
        + separator + remainder
    )
    tampered_manifest = json.loads(original_manifest)
    tampered_manifest["artifacts"]["labels.jsonl"] = hashlib.sha256(labels_path.read_bytes()).hexdigest()
    manifest_path.write_text(
        json.dumps(tampered_manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    semantic_tamper_deps = FakeDependencies((), diagnostic=False)
    with pytest.raises(ValueError, match="labels JSONL|causal Task6"):
        run_gate_train(
            GateTrainPipelineConfig(
                tmp_path / "semantic-label-tamper", full.output_dir,
                _sha("config"), pilot.feasibility_path,
            ),
            semantic_tamper_deps,
        )
    assert semantic_tamper_deps.train_called is False
    labels_path.write_bytes(original_labels)
    manifest_path.write_bytes(original_manifest)
    original_windows = full.data_path.read_bytes()
    for field_name in ("raw_key", "extra"):
        first, separator, remainder = original_windows.partition(b"\n")
        tampered_window = json.loads(first)
        tampered_window[field_name] = "forbidden"
        full.data_path.write_bytes(
            json.dumps(tampered_window, sort_keys=True, separators=(",", ":")).encode()
            + separator + remainder
        )
        tampered_manifest = json.loads(original_manifest)
        window_sha = hashlib.sha256(full.data_path.read_bytes()).hexdigest()
        tampered_manifest["grouped_jsonl_sha256"] = window_sha
        tampered_manifest["artifacts"]["window_groups.jsonl"] = window_sha
        manifest_path.write_text(
            json.dumps(tampered_manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        blocked_window_deps = FakeDependencies((), diagnostic=False)
        with pytest.raises(ValueError, match="window group"):
            run_gate_train(
                GateTrainPipelineConfig(
                    tmp_path / f"window-row-{field_name}", full.output_dir,
                    _sha("config"), pilot.feasibility_path,
                ),
                blocked_window_deps,
            )
        assert blocked_window_deps.train_called is False
        full.data_path.write_bytes(original_windows)
        manifest_path.write_bytes(original_manifest)
    train_deps = FakeDependencies((), diagnostic=False)
    result = run_gate_train(
        GateTrainPipelineConfig(full_dir, full.output_dir, _sha("config"), pilot.feasibility_path),
        train_deps,
    )
    assert result.candidate_bundle_path.exists()
    assert result.manifest["learning_curve_runs_executed"] is False
    assert (result.output_dir / "development_summary.json").exists()
    assert train_deps.train_called is True
    original_window_data = full.data_path.read_bytes()
    mutating_original_trainer = FakeDependencies((), diagnostic=False)
    normal_train = mutating_original_trainer.train_candidate
    def mutate_original_data(**kwargs):
        full.data_path.write_bytes(original_window_data + b"{}\n")
        return normal_train(**kwargs)
    mutating_original_trainer.train_candidate = mutate_original_data
    with pytest.raises(ValueError, match="mutated original"):
        run_gate_train(
            GateTrainPipelineConfig(tmp_path / "mutating-original-trainer", full.output_dir, _sha("config"), pilot.feasibility_path),
            mutating_original_trainer,
        )
    full.data_path.write_bytes(original_window_data)
    mutating_snapshot_trainer = FakeDependencies((), diagnostic=False)
    normal_snapshot_train = mutating_snapshot_trainer.train_candidate
    def mutate_snapshot_data(**kwargs):
        kwargs["data_jsonl"].chmod(0o644)
        kwargs["data_jsonl"].write_bytes(kwargs["data_jsonl"].read_bytes() + b"{}\n")
        return normal_snapshot_train(**kwargs)
    mutating_snapshot_trainer.train_candidate = mutate_snapshot_data
    with pytest.raises(ValueError, match="artifact|hash|window group"):
        run_gate_train(
            GateTrainPipelineConfig(tmp_path / "mutating-snapshot-trainer", full.output_dir, _sha("config"), pilot.feasibility_path),
            mutating_snapshot_trainer,
        )


def test_gate_train_rejects_tampered_grouped_data_before_trainer(tmp_path: Path) -> None:
    pilot_root = tmp_path / "pilot"
    pilot = run_gate_data(
        _data_config(pilot_root),
        FakeDependencies(
            tuple(_group(i) for i in range(100)),
            diagnostic=False,
        ),
    )
    full_root = tmp_path / "full"
    full = run_gate_data(
        _data_config(full_root, scale="full", pilot=pilot.feasibility_path),
        FakeDependencies(
            tuple(_group(i, scale="full") for i in range(300)),
            diagnostic=False,
        ),
    )
    full.data_path.write_text(full.data_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    deps = FakeDependencies((), diagnostic=False)
    with pytest.raises(ValueError, match="artifact|hash"):
        run_gate_train(
            GateTrainPipelineConfig(full_root, full.output_dir, _sha("config"), pilot.feasibility_path),
            deps,
        )
    assert deps.train_called is False
