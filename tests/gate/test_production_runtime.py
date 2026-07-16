from __future__ import annotations

import json
from pathlib import Path

import pytest

from wfcllm.gate.dependencies import build_local_gate_dependencies
from wfcllm.gate.feasibility import FEASIBILITY_CONTRACT_VERSION, FEASIBILITY_THRESHOLD_ITEMS
from wfcllm.gate.pipeline import GateDataPipelineConfig
from wfcllm.gate.production import (
    LOCAL_HF_ADAPTER_NAME,
    GateSourceCatalogRecord,
    LocalHFGateRuntimeOptions,
    experiment_contract_hash,
    load_source_catalog,
    phase_config_hash,
    _validate_cache_against_data,
)
from wfcllm.gate.data import RewriteCandidate


def _runtime_options(tmp_path: Path) -> LocalHFGateRuntimeOptions:
    catalog = tmp_path / "catalog.jsonl"
    catalog.write_text(
        json.dumps(
            {
                "source_family": "oss_python",
                "source_id": "repo-a:module.py:f",
                "code": "a = 1\nb = 2\nc = 3\nd = 4\n",
                "repository_id": "repo-a",
                "task_id": "task-a",
                "function_id": "f",
                "source_model_id": None,
                "license_id": "MIT",
                "contract_or_hard_set": False,
                "prompt": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    model = tmp_path / "model"
    encoder = tmp_path / "encoder"
    gate_model = tmp_path / "gate-model"
    for path in (model, encoder, gate_model):
        path.mkdir()
    return LocalHFGateRuntimeOptions(
        source_catalog=catalog,
        generation_model_path=model,
        rewrite_model_path=None,
        semantic_encoder_model_path=encoder,
        semantic_encoder_checkpoint_path=None,
        semantic_whitening_path=None,
        gate_base_model_path=gate_model,
        model_device="cuda",
        gate_device="cuda",
        cache_dir=tmp_path / "cache",
    )


def test_source_catalog_streams_strict_records(tmp_path: Path) -> None:
    options = _runtime_options(tmp_path)

    records = tuple(load_source_catalog(options.source_catalog))

    assert records == (
        GateSourceCatalogRecord(
            source_family="oss_python",
            source_id="repo-a:module.py:f",
            code="a = 1\nb = 2\nc = 3\nd = 4\n",
            repository_id="repo-a",
            task_id="task-a",
            function_id="f",
            source_model_id=None,
            license_id="MIT",
            contract_or_hard_set=False,
            prompt="",
        ),
    )


def test_source_catalog_rejects_humaneval_without_echoing_code(tmp_path: Path) -> None:
    path = tmp_path / "catalog.jsonl"
    secret_code = "def hidden_secret(): return 'do-not-echo'"
    path.write_text(
        json.dumps(
            {
                "source_family": "main_generation",
                "source_id": "HumanEval/0",
                "code": secret_code,
                "repository_id": None,
                "task_id": "HumanEval/0",
                "function_id": "f",
                "source_model_id": "model-a",
                "license_id": None,
                "contract_or_hard_set": False,
                "prompt": "",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="HumanEval") as exc_info:
        tuple(load_source_catalog(path))

    assert secret_code not in str(exc_info.value)


def test_static_local_hf_adapter_is_allowlisted_and_lazy(tmp_path: Path) -> None:
    options = _runtime_options(tmp_path)
    source_manifest = tmp_path / "source-manifest.json"
    source_manifest.write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-gate-source-manifest/v1",
                "catalog_sha256": "0" * 64,
                "source_count": 1,
            }
        ),
        encoding="utf-8",
    )
    training = tmp_path / "training.json"
    holdout = tmp_path / "holdout.json"
    training.write_text(json.dumps([f"train-{index}" for index in range(32)]))
    holdout.write_text(json.dumps([f"holdout-{index}" for index in range(8)]))

    dependencies = build_local_gate_dependencies(
        source_manifest=source_manifest,
        training_key_file=training,
        training_key_env=None,
        holdout_key_file=holdout,
        holdout_key_env=None,
        base_model_path=options.gate_base_model_path,
        adapter_name=LOCAL_HF_ADAPTER_NAME,
        adapter_options=options,
    )

    assert dependencies.diagnostic_test_backend is False
    assert dependencies.adapter_name == LOCAL_HF_ADAPTER_NAME


def test_experiment_hash_ignores_scale_but_phase_hash_does_not() -> None:
    pilot = {
        "method": {"name": "gated_semantic_window_v1"},
        "gate_data": {"scale": "pilot", "full_independent_group_min": 20_000},
        "runtime": {"run_id": "pilot-run"},
    }
    full = {
        "method": {"name": "gated_semantic_window_v1"},
        "gate_data": {"scale": "full", "full_independent_group_min": 20_000},
        "runtime": {"run_id": "full-run"},
    }

    assert experiment_contract_hash(pilot) == experiment_contract_hash(full)
    assert phase_config_hash(pilot) != phase_config_hash(full)


def test_training_cache_is_bound_to_formal_group_index(tmp_path: Path) -> None:
    data = tmp_path / "gate-data"
    data.mkdir()
    index = {
        "group_id": "g-1",
        "split": "train",
        "close_target": True,
        "suitable_target": False,
    }
    (data / "group_index.jsonl").write_text(json.dumps(index) + "\n", encoding="utf-8")
    rows = [
        {
            "group_id": "g-1",
            "split": "train",
            "context_length": context,
            "budget": budget,
            "close_target": True,
            "suitable_target": False,
        }
        for context in (1, 2, 3)
        for budget in (1, 3, 6)
    ]

    _validate_cache_against_data(rows, data)
    rows[-1]["suitable_target"] = True
    with pytest.raises(ValueError, match="label contradicts"):
        _validate_cache_against_data(rows, data)


class _Rewriter:
    def rewrite(self, request, *, candidate_index):
        return RewriteCandidate(
            code=request.original_window,
            parse_status="ok",
            unit_count=request.window_length,
            same_parent_scope=True,
            boundary_span=(
                request.structural_boundary.start_byte,
                request.structural_boundary.end_byte,
            ),
            generation_seed_id=f"seed-{candidate_index}",
            rewrite_config_id="test-rewriter/v1",
        )


class _SemanticRuntime:
    def signature_and_margin(self, window_text: str):
        assert window_text
        return (0, 0, 0, 0), 1.0, True, True


class _GateTokenizer:
    def __call__(self, text, **kwargs):
        assert kwargs["truncation"] is False
        return {"input_ids": list(range(max(1, len(text.split()))))}


class _KeyView:
    def __init__(self, prefix: str, count: int):
        self.key_ids = tuple(f"{prefix}-key-{index:03d}" for index in range(count))
        self._values = {
            key_id: f"material-{key_id}".encode("utf-8") for key_id in self.key_ids
        }

    def material_for(self, key_id: str):
        return memoryview(self._values[key_id]).toreadonly()


def _pipeline_config(tmp_path: Path) -> GateDataPipelineConfig:
    return GateDataPipelineConfig(
        output_root=tmp_path / "run",
        scale="pilot",
        config_hash="a" * 64,
        parser_contract="python-statement-window/v1",
        rewriter_config_hash="b" * 64,
        semantic_encoder_hash="c" * 64,
        lsh_config_hash="d" * 64,
        feasibility_contract=FEASIBILITY_CONTRACT_VERSION,
        feasibility_thresholds=FEASIBILITY_THRESHOLD_ITEMS,
    )


def test_local_hf_adapter_builds_and_probes_real_pipeline_groups(tmp_path: Path) -> None:
    from wfcllm.gate.production import LocalHFProductionAdapter

    options = _runtime_options(tmp_path)
    source_hash = __import__("hashlib").sha256(options.source_catalog.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "wfcllm-gate-source-manifest/v1",
        "catalog_sha256": source_hash,
        "source_count": 1,
    }
    adapter = LocalHFProductionAdapter(options)
    adapter._rewriter = _Rewriter()
    adapter._semantic_runtime = _SemanticRuntime()
    adapter._gate_tokenizer = _GateTokenizer()
    config = _pipeline_config(tmp_path)

    parsed = adapter.parse_statement_units(manifest, config)
    generated = tuple(adapter.generate_candidate_trajectories(parsed, config))
    assert generated
    assert all(group.window_lengths == (1, 2, 3) for group in generated)
    assert all(
        group.candidate_indices_by_window_length[length] == tuple(range(7))
        for group in generated
        for length in (1, 2, 3)
    )

    training = _KeyView("train", 32)
    holdout = _KeyView("holdout", 8)
    probed = tuple(
        adapter.run_multi_key_lsh_probe(
            generated,
            training_keys=training,
            holdout_keys=holdout,
            config=config,
        )
    )

    assert len(probed) == len(generated)
    assert all(group.observed_training_key_ids == training.key_ids for group in probed)
    assert all(group.observed_holdout_key_ids == holdout.key_ids for group in probed)
    assert all(
        set(group.probe_results_by_length[str(length)][0])
        == set((*training.key_ids, *holdout.key_ids))
        for group in probed
        for length in (1, 2, 3)
    )
    cache = options.cache_dir / f"gate-examples-{config.config_hash}.jsonl"
    cache_rows = [json.loads(line) for line in cache.read_text(encoding="utf-8").splitlines()]
    assert len(cache_rows) == len(probed) * 9
    assert {row["context_length"] for row in cache_rows} == {1, 2, 3}
    assert {row["budget"] for row in cache_rows} == {1, 3, 6}
    split = tuple(adapter.split_groups(probed, config))
    assert split == probed
    adapter.audit_gate_data(
        tmp_path,
        {
            "formal_eligible": True,
            "diagnostic_test_backend": False,
            "config_hash": config.config_hash,
        },
    )
