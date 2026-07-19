from __future__ import annotations

from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import wfcllm.gate.production as gate_production
from wfcllm.gate.dependencies import build_local_gate_dependencies
from wfcllm.gate.feasibility import FEASIBILITY_CONTRACT_VERSION, FEASIBILITY_THRESHOLD_ITEMS
from wfcllm.gate.pipeline import GateDataPipelineConfig
from wfcllm.gate.production import (
    LOCAL_HF_ADAPTER_NAME,
    GateSourceCatalogRecord,
    HFCausalRewriteBackend,
    LocalHFProgramGenerator,
    LocalHFGateRuntimeOptions,
    experiment_contract_hash,
    load_source_catalog,
    phase_config_hash,
    _validate_cache_against_data,
    _balanced_split_assignments,
    _ParsedCatalogSource,
    _select_source_stratified_starts,
    _statement_family,
    _source_stratified_selection_summary,
)
from wfcllm.gate.data import RewriteCandidate
from wfcllm.windowing.contracts import StatementUnit


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


def test_public_semantic_runtime_initialization_is_repeatable_and_rng_isolated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import torch

    class Verifier:
        def __init__(self, bit: int) -> None:
            self.bit = bit

        def verify(self, _code, _valid_set, _margin):
            return SimpleNamespace(
                lsh_signature=(self.bit, 0, 0, 0), min_margin=0.5
            )

    def fake_load(**_kwargs):
        return SimpleNamespace(
            verifier=Verifier(int(torch.randint(0, 2, ()).item()))
        )

    monkeypatch.setattr(
        "wfcllm.semantic.lsh.load_semantic_lsh_components", fake_load
    )
    options = _runtime_options(tmp_path)
    torch.manual_seed(991)
    before = torch.random.get_rng_state().clone()

    first = gate_production._load_semantic_runtime(options)
    second = gate_production._load_semantic_runtime(options)

    assert first.signature_and_margin("x = 1")[0] == second.signature_and_margin(
        "x = 1"
    )[0]
    assert torch.equal(torch.random.get_rng_state(), before)


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


def _parsed_source(source_id: str, repository_id: str) -> _ParsedCatalogSource:
    record = GateSourceCatalogRecord(
        source_family="oss_python",
        source_id=source_id,
        code="a = 1\nb = 2\nc = 3\nd = 4\ne = 5\n",
        repository_id=repository_id,
        task_id=f"task-{source_id}",
        function_id=f"function-{source_id}",
        source_model_id=None,
        license_id="MIT",
        contract_or_hard_set=False,
        prompt="",
    )
    units = tuple(
        StatementUnit(
            unit_id=f"{source_id}-unit-{index}",
            node_type="expression_statement",
            text=f"value_{index} = {index}",
            start_byte=index * 6,
            end_byte=index * 6 + 5,
            start_line=index + 1,
            end_line=index + 1,
            depth=1,
            parent_path=("module",),
            direct_parent_type="module",
            direct_child_ordinal=index,
            eligible=True,
            hard_boundary=False,
            compound_header=False,
        )
        for index in range(5)
    )
    return _ParsedCatalogSource(record, units)


def test_source_stratified_selection_covers_sources_before_reuse() -> None:
    parsed = (
        _parsed_source("source-a1", "repo-a"),
        _parsed_source("source-a2", "repo-a"),
        _parsed_source("source-b1", "repo-b"),
        _parsed_source("source-c1", "repo-c"),
    )

    selected = _select_source_stratified_starts(parsed, max_groups=5)

    source_ids = [item.parsed.record.source_id for item in selected]
    assert len(selected) == 5
    assert len(set(source_ids[:4])) == 4
    assert len({item.parsed.record.repository_id for item in selected[:3]}) == 3
    assert selected == _select_source_stratified_starts(
        tuple(reversed(parsed)), max_groups=5
    )
    summary = _source_stratified_selection_summary(parsed, selected, max_groups=5)
    assert summary["algorithm_version"] == (
        "wfcllm-source-stratified-window-selection/v2"
    )
    assert summary["candidate_source_count"] == 4
    assert summary["selected_source_count"] == 4
    assert summary["selected_repository_count"] == 3
    assert summary["selected_group_count"] == 5
    assert summary["max_selected_per_source"] == 2
    assert len(summary["selection_sha256"]) == 64


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("result = build(value)", "assignment_call"),
        ("result = source.value", "assignment_reference"),
        ("result = {'ok': True}", "assignment_literal"),
        ("result = left + right", "assignment_expression"),
        ("emit(value)", "call"),
        ('"documented value"', "docstring"),
    ],
)
def test_statement_family_uses_public_syntax_only(text: str, expected: str) -> None:
    assert _statement_family(text) == expected


def test_balanced_split_assignments_are_deterministic_and_nonempty() -> None:
    group_ids = tuple(f"repository:repo-{index:02d}" for index in range(13))

    first = _balanced_split_assignments(group_ids)
    second = _balanced_split_assignments(tuple(reversed(group_ids)))

    assert first == second
    assert set(first) == set(group_ids)
    assert Counter(first.values()) == Counter(
        {"train": 7, "validation": 3, "test": 3}
    )


def test_source_stratified_selection_uses_exact_complete_window_contract() -> None:
    parsed = _parsed_source("source-a", "repo-a")
    units = list(parsed.units)
    units[2] = replace(units[2], depth=2)

    selected = _select_source_stratified_starts(
        (_ParsedCatalogSource(parsed.record, tuple(units)),),
        max_groups=10,
    )

    assert selected == ()


def test_local_hf_backend_batches_r3_candidates_in_one_generate_call() -> None:
    import torch

    class Tokenizer:
        def __call__(self, _text, **_kwargs):
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, **_kwargs):
            return f"x = {ids[0]}\n"

    class Model:
        config = type("Config", (), {"is_encoder_decoder": False})()

        def __init__(self):
            self.calls = 0

        def generate(self, **kwargs):
            assert "generator" not in kwargs
            self.calls += 1
            assert kwargs["num_return_sequences"] == 6
            return torch.tensor(
                [[10, 11, index] for index in range(1, 7)]
            )

    model = Model()
    backend = HFCausalRewriteBackend(
        model=model,
        tokenizer=Tokenizer(),
        device="cpu",
        max_new_tokens=16,
    )

    results = backend.generate_windows(
        prompt="",
        completed_prefix="",
        original_window="x = 0\n",
        candidate_indices=(1, 2, 3, 4, 5, 6),
        max_units=3,
    )

    assert model.calls == 1
    assert [result.text for result in results] == [
        "x = 1\n",
        "x = 2\n",
        "x = 3\n",
        "x = 4\n",
        "x = 5\n",
        "x = 6\n",
    ]


def test_local_program_generation_uses_transformers_compatible_seed_state() -> None:
    import torch

    class Tokenizer:
        def __call__(self, _text, **_kwargs):
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, **_kwargs):
            assert ids.tolist() == [7]
            return "x = 7\n"

    class Model:
        def generate(self, **kwargs):
            assert "generator" not in kwargs
            return torch.tensor([[10, 11, 7]])

    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.model = Model()
    runtime.tokenizer = Tokenizer()
    runtime.device = "cpu"
    runtime.max_new_tokens = 16
    runtime.temperature = 0.25
    runtime.top_p = 0.95
    runtime.seed = 7
    runtime.is_encoder_decoder = False

    assert runtime.generate_program(prompt="def f():\n", sample_id="HumanEval/0") == (
        "def f():\nx = 7\n"
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
        "gate_data": {"scale": "pilot", "full_independent_group_min": 1_000},
        "runtime": {"run_id": "pilot-run"},
    }
    full = {
        "method": {"name": "gated_semantic_window_v1"},
        "gate_data": {"scale": "full", "full_independent_group_min": 1_000},
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
        for budget in (3,)
    ]

    _validate_cache_against_data(rows, data)
    rows[-1]["suitable_target"] = True
    with pytest.raises(ValueError, match="label contradicts"):
        _validate_cache_against_data(rows, data)


def test_fast_training_uses_disjoint_test_groups_when_validation_is_empty() -> None:
    rows = [
        {"group_id": "train-a", "split": "train"},
        {"group_id": "train-b", "split": "train"},
        {"group_id": "test-a", "split": "test"},
    ]
    examples = ("train-example-a", "train-example-b", "test-example")

    training, validation, validation_role = gate_production._partition_training_examples(
        examples,
        rows,
        fast_experimental=True,
    )

    assert training == ("train-example-a", "train-example-b")
    assert validation == ("test-example",)
    assert validation_role == "test_fallback"


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
        group.candidate_indices_by_window_length[length] == tuple(range(4))
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
    assert len(cache_rows) == len(probed) * 3
    assert {row["context_length"] for row in cache_rows} == {1, 2, 3}
    assert {row["budget"] for row in cache_rows} == {3}
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


def test_unvalidated_runtime_manifest_loads_hashed_single_statement_profile(
    tmp_path: Path,
) -> None:
    from wfcllm.gate.production import _load_unvalidated_runtime_manifest

    payload = {
        "schema_version": "wfcllm-unvalidated-runtime-thresholds/v1",
        "runtime_profile": "single-statement-structural-accept/v1",
        "close_low_threshold": 0.0,
        "close_high_threshold": 0.000001,
        "suitable_accept_threshold": 0.0,
        "max_units": 1,
    }
    path = tmp_path / "runtime_thresholds.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    manifest, runtime_hash = _load_unvalidated_runtime_manifest(
        tmp_path,
        tokenizer_sha256="a" * 64,
        max_tokens=256,
    )

    assert manifest.runtime_profile == "single-statement-structural-accept/v1"
    assert manifest.close_low_threshold == 0.0
    assert manifest.close_high_threshold == 0.000001
    assert manifest.suitable_accept_threshold == 0.0
    assert manifest.max_units == 1
    assert runtime_hash == hashlib.sha256(path.read_bytes()).hexdigest()


def test_unvalidated_runtime_manifest_rejects_unbound_or_invalid_profile(
    tmp_path: Path,
) -> None:
    from wfcllm.gate.production import _load_unvalidated_runtime_manifest

    (tmp_path / "runtime_thresholds.json").write_text(
        json.dumps(
            {
                "schema_version": "wfcllm-unvalidated-runtime-thresholds/v1",
                "runtime_profile": "single-statement-structural-accept/v1",
                "close_low_threshold": 0.0,
                "close_high_threshold": 1.0,
                "suitable_accept_threshold": 0.0,
                "max_units": 4,
                "unexpected": True,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="schema mismatch"):
        _load_unvalidated_runtime_manifest(
            tmp_path,
            tokenizer_sha256="a" * 64,
            max_tokens=256,
        )


def test_keyed_text_region_scorer_is_stable_keyed_and_non_degenerate() -> None:
    from wfcllm.gate.production import KeyedTextRegionWindowScorer

    scorer = KeyedTextRegionWindowScorer(b"deployment-key")
    repeated = scorer.score(window_text="value = 1\n", parent_descriptor="module")
    assert scorer.score(window_text="value = 1\n", parent_descriptor="module") == repeated
    assert repeated.stable is True
    assert repeated.margin == 1.0
    observed = {
        scorer.score(
            window_text=f"value = {index}\n",
            parent_descriptor="module",
        ).hit
        for index in range(64)
    }
    assert observed == {False, True}
