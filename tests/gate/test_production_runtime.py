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
from wfcllm.gate.data import RewriteCandidate, RewriteRequest, StructuralBoundary
from wfcllm.generation.window_rewriter import CausalWindowRewriter
from wfcllm.windowing import ParentDescriptor, WINDOW_CONTRACT_VERSION
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
    rewrite_model = tmp_path / "rewrite-model"
    encoder = tmp_path / "encoder"
    gate_model = tmp_path / "gate-model"
    for path in (model, rewrite_model, encoder, gate_model):
        path.mkdir()
    encoder_checkpoint = tmp_path / "semantic-projection.pt"
    encoder_checkpoint.write_bytes(b"projection")
    return LocalHFGateRuntimeOptions(
        source_catalog=catalog,
        generation_model_path=model,
        rewrite_model_path=rewrite_model,
        semantic_encoder_model_path=encoder,
        semantic_encoder_checkpoint_path=encoder_checkpoint,
        gate_base_model_path=gate_model,
        model_device="cuda",
        gate_device="cuda",
        cache_dir=tmp_path / "cache",
    )


@pytest.mark.parametrize("gamma", [True, 0.0, 1.0, -0.1, "0.45"])
def test_runtime_options_reject_invalid_lsh_gamma(tmp_path: Path, gamma) -> None:
    with pytest.raises(ValueError, match="lsh_gamma"):
        replace(_runtime_options(tmp_path), lsh_gamma=gamma)


@pytest.mark.parametrize(
    ("contract", "code", "expected_type"),
    [
        (
            "cpp-statement-window/v1",
            "int f() { int x = 1; return x; }",
            "return_statement",
        ),
        (
            "java-statement-window/v1",
            "class A { int f() { int x = 1; return x; } }",
            "return_statement",
        ),
    ],
)
def test_production_adapter_parses_configured_language_contract(
    tmp_path: Path,
    contract: str,
    code: str,
    expected_type: str,
) -> None:
    from wfcllm.gate.production import LocalHFProductionAdapter

    options = replace(_runtime_options(tmp_path), window_contract_version=contract)
    row = json.loads(options.source_catalog.read_text(encoding="utf-8"))
    row["code"] = code
    options.source_catalog.write_text(json.dumps(row) + "\n", encoding="utf-8")
    source_manifest = {
        "catalog_sha256": gate_production._sha256_file(options.source_catalog),
        "source_count": 1,
    }

    parsed = LocalHFProductionAdapter(options).parse_statement_units(
        source_manifest,
        object(),
    )

    assert any(unit.node_type == expected_type for unit in parsed[0].units)


def test_deployment_scorer_and_runtime_hash_bind_lsh_gamma(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured = {}

    def scorer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr(
        gate_production,
        "_load_public_semantic_components",
        lambda _options: SimpleNamespace(verifier=object()),
    )
    monkeypatch.setattr("wfcllm.semantic.window_lsh.SemanticWindowScorer", scorer)
    baseline = replace(_runtime_options(tmp_path), lsh_dimension=12, lsh_gamma=0.25)
    configured = replace(baseline, lsh_gamma=0.45)

    gate_production.build_local_semantic_window_scorer(configured, b"deployment")

    assert captured["k"] == round(0.45 * (2**12))
    assert gate_production.local_semantic_runtime_hash(configured) != (
        gate_production.local_semantic_runtime_hash(baseline)
    )


def test_production_adapter_no_longer_exposes_validate_candidate() -> None:
    assert not hasattr(gate_production.LocalHFProductionAdapter, "validate_candidate")
    assert "validate_candidate" not in gate_production.LocalHFProductionAdapter.capabilities


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

        def signature_and_margin_modes(self, _code):
            return (self.bit, 0, 0, 0), 0.5, True, True

        def semantic_reference_cosine(self, _reference, _candidate):
            return 1.0

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


def test_local_semantic_runtime_uses_real_mode_measurements() -> None:
    class Verifier:
        def verify(self, *_args):
            raise AssertionError("repeat-only verify must not be used")

        def signature_and_margin_modes(self, code_text):
            assert code_text == "x = 1"
            return (1, 0, 1), 0.12, False, True

        def semantic_reference_cosine(self, _reference, _candidate):
            return 1.0

    runtime = gate_production.LocalSemanticRuntime(Verifier())

    assert runtime.signature_and_margin("x = 1") == (
        (1, 0, 1), 0.12, False, True
    )


def test_local_semantic_runtime_canonicalizes_all_encoder_inputs() -> None:
    calls: list[tuple[str, ...]] = []

    class Verifier:
        def signature_and_margin_modes(self, code_text):
            calls.append(("signature", code_text))
            return (1, 0, 1), 0.25, True, True

        def semantic_reference_cosine(self, reference_text, candidate_text):
            calls.append(("cosine", reference_text, candidate_text))
            return 1.0

    runtime = gate_production.LocalSemanticRuntime(Verifier())
    runtime.semantic_reference_cosine("x = 1\r\n", "x = 1   \n\n")
    runtime.signature_and_margin("x = 1\n")

    assert calls == [
        ("cosine", "x = 1", "x = 1"),
        ("signature", "x = 1"),
    ]


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
        def __init__(self):
            self.instructions = None

        def __call__(self, text, **_kwargs):
            self.instructions = text
            batch = len(text) if isinstance(text, list) else 1
            return {
                "input_ids": torch.tensor([[10, 11]] * batch),
                "attention_mask": torch.tensor([[1, 1]] * batch),
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
            assert "num_return_sequences" not in kwargs
            return torch.tensor(
                [[10, 11, index] for index in range(1, 4)]
            )

    model = Model()
    tokenizer = Tokenizer()
    backend = HFCausalRewriteBackend(
        model=model,
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=16,
    )

    results = backend.generate_windows(
        prompt="",
        completed_prefix="",
        original_window="x = 0\n",
        candidate_indices=(1, 2, 3),
        max_units=3,
    )

    assert model.calls == 1
    assert isinstance(tokenizer.instructions, list)
    assert len(tokenizer.instructions) == 3
    assert "Conservative plan A" in tokenizer.instructions[0]
    assert "Conservative plan B" in tokenizer.instructions[1]
    assert "Conservative plan C" in tokenizer.instructions[2]
    assert [result.text for result in results] == [
        "x = 1\n",
        "x = 2\n",
        "x = 3\n",
    ]


def test_rewrite_code_extracts_all_fenced_blocks_after_explanatory_text() -> None:
    text = (
        "Here are the two equivalent statements:\n\n"
        "```python\nleft = item.left\n```\n\n"
        "and then:\n\n"
        "```python\nright = item.right\n```\n"
    )

    assert gate_production._extract_rewrite_code(
        text,
        original_window="left = item.left\n    right = item.right\n",
        completed_prefix="def build(item):\n    ",
    ) == "left = item.left\n    right = item.right\n"


def test_rewrite_code_restores_prefix_indent_after_first_generated_line() -> None:
    assert gate_production._extract_rewrite_code(
        "left = item.left\nright = item.right\nresult = left + right\n",
        original_window=(
            "left = item.left\n"
            "    right = item.right\n"
            "    result = left + right\n"
        ),
        completed_prefix="def build(item):\n    ",
    ) == (
        "left = item.left\n"
        "    right = item.right\n"
        "    result = left + right\n"
    )


def test_hf_rewrite_normalization_preserves_nested_w3_parent_scope() -> None:
    import torch

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            batch = len(text) if isinstance(text, list) else 1
            return {
                "input_ids": torch.tensor([[10, 11]] * batch),
                "attention_mask": torch.tensor([[1, 1]] * batch),
            }

        def decode(self, _ids, **_kwargs):
            return (
                "Here is the equivalent code:\n\n"
                "```python\n"
                "left = item.left\n"
                "right = item.right\n"
                "result = left + right\n"
                "```\n"
            )

    class Model:
        config = type("Config", (), {"is_encoder_decoder": False})()

        def generate(self, **kwargs):
            return torch.tensor(
                [
                    [10, 11, index]
                    for index in range(1, kwargs["input_ids"].shape[0] + 1)
                ]
            )

    completed_prefix = "def build(item):\n    "
    original_window = (
        "left = item.left\n"
        "    right = item.right\n"
        "    result = left + right\n"
    )
    parsed_units = gate_production.PythonStatementUnitExtractor().extract(
        completed_prefix + original_window
    )
    first = next(unit for unit in parsed_units if "left = item.left" in unit.text)
    canonical_parent = ParentDescriptor(
        contract_version=WINDOW_CONTRACT_VERSION,
        ancestor_node_types=first.parent_path[:-1],
        direct_parent_type=first.direct_parent_type,
        first_unit_ordinal=first.direct_child_ordinal,
        compound_header_role="header" if first.compound_header else "body",
    ).canonical
    request = RewriteRequest(
        prompt="",
        completed_prefix=completed_prefix,
        original_window=original_window,
        canonical_parent=canonical_parent,
        window_start_unit_id="unit-0",
        window_length=3,
        structural_boundary=StructuralBoundary(
            21, 93, 1, "block", ("unit-0", "unit-1", "unit-2"), False, True
        ),
    )
    backend = HFCausalRewriteBackend(
        model=Model(), tokenizer=Tokenizer(), device="cpu", max_new_tokens=64
    )

    results = CausalWindowRewriter(backend).rewrite_many(
        request, candidate_indices=(1, 2, 3)
    )

    assert [
        (result.parse_status, result.unit_count, result.same_parent_scope)
        for result in results
    ] == [("ok", 3, True)] * 3


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


def test_encoder_decoder_program_generation_keeps_benchmark_prompt() -> None:
    import torch

    from wfcllm.gate.production import LocalHFProgramGenerator

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, **_kwargs):
            return " return 1;\n}"

    class Model:
        def generate(self, **_kwargs):
            return torch.tensor([[7, 8]])

    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.model = Model()
    runtime.tokenizer = Tokenizer()
    runtime.device = "cpu"
    runtime.max_new_tokens = 16
    runtime.temperature = 0.0
    runtime.top_p = 0.95
    runtime.seed = 7
    runtime.is_encoder_decoder = True

    prompt = "int f() {"
    assert runtime.generate_program(prompt=prompt, sample_id="cpp/0").startswith(
        prompt
    )


def test_mbpp_program_prompt_uses_humaneval_style_interface_prefix() -> None:
    code_prefix = (
        "def target(arg1, arg2):\n"
        '    """Natural task.\n'
        "    Return the result instead of printing it.\n"
        '    """\n'
    )

    prompt = gate_production._program_generation_prompt(
        code_prefix,
        sample_id="mbpp/1",
    )

    assert prompt == code_prefix
    assert "placeholder" not in prompt.lower()


def test_mbpp_generation_combines_interface_prefix_with_completion() -> None:
    import torch

    code_prefix = (
        "def target(arg1, arg2):\n"
        '    """Return the sum."""\n'
    )

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            assert text == code_prefix
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, **_kwargs):
            assert ids.tolist() == [7]
            return "    return arg1 + arg2\n"

    class Model:
        def generate(self, **_kwargs):
            return torch.tensor([[10, 11, 7]])

    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.model = Model()
    runtime.tokenizer = Tokenizer()
    runtime.device = "cpu"
    runtime.max_new_tokens = 16
    runtime.temperature = 0.0
    runtime.top_p = 0.95
    runtime.seed = 7
    runtime.is_encoder_decoder = False

    assert runtime.generate_program(
        prompt=code_prefix,
        sample_id="mbpp/1",
    ) == (code_prefix + "    return arg1 + arg2\n")


def test_mbpp_chat_generation_requests_complete_interface_program() -> None:
    import torch

    code_prefix = (
        "def target(items):\n"
        '    """Return the number of items."""\n'
    )

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {
                "tokenize": False,
                "add_generation_prompt": True,
            }
            assert messages[0]["role"] == "user"
            content = messages[0]["content"]
            assert "one complete executable Python program" in content
            assert "at least four semantically meaningful statements" in content
            assert "contribute to the returned value" in content
            assert "Do not use a one-line implementation" in content
            assert "do not add dead code" in content
            assert code_prefix in content
            assert "tests" in content
            return "<chat-prompt>"

        def __call__(self, text, **_kwargs):
            assert text == "<chat-prompt>"
            return {
                "input_ids": torch.tensor([[10, 11]]),
                "attention_mask": torch.tensor([[1, 1]]),
            }

        def decode(self, ids, **_kwargs):
            assert ids.tolist() == [7]
            return "def targte(items):\n    return len(items)\n"

    class Model:
        def generate(self, **_kwargs):
            return torch.tensor([[10, 11, 7]])

    runtime = object.__new__(LocalHFProgramGenerator)
    runtime.model = Model()
    runtime.tokenizer = Tokenizer()
    runtime.device = "cpu"
    runtime.max_new_tokens = 16
    runtime.temperature = 0.0
    runtime.top_p = 0.95
    runtime.seed = 7
    runtime.is_encoder_decoder = False
    runtime.program_prompt_mode = "mbpp_chat"

    assert runtime.generate_program(
        prompt=code_prefix,
        sample_id="mbpp/1",
    ) == (
        "def target(items):\n"
        '    """Return the number of items."""\n'
        "    return len(items)\n"
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


class _SemanticallyDivergentRewriter:
    def rewrite(self, request, *, candidate_index):
        return RewriteCandidate(
            code=request.original_window.replace(" = ", "="),
            parse_status="ok",
            unit_count=request.window_length,
            same_parent_scope=True,
            boundary_span=(
                request.structural_boundary.start_byte,
                request.structural_boundary.end_byte,
            ),
            generation_seed_id=f"semantic-reject-seed-{candidate_index}",
            rewrite_config_id="semantic-reject-test-rewriter/v1",
        )


class _SemanticRuntime:
    def semantic_reference_cosine(self, reference_text: str, candidate_text: str):
        assert reference_text and candidate_text
        return 1.0 if reference_text == candidate_text else 0.5

    def signature_and_margin(self, window_text: str):
        assert window_text
        return (0, 0, 0, 0), 1.0, True, True


class _CountingSemanticRuntime(_SemanticRuntime):
    def __init__(self) -> None:
        self.calls: list[str] = []

    def signature_and_margin(self, window_text: str):
        self.calls.append(window_text)
        return super().signature_and_margin(window_text)


class _ParseErrorRewriter:
    def rewrite(self, request, *, candidate_index):
        return RewriteCandidate(
            code="unparseable rewrite",
            parse_status="parse_error",
            unit_count=0,
            same_parent_scope=False,
            boundary_span=(0, 0),
            generation_seed_id=f"invalid-seed-{candidate_index}",
            rewrite_config_id="invalid-test-rewriter/v1",
        )


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


def test_local_hf_probe_keeps_parser_invalid_rewrites_evidence_free(
    tmp_path: Path,
) -> None:
    from wfcllm.gate.production import LocalHFProductionAdapter

    options = _runtime_options(tmp_path)
    source_hash = __import__("hashlib").sha256(
        options.source_catalog.read_bytes()
    ).hexdigest()
    manifest = {
        "schema_version": "wfcllm-gate-source-manifest/v1",
        "catalog_sha256": source_hash,
        "source_count": 1,
    }
    adapter = LocalHFProductionAdapter(options)
    adapter._rewriter = _ParseErrorRewriter()
    runtime = _CountingSemanticRuntime()
    adapter._semantic_runtime = runtime
    adapter._gate_tokenizer = _GateTokenizer()
    config = _pipeline_config(tmp_path)

    parsed = adapter.parse_statement_units(manifest, config)
    generated = tuple(adapter.generate_candidate_trajectories(parsed, config))
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

    assert len(runtime.calls) == len(generated) * 3
    for group in probed:
        for length in (1, 2, 3):
            observations = group.candidate_observations_by_length[str(length)]
            results = group.probe_results_by_length[str(length)]
            assert observations[0].lsh_signature == (0, 0, 0, 0)
            assert set(results[0]) == set((*training.key_ids, *holdout.key_ids))
            for observation, candidate_results in zip(
                observations[1:], results[1:], strict=True
            ):
                assert observation.parse_status == "parse_error"
                assert observation.unit_count == 0
                assert observation.lsh_signature is None
                assert not observation.lsh_by_key_id
                assert observation.stable_across_precision_modes is False
                assert observation.stable_across_batch_modes is False
                assert candidate_results == {}


def test_local_hf_probe_keeps_semantically_rejected_rewrites_evidence_free(
    tmp_path: Path,
) -> None:
    from wfcllm.gate.production import LocalHFProductionAdapter

    options = replace(
        _runtime_options(tmp_path), semantic_preservation_threshold=0.9
    )
    source_hash = __import__("hashlib").sha256(
        options.source_catalog.read_bytes()
    ).hexdigest()
    adapter = LocalHFProductionAdapter(options)
    adapter._rewriter = _SemanticallyDivergentRewriter()
    adapter._semantic_runtime = _CountingSemanticRuntime()
    adapter._gate_tokenizer = _GateTokenizer()
    config = _pipeline_config(tmp_path)
    parsed = adapter.parse_statement_units(
        {
            "schema_version": "wfcllm-gate-source-manifest/v1",
            "catalog_sha256": source_hash,
            "source_count": 1,
        },
        config,
    )
    generated = tuple(adapter.generate_candidate_trajectories(parsed, config))

    probed = tuple(
        adapter.run_multi_key_lsh_probe(
            generated,
            training_keys=_KeyView("train", 32),
            holdout_keys=_KeyView("holdout", 8),
            config=config,
        )
    )

    for group in probed:
        for length in (1, 2, 3):
            observations = group.candidate_observations_by_length[str(length)]
            assert observations[0].semantic_preservation_passed is True
            for observation, results in zip(
                observations[1:],
                group.probe_results_by_length[str(length)][1:],
                strict=True,
            ):
                assert observation.semantic_reference_cosine == pytest.approx(0.5)
                assert observation.semantic_preservation_passed is False
                assert observation.lsh_signature is None
                assert not observation.lsh_by_key_id
                assert results == {}
