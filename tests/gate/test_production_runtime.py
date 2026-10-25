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


def test_float_gate_cache_preserves_execution_context_and_invalidates_training() -> None:
    import torch

    class Encoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0
            self.linear = torch.nn.Linear(1, 2, bias=False)
            self.linear.weight.data.fill_(0.25)

        def forward(self, *, input_ids, attention_mask):
            self.calls += 1
            logits = self.linear(input_ids.float())
            return SimpleNamespace(
                close_logits=logits[:, 0], suitable_logits=logits[:, 1]
            )

    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {
                "input_ids": torch.tensor([[min(len(text), kwargs["max_length"])]]),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            }

    model = Encoder().eval()
    predictor = gate_production._FloatGatePredictor(
        model=model, tokenizer=Tokenizer(), max_tokens=256
    )
    expected = predictor.predict("window")
    assert predictor.predict("window") == expected
    assert model.calls == 1
    assert predictor.predict("another window") != expected
    assert model.calls == 2
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        predictor.predict("window")
    assert model.calls == 3
    model.train()
    model.linear.weight.data.fill_(0.5)
    changed = predictor.predict("window")
    assert changed != expected
    model.eval()
    assert predictor.predict("window") == changed
    assert model.calls == 5


def test_float_gate_moves_both_inputs_to_requested_device() -> None:
    import torch

    requested_devices: list[str] = []

    class InputTensor:
        def to(self, device: str):
            requested_devices.append(device)
            return torch.ones((1, 1))

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            return {"input_ids": InputTensor(), "attention_mask": InputTensor()}

    class Model:
        training = False

        def __call__(self, **_kwargs):
            return SimpleNamespace(
                close_logits=torch.tensor([0.0]), suitable_logits=torch.tensor([1.0])
            )

    predictor = gate_production._FloatGatePredictor(
        model=Model(), tokenizer=Tokenizer(), max_tokens=256, device="cuda:0"
    )
    assert predictor.predict("window").close_probability == 0.5
    assert requested_devices == ["cuda:0", "cuda:0"]


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


def test_cpp_structural_factory_requires_explicit_strategy(tmp_path: Path) -> None:
    from wfcllm.generation.cpp_structural_rewriter import KeyBlindCppStructuralWindowRewriter

    base = _runtime_options(tmp_path)
    native = replace(
        base, window_contract_version="cpp-statement-window/v1",
        rewrite_strategy="cpp_structural_equivalent", rewrite_model_path=None,
    )
    assert isinstance(
        gate_production.load_local_causal_rewriter(native), KeyBlindCppStructuralWindowRewriter
    )
    with pytest.raises(ValueError, match="rewrite_model_path is required"):
        replace(native, rewrite_strategy="model_semantic_window")
    with pytest.raises(ValueError, match=r"C\+\+ structural"):
        replace(base, rewrite_strategy="cpp_structural_equivalent")


def test_js_structural_factory_requires_explicit_strategy(tmp_path: Path) -> None:
    from wfcllm.generation.js_structural_rewriter import KeyBlindJavaScriptStructuralWindowRewriter

    base = _runtime_options(tmp_path)
    native = replace(base, window_contract_version="js-statement-window/v1",
                     rewrite_strategy="js_structural_equivalent", rewrite_model_path=None)
    assert isinstance(gate_production.load_local_causal_rewriter(native), KeyBlindJavaScriptStructuralWindowRewriter)
    with pytest.raises(ValueError, match="rewrite_model_path is required"):
        replace(native, rewrite_strategy="model_semantic_window")
    with pytest.raises(ValueError, match="JavaScript structural"):
        replace(base, rewrite_strategy="js_structural_equivalent")


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


def test_deployment_scorer_and_runtime_hash_bind_strict_semantic_margin(
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
    baseline = _runtime_options(tmp_path)
    configured = replace(baseline, semantic_margin=0.005)

    gate_production.build_local_semantic_window_scorer(
        configured, b"deployment"
    )

    assert captured["margin"] == 0.005
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


@pytest.mark.parametrize(
    ("text", "node_type", "expected"),
    [
        ("int value = 1;", "declaration", "assignment_literal"),
        ("auto item = values[index];", "declaration", "assignment_reference"),
        ("auto value = build(source);", "declaration", "assignment_call"),
        ("int value = left + right;", "declaration", "assignment_expression"),
        ("value = source.member;", "expression_statement", "assignment_reference"),
        ("value += 2;", "expression_statement", "assignment_literal"),
        ("++index;", "expression_statement", "assignment_expression"),
        ("emit(value);", "expression_statement", "call"),
        ("if (value > 0) {", "if_statement", "branch"),
        ("for (int i = 0; i < limit; ++i) {", "for_statement", "loop"),
        ("while (value > 0) {", "while_statement", "loop"),
        ("return value;", "return_statement", "return"),
        ("throw error;", "throw_statement", "raise"),
        ("int value;", "declaration", "declaration"),
        ("int left = 1, right = value;", "declaration", "assignment_expression"),
    ],
)
def test_cpp_statement_family_uses_cpp_syntax(text: str, node_type: str, expected: str) -> None:
    assert _statement_family(text, cpp_node_type=node_type) == expected


def test_cpp_statement_family_summary_keeps_selected_windows_identical() -> None:
    base = _parsed_source("cpp-dev", "cpp-repo")
    code = "int dev(int source) { int first = 1; int second = source; int third = first + second; return third; }"
    parsed = _ParsedCatalogSource(
        replace(base.record, source_family="oss_cpp", code=code),
        tuple(gate_production.get_statement_unit_extractor("cpp").extract(code)),
    )
    selected = _select_source_stratified_starts((parsed,), max_groups=2)
    old = _source_stratified_selection_summary((parsed,), selected, max_groups=2)
    new = _source_stratified_selection_summary((parsed,), selected, max_groups=2, language="cpp")
    assert old["statement_family_counts"] == {"other": 2}
    assert new["statement_family_counts"] == {"assignment_literal": 1, "assignment_reference": 1}
    assert {key:value for key,value in old.items() if key != "statement_family_counts"} == {
        key:value for key,value in new.items() if key != "statement_family_counts"
    }


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


@pytest.mark.parametrize("encoder_decoder", [False, True])
@pytest.mark.parametrize("batch_size", [3, 18])
def test_batched_rewrites_use_attended_final_tokens_for_causal_models(encoder_decoder: bool, batch_size: int) -> None:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocabulary = {"<unk>": 0, "<pad>": 1, "value": 2}
    internal = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    internal.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=internal, unk_token="<unk>", pad_token="<pad>",
        padding_side="right",
        chat_template="{{ messages[0]['content'] }}",
    )

    class Model:
        config = SimpleNamespace(is_encoder_decoder=encoder_decoder)

        def generate(self, **kwargs):
            mask = kwargs["attention_mask"]
            assert mask.sum(dim=1).unique().numel() > 1
            if encoder_decoder:
                assert torch.all(mask[:, 0] == 1)
                assert torch.any(mask[:, -1] == 0)
                return torch.tensor([[2]] * batch_size)
            assert torch.all(mask[:, -1] == 1)
            return torch.cat([kwargs["input_ids"], torch.tensor([[2]] * batch_size)], dim=1)

    backend = HFCausalRewriteBackend(
        model=Model(), tokenizer=tokenizer, device="cpu", max_new_tokens=1,
    )
    results = backend.generate_windows(
        prompt="example", completed_prefix="", original_window="value",
        candidate_indices=tuple(range(1, batch_size+1)), max_units=1,
    )
    assert [row.text for row in results] == ["value\n"] * batch_size
    assert tokenizer.padding_side == ("right" if encoder_decoder else "left")


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


@pytest.mark.parametrize(
    ("language", "label", "fence"),
    [("cpp", "C++", "c++"), ("java", "Java", "java"), ("js", "JavaScript", "js")],
)
def test_rewrite_backend_uses_target_language_in_prompt_and_output(
    language: str, label: str, fence: str
) -> None:
    import torch

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            self.instructions = text
            count = len(text) if isinstance(text, list) else 1
            return {"input_ids": torch.tensor([[10, 11]] * count)}

        def decode(self, _ids, **_kwargs):
            return f"Equivalent code:\n```{fence}\nint x = 1;\nx += 2;\n```\n"

    class Model:
        config = SimpleNamespace(is_encoder_decoder=False)

        def generate(self, **kwargs):
            return torch.tensor([[10, 11, 7]] * kwargs["input_ids"].shape[0])

    tokenizer = Tokenizer()
    backend = HFCausalRewriteBackend(
        model=Model(), tokenizer=tokenizer, device="cpu", max_new_tokens=16,
        language=language,
    )
    request = dict(prompt="example", completed_prefix="void f() {\n    ",
                   original_window="int x = 1;\n    x += 2;\n", max_units=2)
    single = backend.generate_window(**request, candidate_index=2)
    prompt = tokenizer.instructions
    assert f"2 complete {label} statements" in prompt
    assert f"independently phrased {label} idiom" in prompt
    assert "Python" not in prompt
    batch = backend.generate_windows(**request, candidate_indices=(1, 2, 3))
    assert [single.text, *(row.text for row in batch)] == [
        "int x = 1;\n    x += 2;\n"
    ] * 4


def test_causal_rewriter_factory_passes_cpp_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import transformers

    class Model:
        def to(self, _device):
            return self

        def eval(self):
            return self

        def generate(self, **_kwargs):
            raise AssertionError("factory must not generate")

    class Tokenizer:
        def __call__(self, *_args, **_kwargs):
            raise AssertionError("factory must not tokenize")

        def decode(self, *_args, **_kwargs):
            raise AssertionError("factory must not decode")

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        lambda *_args, **_kwargs: SimpleNamespace(is_encoder_decoder=False))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        lambda *_args, **_kwargs: Model())
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        lambda *_args, **_kwargs: Tokenizer())
    options = replace(_runtime_options(tmp_path), window_contract_version="cpp-statement-window/v1")
    rewriter = gate_production._load_causal_rewriter(options)
    assert rewriter._backend.language == "cpp"
    instruction = rewriter._backend._render_instruction(
        prompt="", completed_prefix="", original_window="x += 1;", max_units=1,
        candidate_index=2,
    )
    assert "complete C++ statements" in instruction


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


@pytest.mark.parametrize(
    "completion",
    [
        "    return arg1 + arg2\n",
        "    return arg1 + arg2\n```\n```python\n"
        "def target(arg1, arg2):\n    return arg1 + arg2\n```\n",
    ],
)
def test_mbpp_generation_combines_interface_prefix_with_completion(
    completion: str,
) -> None:
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
            return completion

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

    source = runtime.generate_program(
        prompt=code_prefix,
        sample_id="mbpp/1",
    )
    assert source == code_prefix + completion

    from wfcllm.generation.completion_finalizer import finalize_mbpp_program

    finalized = finalize_mbpp_program(code_prefix, source)
    assert finalized.applied is True
    assert "return arg1 + arg2" in finalized.code
    assert "```" not in finalized.code


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


@pytest.mark.parametrize(
    ("code", "arguments", "expected"),
    [
        ("def target(items, count):\n    return items[:count]\n", ([1, 2], 1), [1]),
        ("def target(arg1, arg0):\n    return arg1 - arg0\n", (7, 2), 5),
        (
            "def target(value, step):\n    return (lambda value: value + step)(value)\n",
            (7, 2),
            9,
        ),
    ],
)
def test_mbpp_chat_interface_preserves_generated_parameter_bindings(
    code: str, arguments: tuple, expected: object,
) -> None:
    from wfcllm.gate.production import _normalize_mbpp_chat_interface

    prompt = 'def target(arg0, arg1):\n    """Implement the requested operation."""\n'
    normalized = _normalize_mbpp_chat_interface(prompt, code)
    namespace: dict = {}
    exec(normalized, namespace)

    assert namespace["target"](*arguments) == expected


def test_mbpp_chat_interface_keeps_unsupported_variadic_source_unchanged() -> None:
    from wfcllm.gate.production import _normalize_mbpp_chat_interface

    prompt = 'def target(arg0):\n    """Return the argument."""\n'
    code = "def target(*items):\n    return items[0]\n"

    assert _normalize_mbpp_chat_interface(prompt, code) == code


def test_mbpp_chat_interface_binds_generated_positional_prefix() -> None:
    from wfcllm.gate.production import _normalize_mbpp_chat_interface

    prompt = 'def target(arg0, arg1):\n    """Return the list length."""\n'
    code = "def target(items):\n    return len(items)\n"
    namespace: dict = {}
    exec(_normalize_mbpp_chat_interface(prompt, code), namespace)

    assert namespace["target"]([3, 4, 5], 3) == 3


def test_mbpp_chat_interface_does_not_guess_missing_generated_arguments() -> None:
    from wfcllm.gate.production import _normalize_mbpp_chat_interface

    prompt = 'def target(arg0):\n    """Implement the operation."""\n'
    code = "def target(items, required):\n    return items[:required]\n"

    assert _normalize_mbpp_chat_interface(prompt, code) == code


def test_mbpp_chat_interface_keeps_recursive_arity_mismatch_unchanged() -> None:
    from wfcllm.gate.production import _normalize_mbpp_chat_interface

    prompt = 'def target(arg0, arg1):\n    """Count the items."""\n'
    code = (
        "def target(items):\n"
        "    if items:\n"
        "        return 1 + target(items[1:])\n"
        "    return 0\n"
    )

    assert _normalize_mbpp_chat_interface(prompt, code) == code


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
    training.write_text(
        json.dumps([f"train-{index}" for index in range(32)]),
        encoding="utf-8",
    )
    holdout.write_text(
        json.dumps([f"holdout-{index}" for index in range(8)]),
        encoding="utf-8",
    )

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


def test_static_local_hf_adapter_can_mark_smoke_diagnostic(tmp_path: Path) -> None:
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
        diagnostic_test_backend=True,
    )

    assert dependencies.diagnostic_test_backend is True
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


def test_training_cache_variant_count_follows_gate_data_window_lengths(
    tmp_path: Path,
) -> None:
    data = tmp_path / "gate-data"
    data.mkdir()
    index = {
        "group_id": "g-1",
        "split": "train",
        "close_target": True,
        "suitable_target": False,
    }
    (data / "manifest.json").write_text(
        json.dumps({"window_lengths": [1]}) + "\n",
        encoding="utf-8",
    )
    (data / "group_index.jsonl").write_text(json.dumps(index) + "\n", encoding="utf-8")
    rows = [
        {
            "group_id": "g-1",
            "split": "train",
            "context_length": 1,
            "budget": 1,
            "close_target": True,
            "suitable_target": False,
        }
    ]

    _validate_cache_against_data(rows, data)

    rows[0]["suitable_target"] = True
    with pytest.raises(ValueError, match="label contradicts"):
        _validate_cache_against_data(rows, data)


def test_training_cache_variant_count_follows_gate_data_rewrite_budgets(
    tmp_path: Path,
) -> None:
    data = tmp_path / "gate-data"
    data.mkdir()
    index = {
        "group_id": "g-1",
        "split": "train",
        "close_target": True,
        "suitable_target": False,
    }
    (data / "manifest.json").write_text(
        json.dumps({"window_lengths": [1, 2, 3], "rewrite_budgets": [1, 3, 6, 12, 18]})
        + "\n",
        encoding="utf-8",
    )
    (data / "group_index.jsonl").write_text(json.dumps(index) + "\n", encoding="utf-8")
    rows = [
        {
            "group_id": "g-1",
            "split": "train",
            "context_length": context,
            "budget": budget,
            "close_target": True,
            "suitable_target": False if budget == 18 else True,
        }
        for context in (1, 2, 3)
        for budget in (1, 3, 6, 12, 18)
    ]

    _validate_cache_against_data(rows, data)

    with pytest.raises(ValueError, match="variant count"):
        _validate_cache_against_data(rows[:-1], data)


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
    assert len(cache_rows) == len(probed) * 3 * len(config.rewrite_budgets)
    assert {row["context_length"] for row in cache_rows} == {1, 2, 3}
    assert {row["budget"] for row in cache_rows} == set(config.rewrite_budgets)

    split = tuple(adapter.split_groups(probed, config))
    assert split == probed
    adapter.audit_gate_data(
        tmp_path,
        {
            "formal_eligible": True,
            "diagnostic_test_backend": False,
            "diagnostic_only": False,
            "not_official_method": False,
            "config_hash": config.config_hash,
        },
    )
    adapter.audit_gate_data(
        tmp_path,
        {
            "formal_eligible": False,
            "diagnostic_test_backend": True,
            "diagnostic_only": True,
            "not_official_method": True,
            "config_hash": config.config_hash,
        },
    )
    with pytest.raises(ValueError, match="complete gate-data artifact identity"):
        adapter.audit_gate_data(
            tmp_path,
            {
                "formal_eligible": False,
                "diagnostic_test_backend": True,
                "config_hash": config.config_hash,
            },
        )


def test_production_holdout_transfer_excludes_hits_after_r3(tmp_path: Path, monkeypatch) -> None:
    import wfcllm.gate.production as production

    class LateRewriter(_Rewriter):
        def rewrite(self, request, *, candidate_index):
            result = super().rewrite(request, candidate_index=candidate_index)
            if candidate_index == 18:
                return replace(result, code=result.code.replace(" = ", "  = "))
            return result

    class Runtime(_SemanticRuntime):
        def semantic_reference_cosine(self, reference_text, candidate_text):
            return 1.0

        def signature_and_margin(self, window_text):
            return (int("  = " in window_text), 0, 0, 0), 1.0, True, True

    class Keying:
        def __init__(self, material, dimension):
            self.holdout = "holdout" in material

        def derive_descriptor(self, **kwargs):
            return {(int(self.holdout), 0, 0, 0)}

    monkeypatch.setattr(production, "WatermarkKeying", Keying)
    options = _runtime_options(tmp_path)
    adapter = production.LocalHFProductionAdapter(options)
    adapter._rewriter = LateRewriter()
    adapter._semantic_runtime = Runtime()
    adapter._gate_tokenizer = _GateTokenizer()
    config = replace(_pipeline_config(tmp_path), rewrite_count=18, rewrite_budgets=(1, 3, 6, 12, 18))
    manifest = {
        "schema_version": "wfcllm-gate-source-manifest/v1",
        "catalog_sha256": __import__("hashlib").sha256(options.source_catalog.read_bytes()).hexdigest(),
        "source_count": 1,
    }
    generated = adapter.generate_candidate_trajectories(adapter.parse_statement_units(manifest, config), config)
    groups = tuple(adapter.run_multi_key_lsh_probe(
        generated, training_keys=_KeyView("train", 32), holdout_keys=_KeyView("holdout", 8), config=config,
    ))
    assert groups
    for group in groups:
        assert group.r3_success_rate == 1.0
        assert group.holdout_success_rate == 0.0
        assert all(value.hit for key, value in group.probe_results_by_length["3"][18].items() if key.startswith("holdout-"))


def test_gate_probe_excludes_trailing_comments_but_retains_candidate_bytes(tmp_path: Path) -> None:
    class CommentRewriter(_Rewriter):
        def rewrite(self, request, *, candidate_index):
            return replace(super().rewrite(request, candidate_index=candidate_index),
                           code=request.original_window + "\n# outside scored span")

    options = _runtime_options(tmp_path)
    adapter = gate_production.LocalHFProductionAdapter(options)
    adapter._rewriter = CommentRewriter()
    runtime = _CountingSemanticRuntime()
    adapter._semantic_runtime = runtime
    adapter._gate_tokenizer = _GateTokenizer()
    config = _pipeline_config(tmp_path)
    manifest = {"catalog_sha256": hashlib.sha256(options.source_catalog.read_bytes()).hexdigest(), "source_count": 1}
    generated = adapter.generate_candidate_trajectories(adapter.parse_statement_units(manifest, config), config)
    groups = tuple(adapter.run_multi_key_lsh_probe(generated, training_keys=_KeyView("train", 32),
                   holdout_keys=_KeyView("holdout", 8), config=config))
    assert groups and runtime.calls
    assert all("outside scored span" not in text for text in runtime.calls)
    for group in groups:
        for observations in group.candidate_observations_by_length.values():
            assert all(row.semantic_reference_cosine == 1.0 for row in observations)
            assert all(row.lsh_signature == observations[0].lsh_signature for row in observations)
            assert all("# outside scored span" in row.code for row in observations[1:])
    evidence = adapter.gate_data_selection_summary()["semantic_input_span_audit"]
    path = options.cache_dir / evidence["filename"]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == evidence["candidate_count"]
    assert all(row["audit_only"] and row["not_detector_input"] for row in rows)


def test_pilot_and_full_keep_distinct_span_evidence_in_shared_cache(tmp_path: Path) -> None:
    options = _runtime_options(tmp_path)
    config = _pipeline_config(tmp_path)
    manifest = {"catalog_sha256": hashlib.sha256(options.source_catalog.read_bytes()).hexdigest(), "source_count": 1}
    paths = []
    for scale in ("pilot", "full"):
        adapter = gate_production.LocalHFProductionAdapter(options)
        adapter._rewriter = _Rewriter()
        current = replace(config, scale=scale)
        groups = tuple(adapter.generate_candidate_trajectories(
            adapter.parse_statement_units(manifest, current), current))
        assert groups
        evidence = adapter.gate_data_selection_summary()["semantic_input_span_audit"]
        path = options.cache_dir / evidence["filename"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == evidence["sha256"]
        paths.append((path, path.read_bytes()))
    assert paths[0][0] != paths[1][0]
    assert paths[0][0].read_bytes() == paths[0][1]
    repeated = gate_production.LocalHFProductionAdapter(options)
    repeated._rewriter = _Rewriter()
    with pytest.raises(ValueError, match="evidence already exists"):
        tuple(repeated.generate_candidate_trajectories(
            repeated.parse_statement_units(manifest, config), config))
    assert all(path.read_bytes() == before for path, before in paths)


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
