from __future__ import annotations

from dataclasses import dataclass

from wfcllm.generation.gated_generator import GatedGenerator, RewriteTokens
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindAstEquivalentWindowRewriter,
    KeyBlindCppEquivalentWindowRewriter,
    KeyBlindJavaEquivalentWindowRewriter,
    RewriteGeneration,
)
from wfcllm.windowing import GateScores, GateThresholds, WindowPartitioner
from wfcllm.windowing.multilanguage import get_statement_unit_extractor


class Gate:
    def predict(self, serialized_input: str) -> GateScores:
        assert "secret_key" not in serialized_input
        closes = "[CURRENT_UNIT_COUNT] 3" in serialized_input
        return GateScores(0.9 if closes else 0.1, 0.9, True, 0.0, True)


class Scorer:
    def __init__(self, hits: list[bool], semantic_passes: list[bool] | None = None) -> None:
        self.hits = iter(hits)
        self.semantic_passes = iter(semantic_passes or [True] * 100)
        self.scored_texts: list[str] = []
        self.scored_parents: list[str] = []

    def score(self, *, window_text: str, parent_descriptor: str):
        self.scored_texts.append(window_text)
        self.scored_parents.append(parent_descriptor)
        hit = next(self.hits)
        return type(
            "Evidence",
            (),
            {
                "hit": hit,
                "stable": True,
                "margin": 1.0,
                "signature": (1, 0, 1),
            },
        )()

    def compare_semantics(self, *, reference_text: str, candidate_text: str):
        passed = next(self.semantic_passes)
        return type(
            "SemanticPreservation",
            (),
            {"cosine": 0.95 if passed else 0.5, "passed": passed},
        )()


class Rewriter:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.requests = []

    def rewrite_window(self, request, *, candidate_index: int):
        self.requests.append(request)
        text = self.texts[candidate_index - 1]
        return _parsed_rewrite(request, text, candidate_index)


class BatchRewriter(Rewriter):
    def rewrite_windows(self, request, *, candidate_indices):
        assert candidate_indices == (1, 2, 3)
        self.requests.append(request)
        return tuple(
            _parsed_rewrite(request, text, candidate_index)
            for candidate_index, text in enumerate(self.texts, 1)
        )


def _parsed_rewrite(request, text: str, candidate_index: int):
    strategy = ("a", "b", "c")[candidate_index - 1]

    class Backend:
        def generate_window(self, **_kwargs):
            return RewriteGeneration(
                tuple(range(len(text.encode()))),
                text,
                generation_seed_id=f"local-hf-v1-batch:test:{candidate_index}",
                rewrite_config_id=f"strategy={strategy};count=3",
            )

    return CausalWindowRewriter(Backend()).rewrite_window(
        request, candidate_index=candidate_index
    )


def _generator(rewrites: list[str], hits: list[bool], *, max_rewrites: int = 3):
    return GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=Scorer(hits),
        rewriter=Rewriter(rewrites),
        max_rewrites=max_rewrites,
    )


def test_rewrite_with_changed_unit_count_is_rejected() -> None:
    result = _generator(
        ["a = 10\nb = 20\n"] * 3, [False]
    ).generate(
        prompt="", original="a = 1\nb = 2\nc = 3\n"
    )
    assert result.final_code == "a = 1\nb = 2\nc = 3\n"
    assert result.audit[0].original_unit_count == 3
    assert result.audit[0].selected_unit_count == 3
    assert result.candidates[1].evaluation_status == "structure_rejected"
    assert result.candidates[1].keyed_lsh_scored is False
    assert result.audit[0].rollback_anchor <= result.audit[0].original_span.start
    assert result.audit[0].previous_statement_text_unchanged is True


def test_online_and_gate_data_use_byte_identical_rewrite_request() -> None:
    from wfcllm.gate.data import _make_request
    from wfcllm.windowing import PythonStatementUnitExtractor

    source = "a = 1\nb = 2\nc = 3\n"
    units = tuple(PythonStatementUnitExtractor().extract(source))
    gate_request = _make_request(
        units,
        start_index=0,
        window=units,
        prompt="task",
        source_bytes=source.encode("utf-8"),
    )
    rewriter = BatchRewriter([source, source, source])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=Scorer([True]),
        rewriter=rewriter,
        max_rewrites=3,
    )

    generator.generate(prompt="task", original=source)

    online_request = rewriter.requests[0]
    assert online_request.to_dict() == gate_request.to_dict()


def test_all_failed_rewrites_restore_original_and_continue() -> None:
    result = _generator(
        ["bad one\n", "bad two\n", "bad three\n"],
        [False, False, False, False],
    ).generate(
        prompt="", original="x = 1\ny = 2\n", suffix="return x + y\n"
    )
    assert result.final_code == "x = 1\ny = 2\nreturn x + y\n"
    assert result.audit[0].selected_candidate_index == 0


def test_selected_rewrite_preserves_source_separator_after_window() -> None:
    scorer = Scorer([False, True])
    result = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=Rewriter(["x = 10\ny = 20\nz = 30"] * 3),
        max_rewrites=3,
    ).generate(prompt="", original="x = 1\ny = 2\nz = 3\n")

    assert result.final_code == "x = 10\ny = 20\nz = 30\n"
    assert result.audit[0].selected_candidate_index == 1
    assert len(set(scorer.scored_parents)) == 1


def test_rewrites_must_be_one_to_three_units_and_same_parent() -> None:
    result = _generator(
        ["", "a=1\nb=2\nc=3\nd=4\n", "if ok:\n    x = 1\n"],
        [False, True, True, True],
    ).generate(prompt="", original="a = 1\nb = 2\nc = 3\n")
    assert result.final_code == "a = 1\nb = 2\nc = 3\n"
    assert result.audit[0].selected_candidate_index == 0


def test_compound_header_cannot_change_role_and_overflow_is_not_rewritten() -> None:
    header = _generator(["for x in xs:\n"] * 3, [False, True, True, True]).generate(
        prompt="", original="if ok:\n", suffix="    value = 1\n"
    )
    assert header.final_code == "if ok:\n    value = 1\n"

    rewriter = Rewriter(["x = 2\n"])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8, max_input_tokens=1), lambda text: 2
        ),
        scorer=Scorer([]),
        rewriter=rewriter,
        max_rewrites=1,
    )
    assert generator.generate(prompt="", original="x = 1\n").final_code == "x = 1\n"
    assert rewriter.requests == []


def test_protected_prefix_is_preserved_byte_for_byte() -> None:
    result = _generator(["  x = 2\n"], [False, True]).generate(
        prompt="", original="  x = 1\n", protected_prefix="  "
    )
    assert result.final_code.startswith("  ")


def test_semantic_failure_skips_keyed_score_and_tries_next_candidate() -> None:
    scorer = Scorer([False, True], semantic_passes=[False, True])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=Rewriter(["a = 2\nb = 2\n", "a = 1\nb = 3\n"]),
        max_rewrites=2,
    )

    result = generator.generate(prompt="", original="a = 1\nb = 2\n")

    assert result.final_code == "a = 1\nb = 3\n"
    assert result.audit[0].selected_candidate_index == 2
    assert len(scorer.scored_texts) == 2


def test_ast_equivalence_certificate_overrides_low_encoder_cosine() -> None:
    scorer = Scorer([False, True], semantic_passes=[False])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=KeyBlindAstEquivalentWindowRewriter(),
        max_rewrites=3,
    )

    result = generator.generate(
        prompt="",
        original="label = 'ready'\ncount = 10\nreturn_value = (label, count)\n",
    )

    assert result.audit[0].original_unit_count == 3
    assert result.audit[0].selected_candidate_index == 1
    assert result.audit[0].semantic_validation_rule == "python-ast-equivalent/v1"
    assert result.candidates[1].semantic_reference_cosine == 0.5
    assert result.candidates[1].semantic_preservation_passed is True
    assert result.candidates[1].semantic_validation_rule == "python-ast-equivalent/v1"
    assert result.final_code != "label = 'ready'\ncount = 10\nreturn_value = (label, count)\n"


def test_cpp_keyblind_equivalence_certificate_overrides_low_encoder_cosine() -> None:
    scorer = Scorer([False, True] * 20, semantic_passes=[False] * 20)
    extractor = get_statement_unit_extractor("cpp")
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(),
            GateThresholds(0.2, 0.7, 0.8),
            lambda text: len(text),
            window_contract_version="cpp-statement-window/v1",
        ),
        scorer=scorer,
        rewriter=KeyBlindCppEquivalentWindowRewriter(
            extractor=extractor,
            window_contract_version="cpp-statement-window/v1",
        ),
        max_rewrites=3,
        extractor=extractor,
    )

    result = generator.generate(
        prompt="",
        original="int add(int a, int b) {\n  int sum = a + b;\n  return sum;\n}\n",
    )

    selected = [row for row in result.audit if row.selected_candidate_index > 0]
    certified_candidates = [
        row
        for row in result.candidates
        if row.semantic_validation_rule == "cpp-keyblind-equivalent/v1"
    ]
    assert selected
    assert selected[0].semantic_validation_rule == "cpp-keyblind-equivalent/v1"
    assert certified_candidates[0].semantic_reference_cosine == 0.5
    assert certified_candidates[0].semantic_preservation_passed is True
    assert result.final_code != "int add(int a, int b) {\n  int sum = a + b;\n  return sum;\n}\n"


def test_fast_profile_can_select_certified_cpp_rewrite_without_lsh_hit() -> None:
    scorer = Scorer([False] * 20, semantic_passes=[False] * 20)
    extractor = get_statement_unit_extractor("cpp")
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(),
            GateThresholds(0.2, 0.7, 0.8),
            lambda text: len(text),
            window_contract_version="cpp-statement-window/v1",
        ),
        scorer=scorer,
        rewriter=KeyBlindCppEquivalentWindowRewriter(
            extractor=extractor,
            window_contract_version="cpp-statement-window/v1",
        ),
        max_rewrites=18,
        extractor=extractor,
        accept_certified_rewrite_without_hit=True,
    )

    result = generator.generate(
        prompt="",
        original="int add(int a, int b) {\n  int sum = a + b;\n  return sum;\n}\n",
    )

    selected = [row for row in result.audit if row.selected_candidate_index > 0]
    selected_candidates = [row for row in result.candidates if row.selected]
    assert selected
    assert selected[0].semantic_validation_rule == "cpp-keyblind-equivalent/v1"
    assert any(
        row.candidate_index > 0 and row.semantic_hit is False
        for row in selected_candidates
    )


def test_fast_profile_can_select_certified_java_rewrite_without_lsh_hit() -> None:
    scorer = Scorer([False] * 20, semantic_passes=[False] * 20)
    extractor = get_statement_unit_extractor("java")
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(),
            GateThresholds(0.2, 0.7, 0.8),
            lambda text: len(text),
            window_contract_version="java-statement-window/v1",
        ),
        scorer=scorer,
        rewriter=KeyBlindJavaEquivalentWindowRewriter(
            extractor=extractor,
            window_contract_version="java-statement-window/v1",
        ),
        max_rewrites=18,
        extractor=extractor,
        accept_certified_rewrite_without_hit=True,
    )

    result = generator.generate(
        prompt="",
        original=(
            "class Solution {\n"
            "  int add(int a, int b) {\n"
            "    int sum = a + b;\n"
            "    return sum;\n"
            "  }\n"
            "}\n"
        ),
    )

    selected = [row for row in result.audit if row.selected_candidate_index > 0]
    selected_candidates = [row for row in result.candidates if row.selected]
    assert selected
    assert selected[0].semantic_validation_rule == "java-keyblind-equivalent/v1"
    assert any(
        row.candidate_index > 0
        and row.semantic_hit is False
        and row.evaluation_status == "fast_certified_rewrite_selected"
        for row in selected_candidates
    )


def test_batched_online_trajectory_retains_every_candidate_sidecar() -> None:
    scorer = Scorer(
        [False, False, True], semantic_passes=[False, True, True]
    )
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=BatchRewriter(
            ["a = 2\nb = 2\n", "a = 1\nb = 3\n", "a = 1\nb = 4\n"]
        ),
        max_rewrites=3,
    )

    result = generator.generate(prompt="", original="a = 1\nb = 2\n")

    assert [row.candidate_index for row in result.candidates] == [0, 1, 2, 3]
    first, rejected, missed, accepted = result.candidates
    assert first.selected is False
    assert rejected.semantic_preservation_passed is False
    assert rejected.keyed_lsh_scored is False
    assert rejected.lsh_signature is None
    assert missed.keyed_lsh_scored is True
    assert missed.semantic_hit is False
    assert accepted.selected is True
    assert accepted.semantic_hit is True
