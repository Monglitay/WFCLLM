from __future__ import annotations

from dataclasses import dataclass, replace

import pytest

from wfcllm.generation.gated_generator import GatedGenerator, RewriteTokens
from wfcllm.generation.window_rewriter import (
    CausalWindowRewriter,
    KeyBlindAstEquivalentWindowRewriter,
    KeyBlindCppEquivalentWindowRewriter,
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


class ChannelScorer(Scorer):
    def __init__(self, channel_hits: list[tuple[bool, bool]]) -> None:
        super().__init__([])
        self.channel_hits = iter(channel_hits)

    def score_channels(
        self, *, window_text: str, parent_descriptor: str, channel_count: int
    ):
        self.scored_texts.append(window_text)
        self.scored_parents.append(parent_descriptor)
        assert channel_count == 2
        return tuple(
            type(
                "Evidence",
                (),
                {
                    "hit": hit,
                    "stable": True,
                    "margin": 1.0,
                    "signature": (1, 0, 1),
                },
            )()
            for hit in next(self.channel_hits)
        )


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


@pytest.mark.parametrize(
    ("language", "source"),
    [
        ("python", "a = 1\nb = a + 2\nc = b * 3\n"),
        ("cpp", "int a=1;\nint b=a+2;\nint c=b*3;\n"),
        ("java", "class Demo { static void run(){int a=1; int b=a+2; int c=b*3;} }"),
        ("js", "let a=1;\nlet b=a+2;\nlet c=b*3;\n"),
    ],
)
@pytest.mark.parametrize(
    ("parse_status", "unit_count", "same_parent"),
    [
        ("parse_error", 3, True), ("scope_changed", 3, False),
        ("unit_count_out_of_range", 2, True), ("ok", 2, True), ("ok", 3, False),
    ],
)
def test_online_validation_preserves_supplied_parser_rejections(
    language: str, source: str, parse_status: str, unit_count: int, same_parent: bool,
) -> None:
    extractor = get_statement_unit_extractor(language)
    partitioner = WindowPartitioner(
        Gate(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1,
        window_contract_version=f"{language}-statement-window/v1",
    )
    window = next(window for window in partitioner.partition(extractor.extract(source)).windows
                  if window.suitable and len(window.units) == 3)
    raw = source.encode("utf-8")
    prefix, text, suffix = (
        raw[:window.start_byte].decode("utf-8"),
        raw[window.start_byte:window.end_byte].decode("utf-8"),
        raw[window.end_byte:].decode("utf-8"),
    )
    generator = GatedGenerator(
        partitioner=partitioner, scorer=Scorer([]), rewriter=Rewriter([]),
        extractor=extractor, max_rewrites=1,
    )
    valid = RewriteTokens(
        token_ids=(), text=text, parse_status="ok", unit_count=3,
        same_parent_scope=True,
    )
    kwargs = dict(completed_prefix=prefix, protected_prefix="", original_window=window,
                  completed_suffix=suffix)
    assert generator._validate_candidate(candidate=valid, **kwargs) is not None
    rejected = replace(valid, parse_status=parse_status, unit_count=unit_count,
                       same_parent_scope=same_parent)
    assert generator._validate_candidate(candidate=rejected, **kwargs) is None


def test_rejected_parse_does_not_abort_program_or_receive_semantic_score() -> None:
    class RejectedRewrite:
        def rewrite_window(self, request, *, candidate_index):
            result = _parsed_rewrite(request, request.original_window, candidate_index)
            return replace(result, parse_status="parse_error")

    scorer = Scorer([False])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1,
        ), scorer=scorer, rewriter=RejectedRewrite(), max_rewrites=1,
    )
    original = "a = 1\nb = a + 2\nc = b * 3\n"
    result = generator.generate(prompt="", original=original)
    assert result.final_code == original
    assert len(scorer.scored_texts) == 1
    assert result.candidates[1].parse_status == "parse_error"
    assert result.candidates[1].evaluation_status == "structure_rejected"
    assert result.candidates[1].semantic_reference_cosine is None
    assert result.candidates[1].keyed_lsh_scored is False


def test_subsequent_rewrite_request_uses_committed_prefix() -> None:
    class CloseEach:
        def predict(self, _text):
            return GateScores(0.9, 0.9, True, 0.0, True)

    class SequentialRewriter(Rewriter):
        def rewrite_window(self, request, *, candidate_index):
            self.requests.append(request)
            text = "a = 10\n" if len(self.requests) == 1 else "b = 20\n"
            return _parsed_rewrite(request, text, candidate_index)

    rewriter = SequentialRewriter([])
    result = GatedGenerator(
        partitioner=WindowPartitioner(
            CloseEach(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1
        ),
        scorer=Scorer([False, True, False, True]),
        rewriter=rewriter,
        max_rewrites=1,
    ).generate(prompt="", original="a = 1\nb = 2\n")

    assert result.final_code == "a = 10\nb = 20\n"
    assert rewriter.requests[1].completed_prefix == "a = 10\n"
    assert all(row.previous_statement_text_unchanged for row in result.audit)


def test_candidate_must_close_before_unmodified_following_statement() -> None:
    class ContextGate:
        def predict(self, text):
            current = text.split("[CURRENT]\n", 1)[1]
            defer = "a = 10" in current and "[CURRENT_UNIT_COUNT] 1" in text
            return GateScores(0.1 if defer else 0.9, 0.9, True, 0.0, True)

    result = GatedGenerator(
        partitioner=WindowPartitioner(
            ContextGate(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1
        ),
        scorer=Scorer([False, True, True]),
        rewriter=Rewriter(["a = 10\n"]),
        max_rewrites=1,
    ).generate(prompt="", original="a = 1\nb = 2\n")

    assert result.final_code == "a = 1\nb = 2\n"
    assert result.audit[0].selected_candidate_index == 0
    assert result.candidates[1].evaluation_status == "structure_rejected"


def test_candidate_gate_checks_actual_preceding_statements() -> None:
    class ContextGate:
        def predict(self, text):
            previous, current = text.split("[CURRENT]\n", 1)
            uncertain = "a = 1" in previous and "b = 20" in current
            return GateScores(0.5 if uncertain else 0.9, 0.9, True, 0.0, True)

    result = GatedGenerator(
        partitioner=WindowPartitioner(
            ContextGate(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1
        ),
        scorer=Scorer([True, False, True]),
        rewriter=Rewriter(["b = 20\n"]),
        max_rewrites=1,
    ).generate(prompt="", original="a = 1\nb = 2\n")

    assert result.final_code == "a = 1\nb = 2\n"
    assert result.audit[1].selected_candidate_index == 0
    assert result.candidates[-1].evaluation_status == "structure_rejected"


def test_following_windows_are_repartitioned_after_committing_rewrite() -> None:
    class ContextGate:
        def predict(self, text):
            previous, current = text.split("[CURRENT]\n", 1)
            defer = (
                "a = 10" in previous
                and "b = 2" in current
                and "[CURRENT_UNIT_COUNT] 1" in text
            )
            return GateScores(0.1 if defer else 0.9, 0.9, True, 0.0, True)

    scorer = Scorer([False, True, True, True])
    result = GatedGenerator(
        partitioner=WindowPartitioner(
            ContextGate(), GateThresholds(0.2, 0.7, 0.8), lambda _text: 1
        ),
        scorer=scorer,
        rewriter=Rewriter(["a = 10\n"]),
        max_rewrites=1,
    ).generate(prompt="", original="a = 1\nb = 2\nc = 3\n")

    assert [row.original_unit_count for row in result.audit] == [1, 2]
    assert scorer.scored_texts[-1] == "b = 2\nc = 3"


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


@pytest.mark.parametrize("prefix", ["", "# \u96ea\n"])
def test_candidate_scoring_preserves_exact_final_code_window_bytes(prefix: str) -> None:
    scorer = Scorer([False, True])
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=Rewriter(["x = 0x1\ny = 0x2\nz = 0x3\n"] * 3),
        max_rewrites=3,
    )
    result = generator.generate(
        prompt="", original=prefix + "x = 1\ny = 2\nz = 3\n"
    )
    assert result.audit[0].selected_candidate_index == 1
    windows = generator._partitioner.partition(
        generator._extractor.extract(result.final_code)
    ).windows
    assert len(windows) == 1
    recovered = result.final_code.encode("utf-8")[
        windows[0].start_byte : windows[0].end_byte
    ].decode("utf-8")
    assert scorer.scored_texts[1] == recovered
    assert recovered == "x = 0x1\ny = 0x2\nz = 0x3"


def test_online_selection_requires_multi_channel_hit_quorum() -> None:
    scorer = ChannelScorer(
        [(False, False), (True, False), (True, True)]
    )
    generator = GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=scorer,
        rewriter=Rewriter(
            [
                "x = 2\ny = 2\nz = 3\n",
                "x = 1\ny = 3\nz = 3\n",
                "x = 1\ny = 2\nz = 4\n",
            ]
        ),
        max_rewrites=3,
        evidence_channels=2,
    )

    result = generator.generate(prompt="", original="x = 1\ny = 2\nz = 3\n")

    assert result.audit[0].selected_candidate_index == 2
    assert result.final_code == "x = 1\ny = 3\nz = 3\n"
    assert result.candidates[1].semantic_hit is False
    assert result.candidates[2].semantic_hit is True


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
