from __future__ import annotations

from dataclasses import dataclass

from wfcllm.generation.gated_generator import GatedGenerator, RewriteTokens
from wfcllm.windowing import GateScores, GateThresholds, WindowPartitioner


class Gate:
    def predict(self, serialized_input: str) -> GateScores:
        assert "secret_key" not in serialized_input
        closes = "[CURRENT_UNIT_COUNT] 3" in serialized_input
        return GateScores(0.9 if closes else 0.1, 0.9, True, 0.0, True)


class Scorer:
    def __init__(self, hits: list[bool]) -> None:
        self.hits = iter(hits)

    def score(self, *, window_text: str, parent_descriptor: str):
        hit = next(self.hits)
        return type("Evidence", (), {"hit": hit, "stable": True, "margin": 1.0})()


class Rewriter:
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts
        self.requests = []

    def rewrite_window(self, request, *, candidate_index: int):
        self.requests.append(request)
        text = self.texts[candidate_index - 1]
        return RewriteTokens(tuple(range(len(text.encode()))), text)


def _generator(rewrites: list[str], hits: list[bool], *, max_rewrites: int = 3):
    return GatedGenerator(
        partitioner=WindowPartitioner(
            Gate(), GateThresholds(0.2, 0.7, 0.8), lambda text: len(text)
        ),
        scorer=Scorer(hits),
        rewriter=Rewriter(rewrites),
        max_rewrites=max_rewrites,
    )


def test_rewrite_rolls_back_to_first_unit_not_last_unit() -> None:
    result = _generator(["a = 10\nb = 20\n"], [False, True]).generate(
        prompt="", original="a = 1\nb = 2\nc = 3\n"
    )
    assert result.final_code == "a = 10\nb = 20\n"
    assert result.audit[0].original_unit_count == 3
    assert result.audit[0].selected_unit_count == 2
    assert result.audit[0].rollback_anchor <= result.audit[0].original_span.start
    assert result.audit[0].previous_statement_text_unchanged is True


def test_all_failed_rewrites_restore_original_and_continue() -> None:
    result = _generator(
        ["bad one\n", "bad two\n", "bad three\n"],
        [False, False, False, False],
    ).generate(
        prompt="", original="x = 1\ny = 2\n", suffix="return x + y\n"
    )
    assert result.final_code == "x = 1\ny = 2\nreturn x + y\n"
    assert result.audit[0].selected_candidate_index == 0


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
