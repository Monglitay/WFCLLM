from __future__ import annotations

from wfcllm.sawr.boundary import Candidate, PromptAwareBoundaryDetector


HUMANEVAL_PROMPT = '''def add_one(x):
    """Return x plus one."""
'''


def _feed_all(detector: PromptAwareBoundaryDetector, text: str):
    events = []
    for ch in text:
        events.extend(detector.feed_text(ch))
    return events


def _simple_events(events):
    return [event for event in events if event.kind == "simple_candidate"]


def test_humaneval_detector_emits_generated_simple_statement_after_newline():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    events = _feed_all(detector, "    y = x + 1\n")
    simple_events = _simple_events(events)

    assert len(simple_events) == 1
    assert simple_events[0].candidate == (
        Candidate(
            text="y = x + 1",
            candidate_type="simple_statement",
            node_type="expression_statement",
            position_id="module.add_one.body",
            token_start_idx=0,
            token_count=len("    y = x + 1\n"),
            parent_node_type="function_definition",
            ordinal=0,
            layer_path=("module.add_one.body",),
            start_byte=0,
            end_byte=len("    y = x + 1\n"),
            depth=0,
        )
    )


def test_humaneval_detector_does_not_emit_prompt_existing_docstring():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    events = detector.feed_text("")

    assert events == []


def test_humaneval_detector_reports_prompt_controlled_body_immediately():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    assert detector.saw_controlled_body is True


def test_humaneval_detector_accepts_prompt_supplied_indentation():
    prompt = 'def f(x):\n    """doc"""\n    '
    detector = PromptAwareBoundaryDetector(prompt=prompt, dataset="humaneval")

    events = _feed_all(detector, "return x\n")
    simple_events = _simple_events(events)

    assert len(simple_events) == 1
    assert simple_events[0].candidate is not None
    assert simple_events[0].candidate.text == "return x"
    assert simple_events[0].candidate.position_id == "module.f.body"
    assert simple_events[0].candidate.token_start_idx == 0
    assert simple_events[0].candidate.token_count == len("return x\n")


def test_detector_emits_complete_statement_before_later_incomplete_fragment():
    detector = PromptAwareBoundaryDetector(prompt="def f():\n", dataset="humaneval")

    events = detector.feed_text("    a = 1\n    if ")

    assert [event.candidate.text for event in _simple_events(events)] == ["a = 1"]


def test_humaneval_detector_rejects_prompt_owned_statement_syntax():
    detector = PromptAwareBoundaryDetector(
        prompt="def f():\n    return ",
        dataset="humaneval",
    )

    events = detector.feed_text("x\n")

    assert events == []


def test_humaneval_detector_uses_outer_top_level_function_body():
    prompt = """def outer():
    def inner():
"""
    detector = PromptAwareBoundaryDetector(prompt=prompt, dataset="humaneval")

    events = _feed_all(detector, "        value = 1\n    return 1\n")
    simple_events = _simple_events(events)

    assert [event.candidate.text for event in simple_events] == ["return 1"]
    assert simple_events[0].candidate.position_id == "module.outer.body"


def test_humaneval_detector_ignores_generated_top_level_function_body():
    detector = PromptAwareBoundaryDetector(prompt="def f():\n", dataset="humaneval")

    events = detector.feed_text("    return 1\ndef helper():\n    x = 1\n")
    simple_events = _simple_events(events)

    assert [event.candidate.text for event in simple_events] == ["return 1"]
    assert simple_events[0].candidate.position_id == "module.f.body"


def test_detector_emits_return_statement_on_final_flush():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")
    detector.feed_text("    return x + 1")

    events = detector.flush()
    simple_events = _simple_events(events)

    assert len(simple_events) == 1
    assert simple_events[0].candidate is not None
    assert simple_events[0].candidate.text == "return x + 1"
    assert simple_events[0].candidate.node_type == "return_statement"
    assert simple_events[0].candidate.position_id == "module.add_one.body"
    assert events[-1].kind == "final_flush"


def test_detector_final_flush_sentinel_is_idempotent_and_rollback_restores():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")
    checkpoint = detector.checkpoint()

    first_events = detector.flush()
    second_events = detector.flush()

    assert [event.kind for event in first_events].count("final_flush") == 1
    assert [event.kind for event in second_events].count("final_flush") == 0

    detector.rollback(checkpoint)
    restored_events = detector.flush()

    assert [event.kind for event in restored_events].count("final_flush") == 1


def test_detector_ignores_comment_only_and_whitespace_fragments():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    events = _feed_all(detector, "    # comment\n\n    return x\n")

    assert [event.candidate.text for event in _simple_events(events)] == ["return x"]


def test_detector_ignores_malformed_parse_fragments_until_valid():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    assert detector.feed_text("    if ") == []
    assert _simple_events(detector.flush()) == []


def test_detector_rejects_recovery_multiline_simple_candidate_in_compound():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")

    events = []
    events.extend(detector.feed_text("    if x:\n"))
    events.extend(detector.feed_text("        y = 1\n"))
    events.extend(detector.feed_text("        z = \n"))
    events.extend(detector.feed_text("        z = 2\n"))

    candidate_texts = [
        event.candidate.text
        for event in _simple_events(events)
        if event.candidate is not None
    ]
    assert "z =\n        z = 2" not in candidate_texts
    assert all("\n" not in text for text in candidate_texts)
    assert "y = 1" in candidate_texts


def test_mbpp_detector_waits_for_first_generated_function_body():
    detector = PromptAwareBoundaryDetector(
        prompt="Write a function that returns one.",
        dataset="mbpp",
    )

    events = _feed_all(detector, "def make_one():\n    value = 1\n")
    simple_events = _simple_events(events)

    assert len(simple_events) == 1
    assert simple_events[0].candidate is not None
    assert simple_events[0].candidate.text == "value = 1"
    assert simple_events[0].candidate.position_id == "module.make_one.body"


def test_checkpoint_and_rollback_restore_detector_state():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")
    checkpoint = detector.checkpoint()

    detector.feed_text("    a = 1\n")
    detector.rollback(checkpoint)
    events = _feed_all(detector, "    b = 2\n")

    assert [event.candidate.text for event in _simple_events(events)] == ["b = 2"]


def test_detector_emits_compound_and_nested_simple_events_for_if_else():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")
    events = _feed_all(
        detector,
        "    if x:\n"
        "        y = 1\n"
        "    else:\n"
        "        y = 2\n"
        "    return y\n",
    )
    assert [event.kind for event in events] == [
        "compound_started",
        "simple_candidate",
        "simple_candidate",
        "layer_closed",
        "simple_candidate",
    ]
    assert events[0].node_type == "if_statement"
    assert events[0].parent_node_type == "function_definition"
    assert events[0].text.strip().startswith("if x:")
    assert events[1].candidate is not None
    assert events[1].candidate.text == "y = 1"
    assert events[1].candidate.layer_path[-1].startswith("if_statement:")
    assert events[2].candidate is not None
    assert events[2].candidate.text == "y = 2"
    assert events[2].candidate.layer_path == events[1].candidate.layer_path
    assert events[3].closed_layer_paths == (events[1].layer_path,)
    assert events[3].text == (
        "if x:\n"
        "        y = 1\n"
        "    else:\n"
        "        y = 2"
    )
    assert events[4].candidate is not None
    assert events[4].candidate.text == "return y"
    assert events[4].candidate.layer_path == ("module.f.body",)


def test_detector_emits_nested_for_and_while_layers():
    detector = PromptAwareBoundaryDetector(prompt="def f(items):\n", dataset="humaneval")
    events = _feed_all(
        detector,
        "    total = 0\n"
        "    for item in items:\n"
        "        while item > 0:\n"
        "            total += item\n"
        "            item -= 1\n"
        "    return total\n",
    )
    compound_types = [event.node_type for event in events if event.kind == "compound_started"]
    candidate_texts = [
        event.candidate.text
        for event in events
        if event.kind == "simple_candidate" and event.candidate is not None
    ]
    assert compound_types == ["for_statement", "while_statement"]
    assert candidate_texts == ["total = 0", "total += item", "item -= 1", "return total"]


def test_detector_checkpoint_restores_layer_event_keys():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")
    checkpoint = detector.checkpoint()
    _feed_all(detector, "    if x:\n        return 1\n")
    detector.rollback(checkpoint)
    events = _feed_all(detector, "    if x:\n        return 2\n")
    assert [event.kind for event in events] == ["compound_started", "simple_candidate"]
    assert events[1].candidate is not None
    assert events[1].candidate.text == "return 2"


def test_detector_marks_explicit_compound_close_events_from_flush():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")
    _feed_all(detector, "    if x:\n        return 1\n")

    events = detector.flush()
    close_events = [event for event in events if event.kind == "layer_closed"]

    assert close_events
    assert all(event.final_flush is True for event in close_events)
