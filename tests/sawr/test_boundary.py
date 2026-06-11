from __future__ import annotations

from wfcllm.sawr.boundary import Candidate, PromptAwareBoundaryDetector


HUMANEVAL_PROMPT = '''def add_one(x):
    """Return x plus one."""
'''


def test_humaneval_detector_emits_generated_simple_statement_after_newline():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    events = []
    for ch in "    y = x + 1\n":
        events.extend(detector.feed_text(ch))

    assert events == [
        Candidate(
            text="y = x + 1",
            candidate_type="simple_statement",
            node_type="expression_statement",
            position_id="module.add_one.body",
            token_start_idx=0,
            token_count=len("    y = x + 1\n"),
        )
    ]


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

    events = []
    for ch in "return x\n":
        events.extend(detector.feed_text(ch))

    assert len(events) == 1
    assert events[0].text == "return x"
    assert events[0].position_id == "module.f.body"
    assert events[0].token_start_idx == 0
    assert events[0].token_count == len("return x\n")


def test_detector_emits_return_statement_on_final_flush():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")
    detector.feed_text("    return x + 1")

    events = detector.flush()

    assert len(events) == 1
    assert events[0].text == "return x + 1"
    assert events[0].node_type == "return_statement"
    assert events[0].position_id == "module.add_one.body"


def test_detector_ignores_comment_only_and_whitespace_fragments():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    events = []
    for ch in "    # comment\n\n    return x\n":
        events.extend(detector.feed_text(ch))

    assert [event.text for event in events] == ["return x"]


def test_detector_ignores_malformed_parse_fragments_until_valid():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")

    assert detector.feed_text("    if ") == []
    assert detector.flush() == []


def test_mbpp_detector_waits_for_first_generated_function_body():
    detector = PromptAwareBoundaryDetector(
        prompt="Write a function that returns one.",
        dataset="mbpp",
    )

    events = []
    for ch in "def make_one():\n    value = 1\n":
        events.extend(detector.feed_text(ch))

    assert len(events) == 1
    assert events[0].text == "value = 1"
    assert events[0].position_id == "module.make_one.body"


def test_checkpoint_and_rollback_restore_detector_state():
    detector = PromptAwareBoundaryDetector(prompt=HUMANEVAL_PROMPT, dataset="humaneval")
    checkpoint = detector.checkpoint()

    detector.feed_text("    a = 1\n")
    detector.rollback(checkpoint)
    events = []
    for ch in "    b = 2\n":
        events.extend(detector.feed_text(ch))

    assert [event.text for event in events] == ["b = 2"]
