from __future__ import annotations

from wfcllm.dynamic_semantic.config import ContextConfig
from wfcllm.dynamic_semantic.context import (
    DynamicContextExtractor,
    IncrementalContextTracker,
)


def _extractor() -> DynamicContextExtractor:
    return DynamicContextExtractor(
        ContextConfig(),
        token_counter=lambda text: len(text),
    )


def test_incremental_tracker_emits_only_stable_units_before_flush() -> None:
    tracker = IncrementalContextTracker(_extractor())

    assert tracker.observe("def f():\n    x = 1\n") == ()
    emitted = tracker.observe("def f():\n    x = 1\n    y = 2\n")

    assert [item.canonical_current for item in emitted] == ["x = 1"]
    assert [item.canonical_current for item in tracker.live_contexts] == ["x = 1"]


def test_incremental_tracker_retains_live_units_during_incomplete_suffix() -> None:
    tracker = IncrementalContextTracker(_extractor())
    tracker.observe("def f():\n    x = 1\n    y = 2\n")

    emitted = tracker.observe("def f():\n    x = 1\n    y = 2\n    if")

    assert emitted == ()
    assert [item.canonical_current for item in tracker.live_contexts] == ["x = 1"]


def test_incremental_tracker_rolls_back_live_units_and_reemits_new_context() -> None:
    tracker = IncrementalContextTracker(_extractor())
    tracker.observe("def f():\n    x = 1\n    y = 2\n")
    tracker.observe("def f():\n    x = 1\n    y = 2\n    z = 3\n")

    emitted = tracker.observe("def f():\n    x = 1\n    q = 9\n")

    assert [item.canonical_current for item in tracker.live_contexts] == ["x = 1"]
    assert emitted == ()
    final_emitted = tracker.flush("def f():\n    x = 1\n    q = 9\n")
    assert [item.canonical_current for item in final_emitted] == ["q = 9"]


def test_incremental_flush_exactly_matches_final_code_extraction() -> None:
    code = "def f(x):\n    y = x + 1\n    if y > 2:\n        y *= 2\n    return y\n"
    tracker = IncrementalContextTracker(_extractor())

    tracker.observe("def f(x):\n    y = x + 1\n")
    tracker.observe("def f(x):\n    y = x + 1\n    if y > 2:\n        y *= 2\n")
    tracker.observe(code)
    tracker.flush(code)

    expected = _extractor().extract(code).contexts
    assert [(item.unit_id, item.context_sha256) for item in tracker.live_contexts] == [
        (item.unit_id, item.context_sha256) for item in expected
    ]


def test_incremental_tracker_does_not_reemit_cached_context_hash() -> None:
    tracker = IncrementalContextTracker(_extractor())
    code = "def f():\n    x = 1\n    return x\n"

    first = tracker.observe(code)
    second = tracker.observe(code)
    third = tracker.flush(code)

    assert len(first) == 1
    assert second == ()
    assert len(third) == 1
    assert first[0].context_sha256 != third[0].context_sha256
