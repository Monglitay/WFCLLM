from __future__ import annotations

from wfcllm.dynamic_semantic.config import ContextConfig
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.observer import AttemptContextObserver
from wfcllm.generation.generator import _notify_prefix_observer


class _Scheduler:
    def __init__(self) -> None:
        self.submissions = []
        self.live = []

    def submit(self, **kwargs) -> None:
        self.submissions.append(kwargs)

    def update_live_contexts(self, attempt_index, contexts) -> None:
        self.live.append((attempt_index, tuple(item.unit_id for item in contexts)))


def test_attempt_observer_tracks_generation_rollback_and_final_flush() -> None:
    scheduler = _Scheduler()
    extractor = DynamicContextExtractor(
        ContextConfig(),
        token_counter=lambda text: len(text.split()),
    )
    observer = AttemptContextObserver(
        attempt_index=3,
        extractor=extractor,
        scheduler=scheduler,
    )

    observer.observe_prefix("def f():\n    x = 1\n    y = 2\n")
    observer.observe_prefix("def f():\n    x = 1\n    z = 3\n")
    observer.flush("def f():\n    x = 1\n    z = 3\n")

    assert [row["attempt_index"] for row in scheduler.submissions] == [3, 3]
    assert [row["closure_index"] for row in scheduler.submissions] == [0, 1]
    assert scheduler.submissions[0]["context"].canonical_current == "x = 1"
    assert scheduler.submissions[1]["context"].canonical_current == "z = 3"
    assert len(scheduler.live[-1][1]) == 2


def test_generator_notification_ignores_observer_return_value() -> None:
    class Observer:
        def __init__(self) -> None:
            self.prefixes = []

        def observe_prefix(self, prefix: str):
            self.prefixes.append(prefix)
            return "forbidden-control-signal"

    observer = Observer()

    result = _notify_prefix_observer(observer, "def f():\n    return 1\n")

    assert result is None
    assert observer.prefixes == ["def f():\n    return 1\n"]


def test_attempt_observer_defers_nonboundary_token_fragments() -> None:
    scheduler = _Scheduler()
    observer = AttemptContextObserver(
        attempt_index=0,
        extractor=DynamicContextExtractor(
            ContextConfig(), token_counter=lambda text: len(text.split())
        ),
        scheduler=scheduler,
    )

    observer.observe_prefix("def f():\n")
    live_updates = len(scheduler.live)
    observer.observe_prefix("def f():\n    ret")

    assert len(scheduler.live) == live_updates
