from __future__ import annotations

from typing import Any

from wfcllm.dynamic_semantic.context import (
    DynamicContextExtractor,
    IncrementalContextTracker,
)


class AttemptContextObserver:
    """Read-only bridge from generated public prefixes to the shared scheduler."""

    def __init__(
        self,
        *,
        attempt_index: int,
        extractor: DynamicContextExtractor,
        scheduler: Any,
    ) -> None:
        self.attempt_index = attempt_index
        self._tracker = IncrementalContextTracker(extractor)
        self._scheduler = scheduler
        self._closure_index = 0
        self._last_prefix = ""

    @property
    def live_contexts(self):
        return self._tracker.live_contexts

    def observe_prefix(self, final_code_prefix: str) -> None:
        previous = self._last_prefix
        self._last_prefix = final_code_prefix
        if final_code_prefix.startswith(previous):
            appended = final_code_prefix[len(previous) :]
            if "\n" not in appended and ";" not in appended:
                return
        self._record(self._tracker.observe(final_code_prefix))

    def flush(self, final_code: str) -> None:
        self._last_prefix = final_code
        self._record(self._tracker.flush(final_code))

    def _record(self, emitted) -> None:
        for context in emitted:
            self._scheduler.submit(
                attempt_index=self.attempt_index,
                closure_index=self._closure_index,
                context=context,
            )
            self._closure_index += 1
        self._scheduler.update_live_contexts(
            self.attempt_index,
            self._tracker.live_contexts,
        )
