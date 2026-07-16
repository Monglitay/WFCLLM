from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class RetryAction(str, Enum):
    ACCEPT_ORIGINAL = "accept_original"
    SKIP_ORIGINAL = "skip_original"
    REWRITE = "rewrite"
    ACCEPT_REWRITE = "accept_rewrite"
    RESTORE_ORIGINAL = "restore_original"


@dataclass(frozen=True)
class WindowAttempt:
    candidate_index: int
    suitable: bool
    structure_ok: bool
    semantic_hit: bool
    semantic_stable: bool
    semantic_margin: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.candidate_index, bool)
            or not isinstance(self.candidate_index, int)
            or self.candidate_index < 0
        ):
            raise ValueError("candidate_index must be a non-negative integer")
        if isinstance(self.semantic_margin, bool) or not isinstance(
            self.semantic_margin, (int, float)
        ):
            raise ValueError("semantic_margin must be finite")
        if not math.isfinite(self.semantic_margin):
            raise ValueError("semantic_margin must be finite")


class GatedWindowRetryController:
    """Pure retry state for one complete gated semantic window."""

    def __init__(self, max_rewrites: int) -> None:
        if (
            isinstance(max_rewrites, bool)
            or not isinstance(max_rewrites, int)
            or max_rewrites < 0
        ):
            raise ValueError("max_rewrites must be a non-negative integer")
        self.max_rewrites = max_rewrites
        self.reset()

    @property
    def rewrite_count(self) -> int:
        return self._rewrite_count

    @property
    def selected_candidate_index(self) -> int | None:
        return self._selected_candidate_index

    def reset(self) -> None:
        self._started = False
        self._rewrite_count = 0
        self._selected_candidate_index: int | None = None
        self._action: RetryAction | None = None

    def begin(self, attempt: WindowAttempt) -> None:
        if self._started:
            raise RuntimeError("reset must be called before beginning a new window")
        if attempt.candidate_index != 0:
            raise ValueError("original attempt must have candidate_index=0")

        self._started = True
        if not attempt.suitable:
            self._select_original(RetryAction.SKIP_ORIGINAL)
        elif self._is_acceptable(attempt):
            self._select_original(RetryAction.ACCEPT_ORIGINAL)
        elif self.max_rewrites == 0:
            self._select_original(RetryAction.RESTORE_ORIGINAL)
        else:
            self._action = RetryAction.REWRITE

    def observe(self, attempt: WindowAttempt) -> None:
        if not self._started:
            raise RuntimeError("begin must be called before observe")
        if self._action is not RetryAction.REWRITE:
            raise RuntimeError("a rewrite was not requested")

        expected_index = self._rewrite_count + 1
        if attempt.candidate_index != expected_index:
            raise ValueError(
                f"rewrite attempt must have candidate_index={expected_index}"
            )

        self._rewrite_count += 1
        if self._is_acceptable(attempt):
            self._selected_candidate_index = attempt.candidate_index
            self._action = RetryAction.ACCEPT_REWRITE
        elif self._rewrite_count >= self.max_rewrites:
            self._select_original(RetryAction.RESTORE_ORIGINAL)
        else:
            self._action = RetryAction.REWRITE

    def next_action(self) -> RetryAction:
        if not self._started or self._action is None:
            raise RuntimeError("begin must be called before requesting an action")
        return self._action

    @staticmethod
    def _is_acceptable(attempt: WindowAttempt) -> bool:
        return (
            attempt.suitable
            and attempt.structure_ok
            and attempt.semantic_stable
            and attempt.semantic_hit
        )

    def _select_original(self, action: RetryAction) -> None:
        self._selected_candidate_index = 0
        self._action = action
