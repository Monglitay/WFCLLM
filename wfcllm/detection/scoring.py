from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol


class ScorableWindow(Protocol):
    @property
    def suitable(self) -> bool: ...

    @property
    def window_text(self) -> str: ...

    @property
    def parent_descriptor(self) -> str: ...


@dataclass(frozen=True)
class GatedWindowEvidence:
    start_byte: int
    end_byte: int
    parent_descriptor: str
    close_probability: float | None
    suitable_probability: float | None
    status: str
    margin: float


@dataclass(frozen=True)
class GatedSampleScore:
    hit_count: int
    miss_count: int
    abstain_count: int
    reliable_window_count: int
    hit_rate: float
    evidence: tuple[GatedWindowEvidence, ...]


@dataclass(frozen=True)
class GatedScoreDecision:
    decision: str
    score: GatedSampleScore


class GatedWindowScorer:
    """Score only suitable gated windows and abstain on unstable evidence."""

    def __init__(
        self,
        *,
        semantic_scorer: object,
        minimum_reliable_windows: int,
        evidence_channels: int = 1,
    ) -> None:
        if not callable(getattr(semantic_scorer, "score", None)):
            raise ValueError("semantic_scorer must expose score")
        if (
            isinstance(minimum_reliable_windows, bool)
            or not isinstance(minimum_reliable_windows, int)
            or minimum_reliable_windows <= 0
        ):
            raise ValueError("minimum_reliable_windows must be positive")
        if (
            isinstance(evidence_channels, bool)
            or not isinstance(evidence_channels, int)
            or not 1 <= evidence_channels <= 4
        ):
            raise ValueError("evidence_channels must be an integer in [1, 4]")
        if evidence_channels > 1 and not callable(
            getattr(semantic_scorer, "score_channels", None)
        ):
            raise ValueError(
                "multi-channel semantic_scorer must expose score_channels"
            )
        self._semantic_scorer = semantic_scorer
        self.minimum_reliable_windows = minimum_reliable_windows
        self.evidence_channels = evidence_channels

    def score(self, windows: list[object] | tuple[object, ...]) -> GatedSampleScore:
        hits = misses = abstains = 0
        details: list[GatedWindowEvidence] = []
        for window in windows:
            if getattr(window, "suitable", None) is not True:
                continue
            text = _window_text(window)
            descriptor = _parent_descriptor(window)
            results = (
                self._semantic_scorer.score_channels(
                    window_text=text,
                    parent_descriptor=descriptor,
                    channel_count=self.evidence_channels,
                )
                if self.evidence_channels > 1
                else (
                    self._semantic_scorer.score(
                        window_text=text,
                        parent_descriptor=descriptor,
                    ),
                )
            )
            if not isinstance(results, tuple) or len(results) != self.evidence_channels:
                raise ValueError("semantic channel evidence count does not match config")
            for channel, result in enumerate(results):
                stable = getattr(result, "stable", None)
                hit = getattr(result, "hit", None)
                margin = getattr(result, "margin", None)
                if not isinstance(stable, bool) or not isinstance(hit, bool):
                    raise ValueError("semantic evidence must define boolean stable and hit")
                if (
                    isinstance(margin, bool)
                    or not isinstance(margin, (int, float))
                    or not math.isfinite(float(margin))
                    or float(margin) < 0.0
                ):
                    raise ValueError(
                        "semantic evidence margin must be finite and non-negative"
                    )
                if not stable:
                    status = "abstain"
                    abstains += 1
                elif hit:
                    status = "hit"
                    hits += 1
                else:
                    status = "miss"
                    misses += 1
                details.append(
                    GatedWindowEvidence(
                        start_byte=_optional_int(window, "start_byte"),
                        end_byte=_optional_int(window, "end_byte"),
                        parent_descriptor=(
                            descriptor
                            if channel == 0
                            else f"{descriptor}|wfcllm-evidence-channel={channel}"
                        ),
                        close_probability=_probability(window, "close_probability"),
                        suitable_probability=_probability(
                            window, "suitable_probability"
                        ),
                        status=status,
                        margin=float(margin),
                    )
                )
        reliable = hits + misses
        return GatedSampleScore(
            hit_count=hits,
            miss_count=misses,
            abstain_count=abstains,
            reliable_window_count=reliable,
            hit_rate=(hits / reliable if reliable else 0.0),
            evidence=tuple(details),
        )

    def detect(
        self,
        windows: list[object] | tuple[object, ...],
        *,
        threshold: float | None = None,
    ) -> GatedScoreDecision:
        score = self.score(windows)
        if score.reliable_window_count < self.minimum_reliable_windows:
            decision = "insufficient_evidence"
        elif threshold is None:
            decision = "scored"
        else:
            decision = "watermarked" if score.hit_rate >= threshold else "not_watermarked"
        return GatedScoreDecision(decision=decision, score=score)


def _window_text(window: object) -> str:
    value = getattr(window, "window_text", None)
    if value is None:
        units = getattr(window, "units", None)
        if isinstance(units, tuple):
            value = "\n".join(str(getattr(unit, "text", "")) for unit in units)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scorable window text must be non-empty")
    return value


def _parent_descriptor(window: object) -> str:
    value: Any = getattr(window, "parent_descriptor", None)
    if not isinstance(value, str):
        value = getattr(value, "canonical", None)
    if not isinstance(value, str) or not value:
        raise ValueError("scorable parent descriptor must be non-empty")
    return value


def _optional_int(window: object, name: str) -> int:
    value = getattr(window, name, 0)
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _probability(window: object, name: str) -> float | None:
    value = getattr(window, name, None)
    if value is None:
        value = getattr(getattr(window, "gate_scores", None), name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)
