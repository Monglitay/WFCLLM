from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

from wfcllm.detection.config import WFCLLMDetectionConfig
from wfcllm.detection.proxy_windows import ProxyWindow
from wfcllm.semantic.rules import SemanticLshKeying, SemanticLshVerifier


class ScorableWindow(Protocol):
    """Read-only semantic-window view shared by detector implementations."""

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
    """Score only formal suitable windows and exclude unstable evidence."""

    def __init__(
        self,
        *,
        semantic_scorer: object,
        minimum_reliable_windows: int,
    ) -> None:
        if not callable(getattr(semantic_scorer, "score", None)):
            raise ValueError("semantic_scorer must expose score")
        if (
            isinstance(minimum_reliable_windows, bool)
            or not isinstance(minimum_reliable_windows, int)
            or minimum_reliable_windows <= 0
        ):
            raise ValueError("minimum_reliable_windows must be positive")
        self._semantic_scorer = semantic_scorer
        self.minimum_reliable_windows = minimum_reliable_windows

    def score(self, windows: list[object] | tuple[object, ...]) -> GatedSampleScore:
        hits = misses = abstains = 0
        details: list[GatedWindowEvidence] = []
        for window in windows:
            if getattr(window, "suitable", None) is not True:
                continue
            text = _gated_window_text(window)
            descriptor = _gated_parent_descriptor(window)
            result = self._semantic_scorer.score(
                window_text=text,
                parent_descriptor=descriptor,
            )
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
                raise ValueError("semantic evidence margin must be finite and non-negative")
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
                    start_byte=_optional_int(window, "start_byte", 0),
                    end_byte=_optional_int(window, "end_byte", 0),
                    parent_descriptor=descriptor,
                    close_probability=_gate_probability(window, "close_probability"),
                    suitable_probability=_gate_probability(
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


def _gated_window_text(window: object) -> str:
    value = getattr(window, "window_text", None)
    if value is None:
        units = getattr(window, "units", None)
        if isinstance(units, tuple):
            value = "\n".join(str(getattr(unit, "text", "")) for unit in units)
    if not isinstance(value, str) or not value.strip():
        raise ValueError("scorable window text must be non-empty")
    return value


def _gated_parent_descriptor(window: object) -> str:
    value: Any = getattr(window, "parent_descriptor", None)
    if not isinstance(value, str):
        value = getattr(value, "canonical", None)
    if not isinstance(value, str) or not value:
        raise ValueError("scorable parent descriptor must be non-empty")
    return value


def _optional_int(window: object, name: str, default: int) -> int:
    value = getattr(window, name, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _gate_probability(window: object, name: str) -> float | None:
    value = getattr(window, name, None)
    if value is None:
        scores = getattr(window, "gate_scores", None)
        value = getattr(scores, name, None)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    return float(value)


@dataclass(frozen=True)
class WindowEvidence:
    window_id: str
    context_id: str
    in_valid_set: bool | None
    passed_margin: bool
    min_margin: float
    lsh_signature: tuple[int, ...] | None
    parent_node_type: str
    window_length: int
    structure_type: str
    context_window_count: int
    context_statement_count: int
    window_raw: float


class WFCLLMWindowScorer:
    """Score final-code proxy windows with keyed semantic LSH evidence."""

    def __init__(
        self,
        *,
        verifier: SemanticLshVerifier,
        keying: SemanticLshKeying,
        config: WFCLLMDetectionConfig,
    ) -> None:
        self._verifier = verifier
        self._keying = keying
        self._config = config

    def score_window(self, window: ProxyWindow) -> WindowEvidence:
        key_ordinal = window.ordinal if self._config.use_ordinal_keying else None
        valid_set = self._keying.derive(
            window.parent_node_type,
            k=self._config.k,
            ordinal=key_ordinal,
        )
        result = self._verifier.verify(
            window.normalized_text,
            valid_set,
            self._config.semantic_margin,
        )
        in_valid_set = result.in_valid_set
        passed_margin = result.min_margin > self._config.semantic_margin
        window_raw = self._window_raw(
            in_valid_set=in_valid_set,
            passed_margin=passed_margin,
            min_margin=float(result.min_margin),
        )
        return WindowEvidence(
            window_id=window.window_id,
            context_id=window.context_id,
            in_valid_set=in_valid_set,
            passed_margin=passed_margin,
            min_margin=float(result.min_margin),
            lsh_signature=result.lsh_signature,
            parent_node_type=window.parent_node_type,
            window_length=window.window_length,
            structure_type=window.structure_type,
            context_window_count=window.context_window_count,
            context_statement_count=window.context_statement_count,
            window_raw=window_raw,
        )

    def score_windows(self, windows: list[ProxyWindow]) -> list[WindowEvidence]:
        return [self.score_window(window) for window in windows]

    def _window_raw(
        self,
        *,
        in_valid_set: bool | None,
        passed_margin: bool,
        min_margin: float,
    ) -> float:
        if self._config.evidence_mode == "hit_only":
            return 1.0 if in_valid_set is True and passed_margin else 0.0
        if self._config.evidence_mode == "margin_only":
            return min_margin if passed_margin else 0.0
        return min_margin if in_valid_set is True and passed_margin else 0.0


def load_wfcllm_window_scorer(
    *,
    config: WFCLLMDetectionConfig,
    encoder_model_path: str,
    encoder_checkpoint_path: str | None,
    embed_dim: int,
    device: str,
    use_lora: bool,
    use_bf16: bool,
    whitening_path: str | None,
) -> WFCLLMWindowScorer:
    from wfcllm.semantic.lsh import load_semantic_lsh_components

    components = load_semantic_lsh_components(
        encoder_model_path=encoder_model_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        embed_dim=embed_dim,
        device=device,
        use_lora=use_lora,
        use_bf16=use_bf16,
        secret_key=config.secret_key,
        lsh_d=config.lsh_d,
        whitening_path=whitening_path,
    )
    return WFCLLMWindowScorer(
        verifier=components.verifier,
        keying=components.keying,
        config=config,
    )
