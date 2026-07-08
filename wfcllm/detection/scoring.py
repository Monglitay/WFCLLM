from __future__ import annotations

from dataclasses import dataclass

from wfcllm.detection.config import WFCLLMDetectionConfig
from wfcllm.detection.proxy_windows import ProxyWindow
from wfcllm.semantic.rules import SemanticLshKeying, SemanticLshVerifier


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
