"""Replay lexical token-channel evidence from final code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import norm

from wfcllm.extract.hypothesis import LexicalDetectionResult
from wfcllm.extract.hypothesis import compute_z_score
from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import TokenChannelFeatureContext
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.protocol import make_prefix_key


@dataclass(frozen=True)
class TokenRow:
    token_id: int
    start: int
    end: int
    text: str


class ReplayTokenChannelDetector:
    """Replay token-channel decisions against final tokenizer-visible code."""

    def __init__(self, runtime: Any, tokenizer: object, config: TokenChannelConfig) -> None:
        self._runtime = runtime
        self._tokenizer = tokenizer
        self._config = config

    def detect(self, code: str) -> LexicalDetectionResult:
        token_rows = self._tokenize(code)
        if not token_rows:
            return LexicalDetectionResult.empty()

        seen_prefixes: set[tuple[int, ...]] = set()
        seen_ngrams: set[tuple[int, ...]] = set()
        prefix_ids: list[int] = []
        feature_context = self._prepare_feature_context(code)

        num_positions_scored = 0
        num_green_hits = 0

        for row in token_rows:
            prefix_key = make_prefix_key(prefix_ids)
            if self._config.ignore_repeated_prefixes and prefix_key in seen_prefixes:
                prefix_ids.append(row.token_id)
                continue

            ngram_key = self._make_ngram_key(prefix_ids, row.token_id)
            if self._config.ignore_repeated_ngrams and ngram_key in seen_ngrams:
                prefix_ids.append(row.token_id)
                continue

            features = self._build_features(feature_context, code, row.start, row.end)
            if features is None or not features.structure_mask:
                prefix_ids.append(row.token_id)
                continue

            decision = self._runtime.score_prefix(prefix_ids, features=features)
            seen_prefixes.add(prefix_key)
            seen_ngrams.add(ngram_key)

            if decision.should_switch:
                num_positions_scored += 1
                if row.token_id in decision.partition.green_token_ids:
                    num_green_hits += 1

            prefix_ids.append(row.token_id)

        if num_positions_scored == 0:
            return LexicalDetectionResult.empty()

        expected_hits = num_positions_scored * 0.5
        variance = num_positions_scored * 0.25
        lexical_z_score = compute_z_score(num_green_hits, expected_hits, variance)
        lexical_p_value = float(norm.sf(lexical_z_score))
        return LexicalDetectionResult(
            num_positions_scored=num_positions_scored,
            num_green_hits=num_green_hits,
            green_fraction=num_green_hits / num_positions_scored,
            lexical_z_score=lexical_z_score,
            lexical_p_value=lexical_p_value,
        )

    def _tokenize(self, code: str) -> list[TokenRow]:
        encoded = self._tokenizer(
            code,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        input_ids = self._resolve_token_field(encoded, "input_ids")
        offset_mapping = self._resolve_token_field(encoded, "offset_mapping")
        if len(input_ids) != len(offset_mapping):
            raise ValueError("tokenizer input_ids and offset_mapping must have the same length")

        rows: list[TokenRow] = []
        for token_id, offset in zip(input_ids, offset_mapping, strict=True):
            start, end = self._normalize_offset(offset)
            rows.append(
                TokenRow(
                    token_id=int(token_id),
                    start=start,
                    end=end,
                    text=code[start:end],
                )
            )
        return rows

    @staticmethod
    def _resolve_token_field(encoded: Any, key: str) -> list[Any]:
        value = encoded[key]
        if value and isinstance(value[0], list):
            return value[0]
        return list(value)

    @staticmethod
    def _normalize_offset(offset: Any) -> tuple[int, int]:
        if not isinstance(offset, (tuple, list)) or len(offset) != 2:
            raise ValueError("offset_mapping entries must be (start, end) pairs")
        return int(offset[0]), int(offset[1])

    @staticmethod
    def _prepare_feature_context(code: str) -> TokenChannelFeatureContext | None:
        try:
            return prepare_token_channel_feature_context(code)
        except SyntaxError:
            return None

    @staticmethod
    def _build_features(
        feature_context: TokenChannelFeatureContext | None,
        code: str,
        start: int,
        end: int,
    ) -> TokenChannelFeatures | None:
        if feature_context is not None:
            try:
                return build_token_channel_features_from_context(
                    feature_context,
                    token_start=start,
                    token_end=end,
                )
            except ValueError:
                return None
        return None

    def _make_ngram_key(self, prefix_ids: list[int], token_id: int) -> tuple[int, ...]:
        width = max(1, self._config.context_width)
        suffix = tuple(prefix_ids[-(width - 1) :]) if width > 1 else ()
        return (*suffix, int(token_id))
