"""Replay lexical token-channel evidence from final code."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scipy.stats import norm

from wfcllm.extract.hypothesis import LexicalDetectionResult
from wfcllm.extract.hypothesis import compute_z_score
from wfcllm.watermark.token_channel.core.config import TokenChannelConfig
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatureContext
from wfcllm.watermark.token_channel.core.features import TokenChannelFeatures
from wfcllm.watermark.token_channel.core.features import build_token_channel_features_from_context
from wfcllm.watermark.token_channel.core.features import prepare_token_channel_feature_context
from wfcllm.watermark.token_channel.core.protocol import make_prefix_key


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

    def detect(self, code: str, prompt: str = "") -> LexicalDetectionResult:
        token_rows = self._tokenize(code)
        if not token_rows:
            return LexicalDetectionResult.empty()

        # Determine how many leading tokens are prompt-repeated (not model-generated).
        # These tokens were never biased during generation, so skip them for scoring.
        skip_tokens = self._compute_prompt_overlap_tokens(code, prompt)

        seen_prefixes: set[tuple[int, ...]] = set()
        seen_ngrams: set[tuple[int, ...]] = set()
        prefix_ids: list[int] = []

        # Try to parse code alone; if it fails (indented snippet), prepend prompt.
        # If both fail (e.g. code contains markdown backticks), feature_context stays
        # None and we fall back to structure_mask=True for all tokens — matching the
        # embedding-time fallback in _fallback_runtime_token_features().
        feature_context = self._prepare_feature_context(code)
        prompt_offset = 0
        if feature_context is None and prompt:
            full_code = prompt + code
            feature_context = self._prepare_feature_context(full_code)
            if feature_context is not None:
                prompt_offset = len(prompt)
        if feature_context is None:
            return LexicalDetectionResult.empty()

        num_positions_scored = 0
        num_green_hits = 0

        for idx, row in enumerate(token_rows):
            # Skip prompt-repeated tokens: they were never biased during generation.
            # Still add to prefix_ids so subsequent partitions are computed correctly.
            if idx < skip_tokens:
                prefix_ids.append(row.token_id)
                continue

            prefix_key = make_prefix_key(prefix_ids)
            if self._config.ignore_repeated_prefixes and prefix_key in seen_prefixes:
                prefix_ids.append(row.token_id)
                continue

            ngram_key = self._make_ngram_key(prefix_ids, row.token_id)
            if self._config.ignore_repeated_ngrams and ngram_key in seen_ngrams:
                prefix_ids.append(row.token_id)
                continue

            features = self._build_features(
                feature_context, code if prompt_offset == 0 else prompt + code,
                row.start + prompt_offset, row.end + prompt_offset,
            )
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

    def _is_gap_within_delta(self, decision) -> bool:
        """Check if the gap between best red and best green in top-k is <= delta."""
        pref_logits = decision.preference_logits
        if pref_logits is None:
            return True
        logits_1d = pref_logits[0] if pref_logits.ndim == 2 else pref_logits
        top_k = 50
        topk_vals, topk_idx = logits_1d.topk(min(top_k, logits_1d.shape[0]))
        green_set = decision.partition.green_token_ids
        best_green = None
        best_red = None
        for val, idx in zip(topk_vals.tolist(), topk_idx.tolist()):
            if idx in green_set:
                if best_green is None:
                    best_green = val
            else:
                if best_red is None:
                    best_red = val
            if best_green is not None and best_red is not None:
                break
        if best_green is None or best_red is None:
            return True
        gap = best_red - best_green
        return gap <= self._config.delta

    @staticmethod
    def _prepare_feature_context(code: str) -> TokenChannelFeatureContext | None:
        try:
            return prepare_token_channel_feature_context(code)
        except SyntaxError:
            return None

    def _compute_prompt_overlap_tokens(self, code: str, prompt: str) -> int:
        """Count leading tokens in code that repeat the function signature from prompt.

        The instruct model often regenerates the 'def ...:' line that already exists
        in the prompt. These tokens were produced by the model copying the prompt,
        not by biased sampling, so they should be excluded from scoring.
        """
        if not prompt:
            return 0
        import re
        # Find the def line in the prompt
        def_lines = [l for l in prompt.splitlines() if re.match(r'\s*def\s+', l)]
        if not def_lines:
            return 0
        last_def = def_lines[-1].rstrip()
        # Check if code starts with this def line (possibly with leading whitespace stripped)
        code_lines = code.split('\n')
        overlap_end = 0
        for line in code_lines:
            if line.rstrip() == last_def.rstrip():
                overlap_end = code.index('\n', code.index(line)) + 1
                break
            elif line.strip() and not line.strip().startswith(('def ', '#', '"""', "'''")):
                break
        if overlap_end == 0:
            return 0
        overlap_text = code[:overlap_end]
        overlap_ids = self._tokenizer(overlap_text, add_special_tokens=False)
        ids = overlap_ids["input_ids"]
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

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
