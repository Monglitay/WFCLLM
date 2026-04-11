"""Tests for token-channel replay detection."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from wfcllm.extract.token_channel import ReplayTokenChannelDetector
from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import TokenChannelFeatures


class SimpleTokenizer:
    name_or_path = "offline-tokenizer"

    def __init__(self) -> None:
        self._vocab = {"a": 0, "b": 1, " ": 2}

    def __len__(self) -> int:
        return len(self._vocab)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def __call__(self, text: str, *, add_special_tokens: bool, return_offsets_mapping: bool) -> dict[str, list]:
        assert add_special_tokens is False
        assert return_offsets_mapping is True
        return {
            "input_ids": [self._vocab[ch] for ch in text],
            "offset_mapping": [(index, index + 1) for index, _ in enumerate(text)],
        }


@dataclass
class FakeDecision:
    should_switch: bool
    partition: object


class FakeRuntime:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[int, ...], TokenChannelFeatures]] = []

    def score_prefix(self, prefix_ids, features: TokenChannelFeatures) -> FakeDecision:
        prefix_tuple = tuple(prefix_ids)
        self.calls.append((prefix_tuple, features))
        green_by_prefix = {
            (): {0},
            (0,): {1},
            (0, 1): {2},
            (0, 2): {1},
        }
        partition = SimpleNamespace(green_token_ids=green_by_prefix.get(prefix_tuple, {0}))
        return FakeDecision(
            should_switch=features.structure_mask,
            partition=partition,
        )


def test_replay_detector_replays_green_hits() -> None:
    detector = ReplayTokenChannelDetector(
        runtime=FakeRuntime(),
        tokenizer=SimpleTokenizer(),
        config=TokenChannelConfig(enabled=True, mode="dual-channel", context_width=2),
    )

    result = detector.detect("aba")

    assert result.num_positions_scored == 3
    assert result.num_green_hits == 2
    assert result.green_fraction == pytest.approx(2 / 3)
    assert result.lexical_z_score > 0
    assert result.lexical_p_value < 0.5


def test_replay_detector_respects_repeated_prefixes_and_ngrams() -> None:
    runtime = FakeRuntime()
    detector = ReplayTokenChannelDetector(
        runtime=runtime,
        tokenizer=SimpleTokenizer(),
        config=TokenChannelConfig(
            enabled=True,
            mode="dual-channel",
            context_width=2,
            ignore_repeated_prefixes=True,
            ignore_repeated_ngrams=True,
        ),
    )

    result = detector.detect("aaaa")

    assert result.num_positions_scored == 2
    assert [prefix for prefix, _ in runtime.calls] == [(), (0,)]


def test_replay_detector_skips_structure_masked_tokens() -> None:
    detector = ReplayTokenChannelDetector(
        runtime=FakeRuntime(),
        tokenizer=SimpleTokenizer(),
        config=TokenChannelConfig(enabled=True, mode="dual-channel", context_width=2),
    )

    allowed = TokenChannelFeatures(
        node_type="module",
        parent_node_type="module",
        block_relative_offset=0,
        in_code_body=True,
        structure_mask=True,
    )
    masked = TokenChannelFeatures(
        node_type="module",
        parent_node_type="module",
        block_relative_offset=0,
        in_code_body=False,
        structure_mask=False,
    )

    with patch.object(
        detector,
        "_build_features",
        side_effect=[allowed, masked, allowed],
    ):
        result = detector.detect("aba")

    assert result.num_positions_scored == 2
    assert len(detector._runtime.calls) == 2


def test_replay_detector_fails_closed_when_ast_feature_prep_fails() -> None:
    runtime = FakeRuntime()
    detector = ReplayTokenChannelDetector(
        runtime=runtime,
        tokenizer=SimpleTokenizer(),
        config=TokenChannelConfig(enabled=True, mode="dual-channel", context_width=2),
    )

    with patch(
        "wfcllm.extract.token_channel.prepare_token_channel_feature_context",
        side_effect=SyntaxError("broken parse"),
    ):
        result = detector.detect("ab")

    assert result.num_positions_scored == 0
    assert result.num_green_hits == 0
    assert runtime.calls == []
