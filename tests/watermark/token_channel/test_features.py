"""Tests for token-channel structural features."""

from __future__ import annotations

import pytest

from wfcllm.watermark.token_channel.features import TokenChannelFeatures


def test_token_channel_features_default_language_and_mapping_roundtrip() -> None:
    features = TokenChannelFeatures(
        node_type="if_statement",
        parent_node_type="block",
        block_relative_offset=3,
        in_code_body=True,
        structure_mask=False,
    )

    assert features.language == "python"
    assert TokenChannelFeatures.from_mapping(features.to_dict()) == features


def test_token_channel_features_reject_negative_block_offset() -> None:
    with pytest.raises(ValueError, match="block_relative_offset"):
        TokenChannelFeatures(
            node_type="expression_statement",
            parent_node_type="module",
            block_relative_offset=-1,
            in_code_body=True,
            structure_mask=True,
        )
