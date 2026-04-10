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


def test_token_channel_features_reject_non_bool_mapping_fields() -> None:
    with pytest.raises(ValueError, match="in_code_body"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": "if_statement",
                "parent_node_type": "block",
                "block_relative_offset": 0,
                "in_code_body": "yes",
                "structure_mask": True,
            }
        )

    with pytest.raises(ValueError, match="structure_mask"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": "if_statement",
                "parent_node_type": "block",
                "block_relative_offset": 0,
                "in_code_body": True,
                "structure_mask": 1,
            }
        )


def test_token_channel_features_reject_malformed_string_and_integer_fields() -> None:
    with pytest.raises(ValueError, match="node_type"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": None,
                "parent_node_type": "block",
                "block_relative_offset": 0,
                "in_code_body": True,
                "structure_mask": True,
            }
        )

    with pytest.raises(ValueError, match="parent_node_type"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": "if_statement",
                "parent_node_type": True,
                "block_relative_offset": 0,
                "in_code_body": True,
                "structure_mask": True,
            }
        )

    with pytest.raises(ValueError, match="block_relative_offset"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": "if_statement",
                "parent_node_type": "block",
                "block_relative_offset": "0",
                "in_code_body": True,
                "structure_mask": True,
            }
        )


def test_token_channel_features_reject_non_mapping_payload_and_bad_direct_types() -> None:
    with pytest.raises(ValueError, match="mapping"):
        TokenChannelFeatures.from_mapping([1, 2, 3])

    with pytest.raises(ValueError, match="node_type"):
        TokenChannelFeatures(
            node_type=None,
            parent_node_type="block",
            block_relative_offset=0,
            in_code_body=True,
            structure_mask=True,
        )

    with pytest.raises(ValueError, match="block_relative_offset"):
        TokenChannelFeatures(
            node_type="if_statement",
            parent_node_type="block",
            block_relative_offset="0",
            in_code_body=True,
            structure_mask=True,
        )

    with pytest.raises(ValueError, match="in_code_body"):
        TokenChannelFeatures(
            node_type="if_statement",
            parent_node_type="block",
            block_relative_offset=0,
            in_code_body=1,
            structure_mask=True,
        )


def test_token_channel_features_reject_missing_required_keys() -> None:
    with pytest.raises(ValueError, match="Missing required TokenChannelFeatures keys"):
        TokenChannelFeatures.from_mapping(
            {
                "node_type": "if_statement",
                "parent_node_type": "block",
                "in_code_body": True,
                "structure_mask": True,
            }
        )
