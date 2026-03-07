"""Formatting transformation rules."""

from __future__ import annotations

import re

from wfcllm.common.transform.base import Match, Rule, parse_code


class FixSpacing(Rule):
    """Add spaces around operators that lack them."""
    name = "fix_spacing"
    category = "格式化"
    description = "为运算符添加空格"

    # Operators that should have spaces around them
    _op_pattern = re.compile(r'(?<=[^\s=!<>+\-*/%])([=+\-*/<>]+)(?=[^\s=!<>+\-*/%])')

    def _rebuild_spaced(self, node, source):
        """Recursively rebuild a node with proper spacing around operators."""
        target_types = ("binary_operator", "assignment", "augmented_assignment", "comparison_operator")
        if node.type in target_types:
            parts = []
            for child in node.children:
                parts.append(self._rebuild_spaced(child, source))
            return " ".join(parts)
        return source[node.start_byte:node.end_byte]

    def detect(self, source, tree):
        matches = []
        target_types = ("binary_operator", "assignment", "augmented_assignment", "comparison_operator")
        def walk(node):
            if node.type in target_types:
                # Only match topmost operator nodes (skip if parent is also a target)
                parent = node.parent
                if parent and parent.type in target_types:
                    # Skip — will be handled by the parent
                    for child in node.children:
                        walk(child)
                    return
                original = source[node.start_byte:node.end_byte]
                spaced = self._rebuild_spaced(node, source)
                if spaced != original:
                    matches.append(Match(node.type, node.start_byte, node.end_byte, original, spaced))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class FixCommentSymbols(Rule):
    """Normalize multiple # symbols to single #."""
    name = "fix_comment_symbols"
    category = "格式化"
    description = "规范化注释符号"

    _multi_hash = re.compile(r'^(#{2,})\s*')

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comment":
                text = node.text.decode("utf-8")
                m = self._multi_hash.match(text)
                if m:
                    # Replace multiple # with single #
                    new_text = "# " + text[m.end():].lstrip()
                    matches.append(Match("comment", node.start_byte, node.end_byte, text, new_text))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
