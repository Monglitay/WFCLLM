"""Negative transformation rules — identifier corruption."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule


class ScopeVarCorrupt(Rule):
    """In index accesses lst[i], replace i with j if j exists in scope."""
    name = "neg_scope_var"
    category = "标识符操作"
    description = "作用域变量混淆：lst[i] → lst[j]"

    def detect(self, source, tree):
        """Find subscript accesses with single-char identifier index.
        Collect all single-char identifiers used as indices, then swap i→j or j→i."""
        matches = []
        index_vars = set()

        def collect_indices(node):
            if node.type == "subscript":
                idx = node.child_by_field_name("subscript")
                if idx and idx.type == "identifier" and len(idx.text.decode("utf-8")) == 1:
                    index_vars.add(idx.text.decode("utf-8"))
            for child in node.children:
                collect_indices(child)

        collect_indices(tree.root_node)

        if len(index_vars) < 2:
            return []

        # Build a rotation map among found index vars
        sorted_vars = sorted(index_vars)
        rotate = {sorted_vars[i]: sorted_vars[(i + 1) % len(sorted_vars)]
                  for i in range(len(sorted_vars))}

        def walk(node):
            if node.type == "subscript":
                idx = node.child_by_field_name("subscript")
                if idx and idx.type == "identifier":
                    name = idx.text.decode("utf-8")
                    if name in rotate:
                        matches.append(Match(
                            "subscript", idx.start_byte, idx.end_byte,
                            name, rotate[name],
                        ))
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
