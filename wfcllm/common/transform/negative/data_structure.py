"""Negative transformation rules — data structure corruption."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule


class SliceStepFlip(Rule):
    """Replace lst[::-1] with lst[::1] (reverse to forward)."""
    name = "neg_slice_step"
    category = "数据结构操作"
    description = "切片方向反转：[::-1] → [::1]"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "subscript":
                # Look for slice children with step=-1
                for child in node.children:
                    if child.type == "slice":
                        colons = [c for c in child.children if c.type == ":"]
                        non_colons = [c for c in child.children if c.type != ":"]
                        # step is present when there are 2 colons; it's the last non-colon child
                        if len(colons) == 2 and non_colons:
                            step = non_colons[-1]
                            step_text = step.text.decode("utf-8")
                            if step_text in ("-1", "- 1"):
                                matches.append(Match(
                                    "slice", step.start_byte, step.end_byte,
                                    step_text, "1",
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


class DictViewSwap(Rule):
    """Replace dict.keys() with dict.values() and vice versa."""
    name = "neg_dict_view"
    category = "数据结构操作"
    description = "字典视图错位：.keys() ↔ .values()"

    _flip = {"keys": "values", "values": "keys"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
                        # Verify no args (bare .keys() / .values())
                        args_node = node.child_by_field_name("arguments")
                        args = [c for c in args_node.children
                                if c.type not in ("(", ")", ",")] if args_node else []
                        if not args:
                            name = attr.text.decode("utf-8")
                            matches.append(Match(
                                "attribute", attr.start_byte, attr.end_byte,
                                name, self._flip[name],
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
