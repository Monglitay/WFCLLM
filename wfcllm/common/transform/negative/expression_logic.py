"""Negative transformation rules — expression and logic corruption."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule


class EqNeqFlip(Rule):
    """Flip == to != and vice versa."""
    name = "neg_eq_neq"
    category = "表达式与逻辑"
    description = "关系运算符反转：== ↔ !="

    _flip = {"==": "!=", "!=": "=="}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                if len(children) == 3:
                    op = children[1]
                    if op.type in self._flip:
                        left = children[0].text.decode("utf-8")
                        right = children[2].text.decode("utf-8")
                        matches.append(Match(
                            "comparison_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{left} {self._flip[op.type]} {right}",
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


class ArithmeticOpReplace(Rule):
    """Replace + with - in binary expressions."""
    name = "neg_arithmetic_op"
    category = "表达式与逻辑"
    description = "算术运算符替换：+ → -"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type == "+":
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "binary_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} - {r}",
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


class AndOrSwap(Rule):
    """Replace boolean and with or and vice versa."""
    name = "neg_and_or"
    category = "表达式与逻辑"
    description = "逻辑运算符替换：and ↔ or"

    _flip = {"and": "or", "or": "and"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "boolean_operator":
                op_node = None
                for child in node.children:
                    if child.type in ("and", "or"):
                        op_node = child
                        break
                if op_node:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        new_op = self._flip[op_node.type]
                        matches.append(Match(
                            "boolean_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} {new_op} {r}",
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


class BoundsNarrow(Rule):
    """Replace <= with < and >= with > (remove boundary inclusion)."""
    name = "neg_bounds_narrow"
    category = "表达式与逻辑"
    description = "比较界限收缩：<= → <，>= → >"

    _narrow = {"<=": "<", ">=": ">"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                if len(children) == 3:
                    op = children[1]
                    if op.type in self._narrow:
                        left = children[0].text.decode("utf-8")
                        right = children[2].text.decode("utf-8")
                        matches.append(Match(
                            "comparison_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{left} {self._narrow[op.type]} {right}",
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


class AugAssignCorrupt(Rule):
    """Replace augmented assignment (+=) with simple assignment (=)."""
    name = "neg_aug_assign"
    category = "表达式与逻辑"
    description = "赋值降级：+= → ="

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "augmented_assignment":
                op_node = None
                for child in node.children:
                    if child.type == "+=":
                        op_node = child
                        break
                if op_node:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "augmented_assignment", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} = {r}",
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


class ShiftFlip(Rule):
    """Replace left shift with right shift and vice versa."""
    name = "neg_shift_flip"
    category = "表达式与逻辑"
    description = "移位运算反转：<< ↔ >>"

    _flip = {"<<": ">>", ">>": "<<"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type in self._flip:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "binary_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} {self._flip[op.type]} {r}",
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
