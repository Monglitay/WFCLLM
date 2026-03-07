"""Expression and logic transformation rules."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule, parse_code


class OperandSwap(Rule):
    """Swap operands of commutative operators (+ and *)."""
    name = "operand_swap"
    category = "表达式与逻辑"
    description = "交换可交换运算符的操作数"

    _commutative_ops = {"+", "*"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type in self._commutative_ops:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        left_text = left.text.decode("utf-8")
                        right_text = right.text.decode("utf-8")
                        op_text = op.type
                        replacement = f"{right_text} {op_text} {left_text}"
                        matches.append(Match("binary_operator", node.start_byte, node.end_byte,
                                             source[node.start_byte:node.end_byte], replacement))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ComparisonFlip(Rule):
    """Flip comparison operators and swap operands."""
    name = "comparison_flip"
    category = "表达式与逻辑"
    description = "翻转比较运算符"

    _flip_map = {"<": ">", ">": "<", "<=": ">=", ">=": "<="}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                # Simple case: left op right (3 children)
                if len(children) == 3:
                    left = children[0]
                    op = children[1]
                    right = children[2]
                    op_text = op.type
                    if op_text in self._flip_map:
                        left_text = left.text.decode("utf-8")
                        right_text = right.text.decode("utf-8")
                        new_op = self._flip_map[op_text]
                        replacement = f"{right_text} {new_op} {left_text}"
                        matches.append(Match("comparison_operator", node.start_byte, node.end_byte,
                                             source[node.start_byte:node.end_byte], replacement))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class UnarySimplify(Rule):
    """Simplify double negation: not not x -> x."""
    name = "unary_simplify"
    category = "表达式与逻辑"
    description = "简化双重否定"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "not_operator":
                arg = node.child_by_field_name("argument")
                if arg and arg.type == "not_operator":
                    inner = arg.child_by_field_name("argument")
                    if inner:
                        inner_text = inner.text.decode("utf-8")
                        matches.append(Match("not_operator", node.start_byte, node.end_byte,
                                             source[node.start_byte:node.end_byte], inner_text))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class DeMorgan(Rule):
    """Apply De Morgan's law: x and y -> not (not x or not y)."""
    name = "demorgan"
    category = "表达式与逻辑"
    description = "应用德摩根定律"

    _swap = {"and": "or", "or": "and"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "boolean_operator":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                op_node = None
                for child in node.children:
                    if child.type in ("and", "or"):
                        op_node = child
                        break
                if left and right and op_node:
                    op_text = op_node.type
                    if op_text in self._swap:
                        left_text = left.text.decode("utf-8")
                        right_text = right.text.decode("utf-8")
                        new_op = self._swap[op_text]
                        replacement = f"not (not {left_text} {new_op} not {right_text})"
                        matches.append(Match("boolean_operator", node.start_byte, node.end_byte,
                                             source[node.start_byte:node.end_byte], replacement))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ArithmeticAssociativity(Rule):
    """Distribute multiplication: a * (b + c) -> a * b + a * c."""
    name = "arithmetic_associativity"
    category = "表达式与逻辑"
    description = "乘法分配律展开"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type == "*":
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right and right.type == "parenthesized_expression":
                        inner = None
                        for child in right.children:
                            if child.type == "binary_operator":
                                inner = child
                                break
                        if inner:
                            inner_op = inner.child_by_field_name("operator")
                            if inner_op and inner_op.type == "+":
                                a = left.text.decode("utf-8")
                                b = inner.child_by_field_name("left").text.decode("utf-8")
                                c = inner.child_by_field_name("right").text.decode("utf-8")
                                replacement = f"{a} * {b} + {a} * {c}"
                                matches.append(Match("binary_operator", node.start_byte, node.end_byte,
                                                     source[node.start_byte:node.end_byte], replacement))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
