"""Negative transformation rules — control flow corruption."""

from __future__ import annotations

from rules.base import Match, Rule, parse_code


class OffByOne(Rule):
    """Change range(n) to range(n - 1) — classic off-by-one."""
    name = "neg_off_by_one"
    category = "控制流"
    description = "边界条件偏移：range(n) → range(n - 1)"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.text.decode("utf-8") == "range":
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        positional = [c for c in args_node.children
                                      if c.type not in ("(", ")", ",")]
                        if len(positional) == 1:
                            arg = positional[0]
                            arg_text = arg.text.decode("utf-8")
                            matches.append(Match(
                                "call", arg.start_byte, arg.end_byte,
                                arg_text, f"{arg_text} - 1",
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


class BreakContinueSwap(Rule):
    """Swap break and continue."""
    name = "neg_break_continue"
    category = "控制流"
    description = "循环控制中断：break ↔ continue"

    _flip = {"break_statement": "continue", "continue_statement": "break"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type in self._flip:
                matches.append(Match(
                    node.type, node.start_byte, node.end_byte,
                    source[node.start_byte:node.end_byte],
                    self._flip[node.type],
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


class IfElseBodySwap(Rule):
    """Swap if/else bodies WITHOUT negating the condition (malicious, not equivalent)."""
    name = "neg_if_else_body_swap"
    category = "控制流"
    description = "恶意分支对调：条件不变但对调 if/else 体"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "if_statement":
                alt = node.child_by_field_name("alternative")
                if alt and alt.type == "else_clause":
                    condition = node.child_by_field_name("condition")
                    consequence = node.child_by_field_name("consequence")
                    else_body = alt.child_by_field_name("body")
                    if condition and consequence and else_body:
                        cond_text = condition.text.decode("utf-8")
                        if_body_text = source[consequence.start_byte:consequence.end_byte]
                        else_body_text = source[else_body.start_byte:else_body.end_byte]
                        # Swap bodies, keep condition unchanged
                        replacement = f"if {cond_text}:\n{else_body_text}\nelse:\n{if_body_text}"
                        matches.append(Match(
                            "if_statement", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte], replacement,
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


class MembershipNegate(Rule):
    """Negate membership test: `x in s` → `x not in s`."""
    name = "neg_membership"
    category = "控制流"
    description = "包含关系取反：in ↔ not in"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                # Look for `in` or `not in` operator tokens
                for i, child in enumerate(children):
                    if child.type == "in":
                        # x in s → x not in s
                        left = children[i - 1] if i > 0 else None
                        right = children[i + 1] if i + 1 < len(children) else None
                        if left and right:
                            l = left.text.decode("utf-8")
                            r = right.text.decode("utf-8")
                            matches.append(Match(
                                "comparison_operator", node.start_byte, node.end_byte,
                                source[node.start_byte:node.end_byte],
                                f"{l} not in {r}",
                            ))
                    elif child.type == "not_in":
                        # x not in s → x in s
                        left = children[i - 1] if i > 0 else None
                        right = children[i + 1] if i + 1 < len(children) else None
                        if left and right:
                            l = left.text.decode("utf-8")
                            r = right.text.decode("utf-8")
                            matches.append(Match(
                                "comparison_operator", node.start_byte, node.end_byte,
                                source[node.start_byte:node.end_byte],
                                f"{l} in {r}",
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


class YieldReturnSwap(Rule):
    """Replace yield with return to break generator protocol."""
    name = "neg_yield_return"
    category = "控制流"
    description = "生成器状态截断：yield → return"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "yield":
                # yield expression node: "yield <value>"
                orig = source[node.start_byte:node.end_byte]
                # Replace leading "yield" with "return"
                replacement = "return" + orig[5:]
                matches.append(Match(
                    "yield", node.start_byte, node.end_byte, orig, replacement,
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
