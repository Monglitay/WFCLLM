"""Negative transformation rules — exception handling corruption."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule, parse_code


class ExceptionSwallow(Rule):
    """Replace `raise` inside except block with `pass`."""
    name = "neg_exception_swallow"
    category = "异常处理"
    description = "异常吞噬：except 块中 raise → pass"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "except_clause":
                # Find raise_statement children inside the except body
                body = None
                for child in node.children:
                    if child.type == "block":
                        body = child
                        break
                if body:
                    for stmt in body.children:
                        if stmt.type == "raise_statement":
                            matches.append(Match(
                                "raise_statement", stmt.start_byte, stmt.end_byte,
                                source[stmt.start_byte:stmt.end_byte], "pass",
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
