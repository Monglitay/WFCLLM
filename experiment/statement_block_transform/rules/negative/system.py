"""Negative transformation rules — system interaction corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class SysExitFlip(Rule):
    """Replace sys.exit(0) with sys.exit(1) and vice versa."""
    name = "neg_sys_exit"
    category = "系统交互"
    description = "退出状态码反转：sys.exit(0) ↔ sys.exit(1)"

    _flip = {"0": "1", "1": "0"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    obj = func.child_by_field_name("object")
                    attr = func.child_by_field_name("attribute")
                    if (obj and obj.text.decode("utf-8") == "sys"
                            and attr and attr.text.decode("utf-8") == "exit"):
                        args_node = node.child_by_field_name("arguments")
                        if args_node:
                            positional = [c for c in args_node.children
                                          if c.type not in ("(", ")", ",")]
                            if len(positional) == 1:
                                arg = positional[0]
                                arg_text = arg.text.decode("utf-8")
                                if arg_text in self._flip:
                                    matches.append(Match(
                                        "call", arg.start_byte, arg.end_byte,
                                        arg_text, self._flip[arg_text],
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
