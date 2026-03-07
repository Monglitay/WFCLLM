"""Syntax and initialization transformation rules."""

from __future__ import annotations

from wfcllm.common.transform.base import Match, Rule, parse_code


class ListInit(Rule):
    """Swap between [] and list()."""
    name = "list_init"
    category = "语法与初始化"
    description = "[] <-> list()"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            # [] -> list()
            if node.type == "list" and len(node.children) == 2:
                text = source[node.start_byte:node.end_byte]
                if text.strip() == "[]":
                    matches.append(Match("list", node.start_byte, node.end_byte, "[]", "list()"))
            # list() -> []
            if node.type == "call":
                func = node.child_by_field_name("function")
                args = node.child_by_field_name("arguments")
                if func and func.text.decode("utf-8") == "list" and args:
                    positional = [c for c in args.children if c.type not in ("(", ")", ",")]
                    if len(positional) == 0:
                        matches.append(Match("call", node.start_byte, node.end_byte, "list()", "[]"))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class DictInit(Rule):
    """Swap between {} and dict()."""
    name = "dict_init"
    category = "语法与初始化"
    description = "{} <-> dict()"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            # {} -> dict()
            if node.type == "dictionary" and len(node.children) == 2:
                text = source[node.start_byte:node.end_byte]
                if text.strip() == "{}":
                    matches.append(Match("dictionary", node.start_byte, node.end_byte, "{}", "dict()"))
            # dict() -> {}
            if node.type == "call":
                func = node.child_by_field_name("function")
                args = node.child_by_field_name("arguments")
                if func and func.text.decode("utf-8") == "dict" and args:
                    positional = [c for c in args.children if c.type not in ("(", ")", ",")]
                    if len(positional) == 0:
                        matches.append(Match("call", node.start_byte, node.end_byte, "dict()", "{}"))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class TypeCheck(Rule):
    """Swap between isinstance(x, T) and type(x) == T."""
    name = "type_check"
    category = "语法与初始化"
    description = "isinstance(x, T) <-> type(x) == T"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            # isinstance(x, T) -> type(x) == T
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.text.decode("utf-8") == "isinstance":
                    args = node.child_by_field_name("arguments")
                    if args:
                        positional = [c for c in args.children if c.type not in ("(", ")", ",")]
                        if len(positional) == 2:
                            obj = positional[0].text.decode("utf-8")
                            typ = positional[1].text.decode("utf-8")
                            replacement = f"type({obj}) == {typ}"
                            matches.append(Match("call", node.start_byte, node.end_byte,
                                                 source[node.start_byte:node.end_byte], replacement))
            # type(x) == T -> isinstance(x, T)
            if node.type == "comparison_operator":
                children = [c for c in node.children if c.type != "comment"]
                if len(children) == 3 and children[1].type == "==":
                    left = children[0]
                    right = children[2]
                    if left.type == "call":
                        func = left.child_by_field_name("function")
                        if func and func.text.decode("utf-8") == "type":
                            args = left.child_by_field_name("arguments")
                            if args:
                                positional = [c for c in args.children if c.type not in ("(", ")", ",")]
                                if len(positional) == 1:
                                    obj = positional[0].text.decode("utf-8")
                                    typ = right.text.decode("utf-8")
                                    replacement = f"isinstance({obj}, {typ})"
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


class StringFormat(Rule):
    """Convert '%s' % x to '{}'.format(x)."""
    name = "string_format"
    category = "语法与初始化"
    description = "'%s' % x -> '{}'.format(x)"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type == "%":
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right and left.type == "string":
                        left_text = left.text.decode("utf-8")
                        right_text = right.text.decode("utf-8")
                        # Simple case: replace %s with {}
                        # Determine quote char
                        quote = left_text[0]
                        inner = left_text[1:-1]
                        new_inner = inner.replace("%s", "{}")
                        replacement = f"{quote}{new_inner}{quote}.format({right_text})"
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
