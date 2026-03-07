"""Negative transformation rules — API call semantic corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class MinMaxFlip(Rule):
    """Replace min with max or max with min."""
    name = "neg_min_max_flip"
    category = "API与函数调用"
    description = "函数语义翻转：min ↔ max"

    _flip = {"min": "max", "max": "min"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "identifier":
                    name = func.text.decode("utf-8")
                    if name in self._flip:
                        matches.append(Match(
                            "call", func.start_byte, func.end_byte,
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


class AnyAllFlip(Rule):
    """Replace any() with all() and vice versa."""
    name = "neg_any_all_flip"
    category = "API与函数调用"
    description = "逻辑量词替换：any ↔ all"

    _flip = {"any": "all", "all": "any"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "identifier":
                    name = func.text.decode("utf-8")
                    if name in self._flip:
                        matches.append(Match(
                            "call", func.start_byte, func.end_byte,
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


class SortedReverseFlip(Rule):
    """Flip sorted reverse=False to reverse=True, or add reverse=True to sorted(x)."""
    name = "neg_sorted_reverse"
    category = "API与函数调用"
    description = "关键参数篡改：sorted reverse 方向翻转"

    def _get_call_name(self, node) -> str:
        func = node.child_by_field_name("function")
        return func.text.decode("utf-8") if func else ""

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call" and self._get_call_name(node) == "sorted":
                args_node = node.child_by_field_name("arguments")
                if args_node is None:
                    for child in node.children:
                        walk(child)
                    return
                # Case 1: has reverse=False → flip to True
                for child in args_node.children:
                    if child.type == "keyword_argument":
                        kname = child.child_by_field_name("name")
                        kval = child.child_by_field_name("value")
                        if (kname and kname.text.decode("utf-8") == "reverse"
                                and kval and kval.text.decode("utf-8") == "False"):
                            matches.append(Match(
                                "keyword_argument", kval.start_byte, kval.end_byte,
                                "False", "True",
                            ))
                            return
                # Case 2: no reverse kwarg → insert reverse=True
                existing_args = [c for c in args_node.children if c.type not in ("(", ")", ",")]
                has_reverse = any(
                    c.type == "keyword_argument"
                    and c.child_by_field_name("name")
                    and c.child_by_field_name("name").text.decode("utf-8") == "reverse"
                    for c in args_node.children
                )
                if not has_reverse:
                    close_paren = args_node.end_byte - 1
                    prefix = ", " if existing_args else ""
                    matches.append(Match(
                        "call", close_paren, close_paren,
                        "", f"{prefix}reverse=True",
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


class OpenModeCorrupt(Rule):
    """Change open(f, 'r') to open(f, 'w')."""
    name = "neg_open_mode"
    category = "API与函数调用"
    description = "文件模式破坏：'r' → 'w'"

    _corrupt_map = {"'r'": "'w'", '"r"': '"w"'}

    def _get_call_name(self, node) -> str:
        func = node.child_by_field_name("function")
        return func.text.decode("utf-8") if func else ""

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call" and self._get_call_name(node) == "open":
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    positional = [c for c in args_node.children
                                  if c.type not in ("(", ")", ",", "keyword_argument")]
                    if len(positional) >= 2:
                        mode_node = positional[1]
                        mode_text = mode_node.text.decode("utf-8")
                        if mode_text in self._corrupt_map:
                            matches.append(Match(
                                "call", mode_node.start_byte, mode_node.end_byte,
                                mode_text, self._corrupt_map[mode_text],
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


class ExtendAppendSwap(Rule):
    """Replace .extend(x) with .append(x)."""
    name = "neg_extend_append"
    category = "API与函数调用"
    description = "列表追加模式破坏：extend ↔ append"

    _flip = {"extend": "append", "append": "extend"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
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


class StartsEndsSwap(Rule):
    """Replace .startswith() with .endswith() and vice versa."""
    name = "neg_starts_ends"
    category = "API与函数调用"
    description = "字符串边界颠倒：startswith ↔ endswith"

    _flip = {"startswith": "endswith", "endswith": "startswith"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
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


class CeilFloorFlip(Rule):
    """Replace math.ceil with math.floor and vice versa."""
    name = "neg_ceil_floor"
    category = "API与函数调用"
    description = "舍入方向翻转：ceil ↔ floor"

    _flip = {"ceil": "floor", "floor": "ceil"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    obj = func.child_by_field_name("object")
                    if (attr and obj
                            and obj.text.decode("utf-8") == "math"
                            and attr.text.decode("utf-8") in self._flip):
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
