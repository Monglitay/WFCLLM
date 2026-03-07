"""API call transformation rules — explicit default arguments."""

from __future__ import annotations

import re

from wfcllm.common.transform.base import Match, Rule


class _ExplicitDefaultArgRule(Rule):
    """Base for rules that add an explicit default keyword argument to a function call.

    Subclasses set:
        func_names: set of function name strings to match (e.g. {"print"}, {"random.seed"})
        kwarg_name: the keyword argument to add (e.g. "flush")
        kwarg_value: the default value string (e.g. "False")
    """
    category = "API与函数调用"
    func_names: set[str] = set()
    kwarg_name: str = ""
    kwarg_value: str = ""

    def _get_call_name(self, node) -> str:
        """Extract the function name from a call node."""
        func_node = node.child_by_field_name("function")
        if func_node is None:
            return ""
        return func_node.text.decode("utf-8")

    def _has_kwarg(self, node, kwarg: str) -> bool:
        """Check if the call already has the specified keyword argument."""
        args_node = node.child_by_field_name("arguments")
        if args_node is None:
            return False
        for child in args_node.children:
            if child.type == "keyword_argument":
                name_node = child.child_by_field_name("name")
                if name_node and name_node.text.decode("utf-8") == kwarg:
                    return True
        return False

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                name = self._get_call_name(node)
                if name in self.func_names and not self._has_kwarg(node, self.kwarg_name):
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        # Insert before the closing paren
                        close_paren = args_node.end_byte - 1
                        # Check if there are existing args (need comma)
                        existing_args = [c for c in args_node.children if c.type not in ("(", ")", ",")]
                        prefix = ", " if existing_args else ""
                        insertion = f"{prefix}{self.kwarg_name}={self.kwarg_value}"
                        matches.append(Match(
                            node_type="call",
                            start_byte=close_paren,
                            end_byte=close_paren,
                            original_text="",
                            replacement_text=insertion,
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


class ExplicitDefaultPrint(_ExplicitDefaultArgRule):
    name = "explicit_default_print"
    description = "为 print 补充默认 flush 参数"
    func_names = {"print"}
    kwarg_name = "flush"
    kwarg_value = "False"


class ExplicitDefaultRange(_ExplicitDefaultArgRule):
    """range(x) -> range(0, x). Different pattern: positional arg, not kwarg."""
    name = "explicit_default_range"
    description = "为 range 补充起始值 0"
    category = "API与函数调用"
    func_names = {"range"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                name = self._get_call_name(node)
                if name == "range":
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        positional = [c for c in args_node.children if c.type not in ("(", ")", ",", "keyword_argument")]
                        if len(positional) == 1:
                            # range(x) -> range(0, x)
                            arg = positional[0]
                            matches.append(Match(
                                node_type="call",
                                start_byte=arg.start_byte,
                                end_byte=arg.end_byte,
                                original_text=arg.text.decode("utf-8"),
                                replacement_text=f"0, {arg.text.decode('utf-8')}",
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


class ExplicitDefaultOpen(_ExplicitDefaultArgRule):
    name = "explicit_default_open"
    description = "为 open 补充默认 closefd 参数"
    func_names = {"open"}
    kwarg_name = "closefd"
    kwarg_value = "True"


class ExplicitDefaultSorted(_ExplicitDefaultArgRule):
    name = "explicit_default_sorted"
    description = "为 sorted 补充 reverse 参数"
    func_names = {"sorted"}
    kwarg_name = "reverse"
    kwarg_value = "False"


class ExplicitDefaultMinMax(_ExplicitDefaultArgRule):
    name = "explicit_default_min_max"
    description = "补充 min/max 的 key 参数"
    func_names = {"min", "max"}
    kwarg_name = "key"
    kwarg_value = "None"


class ExplicitDefaultZip(_ExplicitDefaultArgRule):
    name = "explicit_default_zip"
    description = "补充 zip 的 strict 参数"
    func_names = {"zip"}
    kwarg_name = "strict"
    kwarg_value = "False"


class ExplicitDefaultRandomSeed(_ExplicitDefaultArgRule):
    name = "explicit_default_random_seed"
    description = "补充 random.seed 的 version 参数"
    func_names = {"random.seed"}
    kwarg_name = "version"
    kwarg_value = "2"


class ExplicitDefaultHtmlEscape(_ExplicitDefaultArgRule):
    name = "explicit_default_html_escape"
    description = "补充 html.escape 的 quote 参数"
    func_names = {"html.escape"}
    kwarg_name = "quote"
    kwarg_value = "True"


class ExplicitDefaultRound(_ExplicitDefaultArgRule):
    name = "explicit_default_round"
    description = "补充 round 的 ndigits 参数"
    func_names = {"round"}
    kwarg_name = "ndigits"
    kwarg_value = "None"


class ExplicitDefaultJsonDump(_ExplicitDefaultArgRule):
    name = "explicit_default_json_dump"
    description = "补充 json.dump 的 indent 参数"
    func_names = {"json.dump", "json.dumps"}
    kwarg_name = "indent"
    kwarg_value = "None"


class LibraryAliasReplace(Rule):
    """Swap library aliases: np <-> numpy, tf <-> tensorflow, etc."""
    name = "library_alias_replace"
    category = "API与函数调用"
    description = "库别名替换"

    _alias_map = {
        "np": "numpy",
        "numpy": "np",
        "tf": "tensorflow",
        "tensorflow": "tf",
        "pd": "pandas",
        "pandas": "pd",
        "plt": "matplotlib.pyplot",
        "sns": "seaborn",
    }

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "attribute":
                obj = node.child_by_field_name("object")
                if obj and obj.type == "identifier":
                    obj_name = obj.text.decode("utf-8")
                    if obj_name in self._alias_map:
                        new_name = self._alias_map[obj_name]
                        full_text = source[node.start_byte:node.end_byte]
                        replacement = new_name + full_text[len(obj_name):]
                        matches.append(Match("attribute", node.start_byte, node.end_byte,
                                             full_text, replacement))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ThirdPartyFuncReplace(Rule):
    """Replace builtin functions with numpy equivalents: max() -> np.max()."""
    name = "third_party_func_replace"
    category = "API与函数调用"
    description = "内置函数替换为第三方库等价函数"

    _func_map = {
        "max": "np.max",
        "min": "np.min",
        "sum": "np.sum",
        "abs": "np.abs",
        "round": "np.round",
    }

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "identifier":
                    func_name = func.text.decode("utf-8")
                    if func_name in self._func_map:
                        new_name = self._func_map[func_name]
                        matches.append(Match("call", func.start_byte, func.end_byte,
                                             func_name, new_name))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
