"""Identifier transformation rules."""

from __future__ import annotations

import re

from wfcllm.common.transform.base import Match, Rule, parse_code

# Python builtins and keywords to never rename
_BUILTINS = {
    "print", "len", "range", "int", "str", "float", "list", "dict", "set",
    "tuple", "bool", "None", "True", "False", "type", "isinstance", "sum",
    "min", "max", "sorted", "zip", "map", "filter", "enumerate", "open",
    "round", "abs", "all", "any", "chr", "ord", "hex", "id", "input",
    "iter", "next", "super", "getattr", "setattr", "hasattr", "delattr",
    "callable", "repr", "hash", "format", "vars", "dir", "help",
    "return", "if", "else", "elif", "for", "while", "break", "continue",
    "def", "class", "import", "from", "as", "with", "try", "except",
    "finally", "raise", "pass", "yield", "lambda", "and", "or", "not",
    "in", "is", "global", "nonlocal", "assert", "del",
    # Common module names
    "self", "cls", "args", "kwargs",
}


def _snake_to_camel(name: str) -> str:
    """Convert snake_case to camelCase."""
    parts = name.split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])


def _camel_to_snake(name: str) -> str:
    """Convert camelCase to snake_case."""
    s = re.sub(r'([A-Z])', r'_\1', name)
    return s.lower().lstrip("_")


def _is_snake_case(name: str) -> bool:
    return "_" in name and name == name.lower()


def _is_camel_case(name: str) -> bool:
    return not "_" in name and name != name.lower() and name[0].islower()


class VariableRename(Rule):
    """Convert variable names between snake_case and camelCase."""
    name = "variable_rename"
    category = "标识符"
    description = "变量名风格转换"

    def detect(self, source, tree):
        matches = []
        seen = set()

        def walk(node):
            if node.type == "identifier":
                text = node.text.decode("utf-8")
                if text not in _BUILTINS and text not in seen and not text.startswith("__"):
                    if _is_snake_case(text):
                        seen.add(text)
                        new_name = _snake_to_camel(text)
                        # We'll handle all occurrences in apply
                        matches.append(Match("identifier", 0, 0, text, new_name))
                    elif _is_camel_case(text):
                        seen.add(text)
                        new_name = _camel_to_snake(text)
                        matches.append(Match("identifier", 0, 0, text, new_name))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in matches:
            # Replace all occurrences of the old name with the new name
            # Use word boundary replacement to avoid partial matches
            result = re.sub(r'\b' + re.escape(m.original_text) + r'\b', m.replacement_text, result)
        return result


class NameObfuscation(Rule):
    """Replace function/variable names with generic alternatives."""
    name = "name_obfuscation"
    category = "标识符"
    description = "函数/变量名混淆"

    # Mapping of common name patterns to generic replacements
    _prefix_map = {
        "calculate": "compute",
        "compute": "calculate",
        "get": "fetch",
        "fetch": "get",
        "find": "search",
        "search": "find",
        "check": "verify",
        "verify": "check",
        "create": "make",
        "make": "create",
        "update": "modify",
        "modify": "update",
        "remove": "delete",
        "delete": "remove",
        "process": "handle",
        "handle": "process",
    }

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "function_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    func_name = name_node.text.decode("utf-8")
                    if func_name not in _BUILTINS:
                        new_name = self._obfuscate(func_name)
                        if new_name != func_name:
                            matches.append(Match("identifier", 0, 0, func_name, new_name))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def _obfuscate(self, name: str) -> str:
        """Try to replace a prefix with its synonym."""
        parts = name.split("_")
        if parts[0] in self._prefix_map:
            parts[0] = self._prefix_map[parts[0]]
            return "_".join(parts)
        # For camelCase
        for prefix, replacement in self._prefix_map.items():
            if name.startswith(prefix) and len(name) > len(prefix):
                return replacement + name[len(prefix):]
        return name

    def apply(self, source, matches):
        result = source
        for m in matches:
            result = re.sub(r'\b' + re.escape(m.original_text) + r'\b', m.replacement_text, result)
        return result
