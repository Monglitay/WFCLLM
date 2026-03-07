"""Control flow transformation rules."""

from __future__ import annotations

import textwrap

from wfcllm.common.transform.base import Match, Rule, parse_code


class LoopConvert(Rule):
    """Convert for i in range(n) to while loop."""
    name = "loop_convert"
    category = "控制流"
    description = "for-range 循环转 while 循环"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "for_statement":
                # Check if iterating over range()
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                if (left and right and right.type == "call"):
                    func = right.child_by_field_name("function")
                    if func and func.text.decode("utf-8") == "range":
                        args = right.child_by_field_name("arguments")
                        if args:
                            positional = [c for c in args.children if c.type not in ("(", ")", ",")]
                            if len(positional) in (1, 2):
                                matches.append(Match(
                                    "for_statement", node.start_byte, node.end_byte,
                                    source[node.start_byte:node.end_byte], ""))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            tree = parse_code(source)
            # Re-find the node at this position
            node = self._find_for_node(tree.root_node, m.start_byte)
            if not node:
                continue
            replacement = self._convert_for_to_while(node, source)
            result = result[:m.start_byte] + replacement + result[m.end_byte:]
        return result

    def _find_for_node(self, root, start_byte):
        if root.type == "for_statement" and root.start_byte == start_byte:
            return root
        for child in root.children:
            found = self._find_for_node(child, start_byte)
            if found:
                return found
        return None

    def _convert_for_to_while(self, node, source):
        left = node.child_by_field_name("left")
        right = node.child_by_field_name("right")
        body = node.child_by_field_name("body")

        var_name = left.text.decode("utf-8")
        args = right.child_by_field_name("arguments")
        positional = [c for c in args.children if c.type not in ("(", ")", ",")]

        if len(positional) == 1:
            start_val = "0"
            end_val = positional[0].text.decode("utf-8")
        else:
            start_val = positional[0].text.decode("utf-8")
            end_val = positional[1].text.decode("utf-8")

        # Get body content and its indentation
        body_text = source[body.start_byte:body.end_byte]
        # Detect indentation from first line of body
        body_lines = body_text.split("\n")
        indent = ""
        for ch in body_lines[0]:
            if ch in (" ", "\t"):
                indent += ch
            else:
                break

        # Build while loop
        lines = [
            f"{var_name} = {start_val}",
            f"while {var_name} < {end_val}:",
            body_text,
            f"{indent}{var_name} += 1",
        ]
        return "\n".join(lines)


class IterationConvert(Rule):
    """Convert direct iteration to index-based: for x in lst -> for i in range(len(lst))."""
    name = "iteration_convert"
    category = "控制流"
    description = "直接迭代转索引迭代"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "for_statement":
                left = node.child_by_field_name("left")
                right = node.child_by_field_name("right")
                # Match: for x in var (not a call like range())
                if left and right and right.type == "identifier":
                    matches.append(Match(
                        "for_statement", node.start_byte, node.end_byte,
                        source[node.start_byte:node.end_byte], ""))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            tree = parse_code(result)
            node = self._find_for_node(tree.root_node, m.start_byte)
            if not node:
                continue

            left = node.child_by_field_name("left")
            right = node.child_by_field_name("right")
            body = node.child_by_field_name("body")

            var_name = left.text.decode("utf-8")
            iterable = right.text.decode("utf-8")
            body_text = result[body.start_byte:body.end_byte]

            # Detect indentation
            body_lines = body_text.split("\n")
            indent = ""
            for ch in body_lines[0]:
                if ch in (" ", "\t"):
                    indent += ch
                else:
                    break

            idx_var = "_i"
            new_body = f"{indent}{var_name} = {iterable}[{idx_var}]\n{body_text}"
            replacement = f"for {idx_var} in range(len({iterable})):\n{new_body}"
            result = result[:node.start_byte] + replacement + result[node.end_byte:]
        return result

    def _find_for_node(self, root, start_byte):
        if root.type == "for_statement" and root.start_byte == start_byte:
            return root
        for child in root.children:
            found = self._find_for_node(child, start_byte)
            if found:
                return found
        return None


class ComprehensionConvert(Rule):
    """Convert list comprehension to map: [f(x) for x in lst] -> list(map(lambda x: f(x), lst))."""
    name = "comprehension_convert"
    category = "控制流"
    description = "列表推导转 map"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "list_comprehension":
                # Check it has exactly one for_in_clause and no if_clause
                clauses = [c for c in node.children if c.type == "for_in_clause"]
                ifs = [c for c in node.children if c.type == "if_clause"]
                if len(clauses) == 1 and len(ifs) == 0:
                    matches.append(Match(
                        "list_comprehension", node.start_byte, node.end_byte,
                        source[node.start_byte:node.end_byte], ""))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            tree = parse_code(result)
            node = self._find_node(tree.root_node, "list_comprehension", m.start_byte)
            if not node:
                continue

            # Get the body expression and the for_in_clause
            body_node = node.child_by_field_name("body")
            clause = [c for c in node.children if c.type == "for_in_clause"][0]
            var_node = clause.child_by_field_name("left")
            iter_node = clause.child_by_field_name("right")

            body_text = body_node.text.decode("utf-8")
            var_text = var_node.text.decode("utf-8")
            iter_text = iter_node.text.decode("utf-8")

            replacement = f"list(map(lambda {var_text}: {body_text}, {iter_text}))"
            result = result[:node.start_byte] + replacement + result[node.end_byte:]
        return result

    def _find_node(self, root, node_type, start_byte):
        if root.type == node_type and root.start_byte == start_byte:
            return root
        for child in root.children:
            found = self._find_node(child, node_type, start_byte)
            if found:
                return found
        return None


class BranchFlip(Rule):
    """Negate if condition and swap if/else bodies."""
    name = "branch_flip"
    category = "控制流"
    description = "翻转 if/else 分支"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "if_statement":
                alt = node.child_by_field_name("alternative")
                if alt and alt.type == "else_clause":
                    condition = node.child_by_field_name("condition")
                    consequence = node.child_by_field_name("consequence")

                    if not condition or not consequence:
                        for child in node.children:
                            walk(child)
                        return

                    else_body_node = alt.child_by_field_name("body")
                    if not else_body_node:
                        for child in node.children:
                            walk(child)
                        return

                    cond_text = condition.text.decode("utf-8")
                    if_body = source[consequence.start_byte:consequence.end_byte]
                    else_body = source[else_body_node.start_byte:else_body_node.end_byte]

                    replacement = f"if not {cond_text}:\n{else_body}\nelse:\n{if_body}"
                    matches.append(Match(
                        "if_statement", node.start_byte, node.end_byte,
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
