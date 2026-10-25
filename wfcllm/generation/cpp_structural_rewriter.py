"""Key-blind C++ expression rewrites within the selected statement window."""

from __future__ import annotations

from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
from typing import Any

from wfcllm.gate.data import RewriteRequest
from wfcllm.generation.window_rewriter import (
    KeyBlindCppEquivalentWindowRewriter,
    ParsedRewrite,
    RewriteGeneration,
    _parse_generation,
    _validate_call,
)
from wfcllm.lang.cpp.parser import CppParser


def _comments(text: str) -> Counter[bytes]:
    raw = text.encode("utf-8")
    pending = [CppParser().parse(text).root_node]
    found: Counter[bytes] = Counter()
    while pending:
        node = pending.pop()
        if node.type == "comment":
            found[raw[node.start_byte:node.end_byte]] += 1
        pending.extend(node.named_children)
    return found


def _walk_named_nodes(root: Any):
    pending = [root]
    while pending:
        node = pending.pop()
        yield node
        pending.extend(reversed(node.named_children))


def _local_alpha_variants(request: RewriteRequest) -> tuple[str, ...]:
    """Rename primitive locals whose declaration and all references stay in-window.

    Macro arguments, preprocessing definitions, nested bindings, qualified names,
    asm and uses outside the window make the name unavailable. The caller still
    enforces the ordinary semantic guards; this emits no equivalence certificate.
    """
    text = request.completed_prefix + request.original_window + request.completed_suffix
    raw = text.encode("utf-8")
    tree = CppParser().parse(text)
    if tree.root_node.has_error:
        return ()
    nodes = tuple(_walk_named_nodes(tree.root_node))
    if any(
        (node.type.startswith("preproc_") and node.type != "preproc_include")
        or "asm" in node.type for node in nodes
    ):
        return ()
    first = len(request.completed_prefix.encode("utf-8"))
    last = first + len(request.original_window.encode("utf-8"))
    within = tuple(node for node in nodes if first <= node.start_byte and node.end_byte <= last)
    if any(node.type in {"lambda_expression", "class_specifier", "struct_specifier", "compound_statement"}
           for node in within):
        return ()
    names: dict[bytes, list[Any]] = defaultdict(list)
    for node in nodes:
        if node.type == "identifier":
            names[raw[node.start_byte:node.end_byte]].append(node)
    locals_: list[tuple[bytes, tuple[Any, ...]]] = []
    for node in within:
        if node.type != "init_declarator" or node.parent.type != "declaration":
            continue
        declaration = node.parent
        if declaration.parent.type != "compound_statement":
            continue
        result_type = declaration.child_by_field_name("type")
        if result_type is None or result_type.type not in {"primitive_type", "sized_type_specifier"}:
            continue
        declarator = node.child_by_field_name("declarator")
        if declarator is None or declarator.type != "identifier":
            continue
        name = raw[declarator.start_byte:declarator.end_byte]
        occurrences = tuple(names[name])
        if any(not first <= occurrence.start_byte < occurrence.end_byte <= last for occurrence in occurrences):
            continue
        unsafe = False
        for occurrence in occurrences:
            ancestor = occurrence.parent
            while ancestor is not None and ancestor.start_byte >= first and ancestor.end_byte <= last:
                if ancestor.type in {"argument_list", "qualified_identifier", "labeled_statement",
                                     "template_argument_list", "type_descriptor"}:
                    unsafe = True
                    break
                ancestor = ancestor.parent
        if unsafe:
            continue
        locals_.append((name, occurrences))
    variants: list[str] = []
    for ordinal in range(1, 7):
        if not locals_:
            break
        name, occurrences = locals_[(ordinal - 1) % len(locals_)]
        replacement = name + ("_local" + str(ordinal)).encode("ascii")
        if replacement in names or replacement in raw:
            continue
        edits = sorted(((node.start_byte - first, node.end_byte - first) for node in occurrences), reverse=True)
        candidate = request.original_window.encode("utf-8")
        for start, end in edits:
            candidate = candidate[:start] + replacement + candidate[end:]
        rewritten = candidate.decode("utf-8")
        if rewritten != request.original_window and rewritten not in variants:
            variants.append(rewritten)
    return tuple(variants)


class KeyBlindCppStructuralWindowRewriter(KeyBlindCppEquivalentWindowRewriter):
    """Produce deterministic rules without keys, tests or semantic certificates.

    Each expression appears once and no comments are added. The caller must
    still enforce its semantic cosine, precision, margin and scope checks.
    Class returns avoid comma expressions to preserve named return elision.
    """

    def _expression_variants(self, request: RewriteRequest) -> tuple[str, ...]:
        original = request.original_window
        comments = _comments(original)
        pool: list[str] = []

        def add(text: str) -> None:
            if text != original and text not in pool and _comments(text) == comments:
                pool.append(text)

        source_text = request.completed_prefix + original + request.completed_suffix
        source = source_text.encode("utf-8")
        tree = CppParser().parse(source_text)
        offset = len(request.completed_prefix.encode("utf-8"))
        first = request.structural_boundary.start_byte
        last = request.structural_boundary.end_byte
        units = [
            unit for unit in self._extractor.extract(source_text)
            if unit.start_byte >= first and unit.end_byte <= last
        ]
        if len(units) != request.window_length:
            raise ValueError("C++ structural request disagrees with original statement units")
        targets: list[tuple[Any, str, bool]] = []
        kinds = {
            "return_statement", "declaration", "expression_statement", "if_statement",
            "while_statement", "for_statement", "switch_statement",
        }
        for unit in units:
            node = tree.root_node.descendant_for_byte_range(unit.start_byte, unit.end_byte - 1)
            while node is not None and not (
                node.start_byte == unit.start_byte and node.type in kinds
            ):
                node = node.parent
            if node is None:
                continue
            if node.type == "return_statement":
                owner = node.parent
                while owner is not None and owner.type != "function_definition":
                    owner = owner.parent
                result_type = owner.child_by_field_name("type") if owner is not None else None
                primitive = result_type is not None and result_type.type in {
                    "primitive_type", "sized_type_specifier",
                }
                targets.extend(
                    (child, "return", primitive)
                    for child in node.named_children if child.type != "comment"
                )
            elif node.type == "declaration":
                pending = list(node.named_children)
                while pending:
                    child = pending.pop(0)
                    if child.type == "init_declarator":
                        value = child.child_by_field_name("value")
                        declarator = child.child_by_field_name("declarator")
                        if value is not None and declarator is not None and (
                            b"=" in source[declarator.end_byte:value.start_byte]
                        ):
                            targets.append((value, "value", False))
                    elif child.type not in {
                        "lambda_expression", "function_definition", "class_specifier",
                    }:
                        pending.extend(child.named_children)
            elif node.type == "expression_statement":
                for child in node.named_children:
                    if child.type == "assignment_expression":
                        value = child.child_by_field_name("right")
                        if value is not None:
                            targets.append((value, "value", False))
                    elif child.type == "update_expression":
                        targets.append((child, "value", False))
            else:
                value = node.child_by_field_name("condition")
                if value is not None and value.type == "condition_clause":
                    value = value.child_by_field_name("value")
                if value is not None and value.type not in {"declaration", "init_declarator"}:
                    category = "switch" if node.type == "switch_statement" else "condition"
                    targets.append((value, category, False))
        raw = original.encode("utf-8")
        atomic: list[tuple[int, int, tuple[str, ...]]] = []
        for node, category, primitive in sorted(targets, key=lambda item: item[0].start_byte):
            if node.type in {"initializer_list", "argument_list", "lambda_expression"}:
                continue
            expression = source[node.start_byte:node.end_byte].decode("utf-8")
            forms = [f"({expression})", f"(({expression}))", f"((({expression})))"]
            if category != "return" or primitive:
                forms.append(f"((void)0, ({expression}))")
            if category == "condition":
                forms.append(f"static_cast<bool>({expression})")
            start, end = node.start_byte - offset, node.end_byte - offset
            if not 0 <= start < end <= len(raw):
                raise ValueError("C++ expression escaped the replacement window")
            atomic.append((start, end, tuple(forms)))
            for value in forms:
                add((raw[:start] + value.encode("utf-8") + raw[end:]).decode("utf-8"))
        base = tuple(pool) if pool else (original,)
        trajectory = [base[index % len(base)] for index in range(18)]
        # Preserve the first twelve trials and every existing distinct variant.
        # Only repeated late trials can be replaced, without observing any key,
        # signature, semantic score or correctness result.
        if len(base) > 12 or len(atomic) < 2:
            return tuple(trajectory)
        if any(left[1] > right[0] for left, right in zip(atomic, atomic[1:])):
            return tuple(trajectory)
        combined: list[str] = []
        for ordinal in range(64):
            number = ordinal
            edits: list[tuple[int, int, bytes]] = []
            for start, end, alternatives in atomic:
                number, digit = divmod(number, len(alternatives))
                edits.append((start, end, alternatives[digit].encode("utf-8")))
            changed = raw
            for start, end, value in reversed(edits):
                changed = changed[:start] + value + changed[end:]
            text = changed.decode("utf-8")
            if (text != original and text not in base and text not in combined
                    and _comments(text) == comments):
                combined.append(text)
            if len(combined) == 6:
                break
        for index, text in enumerate(combined, start=12):
            trajectory[index] = text
        return tuple(trajectory)

    @lru_cache(maxsize=8)
    def _variants(self, request: RewriteRequest) -> tuple[str, ...]:
        trajectory = list(self._expression_variants(request))
        for index, text in enumerate(_local_alpha_variants(request), start=12):
            trajectory[index] = text
        return tuple(trajectory)

    def rewrite_window(
        self, request: RewriteRequest, *, candidate_index: int
    ) -> ParsedRewrite:
        _validate_call(request, candidate_index)
        pool = self._variants(request)
        text = pool[(candidate_index - 1) % len(pool)]
        generation = RewriteGeneration(
            token_ids=(),
            text=text,
            generation_seed_id=(
                "public-cpp-structural:" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
            ),
            rewrite_config_id=f"cpp-structural-local-alpha/v3:{candidate_index}",
        )
        return _parse_generation(
            request, generation, extractor=self._extractor,
            window_contract_version=self._window_contract_version,
        )
