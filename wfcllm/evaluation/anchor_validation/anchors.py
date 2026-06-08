"""Anchor text helpers for offline validation diagnostics."""

from __future__ import annotations

import io
import keyword
import re
import tokenize

from wfcllm.evaluation.anchor_validation.schema import (
    AnchorMethod,
    CandidateBlock,
    CandidateContext,
)

_ACCUMULATOR_NAMES = {
    "acc",
    "accum",
    "answer",
    "count",
    "counts",
    "result",
    "results",
    "sum",
    "total",
}

_CODET5_CANDIDATE_ANCHOR_METHODS = {
    AnchorMethod.CODET5_VALID_SKELETON,
    AnchorMethod.CODET5_COMMENT_ANCHOR,
    AnchorMethod.CODET5_COMMENT_MINIMAL,
    AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
    AnchorMethod.CODET5_IDENTIFIER_ANCHOR,
}


def mask_code_skeleton(source: str) -> str:
    """Mask identifiers and literals while preserving structural tokens."""
    tokens: list[str] = []
    sentinel = "__WFCLLM_TARGET_BLOCK__"
    source = source.replace("<TARGET_BLOCK>", sentinel)
    stream = io.StringIO(source)
    try:
        generated = tokenize.generate_tokens(stream.readline)
        for token in generated:
            token_type = token.type
            token_text = token.string
            if token_type == tokenize.NAME and token_text == sentinel:
                tokens.append("<TARGET_BLOCK>")
            elif token_type == tokenize.NAME and not keyword.iskeyword(token_text):
                tokens.append("<NAME>")
            elif token_type == tokenize.NUMBER:
                tokens.append("<NUMBER>")
            elif token_type == tokenize.STRING:
                tokens.append("<STRING>")
            elif token_type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NL,
                tokenize.NEWLINE,
            }:
                continue
            elif token_type in {tokenize.INDENT, tokenize.DEDENT}:
                continue
            else:
                tokens.append(token_text)
    except (IndentationError, tokenize.TokenError):
        return source.strip()
    return " ".join(tokens).replace("( ", "(").replace(" )", ")").strip()


def infer_semantic_role(
    block_text: str,
    node_type: str,
    parent_node_type: str,
) -> str:
    """Infer a compact semantic role from block metadata and source text."""
    normalized = block_text.strip()
    lowered = normalized.lower()
    node = node_type.lower()
    parent = parent_node_type.lower()

    if "import" in node or lowered.startswith(("import ", "from ")):
        return "import/dependency"
    if "assert" in node or lowered.startswith("assert "):
        return "assertion/invariant check"
    if any(marker in node for marker in ("raise", "except", "try")):
        return "exception/error handling"
    if lowered.startswith(("raise ", "except ", "try:")) or "except" in parent:
        return "exception/error handling"
    if "return" in node or lowered.startswith("return "):
        return "return final value"
    if "if" in node or lowered.startswith(("if ", "elif ")) or "condition" in node:
        return "branch condition / guard"
    if _looks_like_accumulator_update(normalized):
        return "accumulator update"
    if _looks_like_assignment_or_update(normalized):
        return "variable assignment/update"
    if _looks_like_call(normalized):
        return "function call side effect"
    if ("for" in parent or "while" in parent) and _looks_like_update(normalized):
        return "loop body update"
    return "fallback generic statement"


def _looks_like_update(source: str) -> bool:
    return bool(re.search(r"(\+=|-=|\*=|/=|//=|%=|\.\w+\s*\()", source))


def _looks_like_accumulator_update(source: str) -> bool:
    assignment = re.match(r"\s*([A-Za-z_]\w*)\s*(\+=|=)\s*(.+)", source)
    if assignment is None:
        return False
    name = assignment.group(1).lower()
    rhs = assignment.group(3)
    if name in _ACCUMULATOR_NAMES:
        return True
    return bool(re.search(rf"\b{re.escape(name)}\b", rhs))


def _looks_like_assignment_or_update(source: str) -> bool:
    return bool(re.match(r"\s*[A-Za-z_]\w*(?:\[[^\]]+\])?\s*(?:=|\+=|-=|\*=|/=|//=|%=)", source))


def _looks_like_call(source: str) -> bool:
    return bool(re.search(r"(?:^|\.)[A-Za-z_]\w*\s*\(", source))


def _build_codet5_masked_code_anchor(context: CandidateContext) -> str:
    lines = _context_prefix_lines(context)
    lines.append(_indent_like_context(context.context_before, "<extra_id_0>"))
    lines.extend(_nonempty_lines((context.context_after,)))
    return _format_code_lines(lines)


def _build_codet5_valid_skeleton_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, skeleton),
        ]
    )


def _build_codet5_comment_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    metadata = (
        "# wfcllm "
        f"role={_metadata_token(context.node_type)} "
        f"slot={context.block_ordinal} "
        f"ctx={context.context_hash}"
    )
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, metadata),
            _indent_like_context(context.context_before, skeleton),
        ]
    ).strip()


def _build_codet5_comment_minimal_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    metadata = (
        "# wfcllm: "
        f"{_metadata_literal_token(context.node_type)} "
        f"ordinal_{context.block_ordinal}"
    )
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, metadata),
            _indent_like_context(context.context_before, skeleton),
        ]
    ).strip()


def _build_codet5_comment_contextual_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    metadata = (
        "# wfcllm: "
        f"{_metadata_literal_token(context.node_type)} "
        f"ordinal_{context.block_ordinal} "
        f"parent_{_metadata_literal_token(context.parent_node_type)}"
    )
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(context.context_before, metadata),
            _indent_like_context(context.context_before, skeleton),
            *_nonempty_lines((context.context_after,)),
        ]
    ).strip()


def _build_codet5_identifier_anchor(
    context: CandidateContext,
    candidate: CandidateBlock,
) -> str:
    role = _metadata_token(context.node_type)
    context_hash = _metadata_token(context.context_hash)
    skeleton = _skeleton_for_node(context.node_type, candidate.block_text)
    return _format_code_lines(
        [
            *_context_prefix_lines(context),
            _indent_like_context(
                context.context_before,
                f"_wfcllm_slot_{role}_{context.block_ordinal} = None",
            ),
            _indent_like_context(
                context.context_before,
                f"_wfcllm_ctx_{context_hash} = None",
            ),
            _indent_like_context(context.context_before, skeleton),
        ]
    ).strip()


def _skeleton_for_node(node_type: str, block_text: str) -> str:
    node = node_type.lower()
    stripped = block_text.strip()
    if "import_from" in node or "import_from_statement" in node:
        return _import_from_skeleton(stripped)
    if "import" in node and stripped.startswith("from "):
        return _import_from_skeleton(stripped)
    if "return" in node or stripped.startswith("return"):
        return "return None"
    if "expression" in node:
        return "_ = None"
    if any(marker in node for marker in ("if", "for", "while", "try", "with", "class", "function")):
        return "pass"
    return "pass"


def _import_from_skeleton(source: str) -> str:
    match = re.match(r"from\s+(\.+(?:[A-Za-z_][\w.]*)?|[A-Za-z_][\w.]*)\s+import\s+(.+)", source)
    if match is None:
        return "from __future__ import annotations"
    module = match.group(1)
    names = match.group(2).split("#", 1)[0].strip()
    if not names:
        names = "*"
    return f"from {module} import {names}"


def _metadata_token(value: str) -> str:
    token = re.sub(r"_?statement$", "", value.lower())
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", token)
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return "unknown"
    if token[0].isdigit():
        return f"n_{token}"
    return token


def _metadata_literal_token(value: str) -> str:
    token = re.sub(r"[^0-9a-zA-Z_]+", "_", value.lower())
    token = re.sub(r"_+", "_", token).strip("_")
    if not token:
        return "unknown"
    if token[0].isdigit():
        return f"n_{token}"
    return token


def _indent_like_context(context_before: str, text: str) -> str:
    indent = "    "
    for line in reversed(context_before.splitlines()):
        if line.strip():
            indent = re.match(r"\s*", line).group(0)  # type: ignore[union-attr]
            if line.rstrip().endswith(":"):
                indent += "    "
            break
    return "\n".join(f"{indent}{line}" if line else line for line in text.splitlines())


def _context_prefix_lines(context: CandidateContext) -> list[str]:
    context_lines = _nonempty_lines((context.context_before,))
    normalized_context = {_normalize_code_line(line) for line in context_lines}
    lines: list[str] = []
    for signature in context.import_and_helper_signatures:
        signature_lines = _complete_signature_lines(signature)
        if not signature_lines:
            continue
        if any(_normalize_code_line(line) in normalized_context for line in signature_lines):
            continue
        lines.extend(signature_lines)
    function_signature_lines = _complete_signature_lines(context.function_signature)
    if function_signature_lines and not any(
        _normalize_code_line(line) in normalized_context
        for line in function_signature_lines
    ):
        lines.extend(function_signature_lines)
    lines.extend(context_lines)
    return _dedupe_preserve_order(lines)


def _format_code_lines(lines: list[str]) -> str:
    deduped = _dedupe_preserve_order(lines)
    completed: list[str] = []
    for index, line in enumerate(deduped):
        completed.append(line)
        stripped = line.strip()
        if not stripped.endswith(":"):
            continue
        current_indent = len(_line_indent(line))
        next_indent = _next_code_indent(deduped, index + 1)
        if next_indent is None or next_indent <= current_indent:
            completed.append(f"{_line_indent(line)}    pass")
    return "\n".join(completed).strip()


def _next_code_indent(lines: list[str], start: int) -> int | None:
    for line in lines[start:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        return len(_line_indent(line))
    return None


def _complete_signature_lines(signature: str) -> list[str]:
    lines = _nonempty_lines((signature,))
    if not lines:
        return []
    if len(lines) > 1:
        return lines
    line = lines[0]
    stripped = line.strip()
    if stripped.startswith("def ") and stripped.endswith(":"):
        return [line, f"{_line_indent(line)}    pass"]
    if stripped.startswith("class ") and stripped.endswith(":"):
        return [line, f"{_line_indent(line)}    pass"]
    return [line]


def _line_indent(line: str) -> str:
    return re.match(r"\s*", line).group(0)  # type: ignore[union-attr]


def _normalize_code_line(line: str) -> str:
    return line.strip()


def _nonempty_lines(values: tuple[str, ...]) -> list[str]:
    lines: list[str] = []
    for value in values:
        lines.extend(line.rstrip() for line in value.splitlines() if line.strip())
    return lines


def _dedupe_preserve_order(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for line in lines:
        normalized = _normalize_code_line(line)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(line)
    return deduped


def build_anchor_text(
    method: AnchorMethod,
    context: CandidateContext,
    candidate: CandidateBlock | None = None,
    secret_key: str | None = None,
) -> str:
    """Build deterministic diagnostic anchor material without exposing secrets."""
    if method in _CODET5_CANDIDATE_ANCHOR_METHODS and candidate is None:
        raise ValueError("candidate is required for CodeT5 anchors")
    if method == AnchorMethod.CODET5_MASKED_CODE:
        return _build_codet5_masked_code_anchor(context)
    if method == AnchorMethod.CODET5_VALID_SKELETON:
        return _build_codet5_valid_skeleton_anchor(context, candidate)
    if method == AnchorMethod.CODET5_COMMENT_ANCHOR:
        return _build_codet5_comment_anchor(context, candidate)
    if method == AnchorMethod.CODET5_COMMENT_MINIMAL:
        return _build_codet5_comment_minimal_anchor(context, candidate)
    if method == AnchorMethod.CODET5_COMMENT_CONTEXTUAL:
        return _build_codet5_comment_contextual_anchor(context, candidate)
    if method == AnchorMethod.CODET5_IDENTIFIER_ANCHOR:
        return _build_codet5_identifier_anchor(context, candidate)

    if method in {
        AnchorMethod.VANILLA,
        AnchorMethod.RANDOM,
        AnchorMethod.SEQMARK_ORACLE,
        AnchorMethod.CANDIDATE_CENTROID_ORACLE,
        AnchorMethod.CONTEXT_CENTROID_ORACLE,
    }:
        return ""

    parts: list[str] = []
    if method in {
        AnchorMethod.SLOT,
        AnchorMethod.SLOT_CONTEXT,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
        AnchorMethod.PROMPT_AWARE,
    }:
        parts.extend(
            [
                f"dataset={context.dataset}",
                f"task={context.task_id}",
                f"signature={context.function_signature}",
                f"ast_path={'/'.join(context.ast_path)}",
                f"node={context.node_type}",
                f"parent={context.parent_node_type}",
                f"ordinal={context.block_ordinal}",
            ]
        )
    if method in {
        AnchorMethod.CONTEXT,
        AnchorMethod.SLOT_CONTEXT,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
        AnchorMethod.PROMPT_AWARE,
    }:
        parts.extend(
            [
                f"context_hash={context.context_hash}",
                f"context_before={mask_code_skeleton(context.context_before)}",
                f"context_after={mask_code_skeleton(context.context_after)}",
                f"masked_parent={context.masked_parent_context}",
                "imports_helpers=" + "|".join(context.import_and_helper_signatures),
            ]
        )
    if method in {
        AnchorMethod.SKELETON,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
    }:
        if candidate is None:
            raise ValueError("candidate is required for skeleton anchors")
        parts.append(f"skeleton={mask_code_skeleton(candidate.block_text)}")
    if method in {
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
    }:
        if candidate is None:
            raise ValueError("candidate is required for role-aware anchors")
        parts.append(
            "role="
            + infer_semantic_role(
                candidate.block_text,
                context.node_type,
                context.parent_node_type,
            )
        )
    if method == AnchorMethod.PROMPT_AWARE:
        parts.append(f"prompt={context.prompt}")
    return "\n".join(parts)
