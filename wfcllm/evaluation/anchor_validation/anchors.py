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


def build_anchor_text(
    method: AnchorMethod,
    context: CandidateContext,
    candidate: CandidateBlock | None = None,
    secret_key: str | None = None,
) -> str:
    """Build deterministic diagnostic anchor material without exposing secrets."""
    if method in {
        AnchorMethod.VANILLA,
        AnchorMethod.RANDOM,
        AnchorMethod.SEQMARK_ORACLE,
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
