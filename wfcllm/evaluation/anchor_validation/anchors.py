"""Anchor text helpers for offline validation diagnostics."""

from __future__ import annotations

import io
import keyword
import tokenize

from wfcllm.evaluation.anchor_validation.schema import (
    AnchorMethod,
    CandidateBlock,
    CandidateContext,
)


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
    except tokenize.TokenError:
        return source.strip()
    return " ".join(tokens).replace("( ", "(").replace(" )", ")").strip()


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
    }:
        if candidate is None:
            raise ValueError("candidate is required for skeleton anchors")
        parts.append(f"skeleton={mask_code_skeleton(candidate.block_text)}")
    if method == AnchorMethod.PROMPT_AWARE:
        parts.append(f"prompt={context.prompt}")
    return "\n".join(parts)
