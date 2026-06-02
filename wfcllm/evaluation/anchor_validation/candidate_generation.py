"""Optional offline generation of explicit anchor-validation candidate rows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from wfcllm.evaluation.anchor_validation.pool_builder import (
    _ast_path,
    _context_parts,
    _function_signature,
    _import_and_helper_signatures,
    _parent_node_type,
)
from wfcllm.lang.python.parser import extract_statement_blocks


Sampler = Callable[[str, float, int], str]


@dataclass(frozen=True)
class GenerationContextSource:
    dataset: str
    task_id: str
    prompt: str
    source_code: str


@dataclass(frozen=True)
class BlockGenerationContext:
    candidate_context_id: str
    dataset: str
    task_id: str
    prompt: str
    source_code: str
    block_text: str
    context_hash: str
    context_before: str
    context_after: str
    masked_parent_context: str
    function_signature: str
    ast_path: tuple[str, ...]
    node_type: str
    parent_node_type: str
    block_ordinal: int
    import_and_helper_signatures: tuple[str, ...]


def extract_generation_contexts(
    source: GenerationContextSource,
    max_contexts: int | None = None,
) -> list[BlockGenerationContext]:
    blocks = extract_statement_blocks(source.source_code)
    block_by_id = {block.block_id: block for block in blocks}
    simple_blocks = [block for block in blocks if block.block_type == "simple"]
    contexts: list[BlockGenerationContext] = []
    for ordinal, block in enumerate(simple_blocks):
        parts = _context_parts(source.source_code, block, blocks, block_by_id)
        context_id = f"{source.dataset}:{source.task_id}:{ordinal}:{parts['context_hash'][:12]}"
        contexts.append(
            BlockGenerationContext(
                candidate_context_id=context_id,
                dataset=source.dataset,
                task_id=source.task_id,
                prompt=source.prompt,
                source_code=source.source_code,
                block_text=block.source,
                context_hash=parts["context_hash"],
                context_before=parts["context_before"],
                context_after=parts["context_after"],
                masked_parent_context=parts["masked_parent_context"],
                function_signature=_function_signature(source.prompt),
                ast_path=_ast_path(block, blocks),
                node_type=block.node_type,
                parent_node_type=_parent_node_type(block, block_by_id),
                block_ordinal=ordinal,
                import_and_helper_signatures=tuple(_import_and_helper_signatures(blocks)),
            )
        )
        if max_contexts is not None and len(contexts) >= max_contexts:
            break
    return contexts


def build_block_completion_prompt(
    context: BlockGenerationContext,
    secret_key: str | None = None,
) -> str:
    indent = _infer_target_indent(context)
    return (
        "Complete only the Python statement block that replaces <TARGET_BLOCK>. "
        "Return only the replacement block, without surrounding code.\n\n"
        f"{context.context_before}{indent}<TARGET_BLOCK>\n{context.context_after}"
    )


def generate_candidate_rows(
    sources: tuple[GenerationContextSource, ...],
    sampler: Sampler,
    temperatures: tuple[float, ...] = (0.2, 0.4, 0.7),
    candidates_per_temperature: int = 16,
    max_contexts_per_source: int | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for source in sources:
        contexts = extract_generation_contexts(
            source,
            max_contexts=max_contexts_per_source,
        )
        for context in contexts:
            completion_prompt = build_block_completion_prompt(context)
            for temperature in temperatures:
                for sample_index in range(candidates_per_temperature):
                    raw_candidate = sampler(
                        completion_prompt,
                        temperature,
                        sample_index,
                    ).strip("\n")
                    candidate = indent_candidate_block(context, raw_candidate)
                    full_code = replace_target_block(context, candidate)
                    syntax_valid, parse_valid = _candidate_is_parseable(full_code)
                    if not syntax_valid or not parse_valid:
                        continue
                    rows.append(
                        {
                            "candidate_context_id": context.candidate_context_id,
                            "candidate_id": (
                                f"{context.candidate_context_id}:"
                                f"{temperature}:{sample_index}"
                            ),
                            "task_id": context.task_id,
                            "dataset": context.dataset,
                            "prompt": context.prompt,
                            "function_signature": context.function_signature,
                            "ast_path": list(context.ast_path),
                            "block_text": candidate,
                            "context_hash": context.context_hash,
                            "context_before": context.context_before,
                            "context_after": context.context_after,
                            "masked_parent_context": context.masked_parent_context,
                            "import_and_helper_signatures": list(
                                context.import_and_helper_signatures
                            ),
                            "node_type": context.node_type,
                            "parent_node_type": context.parent_node_type,
                            "block_ordinal": context.block_ordinal,
                            "rank": sample_index,
                            "temperature": temperature,
                            "syntax_valid": syntax_valid,
                            "parse_valid": parse_valid,
                            "quality": {
                                "syntax_valid": syntax_valid,
                                "parse_valid": parse_valid,
                                "block_length": len(candidate),
                            },
                        }
                    )
    return rows


def replace_target_block(
    context: BlockGenerationContext,
    candidate_block: str,
) -> str:
    return context.context_before + candidate_block.rstrip() + "\n" + context.context_after


def indent_candidate_block(
    context: BlockGenerationContext,
    candidate_block: str,
) -> str:
    indent = _infer_target_indent(context)
    lines = candidate_block.splitlines() or [candidate_block]
    indented: list[str] = []
    for line in lines:
        if not line.strip():
            indented.append("")
        elif line.startswith(indent):
            indented.append(line)
        else:
            indented.append(indent + line.lstrip())
    return "\n".join(indented)


def build_hf_sampler(model, tokenizer, max_new_tokens: int = 64) -> Sampler:
    def sample(prompt: str, temperature: float, sample_index: int) -> str:
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].to(model.device)
        output_ids = model.generate(
            input_ids=input_ids,
            do_sample=True,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            pad_token_id=getattr(tokenizer, "eos_token_id", None),
        )
        new_ids = output_ids[0, input_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        return _first_nonempty_line(text)

    return sample


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return text.strip()


def _candidate_is_parseable(source_code: str) -> tuple[bool, bool]:
    try:
        compile(source_code, "<anchor-candidate>", "exec")
        syntax_valid = True
    except SyntaxError:
        syntax_valid = False
    try:
        extract_statement_blocks(source_code)
        parse_valid = True
    except Exception:
        parse_valid = False
    return syntax_valid, parse_valid


def _target_indent(block_text: str) -> str:
    first_line = block_text.splitlines()[0] if block_text.splitlines() else ""
    indent = first_line[: len(first_line) - len(first_line.lstrip())]
    return indent


def _infer_target_indent(context: BlockGenerationContext) -> str:
    indent = _target_indent(context.block_text)
    if indent:
        return indent
    for source in (context.context_after, reversed(context.context_before.splitlines())):
        lines = source if not isinstance(source, str) else source.splitlines()
        for line in lines:
            if line.strip():
                inferred = line[: len(line) - len(line.lstrip())]
                if inferred:
                    return inferred
    return ""
