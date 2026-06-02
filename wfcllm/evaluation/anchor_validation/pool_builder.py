"""Build diagnostic candidate pools from explicit or repeated artifacts."""

from __future__ import annotations

from collections import defaultdict
from hashlib import sha256
from typing import Any

from wfcllm.evaluation.anchor_validation.anchors import mask_code_skeleton
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext
from wfcllm.lang.python.parser import StatementBlock, extract_statement_blocks

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements, fallback for minimal envs.
    tqdm = None  # type: ignore[assignment]


def build_candidate_contexts_from_records(
    records: list[dict[str, Any]],
    min_candidates: int = 2,
    max_contexts_per_task: int | None = None,
    show_progress: bool = False,
) -> list[CandidateContext]:
    explicit_contexts = _build_explicit_candidate_contexts(
        records,
        min_candidates,
        show_progress=show_progress,
    )
    if explicit_contexts:
        return explicit_contexts

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        task_id = str(record.get("id", ""))
        if task_id:
            grouped[task_id].append(record)

    contexts: list[CandidateContext] = []
    grouped_items = sorted(grouped.items())
    task_iterable = _progress(
        grouped_items,
        enabled=show_progress,
        desc="Building candidate pool tasks",
        unit="task",
    )
    for task_id, task_records in task_iterable:
        per_context: dict[str, list[CandidateBlock]] = defaultdict(list)
        context_examples: dict[str, CandidateContext] = {}
        ordinal_hashes: dict[int, list[str]] = defaultdict(list)

        for rank, record in enumerate(task_records):
            prompt = str(record.get("prompt", ""))
            generated = str(record.get("generated_code", ""))
            full_code = prompt + generated if prompt else generated
            try:
                blocks = extract_statement_blocks(full_code)
            except Exception:
                continue
            block_by_id = {block.block_id: block for block in blocks}
            simple_blocks = [block for block in blocks if block.block_type == "simple"]
            for ordinal, block in enumerate(simple_blocks):
                context_parts = _context_parts(full_code, block, blocks, block_by_id)
                context_hash = context_parts["context_hash"]
                ordinal_hashes[ordinal].append(context_hash)
                source_hash = sha256(block.source.encode("utf-8")).hexdigest()[:12]
                candidate = CandidateBlock(
                    candidate_id=f"{task_id}:{ordinal}:{rank}:{source_hash}",
                    block_text=block.source,
                    rank=int(record.get("candidate_index", rank)),
                    syntax_valid=True,
                    parse_valid=True,
                    quality={
                        "source_hash": source_hash,
                        "generated_length": len(generated),
                    },
                )
                per_context[context_hash].append(candidate)
                if context_hash not in context_examples:
                    context_examples[context_hash] = CandidateContext(
                        context_id=(
                            f"{record.get('dataset', 'unknown')}:{task_id}:"
                            f"{ordinal}:{context_hash[:12]}"
                        ),
                        dataset=str(record.get("dataset", "unknown")),
                        task_id=task_id,
                        prompt=prompt,
                        function_signature=_function_signature(prompt),
                        ast_path=_ast_path(block, blocks),
                        node_type=block.node_type,
                        parent_node_type=_parent_node_type(block, block_by_id),
                        block_ordinal=ordinal,
                        context_hash=context_hash,
                        temperature=(
                            float(record["temperature"])
                            if record.get("temperature") is not None
                            else None
                        ),
                        candidates=(),
                        context_before=context_parts["context_before"],
                        context_after=context_parts["context_after"],
                        masked_parent_context=context_parts["masked_parent_context"],
                        import_and_helper_signatures=tuple(
                            _import_and_helper_signatures(blocks)
                        ),
                    )

        task_contexts = 0
        for context_hash, candidates in sorted(per_context.items()):
            unique = _dedupe_candidates(candidates)
            if len(unique) < min_candidates:
                continue
            base = context_examples[context_hash]
            contexts.append(
                CandidateContext(
                    context_id=base.context_id,
                    dataset=base.dataset,
                    task_id=base.task_id,
                    prompt=base.prompt,
                    function_signature=base.function_signature,
                    ast_path=base.ast_path,
                    node_type=base.node_type,
                    parent_node_type=base.parent_node_type,
                    block_ordinal=base.block_ordinal,
                    context_hash=base.context_hash,
                    temperature=base.temperature,
                    candidates=tuple(unique),
                    context_before=base.context_before,
                    context_after=base.context_after,
                    masked_parent_context=base.masked_parent_context,
                    import_and_helper_signatures=base.import_and_helper_signatures,
                )
            )
            task_contexts += 1
            if max_contexts_per_task is not None and task_contexts >= max_contexts_per_task:
                break
        _reject_ambiguous_whole_program_grouping(
            task_id=task_id,
            ordinal_hashes=ordinal_hashes,
            per_context=per_context,
            min_candidates=min_candidates,
        )
    return contexts


def _reject_ambiguous_whole_program_grouping(
    task_id: str,
    ordinal_hashes: dict[int, list[str]],
    per_context: dict[str, list[CandidateBlock]],
    min_candidates: int,
) -> None:
    for ordinal, hashes in ordinal_hashes.items():
        if len(hashes) < min_candidates:
            continue
        unique_hashes = set(hashes)
        if len(unique_hashes) <= 1:
            continue
        if any(len(per_context[context_hash]) >= min_candidates for context_hash in unique_hashes):
            continue
        raise ValueError(
            "ambiguous whole-program candidate grouping "
            f"for task {task_id!r} ordinal {ordinal}: "
            "masked non-target contexts do not match"
        )


def _build_explicit_candidate_contexts(
    records: list[dict[str, Any]],
    min_candidates: int,
    show_progress: bool = False,
) -> list[CandidateContext]:
    if not records or not all("candidate_context_id" in record for record in records):
        return []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["candidate_context_id"])].append(record)

    contexts: list[CandidateContext] = []
    grouped_items = sorted(grouped.items())
    context_iterable = _progress(
        grouped_items,
        enabled=show_progress,
        desc="Building explicit candidate contexts",
        unit="context",
    )
    for context_id, rows in context_iterable:
        if len(rows) < min_candidates:
            continue
        first = rows[0]
        required = (
            "block_text",
            "context_hash",
            "node_type",
            "parent_node_type",
            "block_ordinal",
        )
        missing = [field for field in required if field not in first]
        if missing:
            raise ValueError(f"explicit candidate context {context_id} missing fields: {missing}")
        _validate_explicit_context_consistency(context_id, rows)
        candidates = tuple(
            CandidateBlock(
                candidate_id=str(row.get("candidate_id", f"{context_id}:{idx}")),
                block_text=str(row["block_text"]),
                rank=_load_int(
                    row.get("rank", row.get("candidate_index", idx)),
                    f"explicit candidate {context_id}:{idx} rank",
                ),
                syntax_valid=_load_bool(
                    row.get("syntax_valid", True),
                    f"explicit candidate {context_id}:{idx} syntax_valid",
                ),
                parse_valid=_load_bool(
                    row.get("parse_valid", True),
                    f"explicit candidate {context_id}:{idx} parse_valid",
                ),
                quality=_load_quality(
                    row.get("quality", {}),
                    f"explicit candidate {context_id}:{idx} quality",
                ),
            )
            for idx, row in enumerate(rows)
        )
        contexts.append(
            CandidateContext(
                context_id=context_id,
                dataset=str(first.get("dataset", "unknown")),
                task_id=str(first.get("task_id", first.get("id", ""))),
                prompt=str(first.get("prompt", "")),
                function_signature=str(first.get("function_signature", "")),
                ast_path=tuple(str(part) for part in first.get("ast_path", [])),
                node_type=str(first["node_type"]),
                parent_node_type=str(first["parent_node_type"]),
                block_ordinal=int(first["block_ordinal"]),
                context_hash=str(first["context_hash"]),
                temperature=(
                    float(first["temperature"])
                    if first.get("temperature") is not None
                    else None
                ),
                candidates=candidates,
                context_before=str(first.get("context_before", "")),
                context_after=str(first.get("context_after", "")),
                masked_parent_context=str(first.get("masked_parent_context", "")),
                import_and_helper_signatures=tuple(
                    str(part)
                    for part in first.get("import_and_helper_signatures", [])
                ),
            )
        )
    return contexts


def _validate_explicit_context_consistency(
    context_id: str,
    rows: list[dict[str, Any]],
) -> None:
    first = rows[0]
    fields_to_match = (
        "dataset",
        "task_id",
        "prompt",
        "function_signature",
        "ast_path",
        "context_hash",
        "context_before",
        "context_after",
        "masked_parent_context",
        "import_and_helper_signatures",
        "node_type",
        "parent_node_type",
        "block_ordinal",
    )
    for row in rows[1:]:
        for field in fields_to_match:
            if row.get(field) != first.get(field):
                raise ValueError(
                    "conflicting explicit candidate context "
                    f"{context_id}: field {field!r} differs"
                )


def _load_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise ValueError(f"{label} must be an integer")
    return value


def _load_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{label} must be a boolean")
    return value


def _load_quality(value: object, label: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return dict(value)


def _dedupe_candidates(candidates: list[CandidateBlock]) -> list[CandidateBlock]:
    seen: set[str] = set()
    unique: list[CandidateBlock] = []
    for candidate in sorted(candidates, key=lambda item: item.rank):
        if candidate.block_text in seen:
            continue
        seen.add(candidate.block_text)
        unique.append(candidate)
    return unique


def _parent_node_type(
    block: StatementBlock,
    block_by_id: dict[str, StatementBlock],
) -> str:
    if block.parent_id is None:
        return "module"
    return block_by_id[block.parent_id].node_type


def _context_parts(
    full_code: str,
    block: StatementBlock,
    blocks: list[StatementBlock],
    block_by_id: dict[str, StatementBlock],
) -> dict[str, str]:
    lines = full_code.splitlines(keepends=True)
    start = max(block.start_line - 1, 0)
    end = max(block.end_line, start + 1)
    context_before = "".join(lines[:start])
    context_after = "".join(lines[end:])
    indent = _line_indent(block.source)
    masked_full_context = context_before + f"{indent}<TARGET_BLOCK>\n" + context_after

    parent = block_by_id.get(block.parent_id) if block.parent_id is not None else None
    if parent is not None:
        masked_parent_source = parent.source.replace(block.source, "<TARGET_BLOCK>", 1)
    else:
        masked_parent_source = masked_full_context

    context_fingerprint = "\n".join(
        [
            str(block.start_line),
            block.node_type,
            _parent_node_type(block, block_by_id),
            mask_code_skeleton(masked_full_context),
            mask_code_skeleton(masked_parent_source),
            "\n".join(_import_and_helper_signatures(blocks)),
        ]
    )
    return {
        "context_hash": sha256(context_fingerprint.encode("utf-8")).hexdigest(),
        "context_before": context_before,
        "context_after": context_after,
        "masked_parent_context": mask_code_skeleton(masked_parent_source),
    }


def _import_and_helper_signatures(blocks: list[StatementBlock]) -> list[str]:
    signatures: list[str] = []
    for block in blocks:
        if block.depth != 0:
            continue
        if block.node_type in {"import_statement", "import_from_statement"}:
            signatures.append(block.source.strip())
        elif block.node_type in {"function_definition", "class_definition"}:
            first_line = block.source.splitlines()[0].strip()
            signatures.append(first_line)
    return signatures


def _function_signature(prompt: str) -> str:
    for line in reversed(prompt.splitlines()):
        stripped = line.strip()
        if stripped.startswith("def "):
            return stripped
    return ""


def _ast_path(block: StatementBlock, blocks: list[StatementBlock]) -> tuple[str, ...]:
    by_id = {item.block_id: item for item in blocks}
    path = [block.node_type]
    parent_id = block.parent_id
    while parent_id is not None:
        parent = by_id[parent_id]
        path.append(parent.node_type)
        parent_id = parent.parent_id
    return tuple(reversed(path))


def _line_indent(source: str) -> str:
    first_line = source.splitlines()[0] if source.splitlines() else ""
    return first_line[: len(first_line) - len(first_line.lstrip())]


def _progress(
    values,
    *,
    enabled: bool,
    desc: str,
    unit: str,
):
    if not enabled or tqdm is None:
        return values
    return tqdm(values, total=len(values), desc=desc, unit=unit, dynamic_ncols=True)
