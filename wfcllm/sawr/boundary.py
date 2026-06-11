from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any

from wfcllm.lang.python.parser import PythonParser, SIMPLE_STATEMENT_TYPES


@dataclass(frozen=True)
class Candidate:
    text: str
    candidate_type: str
    node_type: str
    position_id: str
    token_start_idx: int
    token_count: int


@dataclass(frozen=True)
class BoundaryDetectorState:
    generated_text: str
    emitted_keys: tuple[tuple[int, int, str], ...]
    token_boundaries: tuple[int, ...]


@dataclass(frozen=True)
class _SimpleStatement:
    text: str
    node_type: str
    position_id: str
    generated_start: int
    generated_end: int


@dataclass(frozen=True)
class _ControlledBody:
    function_name: str
    body_node: Any

    @property
    def position_id(self) -> str:
        return f"module.{self.function_name}.body"


class PromptAwareBoundaryDetector:
    """Detect generated simple-statement boundaries inside a prompt-owned body."""

    _SUPPORTED_DATASETS = frozenset({"humaneval", "mbpp"})

    def __init__(self, *, prompt: str, dataset: str) -> None:
        normalized_dataset = dataset.lower()
        if normalized_dataset not in self._SUPPORTED_DATASETS:
            raise ValueError(
                "dataset must be one of: "
                f"{', '.join(sorted(self._SUPPORTED_DATASETS))}"
            )

        self._prompt = prompt
        self._dataset = normalized_dataset
        self._parser = PythonParser()
        self._generated_text = ""
        self._emitted_keys: set[tuple[int, int, str]] = set()
        self._token_boundaries: list[int] = []

    @property
    def saw_controlled_body(self) -> bool:
        source, _ = self._source_for_parse()
        return self._controlled_body(source) is not None

    def feed_text(self, token_text: str) -> list[Candidate]:
        self._generated_text += token_text
        self._token_boundaries.append(self._generated_byte_length())
        if "\n" not in token_text:
            return []
        return self._collect_candidates(final_flush=False)

    def flush(self) -> list[Candidate]:
        return self._collect_candidates(final_flush=True)

    def checkpoint(self) -> BoundaryDetectorState:
        return BoundaryDetectorState(
            generated_text=self._generated_text,
            emitted_keys=tuple(sorted(self._emitted_keys)),
            token_boundaries=tuple(self._token_boundaries),
        )

    def rollback(self, checkpoint: BoundaryDetectorState) -> None:
        self._generated_text = checkpoint.generated_text
        self._emitted_keys = set(checkpoint.emitted_keys)
        self._token_boundaries = list(checkpoint.token_boundaries)

    def _collect_candidates(self, *, final_flush: bool) -> list[Candidate]:
        source, generated_base = self._source_for_parse()
        tree = self._parser.parse(source)
        if tree.root_node.has_error:
            return []

        body = self._controlled_body(source)
        if body is None:
            return []

        source_bytes = source.encode("utf-8")
        candidates: list[Candidate] = []
        for statement in self._simple_statements(
            body=body,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=final_flush,
        ):
            key = (
                statement.generated_start,
                statement.generated_end,
                statement.node_type,
            )
            if key in self._emitted_keys:
                continue
            candidate = self._to_candidate(statement)
            if candidate is None:
                continue
            self._emitted_keys.add(key)
            candidates.append(candidate)
        return candidates

    def _source_for_parse(self) -> tuple[str, int]:
        if self._dataset == "humaneval":
            return self._prompt + self._generated_text, len(
                self._prompt.encode("utf-8")
            )
        return self._generated_text, 0

    def _controlled_body(self, source: str) -> _ControlledBody | None:
        tree = self._parser.parse(source)
        functions = list(_walk_nodes(tree.root_node, node_type="function_definition"))
        if not functions:
            return None
        function_node = functions[-1] if self._dataset == "humaneval" else functions[0]
        name_node = function_node.child_by_field_name("name")
        body_node = function_node.child_by_field_name("body")
        if name_node is None or body_node is None:
            return None
        function_name = name_node.text.decode("utf-8")
        return _ControlledBody(function_name=function_name, body_node=body_node)

    def _simple_statements(
        self,
        *,
        body: _ControlledBody,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
    ) -> list[_SimpleStatement]:
        statements: list[_SimpleStatement] = []
        for node in body.body_node.children:
            if node.type not in SIMPLE_STATEMENT_TYPES:
                continue
            generated_start, generated_end = self._generated_offsets(
                node=node,
                source_bytes=source_bytes,
                generated_base=generated_base,
                final_flush=final_flush,
            )
            if generated_start is None or generated_end is None:
                continue
            text = node.text.decode("utf-8").strip()
            if not text:
                continue
            statements.append(
                _SimpleStatement(
                    text=text,
                    node_type=node.type,
                    position_id=body.position_id,
                    generated_start=generated_start,
                    generated_end=generated_end,
                )
            )
        return statements

    def _generated_offsets(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
    ) -> tuple[int | None, int | None]:
        statement_start = max(
            _line_start_byte(source_bytes, node.start_byte),
            generated_base,
        )
        statement_end = node.end_byte
        has_trailing_newline = statement_end < len(source_bytes) and (
            source_bytes[statement_end : statement_end + 1] == b"\n"
        )
        if has_trailing_newline:
            statement_end += 1
        elif not final_flush:
            return None, None

        generated_start = statement_start - generated_base
        generated_end = statement_end - generated_base
        if generated_end <= 0 or generated_start < 0:
            return None, None
        if generated_end > self._generated_byte_length():
            return None, None
        return generated_start, generated_end

    def _to_candidate(self, statement: _SimpleStatement) -> Candidate | None:
        token_start = bisect_right(self._token_boundaries, statement.generated_start)
        token_end = self._token_end_idx(statement.generated_end)
        return Candidate(
            text=statement.text,
            candidate_type="simple_statement",
            node_type=statement.node_type,
            position_id=statement.position_id,
            token_start_idx=token_start,
            token_count=max(1, token_end - token_start),
        )

    def _token_end_idx(self, generated_end: int) -> int:
        if not self._token_boundaries:
            return 0
        insertion_idx = bisect_left(self._token_boundaries, generated_end)
        if insertion_idx == len(self._token_boundaries):
            return len(self._token_boundaries)
        return insertion_idx + 1

    def _generated_byte_length(self) -> int:
        return len(self._generated_text.encode("utf-8"))


def _walk_nodes(node: Any, *, node_type: str) -> list[Any]:
    matches = [node] if node.type == node_type else []
    for child in node.children:
        matches.extend(_walk_nodes(child, node_type=node_type))
    return matches


def _line_start_byte(source_bytes: bytes, byte_offset: int) -> int:
    return source_bytes.rfind(b"\n", 0, byte_offset) + 1
