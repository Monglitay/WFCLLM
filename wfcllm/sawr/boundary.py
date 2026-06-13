from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Any, Literal

from wfcllm.lang.python.parser import PythonParser, SIMPLE_STATEMENT_TYPES

BoundaryEventKind = Literal[
    "compound_started",
    "simple_candidate",
    "layer_closed",
    "final_flush",
]


@dataclass(frozen=True)
class Candidate:
    text: str
    candidate_type: str
    node_type: str
    position_id: str
    token_start_idx: int
    token_count: int
    parent_node_type: str = "module"
    ordinal: int | None = None
    layer_path: tuple[str, ...] = ()
    start_byte: int = 0
    end_byte: int = 0
    depth: int = 0


@dataclass(frozen=True)
class BoundaryEvent:
    kind: BoundaryEventKind
    node_type: str
    parent_node_type: str
    position_id: str
    layer_path: tuple[str, ...]
    token_start_idx: int
    token_count: int
    text: str
    start_byte: int
    end_byte: int
    depth: int
    checkpoint_key: str | None = None
    candidate: Candidate | None = None
    closed_layer_paths: tuple[tuple[str, ...], ...] = ()
    final_flush: bool = False


@dataclass(frozen=True)
class BoundaryDetectorState:
    generated_text: str
    emitted_keys: tuple[tuple[str, int, int, str], ...]
    token_boundaries: tuple[int, ...]
    active_layer_paths: tuple[tuple[str, ...], ...]
    final_flush_emitted: bool


@dataclass(frozen=True)
class _NodeOffsets:
    text: str
    generated_start: int
    generated_end: int


@dataclass(frozen=True)
class _ControlledBody:
    function_name: str
    body_node: Any

    @property
    def position_id(self) -> str:
        return f"module.{self.function_name}.body"

    @property
    def parent_node_type(self) -> str:
        return "function_definition"


class PromptAwareBoundaryDetector:
    """Detect generated structure and simple-statement boundaries."""

    _SUPPORTED_DATASETS = frozenset({"humaneval", "mbpp"})
    _COMPOUND_TYPES = frozenset({"if_statement", "for_statement", "while_statement"})

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
        self._emitted_keys: set[tuple[str, int, int, str]] = set()
        self._token_boundaries: list[int] = []
        self._active_layer_paths: list[tuple[str, ...]] = []
        self._final_flush_emitted = False

    @property
    def saw_controlled_body(self) -> bool:
        source, generated_base = self._source_for_parse()
        return self._controlled_body(source, generated_base) is not None

    def feed_text(self, token_text: str) -> list[BoundaryEvent]:
        self._generated_text += token_text
        self._token_boundaries.append(self._generated_byte_length())
        if "\n" not in token_text:
            return []
        return self._collect_events(final_flush=False)

    def flush(self) -> list[BoundaryEvent]:
        byte_length = self._generated_byte_length()
        events = self._collect_events(final_flush=True)
        if not self._final_flush_emitted:
            events.append(
                BoundaryEvent(
                    kind="final_flush",
                    node_type="function_body",
                    parent_node_type="function_definition",
                    position_id="final_flush",
                    layer_path=(),
                    token_start_idx=len(self._token_boundaries),
                    token_count=0,
                    text="",
                    start_byte=byte_length,
                    end_byte=byte_length,
                    depth=0,
                    final_flush=True,
                )
            )
            self._final_flush_emitted = True
        return events

    def checkpoint(self) -> BoundaryDetectorState:
        return BoundaryDetectorState(
            generated_text=self._generated_text,
            emitted_keys=tuple(sorted(self._emitted_keys)),
            token_boundaries=tuple(self._token_boundaries),
            active_layer_paths=tuple(self._active_layer_paths),
            final_flush_emitted=self._final_flush_emitted,
        )

    def rollback(self, checkpoint: BoundaryDetectorState) -> None:
        self._generated_text = checkpoint.generated_text
        self._emitted_keys = set(checkpoint.emitted_keys)
        self._token_boundaries = list(checkpoint.token_boundaries)
        self._active_layer_paths = list(checkpoint.active_layer_paths)
        self._final_flush_emitted = checkpoint.final_flush_emitted

    def _collect_events(self, *, final_flush: bool) -> list[BoundaryEvent]:
        source, generated_base = self._source_for_parse()
        body = self._controlled_body(source, generated_base)
        if body is None:
            return []

        source_bytes = source.encode("utf-8")
        root_path = (body.position_id,)
        observed_events = self._walk_layer(
            node=body.body_node,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=final_flush,
            parent_node_type=body.parent_node_type,
            layer_path=root_path,
            depth=0,
        )
        observed_paths = tuple(
            event.layer_path
            for event in observed_events
            if event.kind in {"compound_started", "simple_candidate"}
        )
        events = self._dedupe_events(observed_events)
        events.extend(self._close_missing_layers(observed_paths, final_flush=final_flush))
        return events

    def _source_for_parse(self) -> tuple[str, int]:
        if self._dataset == "humaneval":
            return self._prompt + self._generated_text, len(
                self._prompt.encode("utf-8")
            )
        return self._generated_text, 0

    def _controlled_body(
        self,
        source: str,
        generated_base: int,
    ) -> _ControlledBody | None:
        tree = self._parser.parse(source)
        functions = [
            child
            for child in tree.root_node.children
            if child.type == "function_definition"
        ]
        if self._dataset == "humaneval":
            functions = [
                function
                for function in functions
                if function.start_byte < generated_base
            ]
        if not functions:
            return None
        function_node = functions[-1] if self._dataset == "humaneval" else functions[0]
        name_node = function_node.child_by_field_name("name")
        body_node = function_node.child_by_field_name("body")
        if name_node is None or body_node is None:
            return None
        function_name = name_node.text.decode("utf-8")
        return _ControlledBody(function_name=function_name, body_node=body_node)

    def _walk_layer(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
        parent_node_type: str,
        layer_path: tuple[str, ...],
        depth: int,
    ) -> list[BoundaryEvent]:
        events: list[BoundaryEvent] = []
        children = list(node.children)
        for index, child in enumerate(children):
            if child.type in SIMPLE_STATEMENT_TYPES:
                event = self._simple_event(
                    node=child,
                    source_bytes=source_bytes,
                    generated_base=generated_base,
                    final_flush=final_flush,
                    parent_node_type=parent_node_type,
                    layer_path=layer_path,
                    depth=depth,
                )
                if event is not None:
                    events.append(event)
                continue

            if child.type not in self._COMPOUND_TYPES:
                continue

            start_event = self._compound_start_event(
                node=child,
                source_bytes=source_bytes,
                generated_base=generated_base,
                final_flush=final_flush,
                parent_node_type=parent_node_type,
                layer_path=layer_path,
                depth=depth,
                ordinal=index,
            )
            if start_event is None:
                continue

            events.append(start_event)
            compound_path = start_event.layer_path
            for block in self._compound_blocks(child):
                events.extend(
                    self._walk_layer(
                        node=block,
                        source_bytes=source_bytes,
                        generated_base=generated_base,
                        final_flush=final_flush,
                        parent_node_type=child.type,
                        layer_path=compound_path,
                        depth=depth + 1,
                    )
                )

            if self._compound_can_close(
                node=child,
                siblings=children,
                index=index,
                source_bytes=source_bytes,
                generated_base=generated_base,
                final_flush=final_flush,
            ):
                events.extend(
                    self._layer_close_events(
                        node=child,
                        source_bytes=source_bytes,
                        generated_base=generated_base,
                        final_flush=final_flush,
                        parent_node_type=parent_node_type,
                        layer_path=compound_path,
                        depth=depth,
                    )
                )
        return events

    def _simple_event(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
        parent_node_type: str,
        layer_path: tuple[str, ...],
        depth: int,
    ) -> BoundaryEvent | None:
        if self._node_has_recovery_content(node):
            return None
        offsets = self._node_offsets(
            node=node,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=final_flush,
        )
        if offsets is None:
            return None
        if "\n" in offsets.text:
            return None
        token_start = bisect_right(self._token_boundaries, offsets.generated_start)
        token_end = self._token_end_idx(offsets.generated_end)
        event_key = self._event_key(
            kind="simple_candidate",
            start_byte=offsets.generated_start,
            end_byte=offsets.generated_end,
            layer_path=layer_path,
        )
        candidate = Candidate(
            text=offsets.text,
            candidate_type="simple_statement",
            node_type=node.type,
            position_id=layer_path[0],
            token_start_idx=token_start,
            token_count=max(1, token_end - token_start),
            parent_node_type=parent_node_type,
            ordinal=self._ordinal_for_key(event_key),
            layer_path=layer_path,
            start_byte=offsets.generated_start,
            end_byte=offsets.generated_end,
            depth=depth,
        )
        return BoundaryEvent(
            kind="simple_candidate",
            node_type=node.type,
            parent_node_type=parent_node_type,
            position_id=layer_path[0],
            layer_path=layer_path,
            token_start_idx=candidate.token_start_idx,
            token_count=candidate.token_count,
            text=offsets.text,
            start_byte=offsets.generated_start,
            end_byte=offsets.generated_end,
            depth=depth,
            checkpoint_key=self._checkpoint_key(
                node_type=node.type,
                start_byte=offsets.generated_start,
                end_byte=offsets.generated_end,
                depth=depth,
                ordinal=0,
            ),
            candidate=candidate,
        )

    def _compound_start_event(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
        parent_node_type: str,
        layer_path: tuple[str, ...],
        depth: int,
        ordinal: int,
    ) -> BoundaryEvent | None:
        offsets = self._compound_header_offsets(
            node=node,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=final_flush,
        )
        if offsets is None:
            return None
        checkpoint_key = self._checkpoint_key(
            node_type=node.type,
            start_byte=offsets.generated_start,
            end_byte=offsets.generated_end,
            depth=depth,
            ordinal=ordinal,
        )
        compound_path = (*layer_path, checkpoint_key)
        return BoundaryEvent(
            kind="compound_started",
            node_type=node.type,
            parent_node_type=parent_node_type,
            position_id=layer_path[0],
            layer_path=compound_path,
            token_start_idx=bisect_right(self._token_boundaries, offsets.generated_start),
            token_count=0,
            text=offsets.text,
            start_byte=offsets.generated_start,
            end_byte=offsets.generated_end,
            depth=depth,
            checkpoint_key=checkpoint_key,
        )

    def _layer_close_events(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
        parent_node_type: str,
        layer_path: tuple[str, ...],
        depth: int,
    ) -> list[BoundaryEvent]:
        offsets = self._node_offsets(
            node=node,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=final_flush,
            require_trailing_newline=False,
        )
        end_byte = offsets.generated_end if offsets is not None else self._generated_byte_length()
        close_paths = [
            path
            for path in reversed(self._active_layer_paths)
            if path == layer_path or self._is_descendant_path(path, layer_path)
        ]
        if not close_paths:
            close_paths = [layer_path]
        return [
            BoundaryEvent(
                kind="layer_closed",
                node_type=path[-1].split(":", 1)[0] if path else node.type,
                parent_node_type=parent_node_type,
                position_id=path[0] if path else "",
                layer_path=path,
                token_start_idx=self._token_end_idx(end_byte),
                token_count=0,
                text="",
                start_byte=end_byte,
                end_byte=end_byte,
                depth=max(0, len(path) - 1),
                checkpoint_key=path[-1] if path else None,
                closed_layer_paths=(path,),
                final_flush=final_flush,
            )
            for path in close_paths
        ]

    def _compound_can_close(
        self,
        *,
        node: Any,
        siblings: list[Any],
        index: int,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
    ) -> bool:
        if final_flush:
            return True
        offsets = self._node_offsets(
            node=node,
            source_bytes=source_bytes,
            generated_base=generated_base,
            final_flush=False,
        )
        if offsets is None:
            return False
        return any(sibling.start_byte >= node.end_byte for sibling in siblings[index + 1 :])

    def _compound_blocks(self, node: Any) -> list[Any]:
        blocks: list[Any] = []
        pending = list(node.children)
        while pending:
            child = pending.pop(0)
            if child.type == "block":
                blocks.append(child)
                continue
            pending[0:0] = list(child.children)
        return blocks

    def _node_has_recovery_content(self, node: Any) -> bool:
        pending = [node]
        while pending:
            current = pending.pop()
            if current.type == "ERROR":
                return True
            if bool(getattr(current, "is_missing", False)):
                return True
            if bool(getattr(current, "has_error", False)):
                return True
            pending.extend(current.children)
        return False

    def _node_offsets(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
        require_trailing_newline: bool = True,
    ) -> _NodeOffsets | None:
        if node.start_byte < generated_base:
            return None
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
        elif require_trailing_newline and not final_flush:
            return None

        generated_start = statement_start - generated_base
        generated_end = statement_end - generated_base
        if generated_end <= 0 or generated_start < 0:
            return None
        if generated_end > self._generated_byte_length():
            return None
        text = source_bytes[node.start_byte : node.end_byte].decode("utf-8").strip()
        if not text:
            return None
        return _NodeOffsets(
            text=text,
            generated_start=generated_start,
            generated_end=generated_end,
        )

    def _compound_header_offsets(
        self,
        *,
        node: Any,
        source_bytes: bytes,
        generated_base: int,
        final_flush: bool,
    ) -> _NodeOffsets | None:
        if node.start_byte < generated_base:
            return None
        line_end = source_bytes.find(b"\n", node.start_byte, len(source_bytes))
        if line_end == -1:
            if not final_flush:
                return None
            line_end = node.end_byte
        else:
            line_end += 1
        if line_end > self._generated_byte_length() + generated_base:
            return None
        generated_start = max(
            _line_start_byte(source_bytes, node.start_byte),
            generated_base,
        ) - generated_base
        generated_end = line_end - generated_base
        if generated_end <= 0 or generated_start < 0:
            return None
        text = source_bytes[node.start_byte : min(line_end, node.end_byte)].decode(
            "utf-8"
        ).strip()
        if not text:
            return None
        return _NodeOffsets(
            text=text,
            generated_start=generated_start,
            generated_end=generated_end,
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

    def _dedupe_events(self, events: list[BoundaryEvent]) -> list[BoundaryEvent]:
        deduped: list[BoundaryEvent] = []
        for event in events:
            key = self._event_key(
                kind=event.kind,
                start_byte=event.start_byte,
                end_byte=event.end_byte,
                layer_path=event.layer_path,
            )
            if key in self._emitted_keys:
                continue
            self._emitted_keys.add(key)
            if event.kind == "compound_started":
                if event.layer_path not in self._active_layer_paths:
                    self._active_layer_paths.append(event.layer_path)
            elif event.kind == "layer_closed":
                for closed_path in event.closed_layer_paths:
                    if closed_path in self._active_layer_paths:
                        self._active_layer_paths.remove(closed_path)
            deduped.append(event)
        return deduped

    def _close_missing_layers(
        self,
        observed_paths: tuple[tuple[str, ...], ...],
        *,
        final_flush: bool,
    ) -> list[BoundaryEvent]:
        observed = set(observed_paths)
        if final_flush:
            paths_to_close = list(reversed(self._active_layer_paths))
        else:
            paths_to_close = [
                path for path in reversed(self._active_layer_paths) if path not in observed
            ]

        events: list[BoundaryEvent] = []
        byte_length = self._generated_byte_length()
        for path in paths_to_close:
            key = self._event_key(
                kind="layer_closed",
                start_byte=byte_length,
                end_byte=byte_length,
                layer_path=path,
            )
            if key in self._emitted_keys:
                continue
            self._emitted_keys.add(key)
            if path in self._active_layer_paths:
                self._active_layer_paths.remove(path)
            node_type = path[-1].split(":", 1)[0] if path else "function_body"
            events.append(
                BoundaryEvent(
                    kind="layer_closed",
                    node_type=node_type,
                    parent_node_type="function_definition",
                    position_id=path[0] if path else "",
                    layer_path=path,
                    token_start_idx=len(self._token_boundaries),
                    token_count=0,
                    text="",
                    start_byte=byte_length,
                    end_byte=byte_length,
                    depth=max(0, len(path) - 1),
                    checkpoint_key=path[-1] if path else None,
                    closed_layer_paths=(path,),
                    final_flush=final_flush,
                )
            )
        return events

    def _event_key(
        self,
        *,
        kind: BoundaryEventKind,
        start_byte: int,
        end_byte: int,
        layer_path: tuple[str, ...],
    ) -> tuple[str, int, int, str]:
        return kind, start_byte, end_byte, "/".join(layer_path)

    def _ordinal_for_key(self, key: tuple[str, int, int, str]) -> int:
        if key in self._emitted_keys:
            return len(self._emitted_keys)
        return len(self._emitted_keys)

    def _checkpoint_key(
        self,
        *,
        node_type: str,
        start_byte: int,
        end_byte: int,
        depth: int,
        ordinal: int,
    ) -> str:
        return f"{node_type}:{start_byte}:{end_byte}:{depth}:{ordinal}"

    def _is_descendant_path(
        self,
        path: tuple[str, ...],
        ancestor: tuple[str, ...],
    ) -> bool:
        return len(path) > len(ancestor) and path[: len(ancestor)] == ancestor


def _line_start_byte(source_bytes: bytes, byte_offset: int) -> int:
    return source_bytes.rfind(b"\n", 0, byte_offset) + 1
