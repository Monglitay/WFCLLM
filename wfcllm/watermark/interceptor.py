"""Incremental AST parsing for statement block interception."""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Literal

from wfcllm.common.ast_parser import (
    COMPOUND_STATEMENT_TYPES,
    SIMPLE_STATEMENT_TYPES,
    PythonParser,
)


@dataclass
class InterceptEvent:
    """Emitted when a complete statement block is detected."""

    block_text: str
    block_type: Literal["simple", "compound"]
    node_type: str
    parent_node_type: str | None
    token_start_idx: int
    token_count: int


@dataclass
class InterceptorState:
    """Typed snapshot of interceptor internal state for checkpoint/rollback."""

    accumulated: str
    token_idx: int
    prev_all_keys: set[tuple]
    pending_simple: dict  # key → deep-copied _BlockInfo
    emitted_keys: set[tuple]
    token_boundaries: list[int]


class StatementInterceptor:
    """Detect statement block closures by incremental Tree-sitter parsing."""

    def __init__(self):
        self._parser = PythonParser()
        self._accumulated = ""
        self._token_idx = 0
        # Keys of all blocks seen in the previous parse step
        self._prev_all_keys: set[tuple] = set()
        # Simple blocks awaiting a newline terminator: key -> _BlockInfo
        self._pending_simple: dict[tuple, _BlockInfo] = {}
        # Keys of blocks already emitted as events
        self._emitted_keys: set[tuple] = set()
        # Precise token→byte boundary tracking for accurate token_count in events
        self._token_boundaries: list[int] = [0]  # boundaries[i] = UTF-8 byte offset after i tokens
        # Snapshot taken just before each emit (Fix 2: enables clean rollback)
        self._pre_event_state: dict | None = None

    def feed_token(self, token_text: str) -> InterceptEvent | None:
        """Feed a new token; return event if a new block completed."""
        self._accumulated += token_text
        self._token_idx += 1
        self._token_boundaries.append(len(self._accumulated.encode("utf-8")))

        encoded = self._accumulated.encode("utf-8")
        tree = self._parser.parse(self._accumulated)
        root_has_error = tree.root_node.has_error
        current_blocks = self._extract_blocks(tree.root_node)
        current_keys = {(b.node_type, b.start_byte, b.end_byte) for b in current_blocks}

        # Evict pending blocks that disappeared from the AST (e.g. parse changed)
        for key in list(self._pending_simple):
            if key not in current_keys:
                del self._pending_simple[key]

        # Check pending simple blocks — fire if a newline now follows
        for key, block in list(self._pending_simple.items()):
            if (
                not root_has_error
                and encoded[block.end_byte : block.end_byte + 1] == b"\n"
            ):
                del self._pending_simple[key]
                self._pre_event_state = self._make_snapshot()
                self._emitted_keys.add(key)
                self._prev_all_keys = current_keys
                return self._make_event(block)

        # Scan for newly visible blocks not yet emitted.
        # Process simple blocks BEFORE compound so inner statements fire first.
        new_blocks = [
            b for b in current_blocks
            if (b.node_type, b.start_byte, b.end_byte) not in self._prev_all_keys
            and (b.node_type, b.start_byte, b.end_byte) not in self._emitted_keys
        ]

        # Pass 1: simple blocks
        for block in new_blocks:
            if block.is_compound:
                continue
            key = (block.node_type, block.start_byte, block.end_byte)
            self._pending_simple[key] = block

        # Pass 2: compound blocks
        for block in new_blocks:
            if not block.is_compound:
                continue
            key = (block.node_type, block.start_byte, block.end_byte)
            self._pre_event_state = self._make_snapshot()
            self._emitted_keys.add(key)
            self._prev_all_keys = current_keys
            return self._make_event(block)

        self._prev_all_keys = current_keys
        return None

    def reset(self):
        """Clear all accumulated state."""
        self._accumulated = ""
        self._token_idx = 0
        self._prev_all_keys = set()
        self._pending_simple = {}
        self._emitted_keys = set()
        self._token_boundaries = [0]
        self._pre_event_state = None

    def save_state(self) -> dict:
        """Deprecated: use checkpoint() instead. Kept for backward compatibility."""
        state = self.checkpoint()
        return {
            "accumulated": state.accumulated,
            "token_idx": state.token_idx,
            "prev_all_keys": state.prev_all_keys,
            "pending_simple": state.pending_simple,
            "emitted_keys": state.emitted_keys,
            "token_boundaries": state.token_boundaries,
        }

    def restore_state(self, state: dict) -> None:
        """Deprecated: use rollback() instead. Kept for backward compatibility."""
        self.rollback(InterceptorState(
            accumulated=state["accumulated"],
            token_idx=state["token_idx"],
            prev_all_keys=set(state["prev_all_keys"]),
            pending_simple={k: copy.deepcopy(v) for k, v in state["pending_simple"].items()},
            emitted_keys=set(state["emitted_keys"]),
            token_boundaries=list(state["token_boundaries"]),
        ))

    def checkpoint(self) -> InterceptorState:
        """Save current state as a typed, deep-copied snapshot."""
        return InterceptorState(
            accumulated=self._accumulated,
            token_idx=self._token_idx,
            prev_all_keys=set(self._prev_all_keys),
            pending_simple={k: copy.deepcopy(v) for k, v in self._pending_simple.items()},
            emitted_keys=set(self._emitted_keys),
            token_boundaries=list(self._token_boundaries),
        )

    def rollback(self, state: InterceptorState) -> None:
        """Restore to a previously checkpointed state."""
        self._accumulated = state.accumulated
        self._token_idx = state.token_idx
        self._prev_all_keys = set(state.prev_all_keys)
        self._pending_simple = {k: copy.deepcopy(v) for k, v in state.pending_simple.items()}
        self._emitted_keys = set(state.emitted_keys)
        self._token_boundaries = list(state.token_boundaries)

    def finalize_pending_simple_block(self) -> InterceptEvent | None:
        """Flush the final pending simple block when generation ends at EOF.

        Simple statements normally emit only after a trailing newline. If code
        ends immediately after the final simple block, emit it once here so the
        generator can still verify the last block.
        """
        if not self._pending_simple:
            return None

        key, block = max(
            self._pending_simple.items(),
            key=lambda item: item[1].end_byte,
        )
        self._pre_event_state = self._make_snapshot()
        del self._pending_simple[key]
        self._emitted_keys.add(key)
        return self._make_event(block)

    def _make_snapshot(self) -> dict:
        """Internal snapshot identical in structure to save_state output."""
        return {
            "accumulated": self._accumulated,
            "token_idx": self._token_idx,
            "prev_all_keys": set(self._prev_all_keys),
            "pending_simple": dict(self._pending_simple),
            "emitted_keys": set(self._emitted_keys),
            "token_boundaries": list(self._token_boundaries),
        }

    def get_pre_event_state(self) -> dict:
        """Deprecated: use checkpoint() before feed_token() instead.

        Return snapshot taken just before the most recent event was emitted.
        The snapshot's emitted_keys does NOT contain the emitted block's key,
        so restore_state(get_pre_event_state()) allows the sub-loop to re-detect
        the same statement block without the _emitted_keys filter blocking it.
        """
        warnings.warn(
            "get_pre_event_state() is deprecated, use checkpoint()/rollback() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        assert self._pre_event_state is not None, (
            "get_pre_event_state() called before any event was emitted"
        )
        return self._pre_event_state

    def _make_event(self, block: _BlockInfo) -> InterceptEvent:
        import bisect
        start_tok = bisect.bisect_right(self._token_boundaries, block.start_byte) - 1
        end_tok = bisect.bisect_left(self._token_boundaries, block.end_byte)
        token_count = end_tok - start_tok
        return InterceptEvent(
            block_text=block.text,
            block_type="compound" if block.is_compound else "simple",
            node_type=block.node_type,
            parent_node_type=block.parent_type,
            token_start_idx=start_tok,
            token_count=token_count,
        )

    def _extract_blocks(self, root) -> list[_BlockInfo]:
        """Walk AST and collect all error-free statement blocks."""
        blocks: list[_BlockInfo] = []
        self._walk(root, parent_type=None, blocks=blocks)
        return blocks

    def _walk(self, node, parent_type: str | None, blocks: list[_BlockInfo]):
        if node.type in SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES:
            if not node.has_error:
                encoded = self._accumulated.encode("utf-8")
                text = encoded[node.start_byte : node.end_byte].decode("utf-8")
                blocks.append(
                    _BlockInfo(
                        text=text,
                        node_type=node.type,
                        parent_type=parent_type or "module",
                        start_byte=node.start_byte,
                        end_byte=node.end_byte,
                        is_compound=node.type in COMPOUND_STATEMENT_TYPES,
                    )
                )
        for child in node.children:
            child_parent = (
                node.type
                if node.type in COMPOUND_STATEMENT_TYPES | {"module"}
                else parent_type
            )
            self._walk(child, child_parent, blocks)


@dataclass
class _BlockInfo:
    """Internal representation of a detected block."""

    text: str
    node_type: str
    parent_type: str
    start_byte: int
    end_byte: int
    is_compound: bool
