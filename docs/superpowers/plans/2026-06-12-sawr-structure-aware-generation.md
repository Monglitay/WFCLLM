# SAWR Structure-Aware Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make isolated SAWR generation-time embedding structure-aware for Python function-body code while preserving detector-clean final artifacts.

**Architecture:** Keep all behavior inside `wfcllm/sawr` and the isolated `scripts/run_sawr_smoke.py` entry point. Replace the current candidate-only flow with prompt-aware boundary events, a stack-based layer state machine, and generator-level rollback snapshots that include detector and state-machine state plus bounded loop-prevention counters.

**Tech Stack:** Python 3.11, dataclasses, pathlib, JSONL, tree-sitter via existing `wfcllm.lang.python.parser`, PyTorch/HuggingFace interfaces behind fake-model tests, pytest with `HF_HUB_OFFLINE=1`.

---

## Scope Check

This plan implements one cohesive subsystem: SAWR structure-aware generation. It does not split into separate plans because boundary events, layer state, rollback snapshots, and generator safety budgets must be changed together for a working behavior slice.

Do not modify:

```text
wfcllm/watermark/
wfcllm/extract/
```

Do not add SAWR detector claims, TPR/FPR claims, calibration outputs, hidden detector data, or audit data to final rows.

Before implementation, read:

```text
CLAUDE.md
AGENTS.md
docs/superpowers/specs/2026-06-12-sawr-structure-aware-generation-design.md
wfcllm/sawr/boundary.py
wfcllm/sawr/state_machine.py
wfcllm/sawr/generator.py
wfcllm/sawr/config.py
wfcllm/sawr/pipeline.py
scripts/run_sawr_smoke.py
tests/sawr/test_boundary.py
tests/sawr/test_state_machine.py
tests/sawr/test_generator.py
tests/sawr/test_config.py
tests/sawr/test_pipeline.py
tests/integration/test_sawr_smoke_cli.py
```

Run before editing:

```bash
git status --short
```

Expected: output may include unrelated local SAWR changes. Do not revert user changes. Work with the current tree.

## File Structure

Modify:

```text
wfcllm/sawr/boundary.py
wfcllm/sawr/state_machine.py
wfcllm/sawr/generator.py
wfcllm/sawr/config.py
wfcllm/sawr/pipeline.py
wfcllm/sawr/__init__.py
scripts/run_sawr_smoke.py
tests/sawr/test_boundary.py
tests/sawr/test_state_machine.py
tests/sawr/test_generator.py
tests/sawr/test_config.py
tests/sawr/test_pipeline.py
tests/integration/test_sawr_smoke_cli.py
```

Responsibilities:

```text
wfcllm/sawr/boundary.py
  Prompt-aware parse state, recursive Python layer discovery, boundary event emission,
  token-to-byte routing, emitted key tracking, and rollback restoration.

wfcllm/sawr/state_machine.py
  Pure layer-stack logic, direct statement windows, child-hit propagation, retry/fallback
  decisions, attempt hashes, retired checkpoint keys, state snapshots, and audit events.

wfcllm/sawr/generator.py
  Local generation loop, absolute sampled-token budget, global rollback budget,
  full rollback snapshots, stop-sequence-safe event processing, and fake-model-testable
  orchestration between boundary detector and state machine.

wfcllm/sawr/config.py
  Adds bounded generation controls while preserving existing CLI names.

wfcllm/sawr/pipeline.py
  Keeps final rows unchanged and allows the new audit event names only in audit rows.

wfcllm/sawr/__init__.py
  Exports new public SAWR dataclasses used by tests and downstream smoke tooling.

scripts/run_sawr_smoke.py
  Adds CLI flags for bounded generation controls and passes them through config.
```

## Shared Types To Implement

Use these names consistently across tasks.

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Literal, TypeVar

CheckpointT = TypeVar("CheckpointT")

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
```

```python
@dataclass(frozen=True)
class LayerFrame(Generic[CheckpointT]):
    layer_id: tuple[str, ...]
    node_type: str
    parent_node_type: str
    position_id: str
    checkpoint_before_layer: CheckpointT | None
    checkpoint_key: str
    window: tuple[Candidate, ...] = ()
    has_direct_hit: bool = False
    has_child_hit: bool = False
    retry_count: int = 0
    attempt_hashes: frozenset[str] = frozenset()
    closed_reason: str | None = None


@dataclass(frozen=True)
class StateMachineSnapshot(Generic[CheckpointT]):
    layer_stack: tuple[LayerFrame[CheckpointT], ...]
    accepted_hit_count: int
    closed_without_hit_count: int
    fallback_count: int
    candidate_count: int
    retry_count: int
    audit_events: tuple[AuditEvent, ...]
```

The concrete implementation may keep `LayerFrame` mutable internally for simpler stack updates, but `checkpoint()` must return immutable copies with these fields.

## Task 1: Add Bounded Generation Config And CLI Flags

**Files:**
- Modify: `wfcllm/sawr/config.py`
- Modify: `scripts/run_sawr_smoke.py`
- Modify: `tests/sawr/test_config.py`
- Modify: `tests/integration/test_sawr_smoke_cli.py`

- [ ] **Step 1: Write failing config tests**

Append these tests to `tests/sawr/test_config.py`:

```python
def test_pipeline_config_derives_absolute_sampled_token_budget(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        retry_budget=3,
    )

    assert config.global_rollback_budget == 3
    assert config.max_total_sampled_tokens == 200


def test_pipeline_config_accepts_explicit_bounded_generation_controls(tmp_path):
    model_path = tmp_path / "local-model"
    model_path.mkdir()
    generation = SawrGenerationConfig(model_path=str(model_path), max_new_tokens=40)

    config = SawrPipelineConfig(
        dataset="humaneval",
        dataset_path=str(tmp_path / "datasets"),
        output_dir=str(tmp_path / "outputs"),
        generation=generation,
        retry_budget=3,
        global_rollback_budget=7,
        max_total_sampled_tokens=123,
    )

    assert config.global_rollback_budget == 7
    assert config.max_total_sampled_tokens == 123
```

Extend the invalid-values parametrization in `tests/sawr/test_config.py`:

```python
        ({"global_rollback_budget": -1}, "global_rollback_budget must be non-negative"),
        ({"max_total_sampled_tokens": 0}, "max_total_sampled_tokens must be positive"),
```

- [ ] **Step 2: Run config tests and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_config.py -v
```

Expected: FAIL with `TypeError: SawrPipelineConfig.__init__() got an unexpected keyword argument 'global_rollback_budget'` or an assertion showing the fields are missing.

- [ ] **Step 3: Implement config fields**

In `wfcllm/sawr/config.py`, extend `SawrPipelineConfig` with these fields immediately after `retry_budget`:

```python
    global_rollback_budget: int | None = None
    max_total_sampled_tokens: int | None = None
```

Add this validation to `SawrPipelineConfig.__post_init__` after `retry_budget` validation:

```python
        if self.global_rollback_budget is None:
            object.__setattr__(self, "global_rollback_budget", self.retry_budget)
        elif self.global_rollback_budget < 0:
            raise ValueError("global_rollback_budget must be non-negative")
        if self.max_total_sampled_tokens is None:
            derived_budget = self.generation.max_new_tokens * max(2, self.retry_budget + 2)
            object.__setattr__(self, "max_total_sampled_tokens", derived_budget)
        elif self.max_total_sampled_tokens <= 0:
            raise ValueError("max_total_sampled_tokens must be positive")
```

- [ ] **Step 4: Write failing CLI pass-through assertions**

In `tests/integration/test_sawr_smoke_cli.py`, add these CLI arguments to `test_run_sawr_smoke_cli_builds_pipeline_config`:

```python
                "--global-rollback-budget",
                "5",
                "--max-total-sampled-tokens",
                "321",
```

Add these assertions near the existing `config.retry_budget` assertion:

```python
    assert config.global_rollback_budget == 5
    assert config.max_total_sampled_tokens == 321
```

- [ ] **Step 5: Run CLI test and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_sawr_smoke_cli.py::test_run_sawr_smoke_cli_builds_pipeline_config -v
```

Expected: FAIL with argparse rejecting `--global-rollback-budget`.

- [ ] **Step 6: Implement CLI flags**

In `scripts/run_sawr_smoke.py`, add parser arguments immediately after `--retry-budget`:

```python
    parser.add_argument("--global-rollback-budget", type=int, default=None)
    parser.add_argument("--max-total-sampled-tokens", type=int, default=None)
```

Pass them into `SawrPipelineConfig`:

```python
            global_rollback_budget=args.global_rollback_budget,
            max_total_sampled_tokens=args.max_total_sampled_tokens,
```

- [ ] **Step 7: Run focused tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_config.py tests/integration/test_sawr_smoke_cli.py::test_run_sawr_smoke_cli_builds_pipeline_config -v
```

Expected: PASS.

- [ ] **Step 8: Commit**

Run:

```bash
git add wfcllm/sawr/config.py scripts/run_sawr_smoke.py tests/sawr/test_config.py tests/integration/test_sawr_smoke_cli.py
git commit -m "feat: add sawr bounded generation config"
```

Expected: commit succeeds.

## Task 2: Replace Candidate-Only Boundary Detection With Structure Events

**Files:**
- Modify: `wfcllm/sawr/boundary.py`
- Modify: `wfcllm/sawr/__init__.py`
- Modify: `tests/sawr/test_boundary.py`

- [ ] **Step 1: Write failing recursive boundary tests**

Append these tests to `tests/sawr/test_boundary.py`:

```python
def _feed_all(detector: PromptAwareBoundaryDetector, text: str):
    events = []
    for ch in text:
        events.extend(detector.feed_text(ch))
    return events


def test_detector_emits_compound_and_nested_simple_events_for_if_else():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")

    events = _feed_all(
        detector,
        "    if x:\n"
        "        y = 1\n"
        "    else:\n"
        "        y = 2\n"
        "    return y\n",
    )

    assert [event.kind for event in events] == [
        "compound_started",
        "simple_candidate",
        "simple_candidate",
        "layer_closed",
        "simple_candidate",
    ]
    assert events[0].node_type == "if_statement"
    assert events[0].parent_node_type == "function_definition"
    assert events[0].text.strip().startswith("if x:")
    assert events[1].candidate is not None
    assert events[1].candidate.text == "y = 1"
    assert events[1].candidate.layer_path[-1].startswith("if_statement:")
    assert events[2].candidate is not None
    assert events[2].candidate.text == "y = 2"
    assert events[2].candidate.layer_path == events[1].candidate.layer_path
    assert events[3].closed_layer_paths == (events[1].layer_path,)
    assert events[4].candidate is not None
    assert events[4].candidate.text == "return y"
    assert events[4].candidate.layer_path == ("module.f.body",)
```

```python
def test_detector_emits_nested_for_and_while_layers():
    detector = PromptAwareBoundaryDetector(prompt="def f(items):\n", dataset="humaneval")

    events = _feed_all(
        detector,
        "    total = 0\n"
        "    for item in items:\n"
        "        while item > 0:\n"
        "            total += item\n"
        "            item -= 1\n"
        "    return total\n",
    )

    compound_types = [event.node_type for event in events if event.kind == "compound_started"]
    candidate_texts = [
        event.candidate.text
        for event in events
        if event.kind == "simple_candidate" and event.candidate is not None
    ]

    assert compound_types == ["for_statement", "while_statement"]
    assert candidate_texts == ["total = 0", "total += item", "item -= 1", "return total"]
```

```python
def test_detector_checkpoint_restores_layer_event_keys():
    detector = PromptAwareBoundaryDetector(prompt="def f(x):\n", dataset="humaneval")
    checkpoint = detector.checkpoint()

    _feed_all(detector, "    if x:\n        return 1\n")
    detector.rollback(checkpoint)
    events = _feed_all(detector, "    if x:\n        return 2\n")

    assert [event.kind for event in events] == ["compound_started", "simple_candidate"]
    assert events[1].candidate is not None
    assert events[1].candidate.text == "return 2"
```

- [ ] **Step 2: Run boundary tests and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_boundary.py -v
```

Expected: FAIL because `Candidate` objects do not have a `kind` attribute and nested statements are not emitted.

- [ ] **Step 3: Add boundary event dataclasses**

In `wfcllm/sawr/boundary.py`, add the shared `BoundaryEventKind` and `BoundaryEvent` dataclasses from the "Shared Types To Implement" section. Extend `Candidate` with the new defaulted metadata fields:

```python
    layer_path: tuple[str, ...] = ()
    start_byte: int = 0
    end_byte: int = 0
    depth: int = 0
```

Extend `BoundaryDetectorState` to preserve event emission state:

```python
@dataclass(frozen=True)
class BoundaryDetectorState:
    generated_text: str
    emitted_keys: tuple[tuple[str, int, int, str], ...]
    token_boundaries: tuple[int, ...]
    active_layer_paths: tuple[tuple[str, ...], ...]
```

- [ ] **Step 4: Implement recursive event collection**

Replace `_simple_statements()` and `_collect_candidates()` with event-based helpers that follow this exact routing:

```python
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
```

Implement `_walk_layer()` with these rules:

```python
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
    for child in node.children:
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
        if child.type in {"if_statement", "for_statement", "while_statement"}:
            compound_event = self._compound_start_event(
                node=child,
                source_bytes=source_bytes,
                generated_base=generated_base,
                parent_node_type=parent_node_type,
                parent_layer_path=layer_path,
                depth=depth + 1,
            )
            if compound_event is None:
                continue
            events.append(compound_event)
            compound_path = compound_event.layer_path
            for grandchild in child.children:
                if grandchild.type == "block":
                    events.extend(
                        self._walk_layer(
                            node=grandchild,
                            source_bytes=source_bytes,
                            generated_base=generated_base,
                            final_flush=final_flush,
                            parent_node_type=child.type,
                            layer_path=compound_path,
                            depth=depth + 1,
                        )
                    )
            close_event = self._layer_close_event(
                node=child,
                source_bytes=source_bytes,
                generated_base=generated_base,
                layer_path=compound_path,
                parent_node_type=parent_node_type,
                final_flush=final_flush,
                depth=depth + 1,
            )
            if close_event is not None:
                events.append(close_event)
    return events
```

Implement if/elif/else as one rollback unit by deriving one `compound_path` from the outer `if_statement` node and walking every `block` child under that same path. Do not create a separate layer for `else_clause`.

Implement `_close_missing_layers()` so detector rollback can restore active-layer tracking and generated siblings close older compounds:

```python
def _close_missing_layers(
    self,
    observed_paths: tuple[tuple[str, ...], ...],
    *,
    final_flush: bool,
) -> list[BoundaryEvent]:
    observed = set(observed_paths)
    closing = [
        path
        for path in reversed(self._active_layer_paths)
        if path not in observed
    ]
    if final_flush:
        closing = list(reversed(self._active_layer_paths))
    events: list[BoundaryEvent] = []
    for path in closing:
        events.append(
            BoundaryEvent(
                kind="layer_closed",
                node_type=path[-1].split(":", 1)[0] if path else "function_body",
                parent_node_type="function_definition",
                position_id=".".join(path),
                layer_path=path,
                token_start_idx=len(self._token_boundaries),
                token_count=0,
                text="",
                start_byte=self._generated_byte_length(),
                end_byte=self._generated_byte_length(),
                depth=len(path),
                closed_layer_paths=(path,),
                final_flush=final_flush,
            )
        )
        self._active_layer_paths.remove(path)
    return events
```

- [ ] **Step 5: Preserve the public feed and flush names**

Change these methods in `PromptAwareBoundaryDetector`:

```python
def feed_text(self, token_text: str) -> list[BoundaryEvent]:
    self._generated_text += token_text
    self._token_boundaries.append(self._generated_byte_length())
    if "\n" not in token_text:
        return []
    return self._collect_events(final_flush=False)


def flush(self) -> list[BoundaryEvent]:
    events = self._collect_events(final_flush=True)
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
            start_byte=self._generated_byte_length(),
            end_byte=self._generated_byte_length(),
            depth=0,
            final_flush=True,
        )
    )
    return events
```

- [ ] **Step 6: Export boundary event types**

In `wfcllm/sawr/__init__.py`, import and add these names to `__all__`:

```python
BoundaryEvent,
BoundaryEventKind,
```

- [ ] **Step 7: Update old simple-statement tests to inspect `event.candidate`**

For each existing assertion in `tests/sawr/test_boundary.py` that compares direct return values to `Candidate`, filter simple events:

```python
simple_events = [event for event in events if event.kind == "simple_candidate"]
assert simple_events[0].candidate == Candidate(
    text="y = x + 1",
    candidate_type="simple_statement",
    node_type="expression_statement",
    position_id="module.add_one.body",
    token_start_idx=0,
    token_count=len("    y = x + 1\n"),
    parent_node_type="function_definition",
    ordinal=0,
    layer_path=("module.add_one.body",),
    start_byte=0,
    end_byte=len("    y = x + 1\n"),
    depth=0,
)
```

- [ ] **Step 8: Run boundary tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_boundary.py -v
```

Expected: PASS.

- [ ] **Step 9: Commit**

Run:

```bash
git add wfcllm/sawr/boundary.py wfcllm/sawr/__init__.py tests/sawr/test_boundary.py
git commit -m "feat: emit sawr structure boundary events"
```

Expected: commit succeeds.

## Task 3: Replace The Single Group State Machine With Layer Frames

**Files:**
- Modify: `wfcllm/sawr/state_machine.py`
- Modify: `wfcllm/sawr/__init__.py`
- Modify: `tests/sawr/test_state_machine.py`

- [ ] **Step 1: Write failing layer success tests**

Append these helpers and tests to `tests/sawr/test_state_machine.py`:

```python
def _compound_event(
    path: tuple[str, ...],
    checkpoint_key: str,
    node_type: str = "if_statement",
) -> BoundaryEvent:
    return BoundaryEvent(
        kind="compound_started",
        node_type=node_type,
        parent_node_type="function_definition",
        position_id="module.foo.body",
        layer_path=path,
        token_start_idx=0,
        token_count=1,
        text="if x:",
        start_byte=0,
        end_byte=5,
        depth=len(path),
        checkpoint_key=checkpoint_key,
    )


def _close_event(path: tuple[str, ...], final_flush: bool = False) -> BoundaryEvent:
    return BoundaryEvent(
        kind="layer_closed",
        node_type="if_statement",
        parent_node_type="function_definition",
        position_id="module.foo.body",
        layer_path=path,
        token_start_idx=0,
        token_count=0,
        text="",
        start_byte=10,
        end_byte=10,
        depth=len(path),
        closed_layer_paths=(path,),
        final_flush=final_flush,
    )


def _simple_event(text: str, path: tuple[str, ...]) -> BoundaryEvent:
    candidate = _candidate(text, position_id="module.foo.body")
    candidate = Candidate(
        text=candidate.text,
        candidate_type=candidate.candidate_type,
        node_type=candidate.node_type,
        position_id=candidate.position_id,
        token_start_idx=candidate.token_start_idx,
        token_count=candidate.token_count,
        parent_node_type=candidate.parent_node_type,
        ordinal=candidate.ordinal,
        layer_path=path,
        start_byte=0,
        end_byte=len(text),
        depth=len(path),
    )
    return BoundaryEvent(
        kind="simple_candidate",
        node_type=candidate.node_type,
        parent_node_type=candidate.parent_node_type,
        position_id=candidate.position_id,
        layer_path=path,
        token_start_idx=0,
        token_count=1,
        text=text,
        start_byte=0,
        end_byte=len(text),
        depth=len(path),
        candidate=candidate,
    )
```

```python
def test_child_layer_hit_bubbles_to_parent_on_close():
    rule = SequenceRule([True])
    state_machine = _state_machine(rule)
    parent_path = ("module.foo.body",)
    child_path = ("module.foo.body", "if_statement:0")

    state_machine.open_root_layer(parent_path, "function_body", "function_definition")
    state_machine.observe_event(_compound_event(child_path, "if0"), FakeCheckpoint("if0"))
    decision = state_machine.observe_event(_simple_event("return 1", child_path), FakeCheckpoint("stmt"))
    close_decision = state_machine.observe_event(_close_event(child_path), FakeCheckpoint("after"))
    final_decision = state_machine.flush()

    assert decision.action == "accept"
    assert close_decision.action == "close"
    assert final_decision.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert state_machine.closed_without_hit_count == 0
    assert "layer_closed_with_child_hit" in _event_names(state_machine)
```

```python
def test_if_statement_succeeds_when_any_branch_hits():
    rule = SequenceRule([True, False])
    state_machine = _state_machine(rule)
    path = ("module.foo.body", "if_statement:0")

    state_machine.open_root_layer(("module.foo.body",), "function_body", "function_definition")
    state_machine.observe_event(_compound_event(path, "if0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("x = 1", path), FakeCheckpoint("then"))
    state_machine.observe_event(_simple_event("x = 2", path), FakeCheckpoint("else"))
    decision = state_machine.observe_event(_close_event(path), FakeCheckpoint("after"))

    assert decision.action == "close"
    assert state_machine.accepted_hit_count == 1
    assert state_machine.fallback_count == 0
```

```python
def test_layer_close_without_hit_requests_rollback():
    rule = SequenceRule([False])
    state_machine = _state_machine(rule, max_group_statements=2, retry_budget=1)
    path = ("module.foo.body", "if_statement:0")

    state_machine.open_root_layer(("module.foo.body",), "function_body", "function_definition")
    state_machine.observe_event(_compound_event(path, "if0"), FakeCheckpoint("if0"))
    state_machine.observe_event(_simple_event("x = 1", path), FakeCheckpoint("then"))
    decision = state_machine.observe_event(_close_event(path), FakeCheckpoint("after"))

    assert decision.action == "rollback"
    assert decision.rollback_checkpoint == FakeCheckpoint("if0")
    assert state_machine.retry_count == 1
```

- [ ] **Step 2: Run state machine tests and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_state_machine.py -v
```

Expected: FAIL because `BoundaryEvent`, `open_root_layer`, and `observe_event` are not supported.

- [ ] **Step 3: Add layer state dataclasses and snapshot API**

In `wfcllm/sawr/state_machine.py`, import `BoundaryEvent`, keep `Candidate`, and add these dataclasses:

```python
@dataclass
class LayerFrame(Generic[CheckpointT]):
    layer_id: tuple[str, ...]
    node_type: str
    parent_node_type: str
    position_id: str
    checkpoint_before_layer: CheckpointT | None
    checkpoint_key: str
    window: list[Candidate]
    has_direct_hit: bool = False
    has_child_hit: bool = False
    retry_count: int = 0
    attempt_hashes: set[str] = field(default_factory=set)
    closed_reason: str | None = None


@dataclass(frozen=True)
class StateMachineSnapshot(Generic[CheckpointT]):
    layer_stack: tuple[LayerFrame[CheckpointT], ...]
    accepted_hit_count: int
    closed_without_hit_count: int
    fallback_count: int
    candidate_count: int
    retry_count: int
    audit_events: tuple[AuditEvent, ...]
```

Add imports:

```python
from copy import deepcopy
from dataclasses import dataclass, field
```

Add methods:

```python
def checkpoint(self) -> StateMachineSnapshot[CheckpointT]:
    return StateMachineSnapshot(
        layer_stack=tuple(deepcopy(self._layer_stack)),
        accepted_hit_count=self.accepted_hit_count,
        closed_without_hit_count=self.closed_without_hit_count,
        fallback_count=self.fallback_count,
        candidate_count=self.candidate_count,
        retry_count=self._retry_count,
        audit_events=tuple(self._audit_events),
    )


def rollback(self, snapshot: StateMachineSnapshot[CheckpointT]) -> None:
    self._layer_stack = list(deepcopy(snapshot.layer_stack))
    self.accepted_hit_count = snapshot.accepted_hit_count
    self.closed_without_hit_count = snapshot.closed_without_hit_count
    self.fallback_count = snapshot.fallback_count
    self.candidate_count = snapshot.candidate_count
    self._retry_count = snapshot.retry_count
    self._audit_events = list(snapshot.audit_events)
```

- [ ] **Step 4: Implement root and event routing**

In `SawrStateMachine.__init__`, replace `_current_group`, `_group_start_checkpoint`, and `_current_retry_count` as the primary state with:

```python
        self._layer_stack: list[LayerFrame[CheckpointT]] = []
        self._retired_checkpoint_keys: set[str] = set()
```

Keep `current_group_size` by returning the current layer window size:

```python
    @property
    def current_group_size(self) -> int:
        if not self._layer_stack:
            return 0
        return len(self._layer_stack[-1].window)
```

Add:

```python
def open_root_layer(
    self,
    layer_id: tuple[str, ...],
    node_type: str,
    parent_node_type: str,
) -> None:
    if self._layer_stack:
        return
    self._layer_stack.append(
        LayerFrame(
            layer_id=layer_id,
            node_type=node_type,
            parent_node_type=parent_node_type,
            position_id=".".join(layer_id),
            checkpoint_before_layer=None,
            checkpoint_key="root",
            window=[],
        )
    )


def observe_event(
    self,
    event: BoundaryEvent,
    checkpoint: CheckpointT | None,
) -> StateMachineDecision[CheckpointT]:
    if event.kind == "compound_started":
        return self._open_layer(event, checkpoint)
    if event.kind == "simple_candidate":
        if event.candidate is None:
            raise ValueError("simple_candidate event must carry candidate")
        return self._observe_direct_candidate(event.candidate, checkpoint)
    if event.kind == "layer_closed":
        return self._close_layer_paths(event.closed_layer_paths, final_flush=event.final_flush)
    if event.kind == "final_flush":
        return self.flush()
    raise ValueError(f"unsupported boundary event kind: {event.kind}")
```

- [ ] **Step 5: Implement direct-window semantics**

Implement `_observe_direct_candidate()` so it:

1. Opens the root layer if the stack is empty using `candidate.layer_path`.
2. Appends the candidate only to the matching top layer.
3. Builds `RuleRequest` from that layer's window.
4. Marks `has_direct_hit` and clears the window on hit.
5. Keeps a miss below `max_group_statements`.
6. Records a missed full window but keeps a sliding contiguous suffix by dropping the oldest candidate before the next append.

Use this code skeleton with exact event names:

```python
def _observe_direct_candidate(
    self,
    candidate: Candidate,
    checkpoint: CheckpointT | None,
) -> StateMachineDecision[CheckpointT]:
    self._ensure_layer_for_path(candidate.layer_path, candidate.parent_node_type)
    frame = self._layer_stack[-1]
    frame.window.append(candidate)
    self.candidate_count += 1
    self._audit_events.append(self._audit_event(
        event="simple_candidate_observed",
        final_flush=False,
        decision=None,
        reason=None,
        rule_name=None,
        candidate=candidate,
        group_statement_count=len(frame.window),
    ))
    request = self._build_rule_request(frame, final_flush=False)
    decision = self._rule.evaluate(request)
    if decision.hit:
        frame.has_direct_hit = True
        self.accepted_hit_count += 1
        self._audit_events.append(self._audit_event(
            event="accepted_generation_time_window",
            final_flush=False,
            decision="hit",
            reason=decision.reason,
            rule_name=decision.rule_name,
            candidate=candidate,
            group_statement_count=len(request.candidates),
        ))
        frame.window.clear()
        return StateMachineDecision(action="accept")
    self._audit_events.append(self._audit_event(
        event="layer_window_rule_miss",
        final_flush=False,
        decision="miss",
        reason=decision.reason,
        rule_name=decision.rule_name,
        candidate=candidate,
        group_statement_count=len(request.candidates),
    ))
    if len(frame.window) >= self._max_group_statements:
        frame.window = frame.window[-(self._max_group_statements - 1):]
    return StateMachineDecision(action="continue")
```

- [ ] **Step 6: Implement close behavior**

Implement `_close_one_layer()` with this behavior:

```python
def _close_one_layer(
    self,
    frame: LayerFrame[CheckpointT],
    *,
    final_flush: bool,
) -> StateMachineDecision[CheckpointT]:
    flush_decision = self._flush_frame_window(frame, final_flush=final_flush)
    if flush_decision.action == "accept":
        frame.has_direct_hit = True
    if frame.has_direct_hit or frame.has_child_hit:
        if self._layer_stack:
            parent = self._layer_stack[-1]
            parent.has_child_hit = True
        self._audit_events.append(AuditEvent(
            sample_id=self._sample_id,
            position_id=frame.position_id,
            event="layer_closed_with_child_hit" if frame.has_child_hit else "layer_closed_with_direct_hit",
            candidate_type=None,
            group_statement_count=0,
            final_flush=final_flush,
            rule_name=None,
            decision="hit",
            reason=frame.closed_reason,
            candidate_hash=None,
        ))
        return StateMachineDecision(action="close")
    if frame.checkpoint_before_layer is not None and frame.retry_count < self._retry_budget and frame.checkpoint_key not in self._retired_checkpoint_keys:
        frame.retry_count += 1
        self._retry_count += 1
        self._audit_events.append(AuditEvent(
            sample_id=self._sample_id,
            position_id=frame.position_id,
            event="layer_retry_requested",
            candidate_type=None,
            group_statement_count=0,
            final_flush=final_flush,
            rule_name=None,
            decision="miss",
            reason="layer_closed_without_direct_or_child_hit",
            candidate_hash=None,
        ))
        return StateMachineDecision(action="rollback", rollback_checkpoint=frame.checkpoint_before_layer)
    self.closed_without_hit_count += 1
    self.fallback_count += 1
    self._retired_checkpoint_keys.add(frame.checkpoint_key)
    self._audit_events.append(AuditEvent(
        sample_id=self._sample_id,
        position_id=frame.position_id,
        event="layer_fallback_committed_without_hit",
        candidate_type=None,
        group_statement_count=0,
        final_flush=final_flush,
        rule_name=None,
        decision="miss",
        reason="retry_budget_exhausted_or_root_layer",
        candidate_hash=None,
    ))
    return StateMachineDecision(action="fallback")
```

Root layer close must not roll back past the prompt. If the root closes without hit, increment `closed_without_hit_count`, emit `closed_without_hit`, and return `close`.

- [ ] **Step 7: Preserve old wrappers during migration**

Keep `observe_candidate()` and `observe_final_candidate()` as compatibility wrappers used by any remaining old tests:

```python
def observe_candidate(
    self,
    candidate: Candidate,
    group_start_checkpoint: CheckpointT,
) -> StateMachineDecision[CheckpointT]:
    event = BoundaryEvent(
        kind="simple_candidate",
        node_type=candidate.node_type,
        parent_node_type=candidate.parent_node_type,
        position_id=candidate.position_id,
        layer_path=candidate.layer_path or (candidate.position_id,),
        token_start_idx=candidate.token_start_idx,
        token_count=candidate.token_count,
        text=candidate.text,
        start_byte=candidate.start_byte,
        end_byte=candidate.end_byte,
        depth=candidate.depth,
        candidate=candidate,
    )
    return self.observe_event(event, group_start_checkpoint)
```

- [ ] **Step 8: Export state snapshot names**

In `wfcllm/sawr/__init__.py`, export:

```python
LayerFrame,
StateMachineSnapshot,
```

- [ ] **Step 9: Run state machine tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_state_machine.py -v
```

Expected: PASS.

- [ ] **Step 10: Commit**

Run:

```bash
git add wfcllm/sawr/state_machine.py wfcllm/sawr/__init__.py tests/sawr/test_state_machine.py
git commit -m "feat: add sawr layer state machine"
```

Expected: commit succeeds.

## Task 4: Integrate Events, Full Snapshots, And Loop Budgets In The Generator

**Files:**
- Modify: `wfcllm/sawr/generator.py`
- Modify: `wfcllm/sawr/pipeline.py`
- Modify: `tests/sawr/test_generator.py`
- Modify: `tests/sawr/test_pipeline.py`

- [ ] **Step 1: Write failing rollback snapshot tests**

Append to `tests/sawr/test_generator.py`:

```python
def test_generator_retries_nested_if_with_new_early_closed_structure(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer([
        "",
        "    if x:\n",
        "        y = 1\n",
        "    else:\n",
        "        y = 2\n",
        "    if x:\n",
        "        return 3\n",
        "",
    ])
    model = FakeModel([1, 2, 3, 4, 5, 6, 7])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=7, eos_token_id=7),
        rule=SequenceRule([False, False, True]),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo(x):\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=1,
        global_rollback_budget=2,
        max_total_sampled_tokens=20,
    )

    assert result.final_code == "def foo(x):\n    if x:\n        return 3\n"
    assert result.accepted_hit_count == 1
    assert "layer_retry_requested" in [event.event for event in result.audit_events]
```

```python
def test_generator_absolute_sampled_token_count_does_not_decrease_after_rollback(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(["", "    return 1\n", "    return 2\n", "    return 3\n"])
    model = FakeModel([1, 2, 3])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=3),
        rule=AlwaysMissRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=3,
        global_rollback_budget=3,
        max_total_sampled_tokens=2,
    )

    assert result.final_code.startswith("def foo():\n")
    assert result.audit_events[-1].event == "absolute_sampled_token_budget_exhausted"
```

```python
def test_generator_repeated_failed_attempt_consumes_budget_without_unbounded_loop(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(["", "    return 1\n"])
    model = FakeModel([1, 1, 1, 1, 1, 1])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=6),
        rule=AlwaysMissRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=5,
        global_rollback_budget=5,
        max_total_sampled_tokens=6,
    )

    assert result.final_code.startswith("def foo():\n")
    assert "layer_fallback_committed_without_hit" in [event.event for event in result.audit_events]
```

- [ ] **Step 2: Run generator tests and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_generator.py -v
```

Expected: FAIL because `generate()` does not accept `global_rollback_budget` or `max_total_sampled_tokens`, and it still handles candidate lists.

- [ ] **Step 3: Extend result and checkpoint structures**

In `wfcllm/sawr/generator.py`, import `StateMachineSnapshot` and change `SawrCheckpoint`:

```python
@dataclass(frozen=True)
class SawrCheckpoint:
    generated_ids: list[int]
    generated_text: str
    kv_snapshot: Any
    boundary_state: BoundaryDetectorState | None
    state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None
    next_logits: torch.Tensor | None
```

Because `SawrCheckpoint` self-references in a generic annotation, use postponed annotations already enabled by `from __future__ import annotations`.

Change `SawrModelContext.checkpoint()` to accept state-machine state:

```python
def checkpoint(
    self,
    state_machine_state: StateMachineSnapshot[SawrCheckpoint] | None = None,
) -> SawrCheckpoint:
    if self.past_kv is None:
        raise ValueError("cannot checkpoint before prefill")
    boundary_state = self.boundary.checkpoint() if self.boundary is not None else None
    return SawrCheckpoint(
        generated_ids=list(self.generated_ids),
        generated_text=self.generated_text,
        kv_snapshot=self._cache_mgr.snapshot(self.past_kv),
        boundary_state=boundary_state,
        state_machine_state=state_machine_state,
        next_logits=self._next_logits.detach().clone() if self._next_logits is not None else None,
    )
```

- [ ] **Step 4: Restore state-machine state on rollback**

Change `SawrModelContext.rollback()` to return the stored state-machine snapshot:

```python
def rollback(
    self,
    checkpoint: SawrCheckpoint,
) -> StateMachineSnapshot[SawrCheckpoint] | None:
    if self.past_kv is None:
        raise ValueError("cannot rollback before prefill")
    self.past_kv = self._cache_mgr.rollback(self.past_kv, checkpoint.kv_snapshot)
    self.generated_ids = list(checkpoint.generated_ids)
    self.generated_text = checkpoint.generated_text
    if self.boundary is not None and checkpoint.boundary_state is not None:
        self.boundary.rollback(checkpoint.boundary_state)
    self._next_logits = checkpoint.next_logits.detach().clone() if checkpoint.next_logits is not None else None
    self._step_history = self._step_history[: len(self.generated_ids)]
    boundary_count = len(checkpoint.boundary_state.token_boundaries) if checkpoint.boundary_state is not None else 0
    self._boundary_checkpoints = self._boundary_checkpoints[:boundary_count]
    return checkpoint.state_machine_state
```

- [ ] **Step 5: Store boundary checkpoints with state-machine snapshots**

Add this method to `SawrModelContext`:

```python
def record_boundary_checkpoint(
    self,
    state_machine: SawrStateMachine[SawrCheckpoint],
) -> None:
    checkpoint = self.checkpoint(state_machine.checkpoint())
    self._boundary_checkpoints.append(checkpoint)
```

Change `forward_and_sample()` so it increments an absolute counter in the caller rather than using `len(generated_ids)` as the only progress measure. Keep appending `_step_history` for rollback compatibility.

- [ ] **Step 6: Change generator signature and event loop**

Change `SawrGenerator.generate()` signature:

```python
def generate(
    self,
    sample_id: str,
    prompt: str,
    dataset: str,
    max_group_statements: int,
    retry_budget: int,
    global_rollback_budget: int,
    max_total_sampled_tokens: int,
) -> SawrGenerateResult:
```

Initialize:

```python
absolute_sampled_tokens = 0
global_rollback_count = 0
```

Before each sample:

```python
if absolute_sampled_tokens >= max_total_sampled_tokens:
    state_machine.record_budget_exhausted("absolute_sampled_token_budget_exhausted")
    break
```

After `context.forward_and_sample()`:

```python
absolute_sampled_tokens += 1
```

Process events instead of candidates:

```python
events = context.boundary.feed_text(step.token_text)
rolled_back = self._handle_events(events, context, state_machine)
if rolled_back:
    global_rollback_count += 1
    if global_rollback_count >= global_rollback_budget:
        state_machine.record_budget_exhausted("global_rollback_budget_exhausted")
        break
    continue
```

After generation, flush events and route them:

```python
final_events = context.boundary.flush()
self._handle_events(final_events, context, state_machine, allow_rollback=False)
```

- [ ] **Step 7: Implement `_handle_events()`**

Replace `_handle_candidates()` with:

```python
def _handle_events(
    self,
    events: list[BoundaryEvent],
    context: SawrModelContext,
    state_machine: SawrStateMachine[SawrCheckpoint],
    *,
    allow_rollback: bool = True,
) -> bool:
    for event in events:
        checkpoint = None
        if event.kind == "compound_started":
            checkpoint = context.checkpoint(state_machine.checkpoint())
        elif event.kind == "simple_candidate" and event.candidate is not None:
            checkpoint = context.checkpoint_for_token_start(event.candidate.token_start_idx)
        decision = state_machine.observe_event(event, checkpoint)
        if decision.action == "rollback" and decision.rollback_checkpoint is not None:
            if not allow_rollback:
                continue
            snapshot = context.rollback(decision.rollback_checkpoint)
            if snapshot is not None:
                state_machine.rollback(snapshot)
            return True
    return False
```

- [ ] **Step 8: Protect stop-sequence final-code consistency**

Before processing events for a newly sampled token, check whether adding `step.token_text` introduced a stop sequence:

```python
if _contains_stop_sequence(context.generated_text, self.config.stop_sequences):
    break
```

Place this check before routing boundary events from that token. This implements the preferred design option: stop before processing boundary events beyond the retained final-code boundary.

- [ ] **Step 9: Update pipeline generator call**

In `wfcllm/sawr/pipeline.py`, pass the new fields:

```python
                            global_rollback_budget=int(self._config.global_rollback_budget),
                            max_total_sampled_tokens=int(self._config.max_total_sampled_tokens),
```

- [ ] **Step 10: Run generator and pipeline tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_generator.py tests/sawr/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 11: Commit**

Run:

```bash
git add wfcllm/sawr/generator.py wfcllm/sawr/pipeline.py tests/sawr/test_generator.py tests/sawr/test_pipeline.py
git commit -m "feat: bound sawr structure-aware rollback"
```

Expected: commit succeeds.

## Task 5: Update Audit Event Allowlist And Final Artifact Tests

**Files:**
- Modify: `wfcllm/sawr/pipeline.py`
- Modify: `tests/sawr/test_pipeline.py`

- [ ] **Step 1: Write failing audit allowlist test**

Append to `tests/sawr/test_pipeline.py`:

```python
def test_pipeline_allows_structure_aware_audit_events(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    event_names = [
        "compound_layer_started",
        "simple_candidate_observed",
        "layer_window_rule_miss",
        "accepted_generation_time_window",
        "layer_retry_requested",
        "retry_layer_early_closed_after_hit",
        "layer_disappeared_without_hit",
        "layer_closed_with_child_hit",
        "layer_closed_with_direct_hit",
        "layer_fallback_committed_without_hit",
        "global_rollback_budget_exhausted",
        "absolute_sampled_token_budget_exhausted",
    ]
    generator = FakeGenerator()
    generator.generate = lambda **kwargs: SawrGenerateResult(
        final_code="def foo():\n    return 1\n",
        accepted_hit_count=1,
        closed_without_hit_count=0,
        fallback_count=0,
        candidate_count=1,
        audit_events=[
            AuditEvent(
                sample_id="HumanEval/0",
                position_id="module.foo.body",
                event=name,
                candidate_type="simple_statement",
                group_statement_count=1,
                final_flush=False,
                rule_name="hash",
                decision="hit",
                reason="test",
                candidate_hash="abc",
            )
            for name in event_names
        ],
    )
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    audit_path = Path(str(final_path).replace("_final_", "_audit_"))
    rows = _read_jsonl(audit_path)
    assert [row["event"] for row in rows] == event_names
    assert all(row["audit_only"] is True for row in rows)
    assert all(row["detector_input_allowed"] is False for row in rows)
```

- [ ] **Step 2: Run pipeline test and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_pipeline.py::test_pipeline_allows_structure_aware_audit_events -v
```

Expected: FAIL with `unsupported SAWR audit event`.

- [ ] **Step 3: Update audit event allowlist**

In `wfcllm/sawr/pipeline.py`, replace `ALLOWED_AUDIT_EVENTS` with:

```python
ALLOWED_AUDIT_EVENTS = {
    "candidate_observed",
    "group_rule_miss",
    "accepted_generation_time_group",
    "rollback_requested",
    "fallback_committed_without_hit",
    "closed_without_hit",
    "sample_failed",
    "compound_layer_started",
    "simple_candidate_observed",
    "layer_window_rule_miss",
    "accepted_generation_time_window",
    "layer_retry_requested",
    "retry_layer_early_closed_after_hit",
    "layer_disappeared_without_hit",
    "layer_closed_with_child_hit",
    "layer_closed_with_direct_hit",
    "layer_fallback_committed_without_hit",
    "global_rollback_budget_exhausted",
    "absolute_sampled_token_budget_exhausted",
}
```

- [ ] **Step 4: Strengthen final-row clean contract**

Append to `tests/sawr/test_pipeline.py`:

```python
def test_pipeline_final_rows_remain_detector_clean_with_structure_audit(tmp_path: Path) -> None:
    prompts = [{"id": "HumanEval/0", "prompt": "def foo():\n"}]
    generator = FakeGenerator()
    pipeline = SawrPipeline(generator=generator, config=_config(tmp_path))

    with patch("wfcllm.sawr.pipeline.load_prompts", return_value=prompts):
        final_path = Path(pipeline.run())

    row = _read_jsonl(final_path)[0]
    assert set(row) == {
        "artifact_type",
        "schema_version",
        "id",
        "dataset",
        "prompt",
        "final_code",
        "scientific_claims_enabled",
    }
    assert row["artifact_type"] == "sawr_final_code"
    assert row["scientific_claims_enabled"] is False
    assert not (FORBIDDEN_FINAL_FIELDS & set(row))
```

- [ ] **Step 5: Run pipeline tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_pipeline.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add wfcllm/sawr/pipeline.py tests/sawr/test_pipeline.py
git commit -m "feat: allow sawr structure audit events"
```

Expected: commit succeeds.

## Task 6: Update Integration Tests And Public Exports

**Files:**
- Modify: `wfcllm/sawr/__init__.py`
- Modify: `tests/integration/test_sawr_smoke_cli.py`
- Modify: `tests/sawr/test_state_machine.py`

- [ ] **Step 1: Write failing export test**

Extend `test_state_machine_types_are_exported_from_package_root` in `tests/sawr/test_state_machine.py`:

```python
    from wfcllm.sawr import BoundaryEvent, BoundaryEventKind, LayerFrame, StateMachineSnapshot

    assert BoundaryEvent.__name__ == "BoundaryEvent"
    assert BoundaryEventKind is not None
    assert LayerFrame.__name__ == "LayerFrame"
    assert StateMachineSnapshot.__name__ == "StateMachineSnapshot"
```

- [ ] **Step 2: Run export test and verify failure**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/test_state_machine.py::test_state_machine_types_are_exported_from_package_root -v
```

Expected: FAIL if any new type is not exported.

- [ ] **Step 3: Export all new public types**

In `wfcllm/sawr/__init__.py`, import:

```python
from wfcllm.sawr.boundary import BoundaryEvent, BoundaryEventKind, Candidate
from wfcllm.sawr.state_machine import (
    AuditEvent,
    CheckpointT,
    DecisionAction,
    LayerFrame,
    SawrStateMachine,
    StateMachineDecision,
    StateMachineSnapshot,
)
```

Add these strings to `__all__`:

```python
    "BoundaryEvent",
    "BoundaryEventKind",
    "LayerFrame",
    "StateMachineSnapshot",
```

- [ ] **Step 4: Update integration assertions for derived defaults**

Append to `tests/integration/test_sawr_smoke_cli.py`:

```python
def test_run_sawr_smoke_cli_derives_bounded_generation_defaults(tmp_path: Path) -> None:
    model_path = tmp_path / "model"
    model_path.mkdir()
    pipeline = MagicMock()
    pipeline.run.return_value = str(
        tmp_path / "sawr" / "humaneval_sawr_final_20260612_010101.jsonl"
    )

    with (
        patch("scripts.run_sawr_smoke.SawrGenerator"),
        patch("scripts.run_sawr_smoke.SawrPipeline", return_value=pipeline) as pipeline_cls,
    ):
        rc = sawr_cli.main(
            [
                "--dataset",
                "humaneval",
                "--dataset-path",
                "data/datasets",
                "--model-path",
                str(model_path),
                "--max-new-tokens",
                "50",
                "--retry-budget",
                "2",
            ]
        )

    assert rc == 0
    config = pipeline_cls.call_args.kwargs["config"]
    assert config.global_rollback_budget == 2
    assert config.max_total_sampled_tokens == 200
```

- [ ] **Step 5: Run integration tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_sawr_smoke_cli.py -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

Run:

```bash
git add wfcllm/sawr/__init__.py tests/sawr/test_state_machine.py tests/integration/test_sawr_smoke_cli.py
git commit -m "feat: export sawr structure-aware APIs"
```

Expected: commit succeeds.

## Task 7: Full Local Verification

**Files:**
- Read-only validation across `wfcllm/sawr`, `scripts/run_sawr_smoke.py`, and tests.

- [ ] **Step 1: Run SAWR unit and integration tests**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/sawr/ tests/integration/test_sawr_smoke_cli.py -v
```

Expected: PASS.

- [ ] **Step 2: Run compile smoke**

Run:

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tools
```

Expected: exits 0 with compile output and no syntax errors.

- [ ] **Step 3: Inspect final diff for forbidden scope**

Run:

```bash
git diff --stat
git diff --name-only
```

Expected changed paths are limited to:

```text
wfcllm/sawr/boundary.py
wfcllm/sawr/state_machine.py
wfcllm/sawr/generator.py
wfcllm/sawr/config.py
wfcllm/sawr/pipeline.py
wfcllm/sawr/__init__.py
scripts/run_sawr_smoke.py
tests/sawr/test_boundary.py
tests/sawr/test_state_machine.py
tests/sawr/test_generator.py
tests/sawr/test_config.py
tests/sawr/test_pipeline.py
tests/integration/test_sawr_smoke_cli.py
```

If `wfcllm/watermark/` or `wfcllm/extract/` appears, stop and inspect why before committing.

- [ ] **Step 4: Commit verification-only fixes if needed**

If verification required small fixes, run:

```bash
git add wfcllm/sawr scripts/run_sawr_smoke.py tests/sawr tests/integration/test_sawr_smoke_cli.py
git commit -m "test: verify sawr structure-aware generation"
```

Expected: commit succeeds only if there are verification fixes. If there are no changes, skip this commit.

## Task 8: Remote Experiment Gate And Code Transfer

**Files:**
- No repository file changes are required for this task.
- Commands run from local repo root: `/home/monglitay/PycharmProjects/WFCLLM`

This task is intentionally after local verification. Do not ask for server information before local tests and compile smoke pass.

- [ ] **Step 1: Ask the user for remote execution details**

Ask for these exact fields in one message before running any remote command:

```text
1. SSH user and host, for example user@example.com
2. SSH port
3. Remote WFCLLM repository path
4. Remote conda environment name, default WFCLLM if unchanged
5. Remote model path for --model-path
6. Remote dataset path for --dataset-path
7. GPU/device preference, for example cuda, cuda:0, or cpu
8. Whether to sync by rsync/scp or by git push/pull
```

- [ ] **Step 2: Sync local code to remote after user answers**

If the user chooses `rsync`, run from `/home/monglitay/PycharmProjects/WFCLLM` after substituting the user-provided values into shell variables:

```bash
export SAWR_REMOTE=user@example.com
export SAWR_REMOTE_PORT=22
export SAWR_REMOTE_PATH=/absolute/remote/WFCLLM
rsync -az --delete \
  --exclude .git \
  --exclude data/models \
  --exclude data/datasets \
  --exclude data/results \
  --exclude data/checkpoints \
  --exclude data/run_state.json \
  /home/monglitay/PycharmProjects/WFCLLM/ \
  -e "ssh -p ${SAWR_REMOTE_PORT}" \
  "${SAWR_REMOTE}:${SAWR_REMOTE_PATH}/"
```

Expected: rsync exits 0 and prints transferred changed files. If network or sandboxing blocks this command, request escalation for the exact transfer command.

If the user chooses `git`, run:

```bash
git status --short
git log --oneline -5
git push
```

Then on the remote server run:

```bash
cd /absolute/remote/WFCLLM
git pull --ff-only
```

Expected: remote repository contains the local SAWR changes.

- [ ] **Step 3: Run remote smoke command only after code transfer succeeds**

Use the user-provided paths and device:

```bash
cd /absolute/remote/WFCLLM
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python scripts/run_sawr_smoke.py \
  --dataset humaneval \
  --dataset-path /absolute/remote/datasets \
  --model-path /absolute/remote/model \
  --output-dir data/sawr \
  --sample-limit 5 \
  --max-new-tokens 256 \
  --device cuda \
  --max-group-statements 2 \
  --retry-budget 1 \
  --global-rollback-budget 2
```

Expected: stderr includes `[完成] SAWR smoke final rows saved to ...`.

- [ ] **Step 4: Copy remote artifacts back if requested**

If the user wants local copies, run after substituting the same remote values:

```bash
mkdir -p /home/monglitay/PycharmProjects/WFCLLM/data/sawr_remote
rsync -az \
  -e "ssh -p ${SAWR_REMOTE_PORT}" \
  "${SAWR_REMOTE}:${SAWR_REMOTE_PATH}/data/sawr/" \
  /home/monglitay/PycharmProjects/WFCLLM/data/sawr_remote/
```

Expected: local `data/sawr_remote/` contains the remote final and audit JSONL files.

## Self-Review

Spec coverage:

```text
Structure-aware nested layers: Task 2 and Task 3.
If/elif/else one rollback unit: Task 2 and Task 3.
Layer success from direct or child hit: Task 3.
Retry uses newly generated AST structure: Task 4.
Early close after hit succeeds: Task 3 and Task 4.
Disappeared target compounds route sibling candidates to parent: Task 3 close behavior and Task 4 event routing.
Rollback snapshots restore detector and state machine: Task 3 snapshot API and Task 4 checkpoint integration.
Loop prevention: Task 1 config controls, Task 3 attempt/retired keys, Task 4 global and absolute budgets.
Final rows remain clean: Task 5.
Audit events remain audit-only: Task 5.
Remote experiment asks for server info at final run time and transfers local code first: Task 8.
```

Placeholder scan result:

```text
No red-flag phrases remain in executable task steps. User-provided remote values are explicitly requested at the remote experiment gate before remote commands run.
```

Type consistency:

```text
BoundaryEvent, Candidate, LayerFrame, StateMachineSnapshot, SawrCheckpoint, and SawrGenerateResult names are used consistently across tests, implementation steps, and exports.
```

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-12-sawr-structure-aware-generation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - Dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
