from __future__ import annotations

from dataclasses import replace
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import wfcllm.generation.generator as generator_module
from wfcllm.generation.boundary import PromptAwareBoundaryDetector
from wfcllm.generation.generator import GeneratedToken, SawrGenerator, SawrModelContext
from wfcllm.generation.state_machine import StateMachineSnapshot
from wfcllm.method.config import SawrGenerationConfig
from wfcllm.semantic.rules import HashEmbeddingRule


class FakeTokenizer:
    eos_token_id = 127

    def __init__(self) -> None:
        self.decode_calls: list[tuple[int, bool]] = []
        self.text_by_id: dict[int, str] = {
            21: "x = 2",
            22: "\n",
            23: "y = 3\n",
            31: "total += item",
            32: "\n",
            41: "total = total + item\n",
            51: "    prior = 1\n    value",
            52: " = 2\n",
            61: "λ = 1\n",
            70: "",
            71: "",
            72: "    x = 1\n",
            73: "    y = 2\n",
        }

    def encode(self, text: str, return_tensors: str | None = None) -> Any:
        ids = [80 + (byte % 16) for byte in text.encode("utf-8")]
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def decode(self, ids: list[int], *, skip_special_tokens: bool = True) -> str:
        self.decode_calls.extend((token_id, skip_special_tokens) for token_id in ids)
        return "".join(
            ""
            if skip_special_tokens and token_id == self.eos_token_id
            else self.text_by_id.get(token_id, f"<{token_id}>")
            for token_id in ids
        )


class FakeCausalModel(nn.Module):
    def __init__(self, *, fail_on_token: int | None = None) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.fail_on_token = fail_on_token

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: tuple | None = None,
        use_cache: bool,
    ) -> Any:
        assert use_cache is True
        if self.fail_on_token is not None and self.fail_on_token in input_ids.tolist()[0]:
            raise RuntimeError("forced token failure")
        old = (
            torch.empty((1, 1, 0, 1), dtype=torch.float32, device=input_ids.device)
            if past_key_values is None
            else past_key_values[0][0]
        )
        new_values = input_ids.to(dtype=torch.float32).reshape(1, 1, -1, 1)
        cache = torch.cat((old, new_values), dim=2)
        logits = []
        for index in range(input_ids.shape[1]):
            prefix = cache[:, :, : old.shape[2] + index + 1, :]
            score = int(prefix.sum().item()) % 128
            row = torch.arange(128, dtype=torch.float32, device=input_ids.device)
            logits.append(-(row - score).abs())
        output_logits = torch.stack(logits, dim=0).unsqueeze(0)
        return type(
            "FakeOutput",
            (),
            {"past_key_values": ((cache, cache.clone()),), "logits": output_logits},
        )()


class ScriptedFakeCausalModel(FakeCausalModel):
    def __init__(self, next_tokens: tuple[int, ...]) -> None:
        super().__init__()
        self._next_tokens = iter(next_tokens)

    def forward(self, **kwargs: Any) -> Any:
        output = super().forward(**kwargs)
        next_token = next(self._next_tokens)
        output.logits = torch.full_like(output.logits, -1000.0)
        output.logits[..., next_token] = 0.0
        return output


class RecordingBoundary(PromptAwareBoundaryDetector):
    def __init__(self, *, prompt: str, dataset: str) -> None:
        super().__init__(prompt=prompt, dataset=dataset)
        self.feed_calls: list[str] = []

    def feed_text(self, token_text: str):
        self.feed_calls.append(token_text)
        return super().feed_text(token_text)


def _make_context(tmp_path: Path) -> SawrModelContext:
    config = SawrGenerationConfig(
        model_path=str(tmp_path),
        device="cpu",
        retry_repetition_penalty=4.0,
    )
    context = SawrModelContext(FakeCausalModel(), FakeTokenizer(), config)
    context.boundary = PromptAwareBoundaryDetector(
        prompt="def f():\n",
        dataset="humaneval",
    )
    context.prefill("def f():\n")
    return context


@pytest.fixture
def fake_model_context(tmp_path: Path) -> SawrModelContext:
    return _make_context(tmp_path)


def test_replay_from_checkpoint_rebuilds_kv_text_and_boundary(
    fake_model_context: SawrModelContext,
) -> None:
    checkpoint = fake_model_context.checkpoint()
    result = fake_model_context.replay_from_checkpoint(
        checkpoint,
        replacement_steps=[
            GeneratedToken(token_id=21, token_text="x = 2"),
            GeneratedToken(token_id=22, token_text="\n"),
            GeneratedToken(token_id=23, token_text="y = 3\n"),
        ],
    )

    assert result.generated_text == checkpoint.generated_text + "x = 2\ny = 3\n"
    assert fake_model_context.generated_ids == checkpoint.generated_ids + [21, 22, 23]
    assert result.kv_snapshot.seq_len == checkpoint.kv_snapshot.seq_len + 3
    assert fake_model_context.boundary is not None
    assert fake_model_context.boundary.checkpoint().generated_text == fake_model_context.generated_text


def test_failed_rewrites_can_restore_exact_candidate_zero(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    original = [GeneratedToken(31, "total += item"), GeneratedToken(32, "\n")]
    fake_model_context.replay_from_checkpoint(start, replacement_steps=original)
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(41, "total = total + item\n")],
    )
    fake_model_context.replay_from_checkpoint(start, replacement_steps=original)

    assert fake_model_context.generated_text.endswith("total += item\n")
    assert fake_model_context.generated_ids[-2:] == [31, 32]


def test_replay_next_logits_match_one_shot_prefill(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    replacement = [GeneratedToken(21, "x = 2"), GeneratedToken(22, "\n")]
    result = fake_model_context.replay_from_checkpoint(start, replacement_steps=replacement)

    tokenizer = fake_model_context._tokenizer
    prompt_ids = tokenizer.encode("def f():\n")
    input_ids = torch.tensor([prompt_ids + [21, 22]], dtype=torch.long)
    expected = fake_model_context._model(input_ids=input_ids, use_cache=True).logits[:, -1, :]

    assert result.next_logits is not None
    assert torch.allclose(result.next_logits, expected)


def test_replay_after_normally_sampled_prefix_matches_one_shot_prefill(
    fake_model_context: SawrModelContext,
) -> None:
    sampled_prefix = fake_model_context.forward_and_sample()
    assert fake_model_context.boundary is not None
    fake_model_context.boundary.feed_text(sampled_prefix.token_text)
    start = fake_model_context.checkpoint()

    result = fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(21, "x = 2")],
    )

    prompt_ids = fake_model_context._tokenizer.encode("def f():\n")
    expected = fake_model_context._model(
        input_ids=torch.tensor([[*prompt_ids, sampled_prefix.token_id, 21]]),
        use_cache=True,
    ).logits[:, -1, :]
    assert result.next_logits is not None
    assert torch.allclose(result.next_logits, expected)
    assert result.kv_snapshot.seq_len == start.kv_snapshot.seq_len + 1


def test_replay_records_each_token_boundary_and_does_not_accumulate_penalty(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    replacement = [GeneratedToken(21, "x = 2"), GeneratedToken(22, "\n")]
    fake_model_context.replay_from_checkpoint(start, replacement_steps=replacement)
    first_boundary_count = len(fake_model_context._boundary_checkpoints)
    first_penalties = list(fake_model_context._retry_penalty_sequences)

    fake_model_context.replay_from_checkpoint(start, replacement_steps=replacement)

    assert first_boundary_count == 2
    assert len(fake_model_context._boundary_checkpoints) == 2
    assert fake_model_context._retry_penalty_sequences == first_penalties == []


def test_replay_allows_empty_special_token_when_combined_text_is_nonempty(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()

    result = fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(70, ""), GeneratedToken(21, "x = 2")],
    )

    assert result.generated_ids == [70, 21]
    assert result.generated_text == "x = 2"
    assert result.kv_snapshot.seq_len == start.kv_snapshot.seq_len + 2
    assert len(fake_model_context._step_history) == 2
    assert len(fake_model_context._boundary_checkpoints) == 2
    assert fake_model_context.boundary is not None
    assert len(fake_model_context.boundary.checkpoint().token_boundaries) == 2


def test_normal_sampling_and_replay_share_per_token_boundary_semantics(
    tmp_path: Path,
) -> None:
    normal = _make_context(tmp_path)
    sampled_ids = iter((70, 21))
    normal._sample = lambda logits: next(sampled_ids)  # type: ignore[method-assign]
    normal_steps = [normal.forward_and_sample(), normal.forward_and_sample()]
    assert normal.boundary is not None
    for step in normal_steps:
        normal.boundary.feed_text(step.token_text)

    replay = _make_context(tmp_path)
    replay_start = replay.checkpoint()
    replay_result = replay.replay_from_checkpoint(
        replay_start,
        replacement_steps=[GeneratedToken(70, ""), GeneratedToken(21, "x = 2")],
    )

    assert replay.generated_ids == normal.generated_ids == [70, 21]
    assert replay.generated_text == normal.generated_text == "x = 2"
    assert replay_result.kv_snapshot.seq_len == normal.checkpoint().kv_snapshot.seq_len
    assert replay.boundary is not None
    assert replay.boundary.checkpoint() == normal.boundary.checkpoint()
    assert replay.boundary.checkpoint().token_boundaries == (0, len(b"x = 2"))
    assert [item.generated_ids for item in replay._boundary_checkpoints] == [[], [70]]
    assert [item.generated_ids for item in normal._boundary_checkpoints] == [[], [70]]
    assert replay.checkpoint_for_token_start(0).generated_ids == []
    assert replay.checkpoint_for_token_start(1).generated_ids == [70]


def test_generator_feeds_boundary_once_for_empty_and_nonempty_tokens(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    boundaries: list[RecordingBoundary] = []

    def make_boundary(*, prompt: str, dataset: str) -> RecordingBoundary:
        boundary = RecordingBoundary(prompt=prompt, dataset=dataset)
        boundaries.append(boundary)
        return boundary

    monkeypatch.setattr(generator_module, "PromptAwareBoundaryDetector", make_boundary)
    config = SawrGenerationConfig(
        model_path=str(tmp_path),
        device="cpu",
        max_new_tokens=2,
    )
    generator = SawrGenerator(
        config=config,
        rule=HashEmbeddingRule(target_accept_rate=1.0),
        model=ScriptedFakeCausalModel((70, 72, 127)),
        tokenizer=FakeTokenizer(),
    )

    result = generator.generate(
        sample_id="sample",
        prompt="def f():\n",
        dataset="humaneval",
        max_group_statements=2,
        retry_budget=0,
        global_rollback_budget=0,
        max_total_sampled_tokens=2,
    )

    assert result.final_code == "def f():\n    x = 1\n"
    assert len(boundaries) == 1
    assert boundaries[0].feed_calls == ["", "    x = 1\n"]
    assert boundaries[0].checkpoint().token_boundaries == (0, len(b"    x = 1\n"))


def test_empty_tokens_preserve_event_checkpoint_mapping_and_rollback(
    tmp_path: Path,
) -> None:
    context = _make_context(tmp_path)
    sampled_ids = iter((70, 72, 71, 73))
    context._sample = lambda logits: next(sampled_ids)  # type: ignore[method-assign]
    events = []
    assert context.boundary is not None
    for _ in range(4):
        step = context.forward_and_sample()
        events.extend(context.boundary.feed_text(step.token_text))

    simple_events = [event for event in events if event.kind == "simple_candidate"]
    assert [event.token_start_idx for event in simple_events] == [1, 3]
    first_start = context.checkpoint_for_token_start(simple_events[0].token_start_idx)
    second_start = context.checkpoint_for_token_start(simple_events[1].token_start_idx)
    assert first_start.generated_ids == [70]
    assert second_start.generated_ids == [70, 72, 71]

    context.rollback(second_start)

    assert context.generated_ids == [70, 72, 71]
    assert context.generated_text == "    x = 1\n"
    assert context.boundary.checkpoint().generated_text == "    x = 1\n"


def test_replay_failure_restores_start_exactly(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    fake_model_context._model.fail_on_token = 22

    with pytest.raises(RuntimeError, match="forced token failure"):
        fake_model_context.replay_from_checkpoint(
            start,
            replacement_steps=[GeneratedToken(21, "x = 2"), GeneratedToken(22, "\n")],
        )

    restored = fake_model_context.checkpoint()
    assert restored.generated_ids == start.generated_ids
    assert restored.generated_text == start.generated_text
    assert restored.kv_snapshot.seq_len == start.kv_snapshot.seq_len
    assert restored.boundary_state == start.boundary_state
    assert restored.next_logits is not None and start.next_logits is not None
    assert torch.equal(restored.next_logits, start.next_logits)


def test_replay_failure_restores_preexisting_retry_penalties_exactly(
    fake_model_context: SawrModelContext,
) -> None:
    root = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(31, "total += item")],
    )
    start = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(21, "x = 2")],
    )
    fake_model_context.rollback(start, remember_failed_sequence=True)
    penalties_before = list(fake_model_context._retry_penalty_sequences)
    assert penalties_before
    fake_model_context._model.fail_on_token = 22

    with pytest.raises(RuntimeError, match="forced token failure"):
        fake_model_context.replay_from_checkpoint(
            start,
            replacement_steps=[GeneratedToken(23, "y = 3\n"), GeneratedToken(22, "\n")],
        )

    assert fake_model_context._retry_penalty_sequences == penalties_before
    assert fake_model_context.generated_ids == start.generated_ids
    assert fake_model_context.generated_text == start.generated_text


@pytest.mark.parametrize(
    "replacement",
    [[], [GeneratedToken(21, "")]],
)
def test_replay_rejects_empty_steps_or_text(
    fake_model_context: SawrModelContext,
    replacement: list[GeneratedToken],
) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        fake_model_context.replay_from_checkpoint(
            fake_model_context.checkpoint(),
            replacement_steps=replacement,
        )


def test_replay_rejects_token_text_mismatch_before_any_mutation(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    records_before = dict(fake_model_context._checkpoint_records)
    ids_before = list(fake_model_context.generated_ids)
    text_before = fake_model_context.generated_text

    with pytest.raises(ValueError, match="token text does not match") as error:
        fake_model_context.replay_from_checkpoint(
            start,
            replacement_steps=[GeneratedToken(21, "secret mismatch payload")],
        )

    assert "secret mismatch payload" not in str(error.value)
    assert fake_model_context.generated_ids == ids_before
    assert fake_model_context.generated_text == text_before
    assert fake_model_context._checkpoint_records == records_before
    assert fake_model_context._tokenizer.decode_calls[-1] == (21, True)


def test_normal_sampling_uses_shared_special_skipping_token_decode(
    fake_model_context: SawrModelContext,
) -> None:
    fake_model_context._sample = lambda logits: 70  # type: ignore[method-assign]

    step = fake_model_context.forward_and_sample()

    assert step == GeneratedToken(70, "")
    assert fake_model_context._tokenizer.decode_calls[-1] == (70, True)


def test_eos_is_empty_for_normal_sampling_and_replay_attestation(
    fake_model_context: SawrModelContext,
) -> None:
    root = fake_model_context.checkpoint()
    fake_model_context._sample = lambda logits: 127  # type: ignore[method-assign]

    eos = fake_model_context.forward_and_sample()

    assert eos == GeneratedToken(127, "")
    assert fake_model_context.generated_text == ""
    assert fake_model_context._tokenizer.decode_calls[-1] == (127, True)
    fake_model_context.rollback(root)
    result = fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(127, ""), GeneratedToken(21, "x = 2")],
    )
    assert result.generated_ids == [127, 21]
    assert result.generated_text == "x = 2"


def test_replay_rejects_checkpoint_from_another_context(
    fake_model_context: SawrModelContext,
    tmp_path: Path,
) -> None:
    config = replace(fake_model_context._config, model_path=str(tmp_path))
    other = SawrModelContext(FakeCausalModel(), FakeTokenizer(), config)
    other.prefill("def f():\n")

    with pytest.raises(ValueError, match="current context"):
        other.replay_from_checkpoint(
            fake_model_context.checkpoint(),
            replacement_steps=[GeneratedToken(21, "x")],
        )


def test_replay_rejects_old_prefill_and_mutated_checkpoint(
    fake_model_context: SawrModelContext,
) -> None:
    stale = fake_model_context.checkpoint()
    fake_model_context.prefill("different prompt")
    with pytest.raises(ValueError, match="stale"):
        fake_model_context.replay_from_checkpoint(
            stale,
            replacement_steps=[GeneratedToken(21, "x")],
        )

    current = fake_model_context.checkpoint()
    current.generated_ids.append(99)
    with pytest.raises(ValueError, match="modified"):
        fake_model_context.rollback(current)


def test_checkpoint_rejects_boundary_detector_attachment_mismatch(
    fake_model_context: SawrModelContext,
) -> None:
    checkpoint = fake_model_context.checkpoint()
    fake_model_context.boundary = None

    with pytest.raises(ValueError, match="boundary detector"):
        fake_model_context.rollback(checkpoint)

    fake_model_context.boundary = PromptAwareBoundaryDetector(
        prompt="def other():\n",
        dataset="humaneval",
    )
    with pytest.raises(ValueError, match="boundary detector"):
        fake_model_context.rollback(checkpoint)


def _state_snapshot(*, retry_count: int = 0) -> StateMachineSnapshot:
    return StateMachineSnapshot(
        layer_stack=(),
        accepted_hit_count=1,
        closed_without_hit_count=2,
        fallback_count=3,
        candidate_count=4,
        retry_count=retry_count,
        audit_events=(),
    )


def test_controlled_state_machine_attachment_is_provenance_checked(
    fake_model_context: SawrModelContext,
) -> None:
    base = fake_model_context.checkpoint()
    state = _state_snapshot()
    attached = fake_model_context.attach_state_machine_state(base, state)

    restored = fake_model_context.rollback(attached)

    assert restored == state
    assert restored is not state
    assert fake_model_context.rollback(deepcopy(attached)) == state
    forged = replace(attached, state_machine_state=_state_snapshot(retry_count=9))
    with pytest.raises(ValueError, match="modified"):
        fake_model_context.rollback(forged)
    forged_base = replace(base, state_machine_state=state)
    with pytest.raises(ValueError, match="modified"):
        fake_model_context.rollback(forged_base)
    with pytest.raises(ValueError, match="StateMachineSnapshot"):
        fake_model_context.attach_state_machine_state(base, {"retry_count": 0})  # type: ignore[arg-type]
    unsupported = replace(state, audit_events=(object(),))
    with pytest.raises(ValueError, match="unsupported"):
        fake_model_context.attach_state_machine_state(base, unsupported)


def test_repeated_checkpoint_is_interned_without_registry_growth(
    fake_model_context: SawrModelContext,
) -> None:
    first = fake_model_context.checkpoint()
    record_count = len(fake_model_context._checkpoint_records)

    repeated = [fake_model_context.checkpoint() for _ in range(1_000)]

    assert all(checkpoint is first for checkpoint in repeated)
    assert len(fake_model_context._checkpoint_records) == record_count


def test_rollback_invalidates_old_branch_even_if_tokens_overlap(
    fake_model_context: SawrModelContext,
) -> None:
    root = fake_model_context.checkpoint()
    old_branch = fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(21, "x = 2")],
    )
    fake_model_context.rollback(root)
    new_branch = fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(21, "x = 2")],
    )
    assert new_branch._checkpoint_id != old_branch._checkpoint_id

    with pytest.raises(ValueError, match="stale|unknown"):
        fake_model_context.rollback(old_branch)


def test_rollback_pruning_keeps_externally_held_ancestor_anchors(
    fake_model_context: SawrModelContext,
) -> None:
    root = fake_model_context.checkpoint()
    anchor = fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(21, "x = 2")],
    )
    fake_model_context.replay_from_checkpoint(
        anchor,
        replacement_steps=[GeneratedToken(22, "\n")],
    )

    fake_model_context.rollback(anchor)
    fake_model_context.rollback(root)

    assert fake_model_context.generated_ids == []
    assert fake_model_context.generated_text == ""


def test_registry_and_byte_queries_remain_bounded_for_256_tokens(
    fake_model_context: SawrModelContext,
) -> None:
    root = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        root,
        replacement_steps=[GeneratedToken(21, "x = 2") for _ in range(256)],
    )
    assert len(fake_model_context._checkpoint_records) <= 258
    count_before = len(fake_model_context._checkpoint_records)

    for _ in range(100):
        mapped = fake_model_context.checkpoint_for_generated_byte(0)
        assert mapped.checkpoint.generated_text == ""

    assert len(fake_model_context._checkpoint_records) == count_before

def test_checkpoint_for_generated_byte_exact_token_boundary_has_empty_prefix(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(61, "λ = 1\n"), GeneratedToken(52, " = 2\n")],
    )
    start_byte = len("λ = 1\n".encode("utf-8"))

    mapped = fake_model_context.checkpoint_for_generated_byte(start_byte)

    assert mapped.exact_start_byte == start_byte
    assert mapped.protected_prefix == ""
    assert mapped.checkpoint.generated_text == "λ = 1\n"


def test_checkpoint_for_generated_byte_inside_token_protects_exact_utf8_prefix(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    token_text = "    prior = 1\n    value"
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(51, token_text), GeneratedToken(52, " = 2\n")],
    )
    protected = "    prior = 1\n    "
    start_byte = len(protected.encode("utf-8"))

    mapped = fake_model_context.checkpoint_for_generated_byte(start_byte)

    assert mapped.exact_start_byte == start_byte
    assert mapped.checkpoint.generated_text == ""
    assert mapped.protected_prefix == protected
    assert token_text.encode("utf-8").startswith(mapped.protected_prefix.encode("utf-8"))


def test_checkpoint_for_generated_byte_prefers_latest_checkpoint_at_byte_tie(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[
            GeneratedToken(70, ""),
            GeneratedToken(71, ""),
            GeneratedToken(21, "x = 2"),
        ],
    )

    mapped = fake_model_context.checkpoint_for_generated_byte(0)

    assert mapped.checkpoint.generated_ids == [70, 71]
    assert mapped.checkpoint.generated_text == ""
    assert mapped.protected_prefix == ""
    assert mapped.checkpoint.kv_snapshot.seq_len == start.kv_snapshot.seq_len + 2


def test_checkpoint_for_generated_byte_rejects_mid_codepoint_and_bad_offsets(
    fake_model_context: SawrModelContext,
) -> None:
    start = fake_model_context.checkpoint()
    fake_model_context.replay_from_checkpoint(
        start,
        replacement_steps=[GeneratedToken(61, "λ = 1\n")],
    )

    with pytest.raises(ValueError, match="UTF-8 codepoint boundary"):
        fake_model_context.checkpoint_for_generated_byte(1)
    for offset in (-1, len("λ = 1\n".encode("utf-8")) + 1, True):
        with pytest.raises(ValueError):
            fake_model_context.checkpoint_for_generated_byte(offset)  # type: ignore[arg-type]
