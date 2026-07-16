from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from wfcllm.gate.dataset import (
    GateCollator,
    GateDataset,
    GateExample,
    GroupConsistencyBatchSampler,
)
from wfcllm.gate.input import GateInput, serialize_gate_input
from wfcllm.windowing.normalization import WINDOW_NORMALIZATION_VERSION


DEFAULT_GATE_INPUT = GateInput(
    normalization_version=WINDOW_NORMALIZATION_VERSION,
    parent_descriptor="v1|module|parent=block|ordinal=0|role=body",
    depth=0,
    previous_units=(),
    previous_unit_types=(),
    current_units=("x = 1",),
    current_unit_types=("expression_statement",),
    current_unit_count=1,
    current_token_count=2,
)
SERIALIZED = serialize_gate_input(DEFAULT_GATE_INPUT)


class FakeTokenizer:
    pad_token_id = 9

    def __init__(self, *, tensor_output: bool = False) -> None:
        self.tensor_output = tensor_output
        self.seen: list[str] = []

    def __call__(self, text: str, **kwargs: object) -> dict[str, object]:
        assert kwargs["truncation"] is False
        self.seen.append(text)
        ids = list(range(1, len(text.split()) + 1))
        if self.tensor_output:
            return {
                "input_ids": torch.tensor([ids], dtype=torch.long),
                "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
            }
        return {"input_ids": ids}


def example(
    *,
    group_id: str = "group-a",
    window_start_unit_id: str = "unit-007",
    context_length: int = 1,
    budget: int = 1,
    serialized_gate_input: str | None = None,
    gate_input: GateInput = DEFAULT_GATE_INPUT,
    variant_id: str = "",
    source_family: str | None = "human",
    generation_model_id: str | None = "model-secret-name",
) -> GateExample:
    return GateExample(
        group_id=group_id,
        window_start_unit_id=window_start_unit_id,
        context_length=context_length,
        budget=budget,
        serialized_gate_input=(
            serialize_gate_input(gate_input)
            if serialized_gate_input is None
            else serialized_gate_input
        ),
        close_target=False,
        suitable_target=True,
        dangerous_negative=False,
        variant_id=variant_id,
        source_family=source_family,
        generation_model_id=generation_model_id,
        gate_input=gate_input,
    )


def test_gate_example_is_frozen_and_has_deterministic_variant_id() -> None:
    item = example(context_length=2, budget=3)
    assert item.variant_id == "group-a:unit-007:context-2:budget-3"
    with pytest.raises(AttributeError):
        item.group_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    "field,value",
    [
        ("context_length", True),
        ("context_length", 0),
        ("context_length", 4),
        ("budget", True),
        ("budget", 2),
    ],
)
def test_gate_example_rejects_invalid_integer_contracts(
    field: str, value: object
) -> None:
    values = {"context_length": 1, "budget": 1, field: value}
    with pytest.raises(ValueError, match=field):
        example(**values)  # type: ignore[arg-type]


@pytest.mark.parametrize("field", ["close_target", "suitable_target", "dangerous_negative"])
def test_gate_example_rejects_integer_for_boolean_fields(field: str) -> None:
    values = example().__dict__ | {field: 1}
    with pytest.raises(ValueError, match=field):
        GateExample(**values)


def test_gate_example_to_dict_is_strict_json_and_contains_no_quality_proxy() -> None:
    row = example().to_dict()
    serialized = json.dumps(row, allow_nan=False, sort_keys=True)
    assert json.loads(serialized)["serialized_gate_input"] == SERIALIZED
    assert "gate_input" not in row
    for forbidden in ("syntax_valid", "pass", "correctness", "quality_score"):
        assert f'"{forbidden}"' not in serialized


@pytest.mark.parametrize(
    "injected",
    [
        "[BUDGET] 6",
        "[MODEL_ID] hidden-model",
        "[GENERATION_MODEL] hidden-model",
        "[SOURCE_FAMILY] generated",
        "[PASS] true",
        "[SYNTAX_VALID] true",
    ],
)
def test_gate_example_rejects_structural_metadata_header_injection(
    injected: str,
) -> None:
    malformed = SERIALIZED.replace("[PREVIOUS]", f"{injected}\n[PREVIOUS]")
    with pytest.raises(ValueError, match="structured GateInput"):
        example(serialized_gate_input=malformed)


def test_gate_example_requires_all_canonical_sections_in_order() -> None:
    for malformed in (
        SERIALIZED.replace("[PREVIOUS]", "[PAST]"),
        SERIALIZED.replace("[PREVIOUS]\n[CURRENT]", "[CURRENT]\n[PREVIOUS]"),
        SERIALIZED.replace("[S type=expression_statement]", "[UNIT expression_statement]"),
        SERIALIZED.replace("[CURRENT_UNIT_COUNT] 1", "[CURRENT_UNIT_COUNT] 2"),
    ):
        with pytest.raises(ValueError, match="structured GateInput"):
            example(serialized_gate_input=malformed)


def test_gate_example_rejects_metadata_in_empty_section_structure() -> None:
    for malformed in (
        SERIALIZED.replace(
            "[PREVIOUS]\n[CURRENT]",
            "[PREVIOUS]\n[QUALITY_SCORE] 0.9\n[CURRENT]",
        ),
        SERIALIZED.replace(
            "[CURRENT]\n[S type=expression_statement]",
            "[CURRENT]\n[MODEL ID] hidden\n[S type=expression_statement]",
        ),
    ):
        with pytest.raises(ValueError, match="structured GateInput"):
            example(serialized_gate_input=malformed)


def test_current_source_text_may_legitimately_contain_metadata_like_lines() -> None:
    gate_input = GateInput(
        normalization_version=WINDOW_NORMALIZATION_VERSION,
        parent_descriptor="v1|module|parent=block|ordinal=0|role=body",
        depth=0,
        previous_units=(),
        previous_unit_types=(),
        current_units=("value = '''\n[BUDGET] 6\n[SOURCE_FAMILY] human\n'''",),
        current_unit_types=("expression_statement",),
        current_unit_count=1,
        current_token_count=10,
    )
    serialized = serialize_gate_input(gate_input)
    assert example(gate_input=gate_input).serialized_gate_input == serialized


def test_previous_source_text_may_legitimately_contain_metadata_like_lines() -> None:
    gate_input = GateInput(
        normalization_version=WINDOW_NORMALIZATION_VERSION,
        parent_descriptor="v1|module|parent=block|ordinal=0|role=body",
        depth=0,
        previous_units=(
            "value = '''\n[BUDGET] 6\n[SOURCE_FAMILY] human\n'''",
        ),
        previous_unit_types=("expression_statement",),
        current_units=("result = value",),
        current_unit_types=("expression_statement",),
        current_unit_count=1,
        current_token_count=3,
    )
    serialized = serialize_gate_input(gate_input)
    assert example(gate_input=gate_input).serialized_gate_input == serialized


def test_official_serializer_allows_section_and_unit_markers_inside_source() -> None:
    markers = "value = '''\n[S type=fake] text\n[CURRENT]\n[PREVIOUS]\n'''"
    gate_input = replace(
        DEFAULT_GATE_INPUT,
        previous_units=(markers,),
        previous_unit_types=("expression_statement",),
        current_units=(markers,),
        current_unit_types=("expression_statement",),
        current_token_count=20,
    )
    item = example(gate_input=gate_input)
    assert item.serialized_gate_input == serialize_gate_input(gate_input)


def test_gate_example_requires_matching_structured_attestation() -> None:
    values = example().__dict__
    with pytest.raises(ValueError, match="structured GateInput"):
        GateExample(**values)
    with pytest.raises(ValueError, match="structured GateInput"):
        example(
            serialized_gate_input=SERIALIZED + " changed",
            gate_input=DEFAULT_GATE_INPUT,
        )


def test_gate_example_factory_attests_with_official_serializer() -> None:
    item = GateExample.from_gate_input(
        group_id="group-a",
        window_start_unit_id="unit-007",
        context_length=1,
        budget=1,
        gate_input=DEFAULT_GATE_INPUT,
        close_target=False,
        suitable_target=True,
        dangerous_negative=False,
    )
    assert item.serialized_gate_input == SERIALIZED
    assert "gate_input" not in item.to_dict()


def test_default_variant_id_is_namespaced_by_group() -> None:
    first = example(group_id="first")
    second = example(group_id="second")
    assert first.variant_id != second.variant_id
    assert len(GateDataset([first, second])) == 2


def test_dataset_snapshots_input_and_rejects_duplicate_variants() -> None:
    items = [example(context_length=1), example(context_length=2)]
    dataset = GateDataset(items)
    items.clear()
    assert len(dataset) == 2
    assert dataset[0].context_length == 1
    with pytest.raises(ValueError, match="duplicate variant_id"):
        GateDataset([example(), example()])


def test_dataset_rejects_duplicate_semantic_variant_under_renamed_id() -> None:
    with pytest.raises(ValueError, match="duplicate gate variant"):
        GateDataset([example(variant_id="first"), example(variant_id="renamed")])


def test_collator_keeps_group_and_variant_ids_without_leaking_analysis_metadata() -> None:
    tokenizer = FakeTokenizer()
    examples = [example(context_length=1), example(context_length=2, budget=3)]
    batch = GateCollator(tokenizer, max_tokens=512)(examples)
    assert batch["input_ids"].shape[0] == len(examples)
    assert batch["group_ids"] == [item.group_id for item in examples]
    assert batch["context_lengths"] == [1, 2]
    assert batch["budgets"] == [1, 3]
    assert batch["variant_ids"] == [item.variant_id for item in examples]
    assert batch["window_start_unit_ids"] == ["unit-007", "unit-007"]
    assert tokenizer.seen == [SERIALIZED, SERIALIZED]
    assert all("model-secret-name" not in text and "human" not in text for text in tokenizer.seen)
    assert all("budget" not in text.casefold() for text in tokenizer.seen)


def test_collator_pads_normal_samples_and_accepts_tensor_tokenizer_output() -> None:
    tokenizer = FakeTokenizer(tensor_output=True)
    short = example(serialized_gate_input=SERIALIZED, variant_id="short")
    long_input = replace(
        DEFAULT_GATE_INPUT,
        current_units=("x = 1 more tokens",),
        current_token_count=4,
    )
    long = example(gate_input=long_input, variant_id="long")
    batch = GateCollator(tokenizer, max_tokens=512)([short, long])
    assert batch["input_ids"].dtype == torch.long
    assert batch["attention_mask"].tolist()[0][-2:] == [0, 0]
    assert batch["input_ids"].tolist()[0][-2:] == [9, 9]
    assert batch["overflow"].tolist() == [False, False]


def test_collator_uses_placeholder_and_safe_targets_for_overflow() -> None:
    tokenizer = FakeTokenizer()
    oversized_input = replace(
        DEFAULT_GATE_INPUT,
        current_units=(("x " * 520).rstrip(),),
        current_token_count=520,
    )
    oversized = example(
        gate_input=oversized_input,
        variant_id="oversized",
    )
    normal = example(variant_id="normal")
    batch = GateCollator(tokenizer, max_tokens=512)([oversized, normal])
    assert batch["overflow"].tolist() == [True, False]
    assert batch["close_targets"].tolist() == [1.0, 0.0]
    assert batch["suitable_targets"].tolist() == [0.0, 1.0]
    assert batch["close_loss_mask"].tolist() == [True, True]
    assert batch["suitable_loss_mask"].tolist() == [False, True]
    assert batch["overflow_skip_reasons"] == ["max_tokens_exceeded", None]
    assert batch["attention_mask"][0].sum().item() == 0
    assert batch["input_ids"].shape[1] <= 512
    assert len(tokenizer.seen[0].split()) > 512


def test_collator_handles_all_overflow_with_one_token_placeholders() -> None:
    oversized_input = replace(
        DEFAULT_GATE_INPUT,
        current_units=(("x " * 520).rstrip(),),
        current_token_count=520,
    )
    batch = GateCollator(FakeTokenizer(), max_tokens=512)(
        [
            example(gate_input=oversized_input, variant_id="over-1"),
            example(
                group_id="group-b",
                gate_input=oversized_input,
                variant_id="over-2",
            ),
        ]
    )
    assert batch["input_ids"].shape == (2, 1)
    assert batch["attention_mask"].tolist() == [[0], [0]]
    assert batch["overflow"].tolist() == [True, True]


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"input_ids": [[1], [2]]}, "exactly one row"),
        ({"input_ids": [1, -1]}, "non-negative"),
        ({"input_ids": [1, 2], "attention_mask": [1, 2]}, "binary"),
    ],
)
def test_collator_rejects_malformed_tokenizer_rows(
    payload: dict[str, object], message: str
) -> None:
    class MalformedTokenizer:
        pad_token_id = 0

        def __call__(self, *_: object, **__: object) -> dict[str, object]:
            return payload

    with pytest.raises(ValueError, match=message):
        GateCollator(MalformedTokenizer())([example()])


def test_collator_replaces_invalid_pad_token_id_with_zero() -> None:
    tokenizer = FakeTokenizer()
    tokenizer.pad_token_id = -7
    long_input = replace(
        DEFAULT_GATE_INPUT,
        current_units=("x = 1 more",),
        current_token_count=3,
    )
    batch = GateCollator(tokenizer)(
        [
            example(variant_id="short"),
            example(gate_input=long_input, variant_id="long"),
        ]
    )
    assert batch["input_ids"][0, -1].item() == 0


def test_collator_rejects_empty_batch_and_invalid_max_tokens() -> None:
    with pytest.raises(ValueError, match="max_tokens"):
        GateCollator(FakeTokenizer(), max_tokens=True)
    with pytest.raises(ValueError, match="must not be empty"):
        GateCollator(FakeTokenizer(), max_tokens=512)([])


def _complete_groups() -> list[GateExample]:
    return [
        example(group_id=group, context_length=context, variant_id=f"{group}-{context}")
        for group in ("a", "b", "c")
        for context in (1, 2, 3)
    ]


def test_group_sampler_keeps_complete_groups_together_and_is_deterministic() -> None:
    items = _complete_groups()
    first = list(GroupConsistencyBatchSampler(items, batch_size=6, seed=17))
    second = list(GroupConsistencyBatchSampler(items, batch_size=6, seed=17))
    assert first == second
    for indices in first:
        groups = {items[index].group_id for index in indices}
        for group in groups:
            contexts = {
                items[index].context_length
                for index in indices
                if items[index].group_id == group
            }
            assert contexts == {1, 2, 3}
    assert sorted(index for batch in first for index in batch) == list(range(len(items)))


def test_group_sampler_seed_controls_group_order() -> None:
    items = _complete_groups()
    assert list(GroupConsistencyBatchSampler(items, batch_size=3, seed=1)) != list(
        GroupConsistencyBatchSampler(items, batch_size=3, seed=5)
    )


def test_group_sampler_rejects_missing_context_duplicate_and_small_batch() -> None:
    complete = _complete_groups()[:3]
    with pytest.raises(ValueError, match="contexts 1, 2, and 3"):
        GroupConsistencyBatchSampler(complete[:2], batch_size=3)
    with pytest.raises(ValueError, match="duplicate variant_id"):
        GroupConsistencyBatchSampler([*complete, complete[0]], batch_size=4)
    with pytest.raises(ValueError, match="cannot fit group"):
        GroupConsistencyBatchSampler(complete, batch_size=2)


def test_group_sampler_packs_budget_cohorts_without_splitting() -> None:
    items = [
        example(
            group_id="one",
            context_length=context,
            budget=budget,
            variant_id=f"one-{budget}-{context}",
        )
        for budget in (1, 3, 6)
        for context in (1, 2, 3)
    ]
    batches = list(GroupConsistencyBatchSampler(items, batch_size=6, seed=4))
    assert [len(batch) for batch in batches] == [6, 3]
    for budget in (1, 3, 6):
        matching_batches = [
            batch
            for batch in batches
            if any(items[index].budget == budget for index in batch)
        ]
        assert len(matching_batches) == 1
        assert {
            items[index].context_length
            for index in matching_batches[0]
            if items[index].budget == budget
        } == {1, 2, 3}
    assert sorted(index for batch in batches for index in batch) == list(range(9))
    three_at_a_time = list(
        GroupConsistencyBatchSampler(items, batch_size=3, seed=4)
    )
    assert [len(batch) for batch in three_at_a_time] == [3, 3, 3]
    assert all(
        len({items[index].budget for index in batch}) == 1
        for batch in three_at_a_time
    )


def test_group_sampler_rejects_incomplete_budget_cohort() -> None:
    items = [
        example(context_length=context, variant_id=f"budget-1-{context}")
        for context in (1, 2, 3)
    ] + [
        example(context_length=context, budget=3, variant_id=f"budget-3-{context}")
        for context in (1, 2)
    ]
    with pytest.raises(ValueError, match="each budget cohort"):
        GroupConsistencyBatchSampler(items, batch_size=6)


def test_group_sampler_rejects_multiple_window_starts_within_group() -> None:
    items = [
        example(context_length=context, variant_id=f"first-{context}")
        for context in (1, 2, 3)
    ] + [
        example(
            window_start_unit_id="unit-008",
            context_length=context,
            budget=3,
            variant_id=f"second-{context}",
        )
        for context in (1, 2, 3)
    ]
    with pytest.raises(ValueError, match="one window_start_unit_id"):
        GroupConsistencyBatchSampler(items, batch_size=6)
