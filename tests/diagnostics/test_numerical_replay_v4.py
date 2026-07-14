from __future__ import annotations

import hashlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from wfcllm.diagnostics.numerical_replay_v4 import (
    BoundaryCandidate,
    ContextCase,
    LayerCaptureRuntime,
    build_condition_specs,
    capture_downstream_from_saved,
    compose_batch,
    first_divergent_layer,
    quantize_half_away_from_zero,
    projection_sign_rows,
    quantization_boundary_margins,
    select_boundary_cases,
    stable_tensor_sha256,
    tensor_delta,
)


def _case(
    case_id: str,
    *,
    tokens: int,
    category: str = "control",
) -> ContextCase:
    serialized = f"serialized::{case_id}"
    return ContextCase(
        case_id=case_id,
        serialized=serialized,
        context_sha256=hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        role="Return|FunctionDef|body",
        token_count=tokens,
        category=category,
    )


def test_quantization_boundary_margins_are_in_scaled_units() -> None:
    values = torch.tensor([0.125, -0.125, 0.25, -0.25, 0.1875])

    margins = quantization_boundary_margins(values, scale=4)

    assert margins == (0.0, 0.0, 0.5, 0.5, 0.25)


def test_compose_self_repeat_is_shape_isolated() -> None:
    target = _case("1", tokens=9)
    pool = (
        target,
        _case("2", tokens=2),
        _case("3", tokens=30, category="failure"),
    )

    batch = compose_batch(
        target,
        pool,
        batch_size=8,
        composition="self_repeat",
        order="forward",
        seed=17,
    )

    assert len(batch.contexts) == 8
    assert set(batch.contexts) == {target.serialized}
    assert batch.target_index == 0
    assert batch.target_indices == tuple(range(8))


def test_compose_mixed_batches_cover_length_failure_and_order() -> None:
    target = _case("1", tokens=9)
    pool = (
        target,
        _case("2", tokens=1),
        _case("3", tokens=2),
        _case("4", tokens=50),
        _case("5", tokens=60),
        _case("6", tokens=10, category="failure"),
        _case("7", tokens=11, category="failure"),
    )

    short = compose_batch(
        target, pool, batch_size=4, composition="short_mix", order="forward", seed=3
    )
    long = compose_batch(
        target, pool, batch_size=4, composition="long_mix", order="reverse", seed=3
    )
    failure = compose_batch(
        target, pool, batch_size=4, composition="failure_mix", order="random", seed=3
    )

    assert short.target_index == 0
    assert all(item in {"serialized::2", "serialized::3"} for item in short.contexts[1:])
    assert long.target_index == 3
    assert all(item in {"serialized::4", "serialized::5"} for item in long.contexts[:-1])
    assert len(failure.contexts) == 4
    assert failure.contexts.count(target.serialized) == 1
    assert all(
        item == target.serialized or item in {"serialized::6", "serialized::7"}
        for item in failure.contexts
    )


def test_random_batch_order_is_seeded_and_preserves_target() -> None:
    target = _case("1", tokens=9)
    pool = tuple([target, *(_case(str(i), tokens=i) for i in range(2, 12))])

    first = compose_batch(
        target, pool, batch_size=8, composition="short_mix", order="random", seed=99
    )
    second = compose_batch(
        target, pool, batch_size=8, composition="short_mix", order="random", seed=99
    )

    assert first == second
    assert first.contexts[first.target_index] == target.serialized


def test_first_divergent_layer_reports_earliest_numeric_change() -> None:
    reference = {
        "t5_cls_hidden": torch.tensor([1.0, 2.0]),
        "projection_pre_norm": torch.tensor([0.4, 0.8]),
        "model_post_norm": torch.tensor([0.2, 0.9]),
    }
    candidate = {name: value.clone() for name, value in reference.items()}
    candidate["projection_pre_norm"][1] += 1e-7
    candidate["model_post_norm"][0] += 1e-4

    assert first_divergent_layer(reference, candidate) == "projection_pre_norm"
    assert first_divergent_layer(reference, reference) is None

    incomplete = dict(candidate)
    incomplete.pop("projection_pre_norm")
    with pytest.raises(ValueError, match="stage schema"):
        first_divergent_layer(reference, incomplete)


def test_projection_rows_require_v4_diagnostic_domain_separation() -> None:
    material = b"k" * 32

    first = projection_sign_rows(
        material, rows=7, dimensions=64, domain="v4-diagnostic/projection"
    )
    other = projection_sign_rows(
        material, rows=7, dimensions=64, domain="v4-diagnostic/projection-alt"
    )

    assert first.shape == (7, 64)
    assert first.dtype == torch.int8
    assert set(first.flatten().tolist()) == {-1, 1}
    assert not torch.equal(first, other)

    with pytest.raises(ValueError, match="V4 diagnostic"):
        projection_sign_rows(material, rows=7, dimensions=64, domain="projection")


def test_half_away_quantization_matches_v3_boundary_rule() -> None:
    values = torch.tensor([0.0, 0.03125, -0.03125, 0.0625, -0.0625])

    quantized = quantize_half_away_from_zero(values, scale=16)

    assert quantized.tolist() == [0, 1, -1, 1, -1]
    assert quantized.dtype == torch.int64


def test_condition_plan_covers_each_required_axis_without_full_factorial() -> None:
    conditions = build_condition_specs(repeats=20)

    assert {item.batch_size for item in conditions} == {1, 2, 4, 8, 16, 32}
    assert {item.composition for item in conditions} == {
        "self_repeat",
        "short_mix",
        "long_mix",
        "failure_mix",
    }
    assert {item.order for item in conditions} == {"forward", "reverse", "random"}
    assert {item.padding for item in conditions} == {"dynamic", "fixed_256"}
    assert {item.grad_mode for item in conditions} == {"no_grad", "inference_mode"}
    assert {item.deterministic_algorithms for item in conditions} == {False, True}
    assert {item.tf32 for item in conditions} == {False, True}
    assert {item.matmul_precision for item in conditions} == {
        "highest",
        "high",
        "medium",
    }
    assert {item.warmup_count for item in conditions} == {0, 1, 5}
    reference = [item for item in conditions if item.reference]
    assert len(reference) == 1
    assert reference[0].repeats == 20
    assert all(item.repeats == 1 for item in conditions if not item.reference)
    assert len({item.condition_id for item in conditions}) == len(conditions)


def test_order_conditions_use_the_same_multiset_contract() -> None:
    order_conditions = [
        item for item in build_condition_specs(repeats=20) if item.axis == "order"
    ]

    grouped = {}
    for item in order_conditions:
        key = (
            item.batch_size,
            item.composition,
            item.padding,
            item.grad_mode,
            item.deterministic_algorithms,
            item.tf32,
            item.matmul_precision,
            item.warmup_count,
        )
        grouped.setdefault(key, set()).add(item.order)

    assert grouped
    assert all(orders == {"forward", "reverse", "random"} for orders in grouped.values())


def test_boundary_selection_is_deterministic_and_excludes_reserved_cases() -> None:
    candidates = tuple(
        BoundaryCandidate(
            case=_case(str(index), tokens=index + 1),
            minimum_quantization_margin=float(index % 5) / 10,
            minimum_projection_margin=float(index % 7),
        )
        for index in range(40)
    )

    selected = select_boundary_cases(
        candidates,
        count=20,
        excluded_case_ids={"0", "1", "2"},
    )
    repeated = select_boundary_cases(
        tuple(reversed(candidates)),
        count=20,
        excluded_case_ids={"0", "1", "2"},
    )

    assert selected == repeated
    assert len(selected) == 20
    assert not ({item.case.case_id for item in selected} & {"0", "1", "2"})
    assert len({item.case.case_id for item in selected}) == 20


def test_context_case_rejects_noncanonical_or_incorrect_hash() -> None:
    with pytest.raises(ValueError, match="SHA-256"):
        ContextCase(
            case_id="bad",
            serialized="context",
            context_sha256="A" * 64,
            role="Return|FunctionDef|body",
            token_count=3,
            category="control",
        )


def test_tensor_hash_and_delta_are_exact_and_coordinate_addressable() -> None:
    reference = torch.tensor([1.0, -0.0, 3.0], dtype=torch.float32)
    candidate = reference.clone()
    candidate[2] = torch.nextafter(candidate[2], torch.tensor(float("inf")))

    first_hash = stable_tensor_sha256(reference)
    repeated_hash = stable_tensor_sha256(reference.clone())
    delta = tensor_delta(reference, candidate)

    assert first_hash == repeated_hash
    assert first_hash != stable_tensor_sha256(candidate)
    assert delta.mismatch_count == 1
    assert delta.mismatch_indices == (2,)
    assert delta.max_abs > 0.0
    assert delta.max_relative > 0.0
    assert delta.max_ulp == 1
    assert delta.cosine_similarity <= 1.0


class _FakeTokenizer:
    pad_token_id = 0
    vocab_size = 257

    def __call__(
        self,
        texts,
        *,
        padding,
        max_length,
        truncation,
        return_tensors,
    ):
        assert truncation is False
        assert return_tensors == "pt"
        rows = [
            [2, *(3 + byte % 200 for byte in text.encode("utf-8")), 1]
            for text in texts
        ]
        width = max_length if padding == "max_length" else max(map(len, rows))
        input_ids = []
        masks = []
        for row in rows:
            assert len(row) <= width
            padding_size = width - len(row)
            input_ids.append([*row, *([self.pad_token_id] * padding_size)])
            masks.append([*([1] * len(row)), *([0] * padding_size)])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(masks, dtype=torch.int64),
        }


class _FakeEncoder(nn.Module):
    def forward(self, *, input_ids, attention_mask):
        values = input_ids.float() * attention_mask
        hidden = torch.stack(
            (values, values + attention_mask, values * 2, attention_mask.float()),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


class _FakeSemanticModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _FakeEncoder()
        self.projection = nn.Linear(4, 3, bias=False)
        with torch.no_grad():
            self.projection.weight.copy_(
                torch.tensor(
                    [
                        [0.25, 0.5, 0.125, 0.75],
                        [0.5, -0.25, 0.75, 0.125],
                        [-0.25, 0.75, 0.5, -0.125],
                    ]
                )
            )


def _capture_runtime() -> LayerCaptureRuntime:
    return LayerCaptureRuntime(
        model=_FakeSemanticModel(),
        tokenizer=_FakeTokenizer(),
        whitening_mean=torch.zeros(3),
        whitening_projection=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
        projection_rows=torch.tensor(
            [
                [1, 1],
                [1, -1],
                [-1, 1],
                [-1, -1],
                [1, 1],
                [1, -1],
                [-1, 1],
            ],
            dtype=torch.int8,
        ),
        quantization_scale=16,
        device="cpu",
        max_tokens=256,
    )


def test_layer_capture_records_every_self_repeat_row_and_full_stage_schema() -> None:
    target = _case("capture", tokens=8)
    batch = compose_batch(
        target,
        (target, _case("other", tokens=3)),
        batch_size=4,
        composition="self_repeat",
        order="forward",
        seed=9,
    )
    condition = replace(build_condition_specs(repeats=20)[0], batch_size=4)

    captured = _capture_runtime().capture(batch, condition)

    assert captured.sequence_length == 256
    assert captured.model_eval is True
    assert len(captured.target_layers) == 4
    expected = {
        "input_ids",
        "attention_mask",
        "t5_cls_hidden",
        "projection_pre_norm",
        "model_post_norm",
        "runtime_post_norm",
        "cpu_centered",
        "whitening_pre_norm",
        "whitening_post_norm",
        "quantized",
        "projection_dots",
        "signature_bits",
    }
    assert all(set(item) == expected for item in captured.target_layers)
    assert all(
        first_divergent_layer(captured.target_layers[0], item) is None
        for item in captured.target_layers[1:]
    )


def test_masked_tail_variant_changes_only_masked_ids_for_mask_respecting_model() -> None:
    target = _case("tail", tokens=8)
    batch = compose_batch(
        target,
        (target, _case("other-tail", tokens=3)),
        batch_size=2,
        composition="self_repeat",
        order="forward",
        seed=5,
    )
    condition = replace(build_condition_specs(repeats=20)[0], batch_size=2)
    runtime = _capture_runtime()

    padded = runtime.capture(batch, condition).target_layers[0]
    alternate = runtime.capture(
        batch,
        replace(condition, masked_tail_variant="alternate_token"),
    ).target_layers[0]

    assert not torch.equal(padded["input_ids"], alternate["input_ids"])
    assert torch.equal(padded["attention_mask"], alternate["attention_mask"])
    for stage in set(padded) - {"input_ids"}:
        assert torch.equal(padded[stage], alternate[stage])


def test_saved_tensor_downstream_replay_isolates_normalization_and_whitening() -> None:
    projected = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [9.0, 8.0, 7.0]],
        dtype=torch.float32,
    )

    layers = capture_downstream_from_saved(
        projected,
        target_indices=(0, 1),
        normalization_device="cpu",
        whitening_mean=torch.zeros(3),
        whitening_projection=torch.tensor([[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]]),
        projection_rows=torch.tensor(
            [[1, 1], [1, -1], [-1, 1], [-1, -1], [1, 1], [1, -1], [-1, 1]],
            dtype=torch.int8,
        ),
        quantization_scale=16,
    )

    assert len(layers) == 2
    assert first_divergent_layer(layers[0], layers[1]) is None
    assert set(layers[0]) == {
        "projection_pre_norm",
        "model_post_norm",
        "runtime_post_norm",
        "cpu_centered",
        "whitening_pre_norm",
        "whitening_post_norm",
        "quantized",
        "projection_dots",
        "signature_bits",
    }


def test_dynamic_and_fixed_padding_keep_comparable_canonical_input_schema() -> None:
    target = _case("padding-target", tokens=8)
    batch = compose_batch(
        target,
        (target, _case("padding-other", tokens=3)),
        batch_size=2,
        composition="self_repeat",
        order="forward",
        seed=5,
    )
    fixed_condition = replace(build_condition_specs(repeats=20)[0], batch_size=2)
    dynamic_condition = replace(fixed_condition, padding="dynamic")
    runtime = _capture_runtime()

    fixed = runtime.capture(batch, fixed_condition)
    dynamic = runtime.capture(batch, dynamic_condition)

    assert fixed.sequence_length == 256
    assert dynamic.sequence_length < 256
    assert fixed.target_layers[0]["input_ids"].shape == (256,)
    assert dynamic.target_layers[0]["input_ids"].shape == (256,)
    assert first_divergent_layer(fixed.target_layers[0], dynamic.target_layers[0]) is None


def test_matmul_precision_and_tf32_switches_use_one_compatible_api_family() -> None:
    target = _case("precision-target", tokens=8)
    batch = compose_batch(
        target,
        (target, _case("precision-other", tokens=3)),
        batch_size=2,
        composition="self_repeat",
        order="forward",
        seed=5,
    )
    baseline = replace(build_condition_specs(repeats=20)[0], batch_size=2)
    mixed = replace(
        baseline,
        deterministic_algorithms=False,
        tf32=True,
        matmul_precision="medium",
    )
    runtime = _capture_runtime()

    runtime.capture(batch, baseline)
    captured = runtime.capture(batch, mixed)

    assert captured.effective_flags["float32_matmul_precision"] == "medium"
    assert captured.effective_flags["tf32_requested"] is True
    assert captured.effective_flags["precision_api_family"] in {"new", "legacy"}
