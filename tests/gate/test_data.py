from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json
import math

import pytest

from wfcllm.gate.data import (
    GateBuildContext,
    GateDataBuilder,
    LshProbeResult,
    RewriteCandidate,
    StructuralBoundary,
    validate_key_blind_payload,
)
from wfcllm.gate.labels import LabelThresholds, build_gate_labels
from wfcllm.gate.schema import CandidateObservation
from wfcllm.windowing.contracts import StatementUnit


def _unit(
    index: int,
    *,
    parent: tuple[str, ...] = ("module", "function_definition", "block"),
    depth: int = 1,
    eligible: bool = True,
    hard_boundary: bool = False,
    compound_header: bool = False,
) -> StatementUnit:
    start = index * 20
    return StatementUnit(
        unit_id=f"unit-{index}",
        node_type="if_statement" if compound_header else "expression_statement",
        text=f"value_{index} = source" if not compound_header else "if condition:",
        start_byte=start,
        end_byte=start + 16,
        start_line=index + 1,
        end_line=index + 1,
        depth=depth,
        parent_path=parent,
        direct_parent_type=parent[-1],
        direct_child_ordinal=index,
        eligible=eligible,
        hard_boundary=hard_boundary,
        compound_header=compound_header,
    )


class RecordingRewriter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int, dict[str, object]]] = []

    def rewrite(self, request, *, candidate_index: int) -> RewriteCandidate:
        payload = request.to_dict()
        self.calls.append(
            (
                request.window_start_unit_id,
                request.window_length,
                candidate_index,
                payload,
            )
        )
        return RewriteCandidate(
            code=f"rewritten_{request.window_length}_{candidate_index} = source",
            parse_status="ok",
            unit_count=request.window_length,
            same_parent_scope=True,
            boundary_span=(
                request.structural_boundary.start_byte,
                request.structural_boundary.end_byte,
            ),
            generation_seed_id=f"seed-v1:{candidate_index}",
            rewrite_config_id="rewrite-v1",
        )


def _complete_semantic_probe(
    trajectory: tuple[CandidateObservation, ...],
) -> tuple[CandidateObservation, ...]:
    completed = []
    for candidate in trajectory:
        structurally_usable = (
            candidate.parse_status == "ok"
            and candidate.same_parent_scope
            and candidate.unit_count in {1, 2, 3}
        )
        completed.append(
            replace(
                candidate,
                semantic_probe_pending=False,
                semantic_reference_cosine=(
                    1.0 if candidate.candidate_index == 0 else 0.95
                ) if structurally_usable else None,
                semantic_preservation_passed=(
                    True if structurally_usable else None
                ),
            )
        )
    return tuple(completed)


class RecordingProbe:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []

    def probe(self, *, window_text, parent_descriptor, key_ids):
        self.calls.append((window_text, parent_descriptor, key_ids))
        return {
            key_id: LshProbeResult(
                signature=(1, 0, 1, 0),
                margin=0.25,
                hit=True,
                stable=True,
                stable_across_precision_modes=True,
                stable_across_batch_modes=True,
            )
            for key_id in key_ids
        }


def _builder() -> tuple[GateDataBuilder, RecordingRewriter, RecordingProbe]:
    rewriter = RecordingRewriter()
    probe = RecordingProbe()
    return (
        GateDataBuilder(
            rewriter=rewriter,
            lsh_probe=probe,
            key_ids=("train-key-000", "train-key-001"),
            test_only_allow_key_subset=True,
        ),
        rewriter,
        probe,
    )


def test_w1_w2_w3_each_get_candidate_zero_plus_three_ordered_rewrites() -> None:
    builder, rewriter, _ = _builder()
    groups = builder.build([_unit(0), _unit(1), _unit(2)])

    first = groups[0]
    assert first.candidate_window_lengths == (1, 2, 3)
    for length in (1, 2, 3):
        assert [
            item.candidate_index
            for item in first.candidates_by_length[str(length)]
        ] == list(range(4))
        assert first.candidates_by_length[str(length)][0].code == "\n".join(
            unit.text for unit in [_unit(0), _unit(1), _unit(2)][:length]
        )
    assert [call[2] for call in rewriter.calls[:9]] == list(range(1, 4)) * 3
    assert all(call[0] == first.window_start_unit_id for call in rewriter.calls[:9])
    assert len(rewriter.calls) == (3 + 2 + 1) * 3


def test_builder_rewrites_only_explicitly_selected_window_starts() -> None:
    builder, rewriter, _ = _builder()

    groups = builder.build(
        tuple(_unit(index) for index in range(5)),
        selected_start_unit_ids=("unit-2",),
    )

    assert [group.window_start_unit_id for group in groups] == ["unit-2"]
    assert len(rewriter.calls) == 3 * 3


def test_formal_builder_default_uses_all_32_canonical_training_key_ids() -> None:
    probe = RecordingProbe()
    GateDataBuilder(
        rewriter=RecordingRewriter(),
        lsh_probe=probe,
    ).build([_unit(0)])

    assert probe.calls[0][2] == tuple(
        f"train-key-{index:03d}" for index in range(32)
    )


def test_rewrite_request_is_key_blind_and_contains_only_public_context() -> None:
    builder, rewriter, _ = _builder()
    builder.build(
        [_unit(0), _unit(1), _unit(2)],
        context=GateBuildContext(
            prompt="complete the function",
            source_id="source-1",
            source_family="main_generation",
            repository_id="repository-1",
            task_id="task-1",
            function_id="function-1",
        ),
    )

    payload = rewriter.calls[0][3]
    assert set(payload) == {
        "prompt",
        "completed_prefix",
        "original_window",
        "canonical_parent",
        "window_start_unit_id",
        "window_length",
        "structural_boundary",
    }
    serialized = json.dumps(payload, sort_keys=True).casefold()
    for forbidden in (
        "secret",
        "training_key",
        "deployment_key",
        "lsh",
        "target_region",
        "first_hit",
    ):
        assert forbidden not in serialized


def test_rewrite_request_descriptor_drops_repeated_direct_parent() -> None:
    builder, rewriter, _ = _builder()
    unit = _unit(
        0,
        parent=("program", "if_statement", "if_statement", "if_statement"),
        depth=3,
    )

    builder.build([unit])

    assert rewriter.calls[0][3]["canonical_parent"] == (
        "python-statement-window/v1|program|parent=if_statement|ordinal=0|role=body"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": [{"secret": "x"}]},
        {"metadata": {"training_key": "x"}},
        {"metadata": {"deployment-key": "x"}},
        {"metadata": {"lsh_bits": [1, 0]}},
        {"metadata": {"target_region": "x"}},
        {"metadata": {"first_hit_index": 2}},
        {"metadata": {"passed": True}},
        {"metadata": {"key": "x"}},
        {"metadata": {"key_id": "x"}},
        {"metadata": {"keyId": "x"}},
        {"metadata": {"keys": ["x"]}},
        {"metadata": {"apiKey": "x"}},
        {"metadata": {"secretKey": "x"}},
        {"metadata": {"keyMaterial": "x"}},
        {"metadata": {"privateKey": "x"}},
        {"metadata": {"ＰＡＳＳ": True}},
    ],
)
def test_recursive_key_and_quality_leak_scanner_rejects_nested_fields(
    payload: object,
) -> None:
    with pytest.raises(ValueError, match="forbidden"):
        validate_key_blind_payload(payload)


def test_leak_scanner_does_not_reject_unrelated_monkey_field() -> None:
    validate_key_blind_payload({"monkey": "public"})


def test_context_and_budget_views_reuse_one_immutable_candidate_trajectory() -> None:
    builder, rewriter, _ = _builder()
    group = builder.build([_unit(0), _unit(1), _unit(2)])[0]
    calls_before = len(rewriter.calls)

    variants = group.expand_contexts_and_budgets()

    assert len(variants) == 6
    assert {(item.context_length, item.rewrite_budget) for item in variants} == {
        (context_length, budget)
        for context_length in (1, 2, 3)
        for budget in (1, 3)
    }
    assert all(
        item.candidates_by_length is group.candidates_by_length for item in variants
    )
    assert len(rewriter.calls) == calls_before
    with pytest.raises(TypeError):
        group.candidates_by_length["1"] = ()  # type: ignore[index]


def test_structurally_invalid_rewrite_is_never_probed() -> None:
    class InvalidSecondRewrite(RecordingRewriter):
        def rewrite(self, request, *, candidate_index: int) -> RewriteCandidate:
            candidate = super().rewrite(request, candidate_index=candidate_index)
            if candidate_index == 2:
                return replace(
                    candidate,
                    parse_status="scope_changed",
                    same_parent_scope=False,
                )
            return candidate

    rewriter = InvalidSecondRewrite()
    probe = RecordingProbe()
    builder = GateDataBuilder(
        rewriter=rewriter,
        lsh_probe=probe,
        key_ids=("train-key-000",),
        test_only_allow_key_subset=True,
    )

    group = builder.build([_unit(0)])[0]

    assert len(probe.calls) == 3  # candidate 0 plus two valid rewrites
    invalid = group.candidates_by_length["1"][2]
    assert invalid.lsh_by_key_id == {}
    assert invalid.parse_status == "scope_changed"
    assert invalid.lsh_signature is None


def test_empty_parse_error_candidate_is_recorded_without_probe() -> None:
    class EmptyFirstRewrite(RecordingRewriter):
        def rewrite(self, request, *, candidate_index: int) -> RewriteCandidate:
            if candidate_index == 1:
                return RewriteCandidate(
                    code="",
                    parse_status="parse_error",
                    unit_count=0,
                    same_parent_scope=False,
                    boundary_span=(0, 0),
                    generation_seed_id="seed-v1:1",
                    rewrite_config_id="rewrite-v1",
                )
            return super().rewrite(request, candidate_index=candidate_index)

    probe = RecordingProbe()
    group = GateDataBuilder(
        rewriter=EmptyFirstRewrite(),
        lsh_probe=probe,
        key_ids=("train-key-000",),
        test_only_allow_key_subset=True,
    ).build([_unit(0)])[0]

    assert len(probe.calls) == 3
    assert group.candidates_by_length["1"][1].code == ""
    assert group.candidates_by_length["1"][1].lsh_by_key_id == {}


def test_reliable_hit_requires_all_modes_and_margin() -> None:
    class MixedProbe(RecordingProbe):
        def probe(self, *, window_text, parent_descriptor, key_ids):
            return {
                key_ids[0]: LshProbeResult(
                    signature=(1, 1),
                    margin=0.19,
                    hit=True,
                    stable=True,
                    stable_across_precision_modes=True,
                    stable_across_batch_modes=True,
                ),
                key_ids[1]: LshProbeResult(
                    signature=(1, 1),
                    margin=0.9,
                    hit=True,
                    stable=True,
                    stable_across_precision_modes=False,
                    stable_across_batch_modes=True,
                ),
            }

    group = GateDataBuilder(
        rewriter=RecordingRewriter(),
        lsh_probe=MixedProbe(),
        key_ids=("train-key-000", "train-key-001"),
        test_only_allow_key_subset=True,
    ).build([_unit(0)])[0]
    observation = group.candidates_by_length["1"][0]

    assert observation.lsh_by_key_id["train-key-000"] == {
        "hit": True,
        "margin": 0.19,
        "stable": True,
    }
    assert observation.lsh_by_key_id["train-key-001"] == {
        "hit": True,
        "margin": 0.9,
        "stable": True,
    }
    assert observation.stable_across_precision_modes is False
    assert observation.stable_across_batch_modes is True


def test_probe_result_preserves_signature_and_is_frozen() -> None:
    result = LshProbeResult(
        signature=(1, 0, 1),
        margin=0.2,
        hit=True,
        stable=True,
        stable_across_precision_modes=True,
        stable_across_batch_modes=True,
    )
    assert result.signature == (1, 0, 1)
    with pytest.raises(FrozenInstanceError):
        result.margin = 0.9  # type: ignore[misc]
    assert result.is_reliable_hit(configured_margin=0.2) is True
    for margin in (True, -0.1, math.nan, math.inf):
        with pytest.raises(ValueError, match="configured_margin"):
            result.is_reliable_hit(configured_margin=margin)  # type: ignore[arg-type]


def test_builder_persists_raw_probe_facts_for_margin_relabeling() -> None:
    class MarginProbe(RecordingProbe):
        def probe(self, *, window_text, parent_descriptor, key_ids):
            return {
                key_id: LshProbeResult(
                    signature=(1, 0),
                    margin=0.19,
                    hit=True,
                    stable=True,
                    stable_across_precision_modes=True,
                    stable_across_batch_modes=True,
                )
                for key_id in key_ids
            }

    group = GateDataBuilder(
        rewriter=RecordingRewriter(),
        lsh_probe=MarginProbe(),
    ).build([_unit(0)])[0]
    trajectory = _complete_semantic_probe(group.candidates_by_length["1"])

    low = build_gate_labels(
        trajectory,
        training_key_count=32,
        thresholds=LabelThresholds(configured_margin=0.19),
    )
    high = build_gate_labels(
        trajectory,
        training_key_count=32,
        thresholds=LabelThresholds(configured_margin=0.20),
    )

    assert low.budgets[3].success_rate == 1.0
    assert high.budgets[3].success_rate == 0.0
    assert trajectory[0].lsh_by_key_id["train-key-000"]["hit"] is True
    assert trajectory[0].lsh_by_key_id["train-key-000"]["margin"] == 0.19


def test_builder_fails_closed_when_per_key_signatures_disagree() -> None:
    class DisagreeingProbe(RecordingProbe):
        def probe(self, *, window_text, parent_descriptor, key_ids):
            return {
                key_id: LshProbeResult(
                    signature=(index, 0),
                    margin=0.5,
                    hit=False,
                    stable=True,
                    stable_across_precision_modes=True,
                    stable_across_batch_modes=True,
                )
                for index, key_id in enumerate(key_ids)
            }

    with pytest.raises(ValueError, match="canonical signature"):
        GateDataBuilder(
            rewriter=RecordingRewriter(),
            lsh_probe=DisagreeingProbe(),
            key_ids=("train-key-000", "train-key-001"),
            test_only_allow_key_subset=True,
        ).build([_unit(0)])


def test_builder_invalid_rewrite_integrates_with_label_generation() -> None:
    class OneParseError(RecordingRewriter):
        def rewrite(self, request, *, candidate_index: int) -> RewriteCandidate:
            if candidate_index == 2:
                return RewriteCandidate(
                    code="",
                    parse_status="parse_error",
                    unit_count=0,
                    same_parent_scope=False,
                    boundary_span=(0, 0),
                    generation_seed_id="seed-v1:2",
                    rewrite_config_id="rewrite-v1",
                )
            return super().rewrite(request, candidate_index=candidate_index)

    group = GateDataBuilder(
        rewriter=OneParseError(),
        lsh_probe=RecordingProbe(),
    ).build([_unit(0)])[0]

    labels = build_gate_labels(
        _complete_semantic_probe(group.candidates_by_length["1"]),
        training_key_count=32,
    )

    assert labels.budgets[3].structural_valid_rate == 2 / 3
    assert group.candidates_by_length["1"][2].lsh_by_key_id == {}
    assert group.candidates_by_length["1"][2].lsh_signature is None


def test_window_enumeration_stops_at_compound_parent_and_hard_boundaries() -> None:
    builder, _, _ = _builder()
    compound = _unit(1, compound_header=True)
    changed_parent = _unit(
        2,
        parent=("module", "function_definition", "block", "if_statement", "block"),
        depth=2,
    )
    hard = _unit(3, eligible=False, hard_boundary=True)

    groups = builder.build([_unit(0), compound, changed_parent, hard, _unit(4)])

    assert [group.window_start_unit_id for group in groups] == [
        "unit-0",
        "unit-1",
        "unit-2",
        "unit-4",
    ]
    assert groups[0].candidate_window_lengths == (1,)
    assert groups[1].candidate_window_lengths == (1,)
    assert groups[2].candidate_window_lengths == (1,)


def test_source_text_preserves_exact_utf8_crlf_comments_and_blank_lines() -> None:
    prefix = "说明 = '前缀'\r\n# 注释\r\n"
    first_text = "value = '你'"
    gap = "\r\n\r\n# 保留窗口注释\r\n"
    second_text = "next_value = value"
    source = prefix + first_text + gap + second_text + "\r\n"
    source_bytes = source.encode("utf-8")
    first_start = len(prefix.encode("utf-8"))
    first_end = first_start + len(first_text.encode("utf-8"))
    second_start = first_end + len(gap.encode("utf-8"))
    second_end = second_start + len(second_text.encode("utf-8"))
    first = replace(
        _unit(0),
        text=first_text,
        start_byte=first_start,
        end_byte=first_end,
        start_line=3,
        end_line=3,
    )
    second = replace(
        _unit(1),
        text=second_text,
        start_byte=second_start,
        end_byte=second_end,
        start_line=6,
        end_line=6,
    )
    builder, rewriter, _ = _builder()

    group = builder.build([first, second], source_text=source)[0]
    request_payloads = [call[3] for call in rewriter.calls]
    w2_payload = next(item for item in request_payloads if item["window_length"] == 2)

    assert w2_payload["completed_prefix"] == prefix
    assert w2_payload["original_window"] == source_bytes[
        first_start:second_end
    ].decode("utf-8")
    assert "source_text" not in w2_payload
    assert group.candidates_by_length["2"][0].code == (
        first_text + gap + second_text
    )


def test_source_text_none_retains_unit_join_fallback() -> None:
    builder, rewriter, _ = _builder()
    builder.build([_unit(0), _unit(1)], source_text=None)
    w2_payload = next(
        call[3] for call in rewriter.calls if call[1] == 2
    )
    assert w2_payload["original_window"] == "value_0 = source\nvalue_1 = source"


def test_builder_rejects_overlapping_unit_byte_spans() -> None:
    builder, _, _ = _builder()
    overlapping = replace(_unit(1), start_byte=_unit(0).end_byte - 1)
    with pytest.raises(ValueError, match="overlap"):
        builder.build([_unit(0), overlapping])


@pytest.mark.parametrize(
    "source_text",
    ["short", "\ud800 invalid surrogate"],
)
def test_source_text_rejects_invalid_byte_spans_and_utf8(source_text: str) -> None:
    builder, _, _ = _builder()
    with pytest.raises(ValueError, match="source_text|UTF-8|byte span"):
        builder.build([_unit(0)], source_text=source_text)


def test_source_text_rejects_a_span_inside_a_multibyte_character() -> None:
    builder, _, _ = _builder()
    unit = replace(
        _unit(0),
        text="invalid",
        start_byte=1,
        end_byte=3,
    )
    with pytest.raises(ValueError, match="UTF-8"):
        builder.build([unit], source_text="你")


def test_builder_rejects_duplicate_units_deployment_ids_and_mutable_aliases() -> None:
    builder, _, _ = _builder()
    with pytest.raises(ValueError, match="duplicate unit_id"):
        builder.build([_unit(0), _unit(0)])
    with pytest.raises(ValueError, match="deployment"):
        GateDataBuilder(
            rewriter=RecordingRewriter(),
            lsh_probe=RecordingProbe(),
            key_ids=("deployment-key-000",),
        )
    with pytest.raises(ValueError, match="duplicate"):
        GateDataBuilder(
            rewriter=RecordingRewriter(),
            lsh_probe=RecordingProbe(),
            key_ids=("train-key-000", "train-key-000"),
        )

    boundary = StructuralBoundary(
        start_byte=0,
        end_byte=16,
        depth=1,
        direct_parent_type="block",
        unit_ids=("unit-0",),
        compound_singleton=False,
        hard_boundary_after=True,
    )
    with pytest.raises(FrozenInstanceError):
        boundary.depth = 9  # type: ignore[misc]


def test_group_id_binds_function_identity_when_unit_ids_repeat() -> None:
    builder, _, _ = _builder()
    first = builder.build(
        [_unit(0)], context=GateBuildContext(function_id="function-a")
    )[0]
    second = builder.build(
        [_unit(0)], context=GateBuildContext(function_id="function-b")
    )[0]

    assert first.group_id != second.group_id


@pytest.mark.parametrize(
    "context_change",
    [
        {"prompt": "different prompt"},
        {"source_family": "mbpp_train"},
        {"language": "python-variant"},
        {"parser_contract_version": "parser/v2"},
    ],
)
def test_group_id_binds_all_trajectory_context_fields(
    context_change: dict[str, str],
) -> None:
    builder, _, _ = _builder()
    baseline_context = GateBuildContext(function_id="function-a")
    baseline = builder.build([_unit(0)], context=baseline_context)[0]
    changed = builder.build(
        [_unit(0)],
        context=replace(baseline_context, **context_change),
    )[0]
    assert baseline.group_id != changed.group_id


def test_group_id_binds_source_comments_outside_the_window() -> None:
    unit = replace(_unit(0), start_byte=10, end_byte=26)
    first_source = "# first  \n" + unit.text
    second_source = "# second \n" + unit.text
    builder, _, _ = _builder()

    first = builder.build([unit], source_text=first_source)[0]
    second = builder.build([unit], source_text=second_source)[0]

    assert first.group_id != second.group_id


def test_group_id_canonical_json_has_no_nul_field_collision() -> None:
    builder, _, _ = _builder()
    first = builder.build(
        [_unit(0)],
        context=GateBuildContext(
            source_id="a\0b",
            task_id="c",
            function_id=None,
        ),
    )[0]
    second = builder.build(
        [_unit(0)],
        context=GateBuildContext(
            source_id="a",
            task_id="b\0c",
            function_id=None,
        ),
    )[0]
    assert first.group_id != second.group_id


def test_gate_data_group_rejects_wrapper_identity_mismatch() -> None:
    builder, _, _ = _builder()
    group = builder.build(
        [_unit(0)], context=GateBuildContext(function_id="function-a")
    )[0]
    with pytest.raises(ValueError, match="repository_group"):
        replace(group, repository_id="attacker-repository")


def test_variant_numeric_fields_reject_bool() -> None:
    builder, _, _ = _builder()
    variant = builder.build([_unit(0)])[0].expand_contexts_and_budgets()[0]
    with pytest.raises(ValueError, match="context_length"):
        replace(variant, context_length=True)
    with pytest.raises(ValueError, match="rewrite_budget"):
        replace(variant, rewrite_budget=True)


def test_key_modes_are_explicit_and_production_subsets_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly 32"):
        GateDataBuilder(
            rewriter=RecordingRewriter(),
            lsh_probe=RecordingProbe(),
            key_ids=("train-key-000",),
        )
    with pytest.raises(ValueError, match="mixed|mode"):
        GateDataBuilder(
            rewriter=RecordingRewriter(),
            lsh_probe=RecordingProbe(),
            key_ids=("train-key-000", "holdout-key-000"),
            test_only_allow_key_subset=True,
        )

    training_subset = GateDataBuilder(
        rewriter=RecordingRewriter(),
        lsh_probe=RecordingProbe(),
        key_ids=("train-key-000",),
        test_only_allow_key_subset=True,
    )
    assert training_subset.build([_unit(0)])

    holdout_probe = RecordingProbe()
    GateDataBuilder(
        rewriter=RecordingRewriter(),
        lsh_probe=holdout_probe,
        key_mode="holdout",
    ).build([_unit(0)])
    assert holdout_probe.calls[0][2] == tuple(
        f"holdout-key-{index:03d}" for index in range(8)
    )


def test_structurally_usable_rewrite_requires_nonempty_boundary_span() -> None:
    with pytest.raises(ValueError, match="boundary_span"):
        RewriteCandidate(
            code="value = source",
            parse_status="ok",
            unit_count=1,
            same_parent_scope=True,
            boundary_span=(0, 0),
            generation_seed_id="seed:1",
            rewrite_config_id="rewrite-v1",
        )


@pytest.mark.parametrize("code", ["", "   ", "\t\r\n"])
def test_structurally_usable_rewrite_requires_nonblank_code(code: str) -> None:
    with pytest.raises(ValueError, match="code"):
        RewriteCandidate(
            code=code,
            parse_status="ok",
            unit_count=1,
            same_parent_scope=True,
            boundary_span=(0, 1),
            generation_seed_id="seed:1",
            rewrite_config_id="rewrite-v1",
        )


@pytest.mark.parametrize(
    "source_family",
    [
        "main_generation",
        "mbpp_train",
        "mbpp_validation",
        "oss_python",
        "parser_boundary",
    ],
)
def test_build_context_accepts_all_approved_source_families(
    source_family: str,
) -> None:
    assert GateBuildContext(source_family=source_family).source_family == source_family


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_family", "humaneval"),
        ("source_id", "artifact-HUMAN_EVAL"),
        ("repository_id", "ＨｕｍａｎＥｖａｌ-repo"),
        ("task_id", "deployment_detection-task"),
        ("function_id", "final_negative_calibration-fn"),
    ],
)
def test_build_context_uses_shared_source_holdout_policy(
    field: str,
    value: str,
) -> None:
    context = GateBuildContext()
    with pytest.raises(ValueError, match="holdout|approved"):
        replace(context, **{field: value})


@pytest.mark.parametrize("units", [None, "not-units", [], [_unit(0), object()]])
def test_builder_rejects_invalid_unit_containers(units: object) -> None:
    builder, _, _ = _builder()
    with pytest.raises(ValueError, match="units"):
        builder.build(units)  # type: ignore[arg-type]
