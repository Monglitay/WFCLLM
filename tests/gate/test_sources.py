from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import pytest

from wfcllm.gate.data import (
    GateBuildContext,
    GateDataBuilder,
    LshProbeResult,
    RewriteCandidate,
)
from wfcllm.gate.sources import (
    GateDataSplitter,
    GateSourceLoader,
    GateSourceManifest,
    GateSourceRecord,
)
from wfcllm.windowing.contracts import StatementUnit


def _source(
    source_id: str,
    *,
    family: str = "main_generation",
    repository_id: str | None = None,
    task_id: str | None = "task-1",
    function_id: str | None = "function-1",
    model_id: str | None = "model-a",
    license_id: str | None = None,
    hard_set: bool = False,
    code: str = "def f():\n    return 1\n",
) -> GateSourceRecord:
    return GateSourceRecord(
        source_family=family,
        source_id=source_id,
        code=code,
        repository_id=repository_id,
        task_id=task_id,
        function_id=function_id,
        source_model_id=model_id,
        license_id=license_id,
        contract_or_hard_set=hard_set,
    )


@pytest.mark.parametrize(
    "family",
    [
        "humaneval",
        "human_eval",
        "deployment_detection",
        "final_negative_calibration",
    ],
)
def test_final_holdout_families_are_rejected(family: str) -> None:
    loader = GateSourceLoader(())
    with pytest.raises(ValueError, match="holdout"):
        loader.load(source_family=family)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_id", "artifact-HUMAN_EVAL"),
        ("repository_id", "ＨｕｍａｎＥｖａｌ-repo"),
        ("task_id", "deployment_detection-task"),
        ("function_id", "final_negative_calibration-fn"),
        ("source_model_id", "model-human_eval"),
    ],
)
def test_all_source_identity_fields_reject_normalized_holdout_markers(
    field: str, value: str
) -> None:
    source = _source("source")
    with pytest.raises(ValueError, match="holdout"):
        GateSourceRecord(**{**source.__dict__, field: value})


def test_source_code_must_be_utf8_encodable() -> None:
    with pytest.raises(ValueError, match="UTF-8"):
        _source("source", code="\ud800")


def test_loader_supports_only_the_four_approved_source_classes() -> None:
    records = (
        _source("main"),
        _source("mbpp-t", family="mbpp_train", task_id="mbpp-1", model_id=None),
        _source("mbpp-v", family="mbpp_validation", task_id="mbpp-2", model_id=None),
        _source(
            "oss",
            family="oss_python",
            repository_id="repo-1",
            task_id=None,
            license_id="Apache-2.0",
            model_id=None,
        ),
        _source(
            "hard",
            family="parser_boundary",
            task_id="parser-case-1",
            model_id=None,
            hard_set=True,
        ),
    )
    loader = GateSourceLoader(records)

    assert loader.load(source_family="main_generation") == (records[0],)
    assert loader.load(source_family="mbpp_train") == (records[1],)
    assert loader.load(source_family="mbpp_validation") == (records[2],)
    assert loader.load(source_family="oss_python") == (records[3],)
    assert loader.load(source_family="parser_boundary") == (records[4],)


def test_oss_requires_explicit_license_and_parser_samples_are_hard_set_only() -> None:
    with pytest.raises(ValueError, match="license"):
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
        )
    with pytest.raises(ValueError, match="license"):
        _source(
            "oss-blank-license",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="   ",
        )
    with pytest.raises(ValueError, match="contract_or_hard_set"):
        _source(
            "parser",
            family="parser_boundary",
            model_id=None,
            hard_set=False,
        )


def test_manifest_requires_three_models_and_keeps_them_out_of_gate_input() -> None:
    records = tuple(
        _source(f"source-{index}", model_id=f"model-{index}")
        for index in range(3)
    ) + (
        _source(
            "oss",
            family="oss_python",
            repository_id="repo-1",
            task_id=None,
            model_id=None,
            license_id="Apache-2.0",
        ),
        _source(
            "hard",
            family="parser_boundary",
            model_id=None,
            hard_set=True,
        ),
    )
    manifest = GateSourceManifest(records)

    assert manifest.source_model_ids == ("model-0", "model-1", "model-2")
    assert manifest.formal_records == records[:4]
    assert manifest.hard_set_records == records[4:]
    assert all("source_model_id" not in row.gate_input_metadata() for row in records)
    public = json.dumps(manifest.to_dict(), sort_keys=True)
    assert "model-0" in public
    assert "def f" not in public
    assert manifest.to_dict()["split_group_count_by_family"] == {
        "main_generation": 1,
        "oss_python": 1,
    }
    assert manifest.to_dict()["oss_repository_group_count"] == 1
    oss_row = next(
        row for row in manifest.to_dict()["sources"] if row["source_id"] == "oss"
    )
    assert oss_row["code_sha256"] == hashlib.sha256(
        records[3].code.encode("utf-8")
    ).hexdigest()

    with pytest.raises(ValueError, match="three"):
        GateSourceManifest((*records[:2], records[3]))


def test_manifest_rejects_zero_oss_sources() -> None:
    records = tuple(
        _source(f"main-{index}", model_id=f"model-{index}")
        for index in range(3)
    )
    with pytest.raises(ValueError, match="OSS"):
        GateSourceManifest(records)


def test_non_main_model_ids_do_not_satisfy_three_model_requirement() -> None:
    records = (
        _source("main", model_id="main-model"),
        _source(
            "oss-a",
            family="oss_python",
            repository_id="repo-a",
            task_id=None,
            model_id="fake-model-a",
            license_id="MIT",
        ),
        _source(
            "oss-b",
            family="oss_python",
            repository_id="repo-b",
            task_id=None,
            model_id="fake-model-b",
            license_id="MIT",
        ),
    )
    with pytest.raises(ValueError, match="three.*main_generation"):
        GateSourceManifest(records)


def test_oss_repository_groups_must_be_a_primary_formal_source() -> None:
    records = tuple(
        _source(
            f"main-{index}",
            task_id=f"task-{index}",
            function_id=f"function-{index}",
            model_id=f"model-{index}",
        )
        for index in range(3)
    ) + (
        _source(
            "oss",
            family="oss_python",
            repository_id="only-one-oss-repo",
            task_id=None,
            model_id=None,
            license_id="BSD-3-Clause",
        ),
    )
    with pytest.raises(ValueError, match="primary"):
        GateSourceManifest(records)


def test_manifest_identity_binds_code_without_publishing_code() -> None:
    common = tuple(
        _source(f"main-{index}", model_id=f"model-{index}")
        for index in range(3)
    )
    first_records = common + (
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
            code="value = 1\n",
        ),
    )
    second_records = common + (replace(first_records[-1], code="value = 2\n"),)

    first = GateSourceManifest(first_records).to_dict()
    second = GateSourceManifest(second_records).to_dict()

    assert first["manifest_id"] != second["manifest_id"]
    assert "value =" not in json.dumps(first)


def test_manifest_public_rows_and_identity_ignore_input_order() -> None:
    records = tuple(
        _source(f"main-{index}", model_id=f"model-{index}")
        for index in range(3)
    ) + (
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
        ),
    )
    forward = GateSourceManifest(records).to_dict()
    reverse = GateSourceManifest(tuple(reversed(records))).to_dict()
    assert forward == reverse


def test_normalized_duplicate_model_ids_do_not_count_as_diverse() -> None:
    records = (
        _source("main-a", model_id="Model-A"),
        _source("main-b", model_id="model_a"),
        _source("main-c", model_id="MODEL A"),
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
        ),
    )
    with pytest.raises(ValueError, match="three"):
        GateSourceManifest(records)


def test_source_records_and_manifest_snapshot_inputs_and_reject_duplicate_ids() -> None:
    source = _source("source-1")
    records = [
        source,
        _source("source-2", model_id="model-b"),
        _source("source-3", model_id="model-c"),
    ]
    records.append(
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
        )
    )
    manifest = GateSourceManifest(records)
    records.clear()

    assert len(manifest.records) == 4
    with pytest.raises(FrozenInstanceError):
        source.code = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="duplicate source_id"):
        GateSourceManifest((source, source, _source("x", model_id="b")))


def test_split_key_priority_is_repository_then_task_then_function() -> None:
    repository = _source(
        "repository-source",
        repository_id="repo-1",
        task_id="task-1",
        function_id="function-1",
    )
    task = _source(
        "task-source",
        repository_id=None,
        task_id="task-1",
        function_id="function-1",
    )
    function = _source(
        "function-source",
        repository_id=None,
        task_id=None,
        function_id="function-1",
    )

    assert repository.split_group_id == "repository:repo1"
    assert task.split_group_id == "task:task1"
    assert function.split_group_id == "function:function1"


def test_split_assignment_is_exact_seed_plus_group_sha256() -> None:
    splitter = GateDataSplitter(seed="split-seed-v1")
    source = _source("source", repository_id="repository-9")
    digest = hashlib.sha256(
        ("split-seed-v1" + source.split_group_id).encode("utf-8")
    ).digest()
    fraction = int.from_bytes(digest, "big") / (1 << 256)
    expected = "train" if fraction < 0.8 else "validation" if fraction < 0.9 else "test"

    assert splitter.assign(source) == expected
    assert splitter.assign(source) == expected


def test_all_context_and_budget_variants_stay_in_one_split() -> None:
    unit = StatementUnit(
        unit_id="unit-0",
        node_type="expression_statement",
        text="value = source",
        start_byte=0,
        end_byte=14,
        start_line=1,
        end_line=1,
        depth=1,
        parent_path=("module", "function_definition", "block"),
        direct_parent_type="block",
        direct_child_ordinal=0,
        eligible=True,
        hard_boundary=False,
        compound_header=False,
    )

    class Rewriter:
        def rewrite(self, request, *, candidate_index):
            return RewriteCandidate(
                code=f"value = source_{candidate_index}",
                parse_status="ok",
                unit_count=1,
                same_parent_scope=True,
                boundary_span=(0, 14),
                generation_seed_id=f"seed:{candidate_index}",
                rewrite_config_id="rewrite-v1",
            )

    class Probe:
        def probe(self, *, window_text, parent_descriptor, key_ids):
            return {
                key_id: LshProbeResult(
                    signature=(1, 0),
                    margin=0.2,
                    hit=False,
                    stable=True,
                    stable_across_precision_modes=True,
                    stable_across_batch_modes=True,
                )
                for key_id in key_ids
            }

    group = GateDataBuilder(
        rewriter=Rewriter(), lsh_probe=Probe()
    ).build(
        [unit],
        context=GateBuildContext(
            repository_id="repo-1",
            function_id="function-1",
        ),
    )[0]
    splitter = GateDataSplitter(seed="fixed")
    variants = group.expand_contexts_and_budgets()

    assert len({splitter.assign(item) for item in variants}) == 1


def test_same_repository_or_function_never_crosses_splits() -> None:
    splitter = GateDataSplitter(seed="fixed")
    records = (
        _source("a", repository_id="repo", task_id="a", function_id="fa"),
        _source("b", repository_id="repo", task_id="b", function_id="fb"),
        _source("c", repository_id=None, task_id=None, function_id="shared"),
        _source("d", repository_id=None, task_id=None, function_id="shared"),
    )
    assignments = splitter.assign_all(records)

    for split_names in assignments.by_repository_group().values():
        assert len(split_names) == 1
    assert assignments["a"] == assignments["b"]
    assert assignments["c"] == assignments["d"]


def test_splitter_rejects_duplicate_item_ids_and_invalid_seed() -> None:
    with pytest.raises(ValueError, match="seed"):
        GateDataSplitter(seed="")
    splitter = GateDataSplitter(seed="fixed")
    source = _source("duplicate")
    with pytest.raises(ValueError, match="duplicate"):
        splitter.assign_all((source, replace(source, function_id="other")))


def test_splitter_recomputes_identity_and_rejects_malicious_report() -> None:
    class Malicious:
        item_id = "malicious"
        repository_id = "shared-repository"
        task_id = "task"
        function_id = "function"
        split_group_id = "task:attacker-chosen"

    with pytest.raises(ValueError, match="split_group_id"):
        GateDataSplitter(seed="fixed").assign(Malicious())


def test_splitter_snapshots_stateful_identity_properties_once() -> None:
    class Stateful:
        item_id = "stateful"

        def __init__(self) -> None:
            self.reads = {"repository": 0, "task": 0, "function": 0, "split": 0}

        @property
        def repository_id(self):
            self.reads["repository"] += 1
            return "repo-a" if self.reads["repository"] == 1 else "repo-b"

        @property
        def task_id(self):
            self.reads["task"] += 1
            return "task-a"

        @property
        def function_id(self):
            self.reads["function"] += 1
            return "function-a"

        @property
        def split_group_id(self):
            self.reads["split"] += 1
            return "repository:repoa"

    item = Stateful()
    assignments = GateDataSplitter(seed="fixed").assign_all((item,))
    assert assignments["stateful"] in {"train", "validation", "test"}
    assert item.reads == {
        "repository": 1,
        "task": 1,
        "function": 1,
        "split": 1,
    }


def test_split_assignments_direct_constructor_checks_invariants() -> None:
    from wfcllm.gate.sources import SplitAssignments

    with pytest.raises(ValueError, match="same item IDs"):
        SplitAssignments({"a": "train"}, {"b": "repository:r"})
    with pytest.raises(ValueError, match="split"):
        SplitAssignments({"a": "dev"}, {"a": "repository:r"})


def test_source_identity_fields_reject_surrounding_whitespace() -> None:
    with pytest.raises(ValueError, match="whitespace"):
        _source(" source ")
    with pytest.raises(ValueError, match="letters or digits"):
        _source("---")


def test_loader_and_manifest_reject_normalized_duplicate_source_ids() -> None:
    first = _source("Source-A", model_id="model-a")
    duplicate = _source("source_a", model_id="model-b")
    with pytest.raises(ValueError, match="duplicate source_id"):
        GateSourceLoader((first, duplicate))

    records = (
        first,
        duplicate,
        _source("main-c", model_id="model-c"),
        _source(
            "oss",
            family="oss_python",
            repository_id="repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
        ),
    )
    with pytest.raises(ValueError, match="duplicate source_id"):
        GateSourceManifest(records)


@pytest.mark.parametrize(
    ("field", "first_value", "alias_value"),
    [
        ("repository_id", "repo-a", "REPO_A"),
        ("repository_id", "repo-a", "ＲＥＰＯＡ"),
        ("task_id", "task-a", "TASK_A"),
        ("function_id", "function-a", "FUNCTION_A"),
    ],
)
def test_loader_and_manifest_reject_group_identity_aliases(
    field: str,
    first_value: str,
    alias_value: str,
) -> None:
    first = _source("first", model_id="model-a")
    second = _source("second", model_id="model-b")
    first = replace(first, **{field: first_value})
    second = replace(second, **{field: alias_value})
    with pytest.raises(ValueError, match="alias"):
        GateSourceLoader((first, second))

    records = (
        first,
        second,
        _source("third", model_id="model-c"),
        _source(
            "oss",
            family="oss_python",
            repository_id="oss-repo",
            task_id=None,
            model_id=None,
            license_id="MIT",
        ),
    )
    with pytest.raises(ValueError, match="alias"):
        GateSourceManifest(records)


def test_exact_same_repository_across_functions_is_allowed_and_canonical() -> None:
    first = _source(
        "first",
        repository_id="Repo-A",
        task_id="task-a",
        function_id="function-a",
    )
    second = _source(
        "second",
        repository_id="Repo-A",
        task_id="task-b",
        function_id="function-b",
        model_id="model-b",
    )
    loader = GateSourceLoader((first, second))
    assert len(loader.load(source_family="main_generation")) == 2
    assert first.split_group_id == second.split_group_id == "repository:repoa"


def test_group_alias_spellings_cannot_cross_splits_when_considered_separately() -> None:
    first = _source("first", repository_id="repo-a")
    alias = _source("alias", repository_id="REPO_A", model_id="model-b")
    splitter = GateDataSplitter(seed="fixed")
    assert first.split_group_id == alias.split_group_id == "repository:repoa"
    assert splitter.assign(first) == splitter.assign(alias)
