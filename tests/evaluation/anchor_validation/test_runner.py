from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch

import wfcllm.evaluation.anchor_validation.runner as runner_module
from wfcllm.evaluation.anchor_validation.io import write_candidate_contexts
from wfcllm.evaluation.anchor_validation.runner import (
    AnchorValidationConfig,
    AnchorValidationRunner,
)
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext


class _FixedProvider:
    embed_dim = 2

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> torch.Tensor:
        self.calls.append(text)
        values = {
            "valid-a": torch.tensor([1.0, 1.0]),
            "valid-b": torch.tensor([1.0, 1.0]),
            "invalid-a": torch.tensor([-1.0, 1.0]),
            "invalid-b": torch.tensor([-1.0, 1.0]),
        }
        return values.get(text, torch.tensor([0.0, 1.0]))


class _FixedLSHSpace:
    def __init__(self, secret_key: str, embed_dim: int, d: int) -> None:
        self.planes = torch.tensor([[1.0, 0.0]])


class _FixedKeying:
    def __init__(self, secret_key: str, d: int) -> None:
        pass

    def derive(
        self,
        parent_node_type: str,
        k: int,
        ordinal: int | None = None,
    ) -> frozenset[tuple[int, ...]]:
        return frozenset({(1,)})


def test_runner_rejects_invalid_primary_config_before_loading_pool(tmp_path, monkeypatch):
    output_dir = tmp_path / "out"
    load_calls = []
    monkeypatch.setattr(
        runner_module,
        "load_candidate_contexts",
        lambda path: load_calls.append(path),
    )

    runner = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=tmp_path / "missing.jsonl",
            output_dir=output_dir,
            secret_keys=("k0",),
            gammas=(0.5,),
            methods=("vanilla", "random", "seqmark_oracle"),
            retry_budgets=(1,),
            primary_method="codet5_comment_anchor",
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    )

    with pytest.raises(ValueError, match="primary_method.*is not present"):
        runner.run()

    assert load_calls == []
    assert not output_dir.exists()


def test_runner_writes_metrics_simulation_and_summary(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                context_before="def f(x):\n",
                context_after="",
                masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
                import_and_helper_signatures=(),
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                ),
            )
        ],
    )

    runner = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("k0", "k1"),
            gammas=(0.5,),
            methods=("vanilla", "random", "slot_context", "seqmark_oracle"),
            retry_budgets=(1, 2),
            primary_method="slot_context",
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    )

    result = runner.run()

    assert result.metrics_path.exists()
    assert result.selection_path.exists()
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["context_count"] == 1
    assert "go_no_go" in summary
    assert "pool_quality" in summary["summary"]
    assert "oracle_gap" in summary["go_no_go"]["evidence"]
    assert "gamma_calibration" in summary["go_no_go"]["evidence"]
    assert result.anchor_text_debug_path.exists()
    assert result.anchor_diagnostics_path.exists()
    debug_rows = [
        json.loads(line)
        for line in result.anchor_text_debug_path.read_text(encoding="utf-8").splitlines()
    ]
    assert debug_rows
    slot_row = next(row for row in debug_rows if row["method"] == "slot_context")
    assert slot_row["context_id"] == "ctx"
    assert slot_row["key_id"] == "key-00"
    assert slot_row["anchor_text"]
    assert slot_row["tokenized_length"] is None
    assert slot_row["first_token_ids"] == []
    assert "embedding_norm" in slot_row
    assert "cosine_distribution" in slot_row
    assert "entropy_by_gamma" in slot_row
    assert "valid_hit_balance_by_gamma" in slot_row
    assert "k0" not in result.anchor_text_debug_path.read_text(encoding="utf-8")
    diagnostics = json.loads(result.anchor_diagnostics_path.read_text(encoding="utf-8"))
    assert "by_method" in diagnostics
    assert "by_node_type" in diagnostics
    assert "by_candidate_count_bucket" in diagnostics
    assert "by_block_ordinal_bucket" in diagnostics
    assert "seqmark_oracle_minus_method_largest" in diagnostics["top_contexts"]


def test_runner_supports_role_aware_methods(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                context_before="def f(x):\n",
                context_after="",
                masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
                import_and_helper_signatures=("import math",),
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                ),
            )
        ],
    )

    result = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("k0",),
            gammas=(0.5,),
            methods=(
                "vanilla",
                "random",
                "role_aware_slot_context",
                "role_aware_slot_context_skeleton",
                "seqmark_oracle",
            ),
            retry_budgets=(1,),
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    ).run()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["methods"] == [
        "vanilla",
        "random",
        "role_aware_slot_context",
        "role_aware_slot_context_skeleton",
        "seqmark_oracle",
    ]


def test_runner_writes_primary_method_diagnostic_evidence(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                context_before="def f(x):\n",
                context_after="",
                masked_parent_context="def f(<NAME>):\n    <TARGET_BLOCK>",
                import_and_helper_signatures=("import math",),
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                ),
            )
        ],
    )

    result = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("k0",),
            gammas=(0.5,),
            methods=("vanilla", "random", "codet5_comment_anchor", "seqmark_oracle"),
            primary_method="codet5_comment_anchor",
            retry_budgets=(1,),
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    ).run()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["primary_method"] == "codet5_comment_anchor"
    assert summary["go_no_go"]["primary_method"] == "codet5_comment_anchor"
    assert summary["go_no_go"]["primary_method_evidence"]["vs_vanilla"]
    assert summary["go_no_go"]["primary_method_evidence"]["minus_random"]
    assert "diagnostic_positive" in summary["go_no_go"]["primary_method_gates"]
    assert "gate_positive" in summary["go_no_go"]["primary_method_gates"]


def test_runner_supports_centroid_oracle_methods(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                    CandidateBlock("c2", "return x - 1", 2),
                    CandidateBlock("c3", "return 0", 3),
                ),
            )
        ],
    )

    result = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("secret-value",),
            gammas=(0.5,),
            methods=(
                "vanilla",
                "random",
                "candidate_centroid_oracle",
                "context_centroid_oracle",
                "seqmark_oracle",
            ),
            retry_budgets=(1,),
            primary_method="context_centroid_oracle",
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    ).run()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["methods"] == [
        "vanilla",
        "random",
        "candidate_centroid_oracle",
        "context_centroid_oracle",
        "seqmark_oracle",
    ]
    debug_rows = [
        json.loads(line)
        for line in result.anchor_text_debug_path.read_text(encoding="utf-8").splitlines()
    ]
    oracle_row = next(row for row in debug_rows if row["method"] == "candidate_centroid_oracle")
    assert oracle_row["anchor_text"] is None
    assert oracle_row["embedding_norm"] is None
    assert oracle_row["cosine_distribution"] is None
    assert "anchor_embedding_by_gamma" in oracle_row
    assert "cosine_distribution_by_gamma" in oracle_row
    assert "secret-value" not in result.anchor_text_debug_path.read_text(encoding="utf-8")


def test_runner_rejects_primary_method_without_unkeyed_row_before_writing(tmp_path):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                    CandidateBlock("c2", "return x - 1", 2),
                    CandidateBlock("c3", "return 0", 3),
                ),
            )
        ],
    )

    runner = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("secret-value",),
            gammas=(0.5,),
            methods=(
                "vanilla",
                "random",
                "candidate_centroid_oracle",
                "seqmark_oracle",
            ),
            retry_budgets=(1,),
            primary_method="candidate_centroid_oracle",
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    )

    with pytest.raises(ValueError, match="primary_method.*is not present"):
        runner.run()

    assert not (output_dir / "region_metrics.jsonl").exists()
    assert not (output_dir / "selection_simulation.jsonl").exists()
    assert not (output_dir / "anchor_text_debug.jsonl").exists()


def test_candidate_centroid_oracle_is_gamma_scoped_and_debugs_candidate_geometry(
    tmp_path,
    monkeypatch,
):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "valid-a", 0),
                    CandidateBlock("c1", "valid-b", 1),
                    CandidateBlock("c2", "invalid-a", 2),
                    CandidateBlock("c3", "invalid-b", 3),
                ),
            )
        ],
    )
    provider = _FixedProvider()
    monkeypatch.setattr(runner_module, "_build_embedding_provider", lambda config: provider)
    monkeypatch.setattr(runner_module, "LSHSpace", _FixedLSHSpace)
    monkeypatch.setattr(runner_module, "WatermarkKeying", _FixedKeying)

    result = AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("secret-value",),
            gammas=(0.5,),
            methods=(
                "vanilla",
                "random",
                "candidate_centroid_oracle",
                "context_centroid_oracle",
                "seqmark_oracle",
            ),
            retry_budgets=(1,),
            primary_method="context_centroid_oracle",
            lsh_d=1,
            embed_dim=2,
            embedding_mode="hash",
        )
    ).run()

    metrics_rows = [
        json.loads(line)
        for line in result.metrics_path.read_text(encoding="utf-8").splitlines()
    ]
    candidate_rows = [
        row for row in metrics_rows if row["method"] == "candidate_centroid_oracle"
    ]
    context_rows = [
        row for row in metrics_rows if row["method"] == "context_centroid_oracle"
    ]
    assert candidate_rows
    assert all(row["key_id"] is not None and row["gamma"] is not None for row in candidate_rows)
    assert any(row["key_id"] is None for row in context_rows)
    keyed_candidate = next(row for row in candidate_rows if row["key_id"] == "key-00")
    keyed_context = next(
        row
        for row in context_rows
        if row["key_id"] == "key-00" and row["gamma"] == keyed_candidate["gamma"]
    )
    assert keyed_candidate["normalized_entropy"] != keyed_context["normalized_entropy"]

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert "candidate_centroid_oracle" not in summary["summary"]["mean_entropy_by_method"]
    assert "candidate_centroid_oracle" not in summary["anchor_diagnostics"]["by_method"]

    debug_rows = [
        json.loads(line)
        for line in result.anchor_text_debug_path.read_text(encoding="utf-8").splitlines()
    ]
    candidate_debug = next(
        row for row in debug_rows if row["method"] == "candidate_centroid_oracle"
    )
    context_debug = next(
        row for row in debug_rows if row["method"] == "context_centroid_oracle"
    )
    gamma_key = str(keyed_candidate["gamma"]).rstrip("0").rstrip(".")
    assert candidate_debug["embedding_norm"] is None
    assert candidate_debug["cosine_distribution"] is None
    assert candidate_debug["anchor_embedding_by_gamma"][gamma_key]["source"] == (
        "valid_minus_invalid_centroid"
    )
    assert candidate_debug["anchor_embedding_by_gamma"][gamma_key]["embedding_norm"] == 2.0
    assert candidate_debug["cosine_distribution_by_gamma"][gamma_key] != (
        context_debug["cosine_distribution"]
    )


def test_runner_caches_context_only_anchor_embeddings(tmp_path, monkeypatch):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                context_before="def f(x):\n",
                context_after="",
                masked_parent_context="def f(x):\n    <TARGET_BLOCK>",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "valid-a", 0),
                    CandidateBlock("c1", "valid-b", 1),
                    CandidateBlock("c2", "invalid-a", 2),
                ),
            )
        ],
    )
    provider = _FixedProvider()
    monkeypatch.setattr(runner_module, "_build_embedding_provider", lambda config: provider)
    monkeypatch.setattr(runner_module, "LSHSpace", _FixedLSHSpace)

    AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("secret-value",),
            gammas=(0.5,),
            methods=("vanilla", "random", "codet5_masked_code", "seqmark_oracle"),
            retry_budgets=(1,),
            primary_method="codet5_masked_code",
            lsh_d=1,
            embed_dim=2,
            embedding_mode="hash",
        )
    ).run()

    anchor_text = "def f(x):\n    <extra_id_0>"
    assert provider.calls.count(anchor_text) == 1


def test_runner_caches_new_codet5_comment_variants_per_candidate(tmp_path, monkeypatch):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                context_before="def f(x):\n",
                context_after="",
                masked_parent_context="def f(x):\n    <TARGET_BLOCK>",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "valid-a", 0),
                    CandidateBlock("c1", "valid-b", 1),
                    CandidateBlock("c2", "invalid-a", 2),
                ),
            )
        ],
    )
    provider = _FixedProvider()
    monkeypatch.setattr(runner_module, "_build_embedding_provider", lambda config: provider)
    monkeypatch.setattr(runner_module, "LSHSpace", _FixedLSHSpace)

    AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("secret-value",),
            gammas=(0.5,),
            methods=(
                "vanilla",
                "random",
                "codet5_comment_minimal",
                "codet5_comment_contextual",
                "seqmark_oracle",
            ),
            retry_budgets=(1,),
            primary_method="codet5_comment_minimal",
            lsh_d=1,
            embed_dim=2,
            embedding_mode="hash",
        )
    ).run()

    assert sum(
        "# wfcllm: return_statement ordinal_0" in call
        and "parent_function_definition" not in call
        for call in provider.calls
    ) == 3
    assert sum(
        "# wfcllm: return_statement ordinal_0 parent_function_definition" in call
        for call in provider.calls
    ) == 3


def test_runner_can_show_context_progress_bar(tmp_path, monkeypatch):
    pool_path = tmp_path / "candidate_pools.jsonl"
    output_dir = tmp_path / "out"
    write_candidate_contexts(
        pool_path,
        [
            CandidateContext(
                context_id="ctx",
                dataset="humaneval",
                task_id="HumanEval/0",
                prompt="def f(x):\n",
                function_signature="def f(x):",
                ast_path=("function_definition", "return_statement"),
                node_type="return_statement",
                parent_node_type="function_definition",
                block_ordinal=0,
                context_hash="ctxhash",
                temperature=0.2,
                candidates=(
                    CandidateBlock("c0", "return x", 0),
                    CandidateBlock("c1", "return x + 1", 1),
                ),
            )
        ],
    )
    progress_calls = []

    def fake_tqdm(iterable, **kwargs):
        progress_calls.append(SimpleNamespace(total=kwargs.get("total"), desc=kwargs.get("desc"), unit=kwargs.get("unit")))
        return iterable

    monkeypatch.setattr(runner_module, "tqdm", fake_tqdm)

    AnchorValidationRunner(
        AnchorValidationConfig(
            pool_path=pool_path,
            output_dir=output_dir,
            secret_keys=("k0",),
            gammas=(0.5,),
            methods=("vanilla", "random", "seqmark_oracle"),
            retry_budgets=(1,),
            primary_method="vanilla",
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
            show_progress=True,
        )
    ).run()

    assert progress_calls == [
        SimpleNamespace(total=1, desc="Anchor diagnostics contexts", unit="context")
    ]
