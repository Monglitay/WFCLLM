from __future__ import annotations

import json
from types import SimpleNamespace

import wfcllm.evaluation.anchor_validation.runner as runner_module
from wfcllm.evaluation.anchor_validation.io import write_candidate_contexts
from wfcllm.evaluation.anchor_validation.runner import (
    AnchorValidationConfig,
    AnchorValidationRunner,
)
from wfcllm.evaluation.anchor_validation.schema import CandidateBlock, CandidateContext


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
            methods=("role_aware_slot_context", "role_aware_slot_context_skeleton"),
            retry_budgets=(1,),
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
        )
    ).run()

    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
    assert summary["meta"]["methods"] == [
        "role_aware_slot_context",
        "role_aware_slot_context_skeleton",
    ]


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
            methods=("vanilla",),
            retry_budgets=(1,),
            lsh_d=2,
            embed_dim=8,
            embedding_mode="hash",
            show_progress=True,
        )
    ).run()

    assert progress_calls == [
        SimpleNamespace(total=1, desc="Anchor diagnostics contexts", unit="context")
    ]
