from __future__ import annotations

import json

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
