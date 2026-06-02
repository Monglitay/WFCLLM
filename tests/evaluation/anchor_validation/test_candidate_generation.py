from __future__ import annotations

from wfcllm.evaluation.anchor_validation.candidate_generation import (
    GenerationContextSource,
    build_block_completion_prompt,
    extract_generation_contexts,
    generate_candidate_rows,
)


def test_build_block_completion_prompt_contains_masked_context_not_secret():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )
    context = extract_generation_contexts(source, max_contexts=1)[0]

    prompt = build_block_completion_prompt(context, secret_key="do-not-leak")

    assert "<TARGET_BLOCK>" in prompt
    assert "do-not-leak" not in prompt
    assert context.block_text not in prompt


def test_generate_candidate_rows_uses_temperature_sweep_and_k_budget():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )

    def sampler(prompt: str, temperature: float, sample_index: int) -> str:
        return "y = 1 + x" if sample_index == 0 else "y = x + 1"

    rows = generate_candidate_rows(
        sources=(source,),
        sampler=sampler,
        temperatures=(0.2, 0.4),
        candidates_per_temperature=2,
        max_contexts_per_source=1,
    )

    assert len(rows) == 4
    assert {row["temperature"] for row in rows} == {0.2, 0.4}
    assert all(row["candidate_context_id"] for row in rows)
    assert all(row["context_hash"] for row in rows)


def test_generate_candidate_rows_filters_invalid_and_preserves_indentation():
    source = GenerationContextSource(
        dataset="humaneval",
        task_id="HumanEval/0",
        prompt="def f(x):\n",
        source_code="def f(x):\n    y = x + 1\n    return y\n",
    )

    def sampler(prompt: str, temperature: float, sample_index: int) -> str:
        return "not valid python:" if sample_index == 0 else "y = 1 + x"

    rows = generate_candidate_rows(
        sources=(source,),
        sampler=sampler,
        temperatures=(0.2,),
        candidates_per_temperature=2,
        max_contexts_per_source=1,
    )

    assert len(rows) == 1
    assert rows[0]["block_text"] == "    y = 1 + x"
    assert rows[0]["syntax_valid"] is True
    assert rows[0]["parse_valid"] is True
