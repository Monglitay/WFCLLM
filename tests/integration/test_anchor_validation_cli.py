from __future__ import annotations

import json
import subprocess


def test_anchor_validation_cli_build_pool_and_run_diagnostics(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x\n",
                        "candidate_index": 0,
                    }
                ),
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x + 1\n",
                        "candidate_index": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "candidate_pools.jsonl"
    out = tmp_path / "diag"

    build = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "build-pool",
            "--input-jsonl",
            str(candidates),
            "--output",
            str(pool),
            "--min-candidates",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr
    assert pool.exists()

    run = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "run-diagnostics",
            "--pool",
            str(pool),
            "--output-dir",
            str(out),
            "--embedding-mode",
            "hash",
            "--embed-dim",
            "8",
            "--lsh-d",
            "2",
            "--secret-key",
            "k0",
            "--methods",
            "vanilla",
            "random",
            "slot_context",
            "codet5_masked_code",
            "codet5_valid_skeleton",
            "codet5_comment_anchor",
            "codet5_identifier_anchor",
            "candidate_centroid_oracle",
            "context_centroid_oracle",
            "seqmark_oracle",
            "--gammas",
            "0.5",
            "--retry-budgets",
            "1",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    assert (out / "anchor_validation_summary.json").exists()
    assert "[anchor-validation] INFO running diagnostics" in run.stderr
    assert "Anchor diagnostics contexts" in run.stderr


def test_anchor_validation_cli_generate_pool_echo_mode(tmp_path):
    source = tmp_path / "sources.jsonl"
    source.write_text(
        json.dumps(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def f(x):\n",
                "source_code": "def f(x):\n    y = x + 1\n    return y\n",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "per_block_candidates.jsonl"

    result = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "generate-pool",
            "--source-jsonl",
            str(source),
            "--output",
            str(output),
            "--sampler-mode",
            "echo",
            "--temperatures",
            "0.2",
            "--candidates-per-temperature",
            "1",
            "--max-contexts-per-source",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert output.exists()


def test_anchor_validation_cli_run_diagnostics_has_role_aware_default_methods(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x\n",
                        "candidate_index": 0,
                    }
                ),
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x + 1\n",
                        "candidate_index": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "candidate_pools.jsonl"
    out = tmp_path / "diag"
    build = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "build-pool",
            "--input-jsonl",
            str(candidates),
            "--output",
            str(pool),
            "--min-candidates",
            "2",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "run-diagnostics",
            "--pool",
            str(pool),
            "--output-dir",
            str(out),
            "--embedding-mode",
            "hash",
            "--embed-dim",
            "8",
            "--lsh-d",
            "2",
            "--secret-key",
            "k0",
            "--gammas",
            "0.5",
            "--retry-budgets",
            "1",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    summary = json.loads((out / "anchor_validation_summary.json").read_text(encoding="utf-8"))
    assert summary["meta"]["methods"] == [
        "vanilla",
        "random",
        "context",
        "slot_context",
        "slot_context_skeleton",
        "role_aware_slot_context",
        "role_aware_slot_context_skeleton",
        "seqmark_oracle",
    ]


def test_anchor_validation_cli_no_progress_suppresses_progress_bar(tmp_path):
    candidates = tmp_path / "candidates.jsonl"
    candidates.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x\n",
                        "candidate_index": 0,
                    }
                ),
                json.dumps(
                    {
                        "id": "HumanEval/0",
                        "dataset": "humaneval",
                        "prompt": "def f(x):\n",
                        "generated_code": "    return x + 1\n",
                        "candidate_index": 1,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pool = tmp_path / "candidate_pools.jsonl"
    out = tmp_path / "diag"
    build = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "build-pool",
            "--input-jsonl",
            str(candidates),
            "--output",
            str(pool),
            "--min-candidates",
            "2",
            "--no-progress",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert build.returncode == 0, build.stderr

    run = subprocess.run(
        [
            "python",
            "scripts/anchor_validation.py",
            "run-diagnostics",
            "--pool",
            str(pool),
            "--output-dir",
            str(out),
            "--embedding-mode",
            "hash",
            "--embed-dim",
            "8",
            "--lsh-d",
            "2",
            "--secret-key",
            "k0",
            "--gammas",
            "0.5",
            "--retry-budgets",
            "1",
            "--no-progress",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert run.returncode == 0, run.stderr
    assert "[anchor-validation] INFO running diagnostics" in run.stderr
    assert "Anchor diagnostics contexts" not in run.stderr
