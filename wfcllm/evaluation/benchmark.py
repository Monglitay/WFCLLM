"""Offline benchmark evaluation: Pass@1, Pass@10, AUROC for code watermarking."""
from __future__ import annotations

import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wfcllm.evaluation.code_execution import compute_pass_at_k


@dataclass
class TestExecutor:
    """Execute code + test cases via subprocess, return pass/fail."""

    timeout: float = 5.0

    def run_test(self, code: str, test_code: str) -> bool:
        """Run generated code against test assertions. Returns True if all pass."""
        full_source = f"{code}\n{test_code}\n"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(full_source)
            tmp_path = f.name
        try:
            result = subprocess.run(
                ["python", tmp_path],
                capture_output=True,
                timeout=self.timeout,
                check=False,
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
        except Exception:
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def execute_humaneval(
        self, generated_code: str, test_code: str, entry_point: str
    ) -> bool:
        """HumanEval-style execution: code + test + check(entry_point)."""
        full_test = f"{test_code}\ncheck({entry_point})\n"
        return self.run_test(generated_code, full_test)

    def execute_mbpp(self, generated_code: str, test_code: str) -> bool:
        """MBPP-style execution: code + assert statements."""
        return self.run_test(generated_code, test_code)


@dataclass
class BenchmarkConfig:
    """Configuration for the benchmark evaluation."""

    dataset: str
    config_path: str
    dataset_path: str = "data/datasets"
    num_candidates: int = 10
    timeout_per_test: float = 5.0
    watermarked_dirs: list[str] | None = None
    positive_details: str | None = None
    negative_details: str | None = None
    auto_generate: bool = False
    negative_corpus: str | None = None
    output_dir: str = "data/eval/benchmark"


class BenchmarkRunner:
    """Orchestrate benchmark evaluation: Pass@k + AUROC."""

    def __init__(self, config: BenchmarkConfig):
        self._config = config

    def _compute_detection_metrics(
        self,
        positive_details: list[dict[str, Any]],
        negative_details: list[dict[str, Any]],
    ) -> dict[str, dict[str, float]]:
        """Compute AUROC and TPR@1%FPR for each available score field."""
        from wfcllm.evaluation.code_execution import compute_roc_auc, compute_tpr_at_fpr

        score_fields = {
            "semantic": "z_score",
            "lexical": "lexical_z_score",
            "joint": "joint_score",
        }
        result: dict[str, dict[str, float]] = {}
        for channel, field in score_fields.items():
            pos_scores = [r[field] for r in positive_details if field in r]
            neg_scores = [r[field] for r in negative_details if field in r]
            if not pos_scores or not neg_scores:
                continue
            result[channel] = {
                "auroc": compute_roc_auc(pos_scores, neg_scores),
                "tpr_at_1pct_fpr": compute_tpr_at_fpr(pos_scores, neg_scores, target_fpr=0.01),
            }
        return result

    def _compute_pass_at_k(
        self,
        records: list[dict[str, Any]],
        correctness: list[bool],
    ) -> dict[str, float]:
        """Compute pass@1 and pass@10 from records + correctness annotations."""
        annotated = []
        for record, is_correct in zip(records, correctness):
            annotated.append({**record, "is_correct": is_correct})

        result: dict[str, float] = {}
        result["pass_at_1"] = compute_pass_at_k(annotated, k=1)
        k = min(10, self._config.num_candidates)
        result[f"pass_at_{k}"] = compute_pass_at_k(annotated, k=k)
        if k == 10:
            result["pass_at_10"] = result[f"pass_at_{k}"]
        return result
