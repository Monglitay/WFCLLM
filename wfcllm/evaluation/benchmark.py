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
    min_blocks: int = 0


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

    def run(self) -> dict[str, Any]:
        """Execute the full benchmark evaluation and return the report."""
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        test_cases = self._load_test_cases()

        if self._config.auto_generate:
            watermarked_records, pos_path, neg_path = self._auto_generate()
            if not self._config.positive_details:
                self._config.positive_details = pos_path
            if not self._config.negative_details:
                self._config.negative_details = neg_path
        else:
            watermarked_records = self._load_watermarked_records()

        if self._config.min_blocks > 0:
            before = len(watermarked_records)
            watermarked_records = [
                r for r in watermarked_records
                if int(r.get("total_blocks", 0)) >= self._config.min_blocks
            ]
            skipped = before - len(watermarked_records)
            if skipped:
                import sys
                print(f"[bench] min_blocks={self._config.min_blocks}: skipped {skipped}/{before} records", file=sys.stderr)

        executor = TestExecutor(timeout=self._config.timeout_per_test)
        correctness = self._evaluate_correctness(watermarked_records, test_cases, executor)
        pass_metrics = self._compute_pass_at_k(watermarked_records, correctness)

        exec_results_path = output_dir / f"{self._config.dataset}_exec_results.jsonl"
        self._save_execution_results(exec_results_path, watermarked_records, correctness)

        positive_details = self._load_details(self._config.positive_details)
        negative_details = self._load_details(self._config.negative_details)
        detection_metrics = self._compute_detection_metrics(positive_details, negative_details)

        report: dict[str, Any] = {
            "dataset": self._config.dataset,
            "num_candidates": self._config.num_candidates,
            "num_tasks": len(test_cases),
            "metrics": {
                **pass_metrics,
                "detection": detection_metrics,
            },
            "details": {
                "execution_results_path": str(exec_results_path),
                "positive_details_path": self._config.positive_details,
                "negative_details_path": self._config.negative_details,
            },
        }

        report_path = output_dir / f"{self._config.dataset}_benchmark_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return report

    def _auto_generate(self) -> tuple[list[dict[str, Any]], str, str]:
        """Run watermark + extract phases automatically."""
        import os

        output_dir = Path(self._config.output_dir)
        env = dict(os.environ)
        env.setdefault("HF_HUB_OFFLINE", "1")

        all_records: list[dict[str, Any]] = []
        watermarked_paths: list[Path] = []

        for i in range(self._config.num_candidates):
            candidate_dir = output_dir / f"watermarked_candidate_{i + 1}"
            candidate_dir.mkdir(parents=True, exist_ok=True)
            cmd = [
                "python", "run.py",
                "--config", self._config.config_path,
                "--phase", "watermark",
                "--dataset", self._config.dataset,
                "--output-dir", str(candidate_dir),
            ]
            self._run_command(cmd, env)
            jsonl_files = list(candidate_dir.glob("*.jsonl"))
            if jsonl_files:
                path = max(jsonl_files, key=lambda p: p.stat().st_mtime)
                watermarked_paths.append(path)
                records = self._read_jsonl(path)
                for r in records:
                    r["candidate_index"] = i
                all_records.extend(records)

        pos_details_path = ""
        pos_extract_dir = output_dir / "positive_extract"
        if watermarked_paths:
            cmd = [
                "python", "run.py",
                "--config", self._config.config_path,
                "--phase", "extract",
                "--input-file", str(watermarked_paths[0]),
                "--extract-output-dir", str(pos_extract_dir),
            ]
            self._run_command(cmd, env)
            pos_details_path = str(
                pos_extract_dir / f"{watermarked_paths[0].stem}_details.jsonl"
            )

        neg_extract_dir = output_dir / "negative_extract"
        neg_corpus = self._config.negative_corpus or "data/negative_corpus.jsonl"
        neg_corpus_path = Path(neg_corpus)
        cmd = [
            "python", "run.py",
            "--config", self._config.config_path,
            "--phase", "extract",
            "--input-file", str(neg_corpus_path),
            "--extract-output-dir", str(neg_extract_dir),
        ]
        self._run_command(cmd, env)
        neg_details_path = str(
            neg_extract_dir / f"{neg_corpus_path.stem}_details.jsonl"
        )

        return all_records, pos_details_path, neg_details_path

    def _run_command(self, cmd: list[str], env: dict[str, str] | None = None) -> None:
        """Run a subprocess command, raise on failure."""
        import subprocess as _sp
        result = _sp.run(cmd, capture_output=True, text=True, check=False, env=env)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or result.stdout or "command failed")

    def _load_test_cases(self) -> dict[str, Any]:
        from wfcllm.datasets.loaders.local import load_test_cases
        return load_test_cases(self._config.dataset, self._config.dataset_path)

    def _load_watermarked_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        if self._config.watermarked_dirs:
            for dir_path in self._config.watermarked_dirs:
                for jsonl_file in Path(dir_path).glob("*.jsonl"):
                    records.extend(self._read_jsonl(jsonl_file))
        return records

    def _evaluate_correctness(
        self,
        records: list[dict[str, Any]],
        test_cases: dict[str, Any],
        executor: TestExecutor,
    ) -> list[bool]:
        results: list[bool] = []
        for record in records:
            task_id = str(record.get("id", ""))
            prompt = str(record.get("prompt", ""))
            body = str(record.get("generated_code", ""))
            code = prompt + body if prompt else body
            tc = test_cases.get(task_id)
            if tc is None:
                results.append(False)
                continue
            if tc.entry_point is not None:
                passed = executor.execute_humaneval(code, tc.test_code, tc.entry_point)
            else:
                passed = executor.execute_mbpp(code, tc.test_code)
            results.append(passed)
        return results

    def _save_execution_results(
        self,
        path: Path,
        records: list[dict[str, Any]],
        correctness: list[bool],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for record, is_correct in zip(records, correctness):
            lines.append(json.dumps(
                {"id": record.get("id"), "is_correct": is_correct},
                ensure_ascii=False,
            ))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _load_details(self, path: str | None) -> list[dict[str, Any]]:
        if path is None:
            return []
        return self._read_jsonl(Path(path))

    @staticmethod
    def _read_jsonl(path: Path) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records
