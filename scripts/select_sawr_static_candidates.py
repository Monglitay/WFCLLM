#!/usr/bin/env python
"""Select SAWR candidates with static code-quality and code-derived evidence gates."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select between baseline and candidate SAWR final-code rows without executing candidate code.",
    )
    parser.add_argument("--baseline-jsonl", required=True)
    parser.add_argument("--candidate-jsonl", required=True)
    parser.add_argument("--baseline-details", required=True)
    parser.add_argument("--candidate-details", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--analysis-json", required=True)
    parser.add_argument(
        "--policy",
        choices=["candidate_if_syntax_signature_sufficient"],
        default="candidate_if_syntax_signature_sufficient",
    )
    return parser


def select_static_candidates(
    *,
    baseline_rows: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    baseline_details: list[dict[str, Any]],
    candidate_details: list[dict[str, Any]],
    policy: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline_by_id = _by_id(baseline_rows, label="baseline rows")
    candidate_by_id = _by_id(candidate_rows, label="candidate rows")
    baseline_detail_by_id = _by_id(baseline_details, label="baseline details")
    candidate_detail_by_id = _by_id(candidate_details, label="candidate details")

    missing = sorted(set(baseline_by_id) - set(candidate_by_id))
    if missing:
        raise ValueError(f"candidate rows missing ids: {missing[:5]}")

    selected_rows: list[dict[str, Any]] = []
    per_sample: list[dict[str, Any]] = []
    counts = {"baseline": 0, "candidate": 0}

    for sample_id in sorted(baseline_by_id):
        baseline = baseline_by_id[sample_id]
        candidate = candidate_by_id[sample_id]
        baseline_features = _static_features(baseline)
        candidate_features = _static_features(candidate)
        candidate_detail = candidate_detail_by_id[sample_id]
        baseline_detail = baseline_detail_by_id[sample_id]
        use_candidate = _use_candidate(
            policy=policy,
            candidate_features=candidate_features,
            candidate_detail=candidate_detail,
        )
        selected = candidate if use_candidate else baseline
        selected_source = "candidate" if use_candidate else "baseline"
        counts[selected_source] += 1
        selected_rows.append(_sanitized_row(selected))
        per_sample.append(
            {
                "id": sample_id,
                "selected_source": selected_source,
                "policy": policy,
                "baseline": {
                    **baseline_features,
                    "score": float(baseline_detail.get("score", 0.0)),
                    "insufficient_evidence": bool(
                        baseline_detail.get("insufficient_evidence", False)
                    ),
                    "proxy_windows": int(baseline_detail.get("proxy_windows", 0)),
                },
                "candidate": {
                    **candidate_features,
                    "score": float(candidate_detail.get("score", 0.0)),
                    "insufficient_evidence": bool(
                        candidate_detail.get("insufficient_evidence", False)
                    ),
                    "proxy_windows": int(candidate_detail.get("proxy_windows", 0)),
                },
            }
        )

    analysis = {
        "artifact_type": "sawr_static_candidate_selection_analysis",
        "policy": policy,
        "sample_count": len(selected_rows),
        "selected_counts": counts,
        "per_sample": per_sample,
    }
    return selected_rows, analysis


def _use_candidate(
    *,
    policy: str,
    candidate_features: dict[str, Any],
    candidate_detail: dict[str, Any],
) -> bool:
    if policy != "candidate_if_syntax_signature_sufficient":
        raise ValueError(f"unsupported policy: {policy}")
    return (
        bool(candidate_features["syntax_valid"])
        and bool(candidate_features["signature_compatible"])
        and not bool(candidate_detail.get("insufficient_evidence", False))
    )


def _static_features(row: dict[str, Any]) -> dict[str, Any]:
    prompt = str(row.get("prompt") or "")
    code = str(row.get("final_code") or row.get("generated_code") or "")
    try:
        tree = ast.parse(code)
        syntax_valid = True
    except SyntaxError:
        tree = None
        syntax_valid = False

    prompt_function = _target_function(prompt)
    code_function = (
        _function_by_name(tree, prompt_function.name)
        if tree is not None and prompt_function is not None
        else None
    )
    return {
        "syntax_valid": syntax_valid,
        "target_function_present": code_function is not None,
        "signature_compatible": (
            prompt_function is not None
            and code_function is not None
            and ast.dump(prompt_function.args, include_attributes=False)
            == ast.dump(code_function.args, include_attributes=False)
        ),
    }


def _target_function(prompt: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            return node
    return None


def _function_by_name(tree: ast.AST, name: str) -> ast.FunctionDef | None:
    if not isinstance(tree, ast.Module):
        return None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def _by_id(rows: list[dict[str, Any]], *, label: str) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        if not sample_id:
            raise ValueError(f"{label} contain a row without id")
        if sample_id in by_id:
            raise ValueError(f"{label} contain duplicate id: {sample_id}")
        by_id[sample_id] = row
    return by_id


def _sanitized_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(row.get("id") or row.get("task_id") or ""),
        "dataset": str(row.get("dataset") or "humaneval"),
        "prompt": str(row.get("prompt") or ""),
        "final_code": str(row.get("final_code") or row.get("generated_code") or ""),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        rows, analysis = select_static_candidates(
            baseline_rows=_load_jsonl(Path(args.baseline_jsonl)),
            candidate_rows=_load_jsonl(Path(args.candidate_jsonl)),
            baseline_details=_load_jsonl(Path(args.baseline_details)),
            candidate_details=_load_jsonl(Path(args.candidate_details)),
            policy=args.policy,
        )
        _write_jsonl(Path(args.output_jsonl), rows)
        _write_json(Path(args.analysis_json), analysis)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1
    print(f"[完成] static SAWR selected rows saved to {args.output_jsonl}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
