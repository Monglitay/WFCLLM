#!/usr/bin/env python
"""Analyze SAWR multi-candidate selector tradeoffs."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass(frozen=True)
class CandidateInput:
    name: str
    details_path: Path
    pass_path: Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze SAWR candidate pass/detector tradeoffs.",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        metavar="NAME:DETAILS_JSONL:PASS_JSONL",
        help="Candidate source. The first candidate is treated as baseline.",
    )
    parser.add_argument("--reference-details", required=True)
    parser.add_argument("--static-analysis-json", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", type=int, default=5)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    candidates = [_parse_candidate_spec(spec) for spec in args.candidate]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_data = {
        item.name: {
            "details": _by_id(_load_jsonl(item.details_path)),
            "pass": _pass_by_id(_load_jsonl(item.pass_path)),
        }
        for item in candidates
    }
    reference_details = _load_jsonl(Path(args.reference_details))
    static_analysis = (
        _load_json(Path(args.static_analysis_json))
        if args.static_analysis_json is not None
        else None
    )

    matrix = _build_candidate_matrix(candidate_data, reference_details)
    bottleneck = _build_bottleneck_report(candidate_data, static_analysis)
    per_sample = _build_per_sample_delta(candidate_data, static_analysis)
    policies = _sweep_policies(candidate_data, reference_details)
    folds = _cross_validate_policies(
        candidate_data,
        reference_details,
        policies,
        folds=max(2, int(args.folds)),
    )
    pareto = _pareto_frontier(policies)

    _write_json(output_dir / "candidate_metric_matrix.json", matrix)
    _write_json(output_dir / "selector_bottleneck_report.json", bottleneck)
    _write_md(output_dir / "selector_bottleneck_report.md", _bottleneck_md(bottleneck))
    _write_jsonl(output_dir / "per_sample_candidate_delta.jsonl", per_sample)
    _write_json(output_dir / "policy_pareto_frontier.json", pareto)
    _write_json(output_dir / "kfold_selector_results.json", folds)
    _write_md(output_dir / "kfold_selector_results.md", _folds_md(folds))
    print(
        json.dumps(
            {
                "metric_matrix": str(output_dir / "candidate_metric_matrix.json"),
                "bottleneck_report": str(output_dir / "selector_bottleneck_report.md"),
                "kfold_report": str(output_dir / "kfold_selector_results.md"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _parse_candidate_spec(spec: str) -> CandidateInput:
    parts = spec.split(":", maxsplit=2)
    if len(parts) != 3 or not all(parts):
        raise ValueError("--candidate must be NAME:DETAILS_JSONL:PASS_JSONL")
    return CandidateInput(name=parts[0], details_path=Path(parts[1]), pass_path=Path(parts[2]))


def _build_candidate_matrix(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
) -> dict[str, Any]:
    negatives = [_score(row) for row in reference_details]
    rows = []
    for name, data in candidate_data.items():
        details = list(data["details"].values())
        pass_map = data["pass"]
        positives = [_score(row) for row in details]
        operating = _best_tp_at_max_fp(positives, negatives, max_fp=8)
        rows.append(
            {
                "name": name,
                "samples": len(details),
                "pass": sum(1 for value in pass_map.values() if value),
                "pass_rate": _safe_div(sum(1 for value in pass_map.values() if value), len(pass_map)),
                "auroc": _auc(positives, negatives),
                "tp_at_8fp": operating["tp"],
                "fp": operating["fp"],
                "threshold": operating["threshold"],
                "positive_insufficient": sum(
                    1 for row in details if bool(row.get("insufficient_evidence", False))
                ),
                "mean_score": mean(positives) if positives else 0.0,
                "mean_proxy_windows": mean([_proxy_windows(row) for row in details])
                if details
                else 0.0,
                "mean_scoreable_contexts": mean(
                    [int(row.get("scoreable_contexts", 0) or 0) for row in details]
                )
                if details
                else 0.0,
            }
        )
    return {"artifact_type": "sawr_candidate_metric_matrix", "rows": rows}


def _build_bottleneck_report(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    static_analysis: dict[str, Any] | None,
) -> dict[str, Any]:
    names = list(candidate_data)
    if len(names) < 2:
        return {"artifact_type": "sawr_selector_bottleneck", "summary": {}}
    baseline_name, seed_name = names[0], names[1]
    baseline = candidate_data[baseline_name]
    seed = candidate_data[seed_name]
    ids = sorted(set(baseline["details"]) & set(seed["details"]))
    baseline_pass = baseline["pass"]
    seed_pass = seed["pass"]
    groups = {
        "both_pass": [],
        "baseline_only_pass": [],
        "seed_only_pass": [],
        "neither_pass": [],
    }
    for sample_id in ids:
        base_ok = bool(baseline_pass.get(sample_id, False))
        seed_ok = bool(seed_pass.get(sample_id, False))
        if base_ok and seed_ok:
            groups["both_pass"].append(sample_id)
        elif base_ok:
            groups["baseline_only_pass"].append(sample_id)
        elif seed_ok:
            groups["seed_only_pass"].append(sample_id)
        else:
            groups["neither_pass"].append(sample_id)

    static_rows = _static_rows_by_id(static_analysis)
    baseline_only_replaced = [
        sample_id
        for sample_id in groups["baseline_only_pass"]
        if static_rows.get(sample_id, {}).get("selected_source") == "candidate"
    ]
    seed_only_kept_baseline = [
        sample_id
        for sample_id in groups["seed_only_pass"]
        if static_rows.get(sample_id, {}).get("selected_source") == "baseline"
    ]
    return {
        "artifact_type": "sawr_selector_bottleneck",
        "baseline": baseline_name,
        "seed_candidate": seed_name,
        "summary": {key: len(value) for key, value in groups.items()},
        "oracle_union_pass": len(groups["both_pass"])
        + len(groups["baseline_only_pass"])
        + len(groups["seed_only_pass"]),
        "seed_only_detector_delta": _delta_stats(groups["seed_only_pass"], baseline, seed),
        "baseline_only_detector_delta": _delta_stats(
            groups["baseline_only_pass"], baseline, seed
        ),
        "baseline_only_replaced_by_static_selector": {
            "count": len(baseline_only_replaced),
            "ids": baseline_only_replaced,
            "delta_stats": _delta_stats(baseline_only_replaced, baseline, seed),
        },
        "seed_only_rejected_by_static_selector": {
            "count": len(seed_only_kept_baseline),
            "ids": seed_only_kept_baseline,
            "delta_stats": _delta_stats(seed_only_kept_baseline, baseline, seed),
        },
    }


def _build_per_sample_delta(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    static_analysis: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    names = list(candidate_data)
    baseline_name = names[0]
    baseline = candidate_data[baseline_name]
    static_rows = _static_rows_by_id(static_analysis)
    ids = sorted(set().union(*(set(data["details"]) for data in candidate_data.values())))
    rows = []
    for sample_id in ids:
        base_detail = baseline["details"].get(sample_id, {})
        row: dict[str, Any] = {
            "id": sample_id,
            "static_selected_source": static_rows.get(sample_id, {}).get("selected_source"),
        }
        for name, data in candidate_data.items():
            detail = data["details"].get(sample_id, {})
            row[f"{name}_pass"] = bool(data["pass"].get(sample_id, False))
            row[f"{name}_score"] = _score(detail) if detail else None
            row[f"{name}_score_delta_vs_{baseline_name}"] = (
                _score(detail) - _score(base_detail) if detail and base_detail else None
            )
            row[f"{name}_proxy_windows"] = _proxy_windows(detail) if detail else None
            row[f"{name}_scoreable_contexts"] = (
                int(detail.get("scoreable_contexts", 0) or 0) if detail else None
            )
            row[f"{name}_insufficient_evidence"] = (
                bool(detail.get("insufficient_evidence", False)) if detail else None
            )
        rows.append(row)
    return rows


def _sweep_policies(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_name = next(iter(candidate_data))
    all_ids = sorted(candidate_data[baseline_name]["details"])
    policies = []
    for score_min in (-1.5, -1.0, -0.6, -0.4, -0.2, 0.0):
        for max_drop in (0.0, 0.2, 0.5, 1.0, None):
            for min_proxy in (0, 2, 4):
                for min_scoreable in (0, 1, 2):
                    for require_sufficient in (False, True):
                        policy = {
                            "score_min": score_min,
                            "max_drop": max_drop,
                            "min_proxy_windows": min_proxy,
                            "min_scoreable_contexts": min_scoreable,
                            "require_not_insufficient": require_sufficient,
                        }
                        selected = _select_by_policy(candidate_data, all_ids, policy)
                        metrics = _metrics_for_selection(
                            candidate_data,
                            reference_details,
                            selected,
                            all_ids,
                        )
                        policies.append({**policy, **metrics})
    return policies


def _cross_validate_policies(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
    policies: list[dict[str, Any]],
    *,
    folds: int,
) -> dict[str, Any]:
    baseline_name = next(iter(candidate_data))
    ids = sorted(candidate_data[baseline_name]["details"])
    fold_rows = []
    for fold_index in range(folds):
        heldout = [sample_id for i, sample_id in enumerate(ids) if i % folds == fold_index]
        train = [sample_id for sample_id in ids if sample_id not in set(heldout)]
        train_scores = []
        for policy in policies:
            train_selected = _select_by_policy(candidate_data, train, policy)
            train_metrics = _metrics_for_selection(
                candidate_data,
                reference_details,
                train_selected,
                train,
            )
            train_scores.append((_policy_objective(train_metrics), policy, train_metrics))
        _, best_policy, best_train_metrics = max(train_scores, key=lambda item: item[0])
        heldout_selected = _select_by_policy(candidate_data, heldout, best_policy)
        heldout_metrics = _metrics_for_selection(
            candidate_data,
            reference_details,
            heldout_selected,
            heldout,
        )
        fold_rows.append(
            {
                "fold": fold_index,
                "train_ids": len(train),
                "heldout_ids": len(heldout),
                "selected_policy": _policy_only(best_policy),
                "train_metrics": best_train_metrics,
                "heldout_metrics": heldout_metrics,
            }
        )
    return {
        "artifact_type": "sawr_selector_kfold_results",
        "folds": folds,
        "rows": fold_rows,
        "heldout_mean_pass": mean(
            [row["heldout_metrics"]["pass"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
        "heldout_mean_auroc": mean(
            [row["heldout_metrics"]["auroc"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
        "heldout_mean_tp_at_8fp": mean(
            [row["heldout_metrics"]["tp_at_8fp"] for row in fold_rows]
        )
        if fold_rows
        else 0.0,
    }


def _select_by_policy(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    sample_ids: list[str],
    policy: dict[str, Any],
) -> dict[str, str]:
    names = list(candidate_data)
    baseline_name = names[0]
    selected: dict[str, str] = {}
    for sample_id in sample_ids:
        base_detail = candidate_data[baseline_name]["details"][sample_id]
        base_score = _score(base_detail)
        eligible = [baseline_name]
        for name in names[1:]:
            detail = candidate_data[name]["details"].get(sample_id)
            if not detail:
                continue
            score = _score(detail)
            if score < float(policy["score_min"]):
                continue
            max_drop = policy.get("max_drop")
            if max_drop is not None and score < base_score - float(max_drop):
                continue
            if _proxy_windows(detail) < int(policy["min_proxy_windows"]):
                continue
            if int(detail.get("scoreable_contexts", 0) or 0) < int(
                policy["min_scoreable_contexts"]
            ):
                continue
            if policy.get("require_not_insufficient") and bool(
                detail.get("insufficient_evidence", False)
            ):
                continue
            eligible.append(name)
        selected[sample_id] = max(
            eligible,
            key=lambda name: (
                _score(candidate_data[name]["details"][sample_id]),
                _proxy_windows(candidate_data[name]["details"][sample_id]),
                -names.index(name),
            ),
        )
    return selected


def _metrics_for_selection(
    candidate_data: dict[str, dict[str, dict[str, Any]]],
    reference_details: list[dict[str, Any]],
    selected: dict[str, str],
    sample_ids: list[str],
) -> dict[str, Any]:
    positive_details = [
        candidate_data[selected[sample_id]]["details"][sample_id] for sample_id in sample_ids
    ]
    positives = [_score(row) for row in positive_details]
    negatives = [_score(row) for row in reference_details]
    operating = _best_tp_at_max_fp(positives, negatives, max_fp=8)
    passed = sum(
        1
        for sample_id in sample_ids
        if candidate_data[selected[sample_id]]["pass"].get(sample_id, False)
    )
    selected_counts: dict[str, int] = {}
    for name in selected.values():
        selected_counts[name] = selected_counts.get(name, 0) + 1
    return {
        "samples": len(sample_ids),
        "pass": passed,
        "pass_rate": _safe_div(passed, len(sample_ids)),
        "auroc": _auc(positives, negatives),
        "tp_at_8fp": operating["tp"],
        "fp": operating["fp"],
        "threshold": operating["threshold"],
        "positive_insufficient": sum(
            1 for row in positive_details if bool(row.get("insufficient_evidence", False))
        ),
        "selected_counts": selected_counts,
    }


def _policy_objective(metrics: dict[str, Any]) -> tuple[int, float, float, int]:
    meets_detection = (
        int(metrics["tp_at_8fp"]) >= 79
        and int(metrics["fp"]) <= 8
        and float(metrics["auroc"]) >= 0.80
    )
    return (
        1 if meets_detection else 0,
        float(metrics["pass_rate"]),
        float(metrics["auroc"]),
        int(metrics["tp_at_8fp"]),
    )


def _policy_only(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "score_min",
            "max_drop",
            "min_proxy_windows",
            "min_scoreable_contexts",
            "require_not_insufficient",
        )
    }


def _pareto_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    frontier = []
    for row in rows:
        dominated = False
        for other in rows:
            if other is row:
                continue
            if (
                other["pass"] >= row["pass"]
                and other["auroc"] >= row["auroc"]
                and other["tp_at_8fp"] >= row["tp_at_8fp"]
                and other["fp"] <= row["fp"]
                and (
                    other["pass"] > row["pass"]
                    or other["auroc"] > row["auroc"]
                    or other["tp_at_8fp"] > row["tp_at_8fp"]
                    or other["fp"] < row["fp"]
                )
            ):
                dominated = True
                break
        if not dominated:
            frontier.append(row)
    return sorted(frontier, key=lambda item: (item["pass"], item["auroc"]), reverse=True)


def _delta_stats(
    sample_ids: list[str],
    baseline: dict[str, dict[str, Any]],
    candidate: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    rows = []
    for sample_id in sample_ids:
        base = baseline["details"].get(sample_id)
        cand = candidate["details"].get(sample_id)
        if not base or not cand:
            continue
        rows.append(
            {
                "score_delta": _score(cand) - _score(base),
                "proxy_delta": _proxy_windows(cand) - _proxy_windows(base),
                "scoreable_delta": int(cand.get("scoreable_contexts", 0) or 0)
                - int(base.get("scoreable_contexts", 0) or 0),
                "candidate_insufficient": bool(cand.get("insufficient_evidence", False)),
                "baseline_insufficient": bool(base.get("insufficient_evidence", False)),
            }
        )
    return {
        "n": len(rows),
        "mean_score_delta": mean([row["score_delta"] for row in rows]) if rows else 0.0,
        "mean_proxy_delta": mean([row["proxy_delta"] for row in rows]) if rows else 0.0,
        "mean_scoreable_delta": mean([row["scoreable_delta"] for row in rows])
        if rows
        else 0.0,
        "candidate_insufficient": sum(1 for row in rows if row["candidate_insufficient"]),
        "baseline_insufficient": sum(1 for row in rows if row["baseline_insufficient"]),
    }


def _static_rows_by_id(static_analysis: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not static_analysis:
        return {}
    return {
        str(row.get("id") or ""): row
        for row in static_analysis.get("per_sample", [])
        if isinstance(row, dict)
    }


def _auc(positive: list[float], negative: list[float]) -> float:
    if not positive or not negative:
        return 0.0
    wins = 0.0
    for pos in positive:
        for neg in negative:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
    return wins / (len(positive) * len(negative))


def _best_tp_at_max_fp(
    positive: list[float],
    negative: list[float],
    *,
    max_fp: int,
) -> dict[str, Any]:
    best = {"tp": 0, "fp": 0, "threshold": math.inf}
    for threshold in sorted(set(positive + negative), reverse=True):
        fp = sum(score >= threshold for score in negative)
        if fp > max_fp:
            continue
        tp = sum(score >= threshold for score in positive)
        if tp > best["tp"] or (tp == best["tp"] and fp > best["fp"]):
            best = {"tp": tp, "fp": fp, "threshold": threshold}
    return best


def _score(row: dict[str, Any]) -> float:
    return float(row.get("score", 0.0) or 0.0)


def _proxy_windows(row: dict[str, Any]) -> int:
    value = row.get("proxy_windows", 0)
    if isinstance(value, (list, tuple)):
        return len(value)
    return int(value or 0)


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row.get("id") or row.get("task_id") or ""): row for row in rows}


def _pass_by_id(rows: list[dict[str, Any]]) -> dict[str, bool]:
    result = {}
    for row in rows:
        sample_id = str(row.get("id") or row.get("task_id") or "")
        result[sample_id] = bool(row.get("is_correct", row.get("passed", False)))
    return result


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"JSONL row must be an object: {path}:{line_number}")
        rows.append(payload)
    return rows


def _load_json(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _write_md(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _bottleneck_md(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    seed_delta = report.get("seed_only_detector_delta", {})
    base_delta = report.get("baseline_only_detector_delta", {})
    replaced = report.get("baseline_only_replaced_by_static_selector", {})
    rejected = report.get("seed_only_rejected_by_static_selector", {})
    lines = [
        "# SAWR Selector Bottleneck Report",
        "",
        "## Pass Overlap",
        "",
        f"- both pass: {summary.get('both_pass', 0)}",
        f"- baseline-only pass: {summary.get('baseline_only_pass', 0)}",
        f"- seed-only pass: {summary.get('seed_only_pass', 0)}",
        f"- neither pass: {summary.get('neither_pass', 0)}",
        f"- oracle union pass: {report.get('oracle_union_pass', 0)}",
        "",
        "## Detector Delta",
        "",
        f"- seed-only mean score delta vs baseline: {seed_delta.get('mean_score_delta', 0.0):.4f}",
        f"- seed-only candidate insufficient: {seed_delta.get('candidate_insufficient', 0)}",
        f"- baseline-only mean score delta if swapped to seed: {base_delta.get('mean_score_delta', 0.0):.4f}",
        f"- baseline-only replaced by static selector: {replaced.get('count', 0)}",
        f"- seed-only rejected by static selector: {rejected.get('count', 0)}",
        "",
        "## Interpretation",
        "",
        "- Candidate sampling supplies real pass complementarity, but the seed-only pass subset does not consistently preserve code-only detector score.",
        "- Static syntax/signature/evidence gates are too weak: they replace some baseline-only passing rows and keep/choose rows with detector score drops.",
    ]
    return "\n".join(lines) + "\n"


def _folds_md(report: dict[str, Any]) -> str:
    lines = [
        "# SAWR Selector K-Fold Report",
        "",
        f"- folds: {report.get('folds')}",
        f"- heldout mean pass: {report.get('heldout_mean_pass', 0.0):.2f}",
        f"- heldout mean AUROC: {report.get('heldout_mean_auroc', 0.0):.4f}",
        f"- heldout mean TP@<=8FP: {report.get('heldout_mean_tp_at_8fp', 0.0):.2f}",
        "",
        "| fold | heldout pass | heldout AUROC | heldout TP@<=8FP | selected policy |",
        "|---:|---:|---:|---:|---|",
    ]
    for row in report.get("rows", []):
        heldout = row["heldout_metrics"]
        lines.append(
            "| {fold} | {passed}/{total} | {auroc:.4f} | {tp} | `{policy}` |".format(
                fold=row["fold"],
                passed=heldout["pass"],
                total=heldout["samples"],
                auroc=heldout["auroc"],
                tp=heldout["tp_at_8fp"],
                policy=json.dumps(row["selected_policy"], sort_keys=True),
            )
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
