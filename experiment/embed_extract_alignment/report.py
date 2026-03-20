"""Console summary printing and JSON serialization for alignment reports."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path

from experiment.embed_extract_alignment.models import PromptReport, SummaryReport


def print_prompt_summary(report: PromptReport) -> None:
    """Print 3-line summary for a single prompt."""
    print(
        f"[{report.prompt_id}] "
        f"embed={report.embed_total} "
        f"simple={report.embed_simple_passed}✓ "
        f"fallback={report.embed_fallback_passed}✓ "
        f"cascade={report.embed_cascade_passed} "
        f"failed={report.embed_simple_failed}"
    )
    print(
        f"  extract_simple={report.extract_simple_count}  "
        f"aligned={len(report.aligned_pairs)}  "
        f"unmatched_embed={report.embed_unmatched_count}  "
        f"unmatched_extract={report.extract_unmatched_count}"
    )
    print(
        f"  compound_aligned={report.compound_aligned_count}  "
        f"text_mismatch={report.text_mismatch_count} "
        f"(simple={report.text_mismatch_simple_only}, compound={report.text_mismatch_compound_only})  "
        f"parent_mismatch={report.parent_mismatch_count}  "
        f"score_disagree={report.score_disagree_count}  "
        f"z={report.detect_z_score:.2f}"
    )


def print_summary(summary: SummaryReport) -> None:
    """Print aggregated summary across all prompts."""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  prompts:           {summary.n_prompts}")
    print(f"  total_embed_events: {summary.total_embed_events}")
    print(f"  compound_events:   {summary.compound_only_events} ({summary.compound_ratio:.1%})")
    print(
        f"  text_mismatch:     {summary.text_mismatch_total} "
        f"(simple={summary.text_mismatch_simple_only_total}, "
        f"compound={summary.text_mismatch_compound_only_total})"
    )
    print(f"  parent_mismatch:   {summary.parent_mismatch_total}")
    print(f"  score_disagree:    {summary.score_disagree_total}")
    print(f"  avg_embed_rate:    {summary.avg_embed_rate:.1%}")
    print(f"  avg_detect_z:      {summary.avg_detect_z:.2f}")


def build_summary(reports: list[PromptReport]) -> SummaryReport:
    """Aggregate PromptReports into a SummaryReport."""
    n = len(reports)
    if n == 0:
        return SummaryReport(
            n_prompts=0,
            total_embed_events=0, compound_only_events=0, compound_ratio=0.0,
            text_mismatch_total=0,
            text_mismatch_simple_only_total=0,
            text_mismatch_compound_only_total=0,
            parent_mismatch_total=0,
            score_disagree_total=0,
            avg_embed_rate=0.0, avg_detect_z=0.0,
        )

    total_embed = sum(r.embed_total for r in reports)
    compound_total = sum(r.embed_compound_total for r in reports)

    embed_rates = [
        r.embed_simple_passed / r.embed_total
        for r in reports if r.embed_total > 0
    ]
    avg_rate = sum(embed_rates) / len(embed_rates) if embed_rates else 0.0
    avg_z = sum(r.detect_z_score for r in reports) / n

    return SummaryReport(
        n_prompts=n,
        total_embed_events=total_embed,
        compound_only_events=compound_total,
        compound_ratio=compound_total / total_embed if total_embed > 0 else 0.0,
        text_mismatch_total=sum(r.text_mismatch_count for r in reports),
        text_mismatch_simple_only_total=sum(r.text_mismatch_simple_only for r in reports),
        text_mismatch_compound_only_total=sum(r.text_mismatch_compound_only for r in reports),
        parent_mismatch_total=sum(r.parent_mismatch_count for r in reports),
        score_disagree_total=sum(r.score_disagree_count for r in reports),
        avg_embed_rate=avg_rate,
        avg_detect_z=avg_z,
    )


def _to_dict(obj) -> object:
    """Recursively convert dataclasses and lists to JSON-serializable dicts."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_dict(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, list):
        return [_to_dict(i) for i in obj]
    return obj


def save_reports(
    reports: list[PromptReport],
    summary: SummaryReport,
    output_dir: str,
    timestamp: str,
) -> tuple[str, str]:
    """Save summary JSON and details JSONL. Returns (summary_path, details_path)."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / f"summary_{timestamp}.json"
    details_path = out / f"details_{timestamp}.jsonl"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(_to_dict(summary), f, ensure_ascii=False, indent=2)

    with open(details_path, "w", encoding="utf-8") as f:
        for r in reports:
            f.write(json.dumps(_to_dict(r), ensure_ascii=False) + "\n")

    return str(summary_path), str(details_path)
