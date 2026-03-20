"""One-shot local debug CLI for prompt-level embed/extract forensics."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import run


def load_prompt_record(input_file: str, prompt_id: str) -> dict:
    path = Path(input_file)
    if not path.exists():
        raise FileNotFoundError(f"输入文件不存在：{path}")

    with open(path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"输入文件第 {line_no} 行 JSON 解析失败：{exc}") from exc
            if row.get("id") == prompt_id:
                return row
    raise ValueError(f"在输入文件中找不到 prompt_id={prompt_id}")


def resolve_debug_lsh_params(
    record: dict,
    use_embedded_params: bool,
    config_path: str,
) -> tuple[int, float]:
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    ext_cfg = cfg.get("extract", {})
    if use_embedded_params:
        return run.resolve_extract_lsh_params(record, ext_cfg)
    return int(ext_cfg.get("lsh_d", 3)), float(ext_cfg.get("lsh_gamma", 0.5))


def _find_prompt_in_details_file(prompt_id: str, path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("prompt_id") == prompt_id:
                return row
    return None


def _auto_discover_diag_candidates(input_file: str) -> list[Path]:
    input_path = Path(input_file).resolve()
    candidates: list[Path] = []
    try:
        diag_dir = input_path.parents[1] / "diag_reports"
    except IndexError:
        return candidates
    if not diag_dir.exists():
        return candidates
    files = sorted(
        diag_dir.glob("details_*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return files


def find_matching_diag_report(
    prompt_id: str, input_file: str, diag_details: str | None
) -> tuple[dict | None, Path | None]:
    if diag_details:
        path = Path(diag_details)
        return _find_prompt_in_details_file(prompt_id, path), path

    for path in _auto_discover_diag_candidates(input_file):
        report = _find_prompt_in_details_file(prompt_id, path)
        if report is not None:
            return report, path
    return None, None


def collect_mismatch_samples(report: dict, limit: int = 5) -> list[dict]:
    samples: list[dict] = []
    for pair in report.get("aligned_pairs", []):
        text_mismatch = not pair.get("text_match", True)
        parent_mismatch = not pair.get("parent_match", True)
        extract_score = pair.get("extract_score")
        score_disagree = extract_score is not None and not pair.get("score_agree", True)
        if not (text_mismatch or parent_mismatch or score_disagree):
            continue

        embed = pair.get("embed", {})
        extract = pair.get("extract", {})
        samples.append(
            {
                "embed_text": embed.get("block_text", ""),
                "extract_text": extract.get("source", ""),
                "text_mismatch": text_mismatch,
                "parent_mismatch": parent_mismatch,
                "score_disagree": score_disagree,
                "embed_parent": embed.get("parent_node_type", ""),
                "extract_parent": extract.get("node_type", ""),
            }
        )
        if len(samples) >= limit:
            break
    return samples


def build_debug_payload(
    prompt_id: str,
    input_file: str,
    use_embedded_params: bool,
    config_path: str = "configs/base_config.json",
    diag_details: str | None = None,
) -> dict:
    record = load_prompt_record(input_file, prompt_id)
    lsh_d, lsh_gamma = resolve_debug_lsh_params(
        record=record,
        use_embedded_params=use_embedded_params,
        config_path=config_path,
    )

    report, details_path = find_matching_diag_report(prompt_id, input_file, diag_details)
    payload = {
        "prompt_id": prompt_id,
        "input_file": input_file,
        "resolved_lsh_d": lsh_d,
        "resolved_lsh_gamma": lsh_gamma,
        "diagnostic_report_found": report is not None,
        "text_mismatch_count": 0,
        "parent_mismatch_count": 0,
        "score_disagree_count": 0,
        "generated_code_preview": (record.get("generated_code") or "")[:200],
        "mismatch_samples": [],
    }
    if report is None:
        payload["message"] = "未找到可用 prompt-level diagnostic report；仅输出解析参数与基本信息。"
        return payload

    payload["diagnostic_details_file"] = str(details_path) if details_path else None
    payload["text_mismatch_count"] = int(report.get("text_mismatch_count", 0))
    payload["parent_mismatch_count"] = int(report.get("parent_mismatch_count", 0))
    payload["score_disagree_count"] = int(report.get("score_disagree_count", 0))
    payload["mismatch_samples"] = collect_mismatch_samples(report, limit=5)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Prompt-level extract-alignment debug helper")
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--use-embedded-params", action="store_true")
    parser.add_argument("--config", default="configs/base_config.json")
    parser.add_argument("--diag-details", default=None)
    args = parser.parse_args()

    try:
        payload = build_debug_payload(
            prompt_id=args.prompt_id,
            input_file=args.input_file,
            use_embedded_params=args.use_embedded_params,
            config_path=args.config,
            diag_details=args.diag_details,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"[错误] {exc}", file=sys.stderr)
        return 1

    print(
        {
            "text_mismatch_count": payload["text_mismatch_count"],
            "parent_mismatch_count": payload["parent_mismatch_count"],
            "score_disagree_count": payload["score_disagree_count"],
            "resolved_lsh_d": payload["resolved_lsh_d"],
            "resolved_lsh_gamma": payload["resolved_lsh_gamma"],
        }
    )

    if payload["diagnostic_report_found"]:
        print(f"diagnostic_details_file={payload.get('diagnostic_details_file')}")
        for pair in payload["mismatch_samples"][:5]:
            print(
                {
                    "embed_text": pair["embed_text"],
                    "extract_text": pair["extract_text"],
                    "text_mismatch": pair["text_mismatch"],
                    "parent_mismatch": pair["parent_mismatch"],
                    "score_disagree": pair["score_disagree"],
                }
            )
    else:
        print(payload["message"])

    print({"prompt_id": payload["prompt_id"], "input_file": payload["input_file"]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
