"""Merge two watermarked JSONL files from parallel embedding runs into one."""
import argparse
import json
from pathlib import Path


def merge(file_a: Path, file_b: Path, output: Path) -> None:
    records: dict[str, dict] = {}
    for path in (file_a, file_b):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                records[rec["id"]] = rec

    sorted_records = sorted(records.values(), key=lambda r: r["id"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for rec in sorted_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Merged {len(sorted_records)} records → {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge two watermarked JSONL files")
    parser.add_argument("file_a", help="First JSONL file (first half)")
    parser.add_argument("file_b", help="Second JSONL file (second half)")
    parser.add_argument("output", help="Output merged JSONL file")
    args = parser.parse_args()
    merge(Path(args.file_a), Path(args.file_b), Path(args.output))
