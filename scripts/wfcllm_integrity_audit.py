#!/usr/bin/env python
"""Audit final-code contracts, artifact hashes, and exact secret leakage."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.audit.artifact_integrity import reject_secret_key_leak  # noqa: E402
from wfcllm.audit.detector_input_integrity import (  # noqa: E402
    audit_detector_input_file,
)

SCHEMA_VERSION = "wfcllm-integrity-audit/v1"
CHUNK_SIZE = 1024 * 1024


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit strict detector inputs, public artifact hashes, and exact "
            "raw-secret absence without printing the secret."
        ),
    )
    parser.add_argument("--repo", required=True)
    parser.add_argument("--detector-input", action="append", required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    parser.add_argument("--artifact", action="append", default=[])
    parser.add_argument("--scan-root", action="append", default=[])
    parser.add_argument("--scan-git-tracked", action="store_true")
    parser.add_argument("--secret-file", required=True)
    parser.add_argument("--output", required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contains_bytes(path: Path, needle: bytes) -> bool:
    overlap = max(0, len(needle) - 1)
    previous = b""
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            value = previous + chunk
            if needle in value:
                return True
            previous = value[-overlap:] if overlap else b""
    return False


def _files_under(root: Path) -> list[Path]:
    if root.is_file() and not root.is_symlink():
        return [root]
    if not root.is_dir():
        raise ValueError(f"scan root does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    )


def _git_value(repo: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _git_tracked_files(repo: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "-C", str(repo), "ls-files", "-z"],
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise ValueError(f"cannot list Git-tracked files under {repo}")
    return [
        repo / raw_path.decode("utf-8")
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def _public_payloads(path: Path) -> list[Any]:
    if path.suffix == ".json":
        return [json.loads(path.read_text(encoding="utf-8"))]
    if path.suffix == ".jsonl":
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def _artifact_report(path: Path) -> dict[str, Any]:
    report: dict[str, Any] = {
        "path": str(path),
        "exists": path.is_file(),
    }
    if not path.is_file():
        report["ok"] = False
        report["error"] = "artifact is not a file"
        return report
    report["size_bytes"] = path.stat().st_size
    report["sha256"] = _sha256(path)
    try:
        for payload in _public_payloads(path):
            reject_secret_key_leak(payload)
    except (json.JSONDecodeError, ValueError) as exc:
        report["ok"] = False
        report["error"] = str(exc)
        return report
    report["ok"] = True
    return report


def _build_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_records <= 0:
        raise ValueError("expected_records must be positive")
    repo = Path(args.repo).resolve()
    secret_path = Path(args.secret_file).resolve()
    secret = secret_path.read_bytes().strip()
    if not secret:
        raise ValueError("secret file must contain a non-empty value")

    detector_reports: list[dict[str, Any]] = []
    for value in args.detector_input:
        path = Path(value).resolve()
        contract = audit_detector_input_file(path)
        count_ok = contract["records_checked"] == args.expected_records
        detector_reports.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                **contract,
                "expected_records": args.expected_records,
                "record_count_ok": count_ok,
            }
        )

    artifact_reports = [
        _artifact_report(Path(value).resolve()) for value in args.artifact
    ]
    tracked_files = _git_tracked_files(repo) if args.scan_git_tracked else []
    extra_files = [
        path
        for value in args.scan_root
        for path in _files_under(Path(value).resolve())
    ]
    unique_files = sorted({path.resolve() for path in [*tracked_files, *extra_files]})
    if secret_path in unique_files:
        raise ValueError("secret file must not be inside the public scan set")
    matching_paths = [
        str(path) for path in unique_files if _contains_bytes(path, secret)
    ]
    secret = b""

    detector_ok = all(
        report["ok"] and report["record_count_ok"]
        for report in detector_reports
    )
    artifacts_ok = all(report["ok"] for report in artifact_reports)
    return {
        "artifact_type": "wfcllm_integrity_audit",
        "schema_version": SCHEMA_VERSION,
        "ok": detector_ok and artifacts_ok and not matching_paths,
        "repo": str(repo),
        "git": {
            "branch": _git_value(repo, "branch", "--show-current"),
            "head": _git_value(repo, "rev-parse", "HEAD"),
            "tracked_files_scanned": len(tracked_files),
        },
        "detector_inputs": detector_reports,
        "artifacts": artifact_reports,
        "secret_scan": {
            "raw_secret_in_report": False,
            "files_scanned": len(unique_files),
            "extra_files_scanned": len(extra_files),
            "match_count": len(matching_paths),
            "matching_paths": matching_paths,
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    output = Path(args.output)
    try:
        report = _build_report(args)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        report = {
            "artifact_type": "wfcllm_integrity_audit",
            "schema_version": SCHEMA_VERSION,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"integrity audit {'passed' if report['ok'] else 'failed'}: {output}",
        file=sys.stderr,
    )
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
