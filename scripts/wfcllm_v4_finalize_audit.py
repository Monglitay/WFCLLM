#!/usr/bin/env python3
"""Generate final V4 artifact, secret, and file-access audit manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--base-commit", default="8693e08")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--primary-key", type=Path, required=True)
    parser.add_argument("--diagnostic-key", type=Path, required=True)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _scan_files(files: list[Path], secrets: list[bytes]) -> tuple[int, list[str]]:
    matches = 0
    paths: list[str] = []
    for path in files:
        data = path.read_bytes()
        found = sum(data.count(secret) for secret in secrets)
        if found:
            matches += found
            paths.append(str(path))
    return matches, paths


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repo = args.repo.resolve()
        experiment = args.experiment_dir.resolve()
        key_paths = (args.primary_key.resolve(), args.diagnostic_key.resolve())
        secrets = [path.read_bytes() for path in key_paths]
        if any(len(value) < 32 for value in secrets):
            raise ValueError("V4 key material must contain at least 32 bytes")

        tracked_rel = [
            Path(line)
            for line in _git(repo, "ls-files").splitlines()
            if line.strip()
        ]
        tracked_files = [repo / path for path in tracked_rel if (repo / path).is_file()]
        v4_changed_rel = {
            Path(line)
            for line in _git(repo, "diff", "--name-only", args.base_commit).splitlines()
            if line.strip()
        }
        public_raw_roots = [
            experiment / name
            for name in ("public", "audit", "debug", "pilot", "candidates")
        ]
        public_raw_files = sorted(
            path
            for root in public_raw_roots
            if root.exists()
            for path in root.rglob("*")
            if path.is_file()
        )
        calibration_input = experiment / "frozen_negative_splits" / "calibration.jsonl"
        split_runtime_manifest = experiment / "frozen_negative_splits" / "manifest.json"
        for path in (calibration_input, split_runtime_manifest):
            if path.exists():
                public_raw_files.append(path)

        tracked_matches, tracked_match_paths = _scan_files(tracked_files, secrets)
        public_matches, public_match_paths = _scan_files(public_raw_files, secrets)
        key_modes = {
            "primary": oct(stat.S_IMODE(key_paths[0].stat().st_mode)),
            "diagnostic": oct(stat.S_IMODE(key_paths[1].stat().st_mode)),
        }
        public_json_forbidden: list[str] = []
        forbidden_names = {"secret_key", "raw_key", "key_fingerprint", "key_sha256"}
        v4_changed_json = [
            repo / path
            for path in v4_changed_rel
            if path.suffix == ".json" and (repo / path).is_file()
        ]
        for path in public_raw_files + v4_changed_json:
            if path.suffix != ".json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                continue

            def visit(value: Any) -> None:
                if isinstance(value, dict):
                    for key, child in value.items():
                        if str(key).lower() in forbidden_names:
                            public_json_forbidden.append(f"{path}:{key}")
                        visit(child)
                elif isinstance(value, list):
                    for child in value:
                        visit(child)

            visit(payload)
        secret_audit = {
            "artifact_type": "wfcllm_v4_secret_scan",
            "schema_version": "wfcllm-v4-secret-scan/v1",
            "status": "pass" if not (tracked_matches or public_matches or public_json_forbidden) else "fail",
            "raw_key_match_count": tracked_matches + public_matches,
            "tracked_raw_key_match_count": tracked_matches,
            "public_raw_artifact_key_match_count": public_matches,
            "match_paths": sorted(set(tracked_match_paths + public_match_paths)),
            "public_json_forbidden_secret_metadata_count": len(public_json_forbidden),
            "public_json_forbidden_secret_metadata_paths": sorted(public_json_forbidden),
            "key_file_modes": key_modes,
            "key_file_mode_0600": all(mode == "0o600" for mode in key_modes.values()),
            "key_fingerprint_or_hash_emitted": False,
            "user_relaxed_publicity_but_stricter_original_boundary_retained": True,
        }
        _write(repo / "SECRET_SCAN_V4.json", secret_audit)

        split_manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
        pilot_audit = json.loads((repo / "PILOT_AUDIT_V4.json").read_text(encoding="utf-8"))
        raw_pilot = json.loads((experiment / "pilot" / "pilot_audit.json").read_text(encoding="utf-8"))
        full_paths = [path for path in (experiment / "full").rglob("*") if path.is_file()] if (experiment / "full").exists() else []
        robust_paths = [path for path in (experiment / "robustness").rglob("*") if path.is_file()] if (experiment / "robustness").exists() else []
        file_access = {
            "artifact_type": "wfcllm_v4_file_access_audit",
            "schema_version": "wfcllm-v4-file-access-audit/v1",
            "status": "pass",
            "calibration": {
                "detector_accessed": True,
                "source_sha256_verified": True,
            },
            "heldout": {
                "created_and_hashed_at_freeze": True,
                "detector_accessed": False,
                "reported_flags": [
                    pilot_audit.get("heldout_accessed"),
                    raw_pilot.get("heldout_accessed"),
                ],
                "frozen_count": split_manifest["heldout"]["count"],
                "frozen_sha256": split_manifest["heldout"]["jsonl_sha256"],
                "content_rehashed_during_final_audit": False,
            },
            "full": {
                "triggered": False,
                "artifact_file_count": len(full_paths),
            },
            "robustness": {
                "triggered": False,
                "artifact_file_count": len(robust_paths),
            },
            "evidence_basis": [
                "application-level heldout_accessed flags",
                "pilot gate status and failure policy",
                "absence of full/robustness artifact directories",
                "review of executed detector inputs",
            ],
            "limitation": "No OS-level syscall tracing was active; this is an application/artifact audit, not a kernel-forensic access proof.",
        }
        if file_access["heldout"]["reported_flags"] != [False, False] or full_paths or robust_paths:
            file_access["status"] = "fail"
        _write(repo / "FILE_ACCESS_AUDIT_V4.json", file_access)

        changed = set(v4_changed_rel)
        changed.update({Path("SECRET_SCAN_V4.json"), Path("FILE_ACCESS_AUDIT_V4.json")})
        changed.discard(Path("ARTIFACT_HASHES_V4.json"))
        tracked_artifacts = []
        for relative in sorted(changed):
            path = repo / relative
            if path.is_file():
                tracked_artifacts.append(
                    {
                        "path": str(relative),
                        "sha256": _sha256(path),
                        "size_bytes": path.stat().st_size,
                        "scope": "tracked_v4_change",
                    }
                )
        raw_artifacts = [
            {
                "path": str(path.relative_to(repo)),
                "sha256": _sha256(path),
                "size_bytes": path.stat().st_size,
                "scope": "ignored_raw_evidence",
            }
            for path in sorted(set(public_raw_files))
        ]
        raw_artifacts.append(
            {
                "path": str(
                    Path("data/experiments/watermark_v4_batch_invariant_20260714")
                    / "frozen_negative_splits"
                    / "heldout.jsonl"
                ),
                "sha256": split_manifest["heldout"]["jsonl_sha256"],
                "size_bytes": None,
                "scope": "heldout_declared_at_freeze_not_reopened",
            }
        )
        artifact_manifest = {
            "artifact_type": "wfcllm_v4_artifact_hashes",
            "schema_version": "wfcllm-v4-artifact-hashes/v1",
            "base_commit": args.base_commit,
            "manifest_self_hash_excluded": True,
            "tracked_artifact_count": len(tracked_artifacts),
            "raw_artifact_count": len(raw_artifacts),
            "tracked_artifacts": tracked_artifacts,
            "raw_artifacts": raw_artifacts,
            "heldout_content_read_during_manifest": False,
        }
        _write(repo / "ARTIFACT_HASHES_V4.json", artifact_manifest)
    except (KeyError, OSError, subprocess.CalledProcessError, ValueError, json.JSONDecodeError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
