from __future__ import annotations

import json
from pathlib import Path

import scripts.wfcllm_integrity_audit as audit_cli


def _write_final(path: Path, count: int) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "id": f"HumanEval/{index}",
                    "dataset": "humaneval",
                    "prompt": f"def f{index}():\n",
                    "final_code": f"def f{index}():\n    return {index}\n",
                }
            )
            + "\n"
            for index in range(count)
        ),
        encoding="utf-8",
    )


def test_integrity_audit_records_hashes_contract_and_clean_secret_scan(
    tmp_path: Path,
) -> None:
    final = tmp_path / "final.jsonl"
    artifact = tmp_path / "report.json"
    scan_root = tmp_path / "public"
    scan_root.mkdir()
    (scan_root / "config.json").write_text('{"threshold": 1.0}\n')
    artifact.write_text('{"schema_version": "example/v1"}\n')
    _write_final(final, 2)
    secret_file = tmp_path / "private.key"
    secret_file.write_text("do-not-publish-this-value\n")
    output = tmp_path / "audit.json"

    rc = audit_cli.main(
        [
            "--repo",
            str(tmp_path),
            "--detector-input",
            str(final),
            "--expected-records",
            "2",
            "--artifact",
            str(artifact),
            "--scan-root",
            str(scan_root),
            "--secret-file",
            str(secret_file),
            "--output",
            str(output),
        ]
    )

    assert rc == 0
    report = json.loads(output.read_text())
    assert report["schema_version"] == "wfcllm-integrity-audit/v1"
    assert report["ok"] is True
    assert report["detector_inputs"][0]["records_checked"] == 2
    assert report["detector_inputs"][0]["sha256"]
    assert report["artifacts"][0]["sha256"]
    assert report["secret_scan"]["match_count"] == 0
    assert "do-not-publish-this-value" not in output.read_text()


def test_integrity_audit_fails_without_echoing_leaked_secret(tmp_path: Path) -> None:
    final = tmp_path / "final.jsonl"
    _write_final(final, 1)
    secret_file = tmp_path / "private.key"
    secret = "uniquely-sensitive-test-value"
    secret_file.write_text(secret + "\n")
    scan_root = tmp_path / "public"
    scan_root.mkdir()
    leaked = scan_root / "leaked.json"
    leaked.write_text(json.dumps({"value": secret}) + "\n")
    output = tmp_path / "audit.json"

    rc = audit_cli.main(
        [
            "--repo",
            str(tmp_path),
            "--detector-input",
            str(final),
            "--expected-records",
            "1",
            "--scan-root",
            str(scan_root),
            "--secret-file",
            str(secret_file),
            "--output",
            str(output),
        ]
    )

    assert rc == 1
    report_text = output.read_text()
    report = json.loads(report_text)
    assert report["ok"] is False
    assert report["secret_scan"]["match_count"] == 1
    assert str(leaked) in report["secret_scan"]["matching_paths"]
    assert secret not in report_text


def test_integrity_audit_fails_on_wrong_detector_input_count(tmp_path: Path) -> None:
    final = tmp_path / "final.jsonl"
    _write_final(final, 1)
    secret_file = tmp_path / "private.key"
    secret_file.write_text("private-value\n")
    output = tmp_path / "audit.json"

    rc = audit_cli.main(
        [
            "--repo",
            str(tmp_path),
            "--detector-input",
            str(final),
            "--expected-records",
            "2",
            "--secret-file",
            str(secret_file),
            "--output",
            str(output),
        ]
    )

    assert rc == 1
    report = json.loads(output.read_text())
    assert report["detector_inputs"][0]["record_count_ok"] is False
