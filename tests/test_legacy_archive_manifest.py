from __future__ import annotations

import json
from pathlib import Path


def test_legacy_archive_manifest_exists_and_names_archived_roots() -> None:
    manifest_path = Path("archive/legacy_wfcllm_2026_07/MANIFEST.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["archive_id"] == "legacy_wfcllm_2026_07"
    archived_paths = {entry["path"] for entry in manifest["entries"]}
    expected_paths = {
        "archive/legacy_wfcllm_2026_07/code/watermark",
        "archive/legacy_wfcllm_2026_07/code/extract",
        "archive/legacy_wfcllm_2026_07/code/ablation",
        "archive/legacy_wfcllm_2026_07/code/pretrain",
        "archive/legacy_wfcllm_2026_07/code/dual_channel",
        "archive/legacy_wfcllm_2026_07/tests",
        "configs/legacy",
        "scripts/legacy",
        "docs/archive",
    }
    assert expected_paths <= archived_paths
