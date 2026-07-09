from __future__ import annotations

import json
from pathlib import Path


def test_legacy_archive_manifest_exists_and_names_archived_roots() -> None:
    manifest_path = Path("archive/legacy_wfcllm_2026_07/MANIFEST.json")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["archive_id"] == "legacy_wfcllm_2026_07"
    archived_paths = {entry["path"] for entry in manifest["entries"]}
    assert "archive/legacy_wfcllm_2026_07/code/watermark" in archived_paths
    assert "archive/legacy_wfcllm_2026_07/code/extract" in archived_paths
    assert "configs/legacy" in archived_paths
    assert "scripts/legacy" in archived_paths
