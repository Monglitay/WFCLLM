from __future__ import annotations

import pytest

from wfcllm.cli import runners


def test_public_audit_reader_has_32_mib_bounded_limit(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert runners._MAX_PUBLIC_AUDIT_ARTIFACT_BYTES == 32 * 1024 * 1024

    monkeypatch.setattr(runners, "_MAX_PUBLIC_AUDIT_ARTIFACT_BYTES", 32)
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}" + " " * 30, encoding="utf-8")
    assert runners._read_public_json_artifacts(artifact) == [{}]

    artifact.write_text("{}" + " " * 31, encoding="utf-8")
    with pytest.raises(ValueError, match="exceeds size limit"):
        runners._read_public_json_artifacts(artifact)
