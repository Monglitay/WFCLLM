from __future__ import annotations

import inspect
import json
from pathlib import Path

from wfcllm.dynamic_semantic.config import load_public_config
from wfcllm.dynamic_semantic.detector import R3Detector


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_v3_public_config_contains_no_secret_or_key_fingerprint() -> None:
    path = REPO_ROOT / "configs/wfcllm/v3_dynamic_semantic_public.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    text = json.dumps(payload, sort_keys=True).lower()

    load_public_config(path)
    assert "secret" not in text
    assert "fingerprint" not in text
    assert "private_key" not in text


def test_r3_per_sample_api_accepts_only_final_code() -> None:
    signature = inspect.signature(R3Detector.detect)

    assert list(signature.parameters) == ["self", "final_code"]


def test_v3_production_modules_do_not_import_legacy_or_syntax_codebook() -> None:
    root = REPO_ROOT / "wfcllm/dynamic_semantic"
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(root.glob("*.py"))
    )

    assert "from experiment" not in source
    assert "import experiment" not in source
    assert "code-watermark-main" not in source
    assert "syntax_codebook" not in source
