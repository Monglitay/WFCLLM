from __future__ import annotations

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "statement",
    [
        "import wfcllm.semantic",
        "import wfcllm.semantic.rules",
        "import wfcllm.semantic.lsh",
        "import wfcllm.semantic.keying",
        "import wfcllm.semantic.verifier",
        "import wfcllm.semantic.window_lsh",
        (
            "from wfcllm.semantic import "
            "SemanticWindowEvidence, SemanticWindowScorer"
        ),
    ],
)
def test_semantic_imports_succeed_in_clean_process(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_semantic_dir_lists_lazy_window_exports() -> None:
    import wfcllm.semantic as semantic

    assert "SemanticWindowEvidence" in dir(semantic)
    assert "SemanticWindowScorer" in dir(semantic)


def test_semantic_unknown_attribute_raises_attribute_error() -> None:
    import wfcllm.semantic as semantic

    with pytest.raises(AttributeError, match="does_not_exist"):
        getattr(semantic, "does_not_exist")
