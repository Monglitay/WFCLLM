from __future__ import annotations

import pytest

from wfcllm.detection.code_only import validate_final_code_record_exact


def test_official_detector_rejects_generated_code_even_with_prompt() -> None:
    with pytest.raises(ValueError, match="generated_code"):
        validate_final_code_record_exact(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def target():\n",
                "generated_code": "    return 1\n",
            }
        )
