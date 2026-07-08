from __future__ import annotations

import pytest

from wfcllm.detection.pipeline import validate_final_code_detector_input_record


def test_official_detector_rejects_generated_code_even_with_prompt() -> None:
    with pytest.raises(ValueError, match="generated_code"):
        validate_final_code_detector_input_record(
            {
                "id": "HumanEval/0",
                "dataset": "humaneval",
                "prompt": "def target():\n",
                "generated_code": "    return 1\n",
            }
        )
