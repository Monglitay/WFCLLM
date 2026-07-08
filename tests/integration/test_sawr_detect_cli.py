from __future__ import annotations

from unittest.mock import patch

import scripts.run_sawr_detect as sawr_detect_cli


def test_run_sawr_detect_deprecated_wrapper_delegates(capsys) -> None:
    with patch("scripts.run_sawr_detect.wfcllm_detect_main", return_value=0) as main:
        rc = sawr_detect_cli.main(["split", "--input", "rows.jsonl"])

    assert rc == 0
    main.assert_called_once_with(["split", "--input", "rows.jsonl"])
    assert "[deprecated] use scripts/wfcllm_detect.py" in capsys.readouterr().err
