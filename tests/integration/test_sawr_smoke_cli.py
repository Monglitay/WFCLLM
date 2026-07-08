from __future__ import annotations

from unittest.mock import patch

import scripts.run_sawr_smoke as sawr_smoke_cli


def test_run_sawr_smoke_deprecated_wrapper_delegates(capsys) -> None:
    with patch("scripts.run_sawr_smoke.wfcllm_generate_main", return_value=0) as main:
        rc = sawr_smoke_cli.main(["--model-path", "model"])

    assert rc == 0
    main.assert_called_once_with(["--model-path", "model"])
    assert "[deprecated] use scripts/wfcllm_generate.py" in capsys.readouterr().err
