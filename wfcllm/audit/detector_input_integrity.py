from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from wfcllm.detection.code_only import validate_final_code_record_exact


def audit_detector_input_file(path: str | Path) -> dict[str, Any]:
    input_path = Path(path)
    errors: list[dict[str, object]] = []
    records_checked = 0

    for line_number, line in enumerate(
        input_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue

        records_checked += 1
        try:
            record = json.loads(line)
            validate_final_code_record_exact(record)
        except (json.JSONDecodeError, ValueError) as exc:
            errors.append({"line": line_number, "error": str(exc)})

    return {"ok": not errors, "records_checked": records_checked, "errors": errors}
