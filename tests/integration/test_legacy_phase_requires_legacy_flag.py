from __future__ import annotations

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.entry import validate_legacy_phase_request


def test_legacy_phase_requires_legacy_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--phase", "legacy-watermark"])

    assert validate_legacy_phase_request(args) == "[错误] legacy phase requires --legacy"


def test_legacy_phase_allows_explicit_legacy_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--phase", "legacy-watermark", "--legacy"])

    assert validate_legacy_phase_request(args) is None
