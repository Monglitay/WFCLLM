from __future__ import annotations

from wfcllm.cli.arguments import build_parser
from wfcllm.cli.entry import _populate_phase_registry
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.state import PHASES


def test_default_phase_list_is_new_wfcllm_mainline() -> None:
    assert PHASES == ["generate", "calibrate", "detect", "report", "audit"]


def test_parser_accepts_new_official_phases() -> None:
    parser = build_parser()

    for phase in [
        "generate",
        "calibrate",
        "detect",
        "report",
        "audit",
        "posthoc-pass-report",
        "diagnostic-selector",
    ]:
        args = parser.parse_args(["--phase", phase])
        assert args.phase == phase


def test_parser_does_not_expose_allow_quality_proxy() -> None:
    parser = build_parser()

    option_strings = {
        option
        for action in parser._actions
        for option in action.option_strings
    }
    assert "--allow-quality-proxy" not in option_strings


def test_phase_registry_registers_new_mainline_runners() -> None:
    registry = PhaseRegistry()
    _populate_phase_registry(registry)

    for phase in ["generate", "calibrate", "detect", "report", "audit"]:
        assert callable(registry.get(phase))
