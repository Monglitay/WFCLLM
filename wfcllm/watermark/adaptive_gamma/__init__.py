"""Adaptive gamma scheduling: entropy estimation, profile, schedule, calibrate.

Importing this package as a side effect registers the `entropy-profile` prereq
on the `legacy-watermark` phase. The official prereq's `fix` callback delegates
to the `legacy-build-entropy-profile` phase via PhaseOrchestrator.dispatch_phase.
The old `watermark` -> `build-entropy-profile` registration is retained for
compatibility with lower-level callers that still use legacy internal names.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.orchestration.prereq import Prereq, PrereqRegistry

_PREREQ_REGISTERED = False


def _config_block(config: dict) -> dict:
    config = config or {}
    return (
        ((config.get("watermark") or {}).get("adaptive_gamma") or {})
        or (config.get("adaptive_gamma") or {})
    )


def _check_entropy_profile(config: dict) -> bool:
    block = _config_block(config)
    if not block.get("enabled", False):
        return True
    profile_path = block.get("profile_path")
    if not profile_path:
        return False
    return Path(profile_path).exists()


def _make_entropy_profile_fixer(phase: str):
    def _fix_entropy_profile(config: dict, runner) -> None:
        rc = runner.dispatch_phase(phase, _build_entropy_profile_args(config))
        if rc != 0:
            raise RuntimeError(f"{phase} phase failed with exit code {rc}")

    return _fix_entropy_profile


def _build_entropy_profile_args(config: dict) -> argparse.Namespace:
    adaptive_gamma = _config_block(config)
    args = argparse.Namespace(
        build_profile_input_log=None,
        build_profile_output=None,
        build_profile_language=None,
        build_profile_model_family=None,
        build_profile_strategy=None,
        build_profile_id=None,
    )
    setattr(args, "_config_cache", {"adaptive_gamma": adaptive_gamma})
    return args


_fix_entropy_profile = _make_entropy_profile_fixer("legacy-build-entropy-profile")
_fix_old_entropy_profile = _make_entropy_profile_fixer("build-entropy-profile")


def _register_prereqs() -> None:
    global _PREREQ_REGISTERED
    if _PREREQ_REGISTERED:
        return
    registry = PrereqRegistry()
    registry.register(
        "legacy-watermark",
        Prereq(
            name="entropy-profile",
            check=_check_entropy_profile,
            fix=_fix_entropy_profile,
        ),
    )
    registry.register(
        "watermark",
        Prereq(
            name="entropy-profile",
            check=_check_entropy_profile,
            fix=_fix_old_entropy_profile,
        ),
    )
    _PREREQ_REGISTERED = True


_register_prereqs()
