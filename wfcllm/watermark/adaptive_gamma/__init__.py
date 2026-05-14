"""Adaptive gamma scheduling: entropy estimation, profile, schedule, calibrate.

Importing this package as a side effect registers the `entropy-profile` prereq
on the `watermark` phase. The prereq's `fix` callback delegates to the
`build-entropy-profile` phase via PhaseOrchestrator.dispatch_phase.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from wfcllm.orchestration.prereq import Prereq, PrereqRegistry

_PREREQ_REGISTERED = False


def _config_block(config: dict) -> dict:
    return ((config or {}).get("watermark") or {}).get("adaptive_gamma") or {}


def _check_entropy_profile(config: dict) -> bool:
    block = _config_block(config)
    if not block.get("enabled", False):
        return True
    profile_path = block.get("profile_path")
    if not profile_path:
        return False
    return Path(profile_path).exists()


def _fix_entropy_profile(config: dict, runner) -> None:
    rc = runner.dispatch_phase("build-entropy-profile", argparse.Namespace())
    if rc != 0:
        raise RuntimeError(
            f"build-entropy-profile phase failed with exit code {rc}"
        )


def _register_prereqs() -> None:
    global _PREREQ_REGISTERED
    if _PREREQ_REGISTERED:
        return
    PrereqRegistry().register(
        "watermark",
        Prereq(
            name="entropy-profile",
            check=_check_entropy_profile,
            fix=_fix_entropy_profile,
        ),
    )
    _PREREQ_REGISTERED = True


_register_prereqs()
