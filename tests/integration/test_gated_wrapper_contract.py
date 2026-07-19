from __future__ import annotations

from pathlib import Path


def test_gated_wrapper_requires_separate_pilot_and_full_catalogs() -> None:
    script = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "run_gated_single_gpu.sh"
    ).read_text(encoding="utf-8")

    assert 'PILOT_SOURCE_CATALOG:?set PILOT_SOURCE_CATALOG' in script
    assert 'FULL_SOURCE_CATALOG:?set FULL_SOURCE_CATALOG' in script
    assert '--gate-source-catalog "${PILOT_SOURCE_CATALOG}"' in script
    assert '--gate-source-catalog "${FULL_SOURCE_CATALOG}"' in script
    assert 'CONDA_ENVS_PATH="${CONDA_ENVS_PATH:-/root/autodl-tmp/conda/envs}"' in script
