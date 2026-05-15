"""AblationRunner: iterate Cartesian product of a SweepSpec and aggregate metrics."""
from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Any, Callable, Iterable, Protocol

from wfcllm.ablation.metrics import get_metric
from wfcllm.ablation.sweep import ResolvedConfig, SweepSpec
from wfcllm.orchestration.phase_registry import PhaseRegistry
from wfcllm.orchestration.pipeline import PhaseOrchestrator
from wfcllm.orchestration.prereq import PrereqRegistry
from wfcllm.orchestration.state import RunStateManager

logger = logging.getLogger(__name__)


class _OrchestratorLike(Protocol):
    def dispatch_phase(self, phase: str, args: argparse.Namespace) -> int: ...


def _default_orchestrator_factory(state: RunStateManager) -> _OrchestratorLike:
    """Build a real PhaseOrchestrator backed by the standard CLI runner registrations."""
    from wfcllm.cli.entry import _populate_phase_registry

    registry = PhaseRegistry()
    _populate_phase_registry(registry)
    return PhaseOrchestrator(
        state=state,
        phase_registry=registry,
        prereq_registry=PrereqRegistry(),
    )


class AblationRunner:
    """Sequentially execute a SweepSpec, one combination at a time.

    Crash-isolated: a single combination's exception is caught, logged, and
    written to results.jsonl as `{"run_id": ..., "error": ...}`. The next
    combination is then attempted.
    """

    def __init__(
        self,
        orchestrator_factory: Callable[[RunStateManager], _OrchestratorLike] | None = None,
    ) -> None:
        self._factory = orchestrator_factory or _default_orchestrator_factory

    def run(self, spec: SweepSpec) -> None:
        spec.output_dir.mkdir(parents=True, exist_ok=True)
        results_path = spec.output_dir / "results.jsonl"
        already = self._load_done_run_ids(results_path)

        state = RunStateManager(path=spec.output_dir / "state.json")
        orchestrator = self._factory(state)

        for combo in spec.expand():
            if combo.run_id in already:
                logger.info("ablation: skip %s (resume)", combo.run_id)
                continue
            row = self._run_one(spec, combo, orchestrator)
            with open(results_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[ablation] {combo.run_id} {row.get('metrics') or row.get('error')}")

    def _run_one(
        self, spec: SweepSpec, combo: ResolvedConfig, orchestrator: _OrchestratorLike
    ) -> dict[str, Any]:
        run_dir = spec.output_dir / "runs" / combo.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "config.json").write_text(
            json.dumps(combo.config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        try:
            self._dispatch_phases(spec, combo, run_dir, orchestrator)
            metrics = self._collect_metrics(spec, combo, run_dir)
            return {
                "run_id": combo.run_id,
                "axes_values": combo.axes_values,
                "metrics": metrics,
            }
        except Exception as exc:  # noqa: BLE001 — log-and-continue is the contract
            logger.exception("ablation: combo %s failed", combo.run_id)
            return {
                "run_id": combo.run_id,
                "axes_values": combo.axes_values,
                "error": repr(exc),
            }

    def _dispatch_phases(
        self,
        spec: SweepSpec,
        combo: ResolvedConfig,
        run_dir: Path,
        orchestrator: _OrchestratorLike,
    ) -> None:
        for phase in spec.phases:
            args = self._build_args(combo, run_dir)
            rc = orchestrator.dispatch_phase(phase, args)
            if rc != 0:
                raise RuntimeError(f"phase '{phase}' returned non-zero rc={rc}")

    def _collect_metrics(
        self, spec: SweepSpec, combo: ResolvedConfig, run_dir: Path
    ) -> dict[str, Any]:
        artefacts = self._build_artefact_map(combo, run_dir)
        out: dict[str, Any] = {}
        for name in spec.metrics:
            extractor = get_metric(name)
            out[name] = extractor(artefacts)
        return out

    @staticmethod
    def _build_args(combo: ResolvedConfig, run_dir: Path) -> argparse.Namespace:
        """Synthesise the argparse.Namespace expected by phase runners.

        Mirrors what wfcllm.cli.entry:main attaches: `_config_cache`, `force=True`,
        `eval_only=False`, plus the small set of fields runners read directly from
        args (we copy the resolved config's relevant scalars so runners that
        prefer args-overrides over config still see the right values)."""
        ns = argparse.Namespace()
        ns._config_cache = copy.deepcopy(combo.config)
        ns.force = True
        ns.eval_only = False
        ns.resume = None
        ns.checkpoint = None
        ns.input_file = None
        ns.config = run_dir / "config.json"
        return ns

    @staticmethod
    def _build_artefact_map(combo: ResolvedConfig, run_dir: Path) -> dict[str, Path]:
        """Return the canonical artefact paths produced by the phases for this run.

        Keys here must match those used inside `wfcllm/ablation/metrics.py`.
        """
        cfg = combo.config
        wm_dir = Path(cfg.get("watermark", {}).get("output_dir") or run_dir / "watermark")
        ext_dir = Path(cfg.get("extract", {}).get("output_dir") or run_dir / "extract")
        # Best-effort artefact discovery — extractors return None when missing.
        extract_summary = next(iter(ext_dir.glob("*_summary.json")), None) if ext_dir.exists() else None
        evaluation_summary = ext_dir / "evaluation_summary.json"
        return {
            "extract_summary": extract_summary or ext_dir / "_missing.json",
            "evaluation_summary": evaluation_summary,
            "calibration": ext_dir / "calibration.json",
            "watermark_dir": wm_dir,
            "run_dir": run_dir,
        }

    @staticmethod
    def _load_done_run_ids(results_path: Path) -> set[str]:
        if not results_path.exists():
            return set()
        ids: set[str] = set()
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ids.add(json.loads(line)["run_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
        return ids
