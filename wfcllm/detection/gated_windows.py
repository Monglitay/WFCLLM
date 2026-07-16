"""Final-code-only recovery of formal gated semantic windows."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

from wfcllm.detection.config import GatedDetectionConfig
from wfcllm.gate.input import GATE_INPUT_CONTRACT_VERSION
from wfcllm.windowing import (
    WINDOW_CONTRACT_VERSION,
    CloseReason,
    GateDecision,
    GateScores,
    GateThresholds,
    ParentDescriptor,
    PythonStatementUnitExtractor,
    SemanticWindow,
    SkipReason,
    SkippedContext,
    StatementUnit,
    WindowPartitioner,
)


@dataclass(frozen=True)
class RecoveredGatedWindow:
    """Detection projection of one closed shared-partitioner window."""

    units: tuple[StatementUnit, ...]
    parent_descriptor: ParentDescriptor
    start_byte: int
    end_byte: int
    gate_scores: GateScores | None
    gate_decision: GateDecision
    close_reason: CloseReason
    suitable: bool
    skip_reason: SkipReason | None
    previous_context: tuple[StatementUnit, ...]

    @classmethod
    def from_semantic_window(cls, window: SemanticWindow) -> RecoveredGatedWindow:
        if not isinstance(window, SemanticWindow):
            raise ValueError("window must be a SemanticWindow")
        return cls(
            units=window.units,
            parent_descriptor=window.parent_descriptor,
            start_byte=window.start_byte,
            end_byte=window.end_byte,
            gate_scores=window.gate_scores,
            gate_decision=window.gate_decision,
            close_reason=window.close_reason,
            suitable=window.suitable,
            skip_reason=window.skip_reason,
            previous_context=window.previous_context,
        )

    @property
    def byte_span(self) -> tuple[int, int]:
        return self.start_byte, self.end_byte

    @property
    def closed(self) -> bool:
        return True

    @property
    def uncertain(self) -> bool:
        return self.close_reason in {
            CloseReason.UNCERTAIN,
            CloseReason.UNRELIABLE_GATE,
        }

    @property
    def overflow(self) -> bool:
        return self.close_reason is CloseReason.INPUT_OVERFLOW


@dataclass(frozen=True)
class GatedWindowExtraction:
    windows: tuple[RecoveredGatedWindow, ...]
    skipped_context: tuple[SkippedContext, ...]


class GatedWindowExtractor:
    """Recover gated windows from final code and no generation-time evidence.

    The bundle passed here is already a validated runtime bundle.  It must
    expose its formal manifest, one bound stable predictor, and the tokenizer
    counter used by the gate.  This keeps ``extract`` deliberately limited to
    one argument: the final code string.
    """

    def __init__(
        self,
        bundle: object,
        config: GatedDetectionConfig | None = None,
    ) -> None:
        self._bundle = bundle
        self._config = config
        manifest = _validate_bundle(bundle, config)
        predictor = _bound_stable_predictor(bundle)
        tokenizer_counter = _tokenizer_counter(bundle)
        self.unit_extractor = PythonStatementUnitExtractor()
        self.partitioner = WindowPartitioner(
            predictor=predictor,
            thresholds=GateThresholds(
                close_low=manifest.close_low_threshold,
                close_high=manifest.close_high_threshold,
                suitable_accept=manifest.suitable_accept_threshold,
                max_units=3,
                max_input_tokens=manifest.max_tokens,
            ),
            tokenizer_counter=tokenizer_counter,
        )

    def extract(self, final_code: str) -> GatedWindowExtraction:
        if not isinstance(final_code, str):
            raise ValueError("final_code must be a string")
        units = self.unit_extractor.extract(final_code)
        partitioned = self.partitioner.partition(units)
        return GatedWindowExtraction(
            windows=self.project_windows(partitioned.windows),
            skipped_context=partitioned.skipped_context,
        )

    @staticmethod
    def project_windows(
        windows: Sequence[SemanticWindow],
    ) -> tuple[RecoveredGatedWindow, ...]:
        return tuple(RecoveredGatedWindow.from_semantic_window(item) for item in windows)


def gate_bundle_tree_sha256(root: Path) -> str:
    """Return the artifact-tree digest used by gate publication manifests."""

    if not isinstance(root, Path) or not root.is_dir():
        raise ValueError("gate bundle directory is missing")
    digest = hashlib.sha256(b"wfcllm-artifact-tree/v1\0")
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        if path.is_symlink():
            raise ValueError("gate bundle contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("gate bundle contains an unsupported entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _validate_bundle(bundle: object, config: GatedDetectionConfig | None) -> Any:
    summary = getattr(bundle, "validation_summary", None)
    if not isinstance(summary, Mapping) or summary.get("validated") is not True:
        raise ValueError("gated extraction requires a validated gate bundle")
    manifest = getattr(bundle, "manifest", None)
    if manifest is None:
        raise ValueError("validated gate bundle manifest is required")

    if getattr(manifest, "window_contract_version", None) != WINDOW_CONTRACT_VERSION:
        raise ValueError("gate bundle window contract mismatch")
    if (
        getattr(manifest, "gate_input_contract_version", None)
        != GATE_INPUT_CONTRACT_VERSION
    ):
        raise ValueError("gate bundle input contract mismatch")
    tokenizer_sha256 = getattr(manifest, "tokenizer_sha256", None)
    declared_tokenizer_sha256 = getattr(bundle, "tokenizer_sha256", tokenizer_sha256)
    if tokenizer_sha256 != declared_tokenizer_sha256:
        raise ValueError("gate bundle tokenizer hash mismatch")

    root = getattr(bundle, "root", None)
    if config is not None:
        if not isinstance(config, GatedDetectionConfig):
            raise ValueError("config must be a GatedDetectionConfig")
        if not isinstance(root, Path) or root != config.gate_bundle_path:
            raise ValueError("gate bundle path does not match detection config")
        if config.window_contract_version != getattr(
            manifest, "window_contract_version", None
        ):
            raise ValueError("gate bundle window contract mismatch")
        if config.gate_input_contract_version != getattr(
            manifest, "gate_input_contract_version", None
        ):
            raise ValueError("gate bundle input contract mismatch")
        expected_hash = config.gate_bundle_sha256
    else:
        expected_hash = getattr(bundle, "bundle_sha256", None)

    actual_hash: object
    if isinstance(root, Path) and root.is_dir():
        actual_hash = gate_bundle_tree_sha256(root)
    else:
        actual_hash = getattr(bundle, "bundle_sha256", None)
    if expected_hash is not None and actual_hash != expected_hash:
        raise ValueError("gate bundle hash mismatch")
    return manifest


def _bound_stable_predictor(bundle: object) -> Any:
    candidate = getattr(bundle, "stable_gate_predictor", None)
    if candidate is None:
        candidate = getattr(bundle, "stable_predictor", None)
    if callable(candidate) and not callable(getattr(candidate, "predict", None)):
        try:
            candidate = candidate()
        except TypeError as exc:
            raise ValueError(
                "gate bundle must expose a bound StableGatePredictor"
            ) from exc
    if not callable(getattr(candidate, "predict", None)):
        raise ValueError("gate bundle must expose a bound StableGatePredictor")
    return candidate


def _tokenizer_counter(bundle: object) -> Callable[[str], int]:
    counter = getattr(bundle, "tokenizer_counter", None)
    if counter is None:
        counter = getattr(bundle, "count_tokens", None)
    if not callable(counter):
        raise ValueError("gate bundle must expose its tokenizer counter")
    return counter


__all__ = [
    "GatedWindowExtraction",
    "GatedWindowExtractor",
    "RecoveredGatedWindow",
    "gate_bundle_tree_sha256",
]
