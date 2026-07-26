#!/usr/bin/env python3
"""Thin offline CLI for the low-level semantic-gate data/train pipelines."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
from typing import Any
from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.gate.pipeline import (  # noqa: E402
    GateDataPipelineConfig,
    GatePipelineGroup,
    GateTrainPipelineConfig,
    KeyBankSnapshot,
    run_gate_data,
    run_gate_train,
)
from wfcllm.gate.feasibility import FEASIBILITY_THRESHOLD_ITEMS  # noqa: E402
from wfcllm.gate.data import LshProbeResult  # noqa: E402
from wfcllm.gate.schema import CandidateObservation  # noqa: E402
from wfcllm.common.secrets import load_secret  # noqa: E402
from wfcllm.gate.dependencies import build_local_gate_dependencies  # noqa: E402

_MAX_CONFIG_BYTES = 1024 * 1024


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline semantic-gate artifact pipelines")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("data", "train"):
        child = subparsers.add_parser(name)
        child.add_argument("--config", type=Path, required=True)
        child.add_argument("--backend", choices=("real", "fake"), default="real")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        raw = _load_config(args.config)
        dependencies = (
            _FakeDependencies(raw)
            if args.backend == "fake"
            else build_local_gate_dependencies(
                source_manifest=_optional_path(raw, "source_manifest"),
                training_key_file=raw.get("training_key_source"),
                training_key_env=None,
                holdout_key_file=raw.get("holdout_key_source"),
                holdout_key_env=None,
                base_model_path=_optional_path(raw, "base_model_path"),
            )
        )
        if args.command == "data":
            _require_key_sources(raw)
            config = GateDataPipelineConfig(
                output_root=_required_path(raw, "output_root"),
                scale=_required_string(raw, "scale"),
                config_hash=_digest_value(raw, "config_hash"),
                parser_contract=str(raw.get("parser_contract", "wfcllm-window/v1")),
                rewriter_config_hash=_digest_value(raw, "rewriter_config_hash"),
                semantic_encoder_hash=_digest_value(raw, "semantic_encoder_hash"),
                lsh_config_hash=_digest_value(raw, "lsh_config_hash"),
                feasibility_contract=_required_string(raw, "feasibility_contract_version"),
                feasibility_thresholds=_feasibility_thresholds(raw),
                pilot_feasibility_path=_optional_path(raw, "pilot_feasibility_path"),
            )
            result = run_gate_data(config, dependencies)
            output = {"manifest_path": str(result.manifest_path), "group_count": result.group_count, "diagnostic_test_backend": args.backend == "fake"}
        else:
            config = GateTrainPipelineConfig(
                output_root=_required_path(raw, "output_root"),
                data_dir=_required_path(raw, "data_dir"),
                pilot_feasibility_path=_optional_path(raw, "pilot_feasibility_path"),
                config_hash=_digest_value(raw, "config_hash"),
            )
            result = run_gate_train(config, dependencies)
            output = {"candidate_bundle": str(result.candidate_bundle_path), "diagnostic_test_backend": args.backend == "fake"}
        print(json.dumps(output, allow_nan=False, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, UnicodeError, ValueError, RuntimeError) as exc:
        print(f"wfcllm_gate: {exc}", file=sys.stderr)
        return 2


class _FakeDependencies:
    """Deterministic CI wiring backend; its artifacts are never formal eligible."""

    diagnostic_test_backend = True

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw

    def load_source_manifest(self, config):
        return {"schema_version": "wfcllm-gate-source-manifest/v1", "sources": [{"source_id": "diagnostic-local-source"}]}

    def load_key_bank(self, *, role, expected_count, config):
        source = _required_path(self.raw, f"{role}_key_source")
        if not source.is_file() or source.is_symlink():
            raise ValueError(f"{role} key source is missing or unsafe")
        digest = hashlib.sha256(load_secret(secret_file=source, env_name=None)).hexdigest()
        return KeyBankSnapshot(
            tuple(f"{'train' if role == 'training' else 'holdout'}-key-{index:03d}" for index in range(expected_count)),
            f"{role}-key-bank/v1:sha256:{digest}",
        )

    def parse_statement_units(self, source_manifest, config):
        return ("diagnostic",)

    def generate_candidate_trajectories(self, parsed_sources, config):
        count = self.raw.get("fake_group_count", 100 if config.scale == "pilot" else 1_000)
        if type(count) is not int or count <= 0:
            raise ValueError("fake_group_count must be a positive integer")
        positive_count = 15 if config.scale == "pilot" else 150
        for index in range(count):
            split = "validation" if index % 10 == 0 else "test" if index % 10 == 1 else "train"
            suitable = index < positive_count
            group_id = f"diagnostic-group-{index:05d}"
            yield GatePipelineGroup(
                group_id=group_id,
                    split_group_id=f"repository:diagnostic-repo-{index:05d}",
                    split=split,
                    suitable_target=suitable,
                    close_target=True,
                    window_lengths=(1, 2, 3),
                    statement_family=("assignment", "branch", "loop", "return")[index % 4],
                    r1_success_rate=0.25 if suitable else 0.0,
                    r3_success_rate=0.625 if suitable else 0.0,
                    holdout_success_rate=0.625 if suitable else 0.0,
                    repository_id=f"diagnostic-repo-{index:05d}",
                    task_id=f"diagnostic-task-{index:05d}",
                    generation_model_id=f"diagnostic-model-{index % 4}",
                    structural_invalid_rate=0.0,
                    numeric_instability_rate=0.0,
                    first_hit_candidate_position=0 if suitable else None,
                    candidate_indices_by_window_length={length: tuple(range(4)) for length in (1, 2, 3)},
                    observed_training_key_ids=tuple(f"train-key-{key_index:03d}" for key_index in range(32)),
                    observed_holdout_key_ids=tuple(f"holdout-key-{key_index:03d}" for key_index in range(8)),
                    candidate_observations_by_length=_fake_evidence(suitable)[0],
                    probe_results_by_length=_fake_evidence(suitable)[1],
                row={"schema_version": "wfcllm-gate-data/v1", "group_id": group_id, "split": split, "diagnostic_test_backend": True},
            )

    def run_multi_key_lsh_probe(self, groups, *, training_key_ids, holdout_key_ids, config):
        if len(training_key_ids) != 32 or len(holdout_key_ids) != 8:
            raise ValueError("fake probe requires exact 32/8 key IDs")
        return groups

    def split_groups(self, groups, config):
        return groups

    def audit_gate_data(self, staging_dir, manifest):
        if manifest.get("diagnostic_test_backend") is not True or manifest.get("formal_eligible") is not False:
            raise ValueError("fake gate-data audit marker mismatch")

    def train_candidate(self, *, config, data_manifest, data_jsonl, output_dir, learning_curve_plan):
        output_dir.mkdir(parents=True)
        marker = {"diagnostic_test_backend": True, "formal_eligible": False, "data_manifest_sha256": _sha_path(config.data_dir / "manifest.json")}
        (output_dir / "diagnostic_candidate.json").write_text(json.dumps(marker, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
        return marker


@lru_cache(maxsize=2)
def _fake_evidence(suitable: bool):
    training_ids = tuple(f"train-key-{index:03d}" for index in range(32))
    holdout_ids = tuple(f"holdout-key-{index:03d}" for index in range(8))
    observations_by_length: dict[str, tuple[CandidateObservation, ...]] = {}
    probes_by_length: dict[str, tuple[Mapping[str, LshProbeResult], ...]] = {}
    for length in (1, 2, 3):
        observations = []
        probes = []
        for candidate_index in range(4):
            hit_ids: set[str] = set()
            if suitable:
                if candidate_index == 0:
                    hit_ids.update((*training_ids[:8], *holdout_ids[:2]))
                elif candidate_index == 2:
                    hit_ids.update((*training_ids[8:20], *holdout_ids[2:5]))
            results = {
                key_id: LshProbeResult((1, 0), 1.0, key_id in hit_ids, True, True, True)
                for key_id in (*training_ids, *holdout_ids)
            }
            observations.append(CandidateObservation(
                candidate_index, f"candidate_{candidate_index} = {candidate_index}", "ok", length,
                True, (0, 1), True, True,
                {key_id: {"hit": results[key_id].hit, "stable": True, "margin": 1.0} for key_id in training_ids},
                f"seed-{candidate_index}", "rewrite-v1", (1, 0),
                semantic_reference_cosine=(
                    1.0 if candidate_index == 0 else 0.95
                ),
                semantic_preservation_passed=True,
            ))
            probes.append(MappingProxyType(results))
        observations_by_length[str(length)] = tuple(observations)
        probes_by_length[str(length)] = tuple(probes)
    return observations_by_length, probes_by_length


def _load_config(path: Path) -> dict[str, Any]:
    if not isinstance(path, Path):
        raise ValueError("config must be a local non-symlink JSON file")
    try:
        value = json.loads(_safe_read_public_file(path).decode("utf-8"))
    except (json.JSONDecodeError, UnicodeError, OSError) as exc:
        raise ValueError("config is not valid UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise ValueError("config must be a JSON object")
    return value


def _require_key_sources(raw: dict[str, Any]) -> None:
    for name in ("training_key_source", "holdout_key_source"):
        if name not in raw:
            raise ValueError(f"{name.replace('_', ' ')} is required; no key source was provided")


def _required_path(raw: dict[str, Any], name: str) -> Path:
    value = raw.get(name)
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty local path")
    path = Path(value)
    _reject_symlink_ancestors(path)
    return path


def _optional_path(raw: dict[str, Any], name: str) -> Path | None:
    return None if raw.get(name) is None else _required_path(raw, name)


def _required_string(raw: dict[str, Any], name: str) -> str:
    value = raw.get(name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _digest_value(raw: dict[str, Any], name: str) -> str:
    if name == "config_hash":
        return _resolved_config_hash(raw)
    value = raw.get(name)
    if value is None:
        raise ValueError(f"{name} is required")
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a SHA-256 digest")
    return value


def _resolved_config_hash(raw: dict[str, Any]) -> str:
    resolved = raw.get("resolved_config")
    if not isinstance(resolved, dict):
        raise ValueError("resolved_config mapping is required for config provenance")
    computed = hashlib.sha256(
        json.dumps(resolved, allow_nan=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    supplied = raw.get("config_hash")
    if supplied is not None and supplied != computed:
        raise ValueError("config_hash does not match resolved canonical config")
    return computed


def _reject_symlink_ancestors(path: Path) -> None:
    absolute = path.absolute()
    for parent in (absolute, *absolute.parents):
        try:
            if parent.is_symlink():
                raise ValueError("path cannot traverse symlinks")
        except OSError as exc:
            raise ValueError("path cannot be safely inspected") from exc


def _safe_read_public_file(path: Path) -> bytes:
    _reject_symlink_ancestors(path)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        before = os.fstat(descriptor)
        if before.st_size > _MAX_CONFIG_BYTES:
            raise ValueError("public config/source file exceeds 1 MiB")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read(_MAX_CONFIG_BYTES + 1)
            after = os.fstat(handle.fileno())
        if len(content) > _MAX_CONFIG_BYTES or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("public config/source file changed while reading")
        return content
    except OSError as exc:
        raise ValueError("public config/source file is missing or unsafe") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _sha_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _feasibility_thresholds(raw: dict[str, Any]) -> tuple[tuple[str, int | float], ...]:
    value = raw.get("feasibility_thresholds")
    if not isinstance(value, dict):
        raise ValueError("feasibility_thresholds must be the resolved v1 mapping")
    if set(value) != {name for name, _expected in FEASIBILITY_THRESHOLD_ITEMS}:
        raise ValueError("feasibility_thresholds schema must exactly match v1")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value.values()):
        raise ValueError("feasibility_thresholds values must be numbers")
    return tuple((name, value[name]) for name, _expected in FEASIBILITY_THRESHOLD_ITEMS)


if __name__ == "__main__":
    raise SystemExit(main())
