"""Prepare local-only manifests and private key banks for gated experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets

from wfcllm.gate.production import load_source_catalog
from wfcllm.gate.sources import GateSourceManifest, GateSourceRecord


def prepare_gated_experiment(
    *,
    source_catalog: Path,
    source_manifest: Path,
    training_key_bank: Path,
    holdout_key_bank: Path,
    deployment_key: Path,
) -> dict[str, object]:
    """Validate a catalog and create provenance plus fresh non-public key banks."""

    records = tuple(load_source_catalog(source_catalog))
    if not records:
        raise ValueError("gate source catalog must contain at least one record")
    formal_manifest = GateSourceManifest(
        tuple(
            GateSourceRecord(
                source_family=record.source_family,
                source_id=record.source_id,
                code=record.code,
                repository_id=record.repository_id,
                task_id=record.task_id,
                function_id=record.function_id,
                source_model_id=record.source_model_id,
                license_id=record.license_id,
                contract_or_hard_set=record.contract_or_hard_set,
            )
            for record in records
        )
    ).to_dict()
    outputs = (source_manifest, training_key_bank, holdout_key_bank, deployment_key)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("gated experiment preparation refuses to overwrite outputs")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source_catalog.read_bytes()).hexdigest()
    manifest = {
        **formal_manifest,
        "catalog_sha256": digest,
    }
    source_manifest.write_text(
        json.dumps(manifest, allow_nan=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    training = [secrets.token_hex(32) for _ in range(32)]
    holdout = [secrets.token_hex(32) for _ in range(8)]
    if not set(training).isdisjoint(holdout):
        raise RuntimeError("generated key banks unexpectedly overlap")
    for path, values in ((training_key_bank, training), (holdout_key_bank, holdout)):
        path.write_text(
            json.dumps(values, allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        os.chmod(path, 0o600)
    deployment_key.write_text(secrets.token_hex(32) + "\n", encoding="utf-8")
    os.chmod(deployment_key, 0o600)
    return manifest


__all__ = ["prepare_gated_experiment"]
