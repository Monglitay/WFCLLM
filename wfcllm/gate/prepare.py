"""Prepare local-only manifests and private key banks for gated experiments."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import secrets

from wfcllm.gate.production import load_source_catalog


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
    outputs = (source_manifest, training_key_bank, holdout_key_bank, deployment_key)
    if any(path.exists() or path.is_symlink() for path in outputs):
        raise ValueError("gated experiment preparation refuses to overwrite outputs")
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(source_catalog.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "wfcllm-gate-source-manifest/v1",
        "catalog_sha256": digest,
        "source_count": len(records),
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
