from __future__ import annotations

import hashlib
import re
from typing import Callable, Sequence

from wfcllm.batch_invariant_v4.context import StructuralContext


_SHA256 = re.compile(r"[0-9a-f]{64}")


def public_cache_key(
    *,
    schema_version: str,
    public_config_sha256: str,
    context_sha256: str,
) -> str:
    if not isinstance(schema_version, str) or not schema_version:
        raise ValueError("schema_version must be non-empty")
    if not _SHA256.fullmatch(public_config_sha256):
        raise ValueError("public_config_sha256 must be lowercase SHA-256")
    if not _SHA256.fullmatch(context_sha256):
        raise ValueError("context_sha256 must be lowercase SHA-256")
    return hashlib.sha256(
        (schema_version + "\0" + public_config_sha256 + "\0" + context_sha256).encode(
            "utf-8"
        )
    ).hexdigest()


class PublicContextCache:
    def __init__(self, *, public_config_sha256: str) -> None:
        if not _SHA256.fullmatch(public_config_sha256):
            raise ValueError("public_config_sha256 must be lowercase SHA-256")
        self.public_config_sha256 = public_config_sha256
        self._values: dict[str, tuple[int, ...]] = {}
        self.hits = 0
        self.misses = 0
        self.last_flush_order: tuple[str, ...] = ()

    def get_or_create(
        self,
        context: StructuralContext,
        factory: Callable[[], Sequence[int]],
    ) -> tuple[int, ...]:
        key = public_cache_key(
            schema_version="wfcllm-batch-invariant-structural-context/v4",
            public_config_sha256=self.public_config_sha256,
            context_sha256=context.context_sha256,
        )
        if key in self._values:
            self.hits += 1
            return self._values[key]
        value = tuple(int(item) for item in factory())
        if value != context.representation_bytes:
            raise ValueError("cache factory returned non-canonical representation")
        self._values[key] = value
        self.misses += 1
        return value

    def flush_order(self, candidate_ids: Sequence[str]) -> None:
        if any(not isinstance(item, str) or not item for item in candidate_ids):
            raise ValueError("flush order IDs must be non-empty strings")
        self.last_flush_order = tuple(candidate_ids)
