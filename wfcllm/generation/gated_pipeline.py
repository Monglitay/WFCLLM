"""Batch generation for the formal gated semantic-window method."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping

from wfcllm.generation.gated_generator import GatedGenerationResult
from wfcllm.generation.outputs import (
    write_final_code_rows,
    write_generation_manifest,
    write_generation_sidecar_rows,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GatedGenerationPipelineConfig:
    output_dir: Path
    dataset: str
    bundle_path: Path
    bundle_sha256: str
    parser_contract: str
    gate_input_contract: str
    tokenizer_sha256: str
    semantic_encoder_sha256: str
    lsh_config_sha256: str
    generation_config_sha256: str
    secret_source_type: str
    fail_fast: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_dir, Path) or not isinstance(self.bundle_path, Path):
            raise ValueError("output_dir and bundle_path must be pathlib.Path values")
        if not isinstance(self.dataset, str) or not self.dataset:
            raise ValueError("dataset must be a non-empty string")
        for name in (
            "bundle_sha256",
            "tokenizer_sha256",
            "semantic_encoder_sha256",
            "lsh_config_sha256",
            "generation_config_sha256",
        ):
            if _DIGEST.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"{name} must be lowercase SHA-256")
        for name in ("parser_contract", "gate_input_contract"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.secret_source_type not in {"file", "environment"}:
            raise ValueError("secret_source_type must be file or environment")
        if type(self.fail_fast) is not bool:
            raise ValueError("fail_fast must be boolean")


class GatedGenerationPipeline:
    """Produce one final program per sample and keep all window data sidecar-only.

    Runtime-heavy components are explicit dependencies.  Production wiring can
    load the local base model, dataset adapter, semantic encoder/LSH scorer and
    bundle-backed gate before construction; tests use lightweight fakes.  The
    deployment key is retained only as private in-memory bytes.
    """

    def __init__(
        self,
        *,
        config: GatedGenerationPipelineConfig,
        base_model: Any,
        generator: Any,
        data_adapter: Callable[[], Iterable[Mapping[str, Any]]] | Iterable[Mapping[str, Any]],
        deployment_key: bytes,
        bundle_loader: Callable[[Path], Any] | None = None,
        bundle_hasher: Callable[[Path], str] | None = None,
    ) -> None:
        if not isinstance(config, GatedGenerationPipelineConfig):
            raise ValueError("config must be GatedGenerationPipelineConfig")
        if not isinstance(deployment_key, bytes) or not deployment_key:
            raise ValueError("deployment key must be non-empty bytes")
        if not callable(getattr(generator, "generate", None)):
            raise ValueError("gated generator must define generate")
        if not callable(data_adapter) and not hasattr(data_adapter, "__iter__"):
            raise ValueError("data adapter must be callable or iterable")
        if not (
            callable(base_model)
            or callable(getattr(base_model, "generate_program", None))
            or callable(getattr(base_model, "generate", None))
        ):
            raise ValueError("base model must generate a complete program")
        if bundle_loader is None:
            from wfcllm.gate.bundle import GateBundle

            bundle_loader = GateBundle.load
        hasher = bundle_hasher or _hash_tree
        before = hasher(config.bundle_path)
        if before != config.bundle_sha256:
            raise ValueError("gate bundle hash mismatch")
        bundle = bundle_loader(config.bundle_path)
        after = hasher(config.bundle_path)
        if after != before:
            raise ValueError("gate bundle changed while loading")
        self._validate_bundle(bundle, config)
        self._config = config
        self._bundle = bundle
        self._base_model = base_model
        self._generator = generator
        self._data_adapter = data_adapter
        self._deployment_key = deployment_key

    @staticmethod
    def _validate_bundle(bundle: Any, config: GatedGenerationPipelineConfig) -> None:
        summary = getattr(bundle, "validation_summary", None)
        if not isinstance(summary, Mapping) or summary.get("validated") is not True:
            raise ValueError("gate bundle must be validated")
        manifest = getattr(bundle, "manifest", None)
        if getattr(manifest, "window_contract_version", None) != config.parser_contract:
            raise ValueError("parser contract hash/version mismatch")
        if getattr(manifest, "gate_input_contract_version", None) != config.gate_input_contract:
            raise ValueError("gate input contract mismatch")
        if getattr(manifest, "tokenizer_sha256", None) != config.tokenizer_sha256:
            raise ValueError("tokenizer hash mismatch")

    def run(self, output_dir: str | Path | None = None) -> str:
        root = Path(output_dir) if output_dir is not None else self._config.output_dir
        final_rows: list[dict[str, str]] = []
        audits: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        samples = self._data_adapter() if callable(self._data_adapter) else self._data_adapter
        for raw in samples:
            sample_id, prompt = self._sample_identity(raw)
            try:
                original = self._generate_base_program(prompt, sample_id)
                result = self._generator.generate(prompt=prompt, original=original)
                if not isinstance(result, GatedGenerationResult):
                    raise ValueError("gated generator returned an invalid result")
            except Exception as exc:
                if self._config.fail_fast:
                    raise
                audits.append(
                    {
                        "id": sample_id,
                        "dataset": self._config.dataset,
                        "sample_generation_failed": True,
                        "reason": str(exc),
                    }
                )
                continue
            final_rows.append(
                {
                    "id": sample_id,
                    "dataset": self._config.dataset,
                    "prompt": prompt,
                    "final_code": result.final_code,
                }
            )
            for window_index, audit in enumerate(result.audit):
                value = asdict(audit)
                value.update(
                    {
                        "id": sample_id,
                        "dataset": self._config.dataset,
                        "window_index": window_index,
                    }
                )
                audits.append(value)
                candidates.append(dict(value))

        final_path = root / "inputs" / "final_code.jsonl"
        write_final_code_rows(final_path, final_rows)
        write_generation_sidecar_rows(root / "generation" / "audit.jsonl", audits)
        write_generation_sidecar_rows(
            root / "generation" / "candidate_sidecar.jsonl", candidates
        )
        write_generation_manifest(
            root / "generation" / "manifest.json",
            {
                "schema_version": "wfcllm-gated-generation-manifest/v1",
                "formal": True,
                "gate_bundle_sha256": self._config.bundle_sha256,
                "parser_contract": self._config.parser_contract,
                "gate_input_contract": self._config.gate_input_contract,
                "tokenizer_sha256": self._config.tokenizer_sha256,
                "semantic_encoder_sha256": self._config.semantic_encoder_sha256,
                "lsh_config_sha256": self._config.lsh_config_sha256,
                "generation_config_sha256": self._config.generation_config_sha256,
                "secret_source_type": self._config.secret_source_type,
                "sample_count": len(final_rows),
                "sample_failure_count": sum(
                    row.get("sample_generation_failed") is True for row in audits
                ),
            },
        )
        return str(final_path)

    @staticmethod
    def _sample_identity(raw: Mapping[str, Any]) -> tuple[str, str]:
        if not isinstance(raw, Mapping):
            raise ValueError("dataset samples must be mappings")
        sample_id = raw.get("id")
        prompt = raw.get("prompt")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError("sample id must be a non-empty string")
        if not isinstance(prompt, str):
            raise ValueError("sample prompt must be a string")
        return sample_id, prompt

    def _generate_base_program(self, prompt: str, sample_id: str) -> str:
        if callable(getattr(self._base_model, "generate_program", None)):
            value = self._base_model.generate_program(prompt=prompt, sample_id=sample_id)
        elif callable(getattr(self._base_model, "generate", None)):
            value = self._base_model.generate(prompt=prompt, sample_id=sample_id)
        else:
            value = self._base_model(prompt=prompt, sample_id=sample_id)
        if not isinstance(value, str):
            raise ValueError("base model must return one complete program string")
        return value


def _hash_tree(root: Path) -> str:
    if not root.is_dir():
        raise ValueError("gate bundle directory is missing")
    digest = hashlib.sha256(b"wfcllm-artifact-tree/v1\0")
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError("gate bundle cannot contain symlinks")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("gate bundle contains an unsupported entry")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = path.read_bytes()
        digest.update(len(relative).to_bytes(8, "big") + relative)
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()
