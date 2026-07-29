"""Batch generation for the formal gated semantic-window method."""

from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from wfcllm.generation.completion_finalizer import ProgramFinalizationResult
from wfcllm.generation.gated_generator import GatedGenerationResult
from wfcllm.generation.outputs import (
    write_final_code_rows,
    write_generation_manifest,
    write_generation_sidecar_rows,
)

_DIGEST = re.compile(r"^[0-9a-f]{64}$")


class FinalizerIntegrityError(RuntimeError):
    """Raised when finalizer output cannot be traced to input AST statements."""


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
    generation_model_identifier: str | None = None
    embedding_passes: int = 1
    fail_fast: bool = False
    diagnostic_test_backend: bool = False
    supplementary_binding: Mapping[str, object] | None = None

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
        if self.generation_model_identifier is not None and (
            not isinstance(self.generation_model_identifier, str)
            or not self.generation_model_identifier
        ):
            raise ValueError(
                "generation_model_identifier must be a non-empty string or None"
            )
        if (
            type(self.embedding_passes) is not int
            or not 1 <= self.embedding_passes <= 3
        ):
            raise ValueError("embedding_passes must be an integer in [1, 3]")
        if type(self.fail_fast) is not bool:
            raise ValueError("fail_fast must be boolean")
        if type(self.diagnostic_test_backend) is not bool:
            raise ValueError("diagnostic_test_backend must be boolean")
        if self.supplementary_binding is not None and not isinstance(
            self.supplementary_binding, Mapping
        ):
            raise ValueError("supplementary_binding must be an object or None")


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
        program_finalizer: Callable[[str, str], ProgramFinalizationResult] | None = None,
        program_finalizer_name: str = "none",
        bundle_loader: Callable[[Path], Any] | None = None,
        bundle_hasher: Callable[[Path], str] | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, GatedGenerationPipelineConfig):
            raise ValueError("config must be GatedGenerationPipelineConfig")
        if not isinstance(deployment_key, bytes) or not deployment_key:
            raise ValueError("deployment key must be non-empty bytes")
        if not callable(getattr(generator, "generate", None)):
            raise ValueError("gated generator must define generate")
        if not callable(data_adapter) and not hasattr(data_adapter, "__iter__"):
            raise ValueError("data adapter must be callable or iterable")
        if program_finalizer is not None and not callable(program_finalizer):
            raise ValueError("program_finalizer must be callable or None")
        if not isinstance(program_finalizer_name, str) or not program_finalizer_name:
            raise ValueError("program_finalizer_name must be a non-empty string")
        if (program_finalizer is None) != (program_finalizer_name == "none"):
            raise ValueError("program_finalizer and name must be configured together")
        if program_finalizer_name == "mbpp_target_interface_wrapper_v1":
            from wfcllm.generation.completion_finalizer import (
                finalize_mbpp_program_with_interface_wrapper,
            )

            if program_finalizer is not finalize_mbpp_program_with_interface_wrapper:
                raise ValueError(
                    "mbpp_target_interface_wrapper_v1 requires the trusted "
                    "MBPP interface wrapper"
                )
        if not (
            callable(base_model)
            or callable(getattr(base_model, "generate_program", None))
            or callable(getattr(base_model, "generate", None))
        ):
            raise ValueError("base model must generate a complete program")
        if bundle_loader is None:
            raise ValueError("current-run gate candidate loader is required")
        if not callable(monotonic_clock):
            raise ValueError("monotonic_clock must be callable")
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
        self._program_finalizer = program_finalizer
        self._program_finalizer_name = program_finalizer_name
        self._monotonic_clock = monotonic_clock

    @staticmethod
    def _validate_bundle(bundle: Any, config: GatedGenerationPipelineConfig) -> None:
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
        finalizer_rows: list[dict[str, Any]] = []
        latency_rows: list[dict[str, Any]] = []
        samples_iter = (
            self._data_adapter() if callable(self._data_adapter) else self._data_adapter
        )
        samples = list(samples_iter)
        total_samples = len(samples)
        progress_path = root / "generation" / "progress.json"
        self._write_progress(
            progress_path,
            status="running",
            total_samples=total_samples,
            completed_samples=0,
            failed_samples=0,
            final_code_rows=0,
            audit_rows=0,
            candidate_sidecar_rows=0,
            current_sample_id=None,
        )
        for raw in samples:
            sample_id, prompt = self._sample_identity(raw)
            self._write_progress(
                progress_path,
                status="running",
                total_samples=total_samples,
                completed_samples=len(final_rows),
                failed_samples=sum(
                    row.get("sample_generation_failed") is True for row in audits
                ),
                final_code_rows=len(final_rows),
                audit_rows=len(audits),
                candidate_sidecar_rows=len(candidates),
                current_sample_id=sample_id,
            )
            # Base generation and finalization are required inference work, not
            # optional watermark transforms.  A failure here means there is no
            # program to keep in the full denominator, so it must abort the
            # phase instead of publishing an empty formal detector input.
            sample_started_at = self._monotonic_clock()
            original = self._generate_base_program(prompt, sample_id)
            if self._program_finalizer is not None:
                before = original
                finalized = self._program_finalizer(prompt, before)
                if not isinstance(finalized, ProgramFinalizationResult):
                    raise ValueError(
                        "program_finalizer must return ProgramFinalizationResult"
                    )
                original = finalized.code
                provenance = _verify_finalizer_statement_provenance(
                    before=before,
                    after=original,
                    prompt=prompt,
                    program_finalizer_name=self._program_finalizer_name,
                )
                finalizer_rows.append(
                    {
                        "id": sample_id,
                        "dataset": self._config.dataset,
                        "applied": finalized.applied,
                        "reason": finalized.reason,
                        "before_sha256": hashlib.sha256(
                            before.encode("utf-8")
                        ).hexdigest(),
                        "after_sha256": hashlib.sha256(
                            original.encode("utf-8")
                        ).hexdigest(),
                        "before_character_count": len(before),
                        "after_character_count": len(original),
                        "input_source": before,
                        "output_source": original,
                        "carrier_count": 0,
                        **provenance,
                    }
                )
            try:
                pass_audits = []
                pass_candidates = []
                for _embedding_pass in range(self._config.embedding_passes):
                    result = self._generator.generate(
                        prompt=prompt,
                        original=original,
                    )
                    if not isinstance(result, GatedGenerationResult):
                        raise ValueError("gated generator returned an invalid result")
                    window_offset = len(pass_audits)
                    pass_audits.extend(result.audit)
                    pass_candidates.extend(
                        replace(
                            candidate,
                            window_index=candidate.window_index + window_offset,
                        )
                        for candidate in result.candidates
                    )
                    original = result.final_code
                result = GatedGenerationResult(
                    final_code=original,
                    audit=tuple(pass_audits),
                    candidates=tuple(pass_candidates),
                )
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
                final_rows.append(
                    {
                        "id": sample_id,
                        "dataset": self._config.dataset,
                        "prompt": prompt,
                        "final_code": original,
                    }
                )
                latency_rows.append(
                    self._latency_row(
                        sample_id=sample_id,
                        started_at=sample_started_at,
                    )
                )
                self._publish_partial_outputs(
                    root,
                    final_rows=final_rows,
                    audits=audits,
                    candidates=candidates,
                    finalizer_rows=finalizer_rows,
                    latency_rows=latency_rows,
                )
                self._write_progress(
                    progress_path,
                    status="running",
                    total_samples=total_samples,
                    completed_samples=len(final_rows),
                    failed_samples=sum(
                        row.get("sample_generation_failed") is True for row in audits
                    ),
                    final_code_rows=len(final_rows),
                    audit_rows=len(audits),
                    candidate_sidecar_rows=len(candidates),
                    current_sample_id=sample_id,
                )
                self._log_progress(
                    total_samples=total_samples,
                    completed_samples=len(final_rows),
                    failed_samples=sum(
                        row.get("sample_generation_failed") is True for row in audits
                    ),
                    final_code_rows=len(final_rows),
                    audit_rows=len(audits),
                    candidate_sidecar_rows=len(candidates),
                    sample_id=sample_id,
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
            for candidate in result.candidates:
                value = asdict(candidate)
                value.update(
                    {
                        "id": sample_id,
                        "dataset": self._config.dataset,
                    }
                )
                candidates.append(value)
            latency_rows.append(
                self._latency_row(
                    sample_id=sample_id,
                    started_at=sample_started_at,
                )
            )
            self._publish_partial_outputs(
                root,
                final_rows=final_rows,
                audits=audits,
                candidates=candidates,
                finalizer_rows=finalizer_rows,
                latency_rows=latency_rows,
            )
            self._write_progress(
                progress_path,
                status="running",
                total_samples=total_samples,
                completed_samples=len(final_rows),
                failed_samples=sum(
                    row.get("sample_generation_failed") is True for row in audits
                ),
                final_code_rows=len(final_rows),
                audit_rows=len(audits),
                candidate_sidecar_rows=len(candidates),
                current_sample_id=sample_id,
            )
            self._log_progress(
                total_samples=total_samples,
                completed_samples=len(final_rows),
                failed_samples=sum(
                    row.get("sample_generation_failed") is True for row in audits
                ),
                final_code_rows=len(final_rows),
                audit_rows=len(audits),
                candidate_sidecar_rows=len(candidates),
                sample_id=sample_id,
            )

        final_path = root / "inputs" / "final_code.jsonl"
        self._publish_partial_outputs(
            root,
            final_rows=final_rows,
            audits=audits,
            candidates=candidates,
            finalizer_rows=finalizer_rows,
            latency_rows=latency_rows,
        )
        failed_samples = sum(
            row.get("sample_generation_failed") is True for row in audits
        )
        manifest = {
                "schema_version": "wfcllm-gated-generation-manifest/v1",
                "formal": not self._config.diagnostic_test_backend,
                "diagnostic_test_backend": self._config.diagnostic_test_backend,
                "formal_eligible": not self._config.diagnostic_test_backend,
                "diagnostic_only": self._config.diagnostic_test_backend,
                "not_official_method": self._config.diagnostic_test_backend,
                "gate_bundle_sha256": self._config.bundle_sha256,
                "parser_contract": self._config.parser_contract,
                "gate_input_contract": self._config.gate_input_contract,
                "tokenizer_sha256": self._config.tokenizer_sha256,
                "semantic_encoder_sha256": self._config.semantic_encoder_sha256,
                "lsh_config_sha256": self._config.lsh_config_sha256,
                "generation_config_sha256": self._config.generation_config_sha256,
                "generation_model_identifier": (
                    self._config.generation_model_identifier
                ),
                "final_code_sha256": hashlib.sha256(
                    final_path.read_bytes()
                ).hexdigest(),
                "final_code_row_count": len(final_rows),
                "embedding_passes": self._config.embedding_passes,
                "program_finalizer": self._program_finalizer_name,
                "finalizer_applied_count": sum(
                    row["applied"] is True for row in finalizer_rows
                ),
                "finalizer_fallback_count": sum(
                    row["applied"] is False for row in finalizer_rows
                ),
                "carrier_count": 0,
                "finalizer_added_ast_statement_count": sum(
                    int(row["added_ast_statement_count"])
                    for row in finalizer_rows
                ),
                "finalizer_provenance_verified_count": sum(
                    row["statement_provenance_verified"] is True
                    for row in finalizer_rows
                ),
                "secret_source_type": self._config.secret_source_type,
                "sample_count": len(final_rows),
                "candidate_attempt_count": len(candidates),
                "candidate_evaluated_count": sum(
                    row["evaluation_status"]
                    != "generated_not_evaluated_after_accept"
                    for row in candidates
                ),
                "candidate_not_evaluated_after_accept_count": sum(
                    row["evaluation_status"]
                    == "generated_not_evaluated_after_accept"
                    for row in candidates
                ),
                "sample_failure_count": sum(
                    row.get("sample_generation_failed") is True for row in audits
                ),
            }
        if self._config.supplementary_binding is not None:
            binding = json.loads(
                json.dumps(
                    dict(self._config.supplementary_binding),
                    allow_nan=False,
                    ensure_ascii=False,
                )
            )
            study = binding.get("supplementary_ablation")
            if isinstance(study, dict):
                diagnostic = self._config.diagnostic_test_backend
                study.update(
                    {
                        "formal": not diagnostic,
                        "formal_eligible": not diagnostic,
                        "diagnostic_test_backend": diagnostic,
                        "diagnostic_only": diagnostic,
                        "not_official_method": diagnostic,
                    }
                )
            manifest.update(binding)
        write_generation_manifest(
            root / "generation" / "manifest.json",
            manifest,
        )
        self._write_progress(
            progress_path,
            status="completed",
            total_samples=total_samples,
            completed_samples=len(final_rows),
            failed_samples=failed_samples,
            final_code_rows=len(final_rows),
            audit_rows=len(audits),
            candidate_sidecar_rows=len(candidates),
            current_sample_id=None,
        )
        return str(final_path)

    def _publish_partial_outputs(
        self,
        root: Path,
        *,
        final_rows: list[dict[str, str]],
        audits: list[dict[str, Any]],
        candidates: list[dict[str, Any]],
        finalizer_rows: list[dict[str, Any]],
        latency_rows: list[dict[str, Any]],
    ) -> None:
        write_final_code_rows(root / "inputs" / "final_code.jsonl", final_rows)
        write_generation_sidecar_rows(root / "generation" / "audit.jsonl", audits)
        write_generation_sidecar_rows(
            root / "generation" / "candidate_sidecar.jsonl", candidates
        )
        write_generation_sidecar_rows(
            root / "generation" / "latency_sidecar.jsonl", latency_rows
        )
        if self._program_finalizer is not None:
            write_generation_sidecar_rows(
                root / "generation" / "finalizer.jsonl", finalizer_rows
            )

    def _latency_row(self, *, sample_id: str, started_at: float) -> dict[str, Any]:
        finished_at = self._monotonic_clock()
        if (
            not isinstance(started_at, (int, float))
            or isinstance(started_at, bool)
            or not isinstance(finished_at, (int, float))
            or isinstance(finished_at, bool)
        ):
            raise ValueError("monotonic clock must return finite numeric values")
        elapsed = float(finished_at) - float(started_at)
        if not math.isfinite(elapsed) or elapsed < 0.0:
            raise ValueError("monotonic clock must not move backwards")
        return {
            "id": sample_id,
            "dataset": self._config.dataset,
            "generation_latency_seconds": elapsed,
        }

    @staticmethod
    def _write_progress(
        path: Path,
        *,
        status: str,
        total_samples: int,
        completed_samples: int,
        failed_samples: int,
        final_code_rows: int,
        audit_rows: int,
        candidate_sidecar_rows: int,
        current_sample_id: str | None,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "wfcllm-gated-generation-progress/v1",
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "total_samples": total_samples,
            "completed_samples": completed_samples,
            "failed_samples": failed_samples,
            "final_code_rows": final_code_rows,
            "audit_rows": audit_rows,
            "candidate_sidecar_rows": candidate_sidecar_rows,
            "current_sample_id": current_sample_id,
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)

    @staticmethod
    def _log_progress(
        *,
        total_samples: int,
        completed_samples: int,
        failed_samples: int,
        final_code_rows: int,
        audit_rows: int,
        candidate_sidecar_rows: int,
        sample_id: str,
    ) -> None:
        print(
            "[progress] gated generate "
            f"completed={completed_samples}/{total_samples} "
            f"failed={failed_samples} "
            f"final_code_rows={final_code_rows} "
            f"audit_rows={audit_rows} "
            f"candidate_sidecar_rows={candidate_sidecar_rows} "
            f"last_sample_id={sample_id}",
            file=sys.stderr,
            flush=True,
        )

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


def _verify_finalizer_statement_provenance(
    *,
    before: str,
    after: str,
    prompt: str,
    program_finalizer_name: str,
) -> dict[str, Any]:
    """Prove every output statement already occurs in a parseable input prefix."""

    if program_finalizer_name == "mbpp_target_interface_wrapper_v1":
        from wfcllm.generation.completion_finalizer import (
            finalize_mbpp_program,
            finalize_mbpp_program_with_interface_wrapper,
        )

        expected = finalize_mbpp_program_with_interface_wrapper(prompt, before)
        if expected.code != after:
            raise FinalizerIntegrityError(
                "trusted MBPP interface wrapper output mismatch"
            )
        finalized_base = finalize_mbpp_program(prompt, before)
        input_count = _statement_count_if_parseable(finalized_base.code)
        output_count = _statement_count_if_parseable(after)
        if input_count is None or output_count is None or output_count < input_count:
            raise FinalizerIntegrityError(
                "trusted MBPP interface wrapper AST accounting failed"
            )
        return {
            "statement_provenance_verified": True,
            "provenance_mode": "trusted_mbpp_interface_wrapper/v1",
            "input_ast_statement_count": input_count,
            "output_ast_statement_count": output_count,
            "added_ast_statement_count": output_count - input_count,
        }

    if before == after:
        statement_count = _statement_count_if_parseable(after)
        return {
            "statement_provenance_verified": True,
            "provenance_mode": "byte_identical",
            "input_ast_statement_count": statement_count,
            "output_ast_statement_count": statement_count,
            "added_ast_statement_count": 0,
        }

    try:
        output_tree = ast.parse(after)
    except SyntaxError as exc:
        raise FinalizerIntegrityError(
            "non-identical finalizer output must be parseable for AST provenance"
        ) from exc

    output_statements = _statement_fingerprints(output_tree)
    smallest_deficit = sum(output_statements.values())
    matched_input_count: int | None = None
    for input_tree in _parseable_line_prefix_trees(before):
        input_statements = _statement_fingerprints(input_tree)
        deficit = output_statements - input_statements
        deficit_count = sum(deficit.values())
        smallest_deficit = min(smallest_deficit, deficit_count)
        if deficit_count == 0:
            matched_input_count = sum(input_statements.values())
            break

    if matched_input_count is None:
        raise FinalizerIntegrityError(
            "finalizer introduced AST statements: "
            f"at least {smallest_deficit} output statements lack input provenance"
        )

    return {
        "statement_provenance_verified": True,
        "provenance_mode": "ast_statement_multiset_subset",
        "input_ast_statement_count": matched_input_count,
        "output_ast_statement_count": sum(output_statements.values()),
        "added_ast_statement_count": 0,
    }


def _statement_count_if_parseable(source: str) -> int | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    return sum(_statement_fingerprints(tree).values())


def _statement_fingerprints(tree: ast.AST) -> Counter[str]:
    return Counter(
        ast.dump(node, annotate_fields=True, include_attributes=False)
        for node in ast.walk(tree)
        if isinstance(node, ast.stmt)
    )


def _parseable_line_prefix_trees(source: str) -> Iterable[ast.Module]:
    lines = source.splitlines(keepends=True)
    for end in range(len(lines), 0, -1):
        try:
            yield ast.parse("".join(lines[:end]))
        except SyntaxError:
            continue


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
