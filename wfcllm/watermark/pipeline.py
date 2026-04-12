"""Batch watermarking pipeline over HumanEval/MBPP datasets."""
from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None  # type: ignore[assignment]

import torch

from wfcllm.common.checkpoint import load_processed_ids, resolve_resume_path
from wfcllm.watermark.generator import WatermarkGenerator
from wfcllm.common.dataset_loader import SUPPORTED_DATASETS, load_prompts


@dataclass
class WatermarkPipelineConfig:
    """Configuration for batch watermark embedding pipeline."""

    dataset: str            # "humaneval" or "mbpp"
    output_dir: str         # e.g. "data/watermarked"
    dataset_path: str       # local datasets root, e.g. "data/datasets"
    resume: str | None = None
    sample_limit: int | None = None

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )


class WatermarkPipeline:
    """Batch watermark embedding over a HumanEval or MBPP dataset."""

    _ROUTE_ONE_RESUME_SENTINEL_FIELDS = (
        "diagnostics_version",
        "retry_summary",
        "cascade_summary",
    )
    _DIAGNOSTIC_SUMMARY_ALLOWLIST = (
        "diagnostics_version",
        "retry_summary",
        "cascade_summary",
        "failure_reason_counts",
        "rescued_blocks",
        "unrescued_blocks",
    )

    def __init__(self, generator: WatermarkGenerator, config: WatermarkPipelineConfig):
        self._generator = generator
        self._config = config

    def _load_prompts(self) -> list[dict]:
        """Load prompts from local dataset. Returns list of {"id", "prompt"}."""
        return load_prompts(self._config.dataset, self._config.dataset_path)

    def _validate_resume_path(self, resume_path: Path) -> None:
        expected_prefix = f"{self._config.dataset}_"
        if not resume_path.name.startswith(expected_prefix):
            raise ValueError(
                f"Resume file {resume_path.name} does not match dataset {self._config.dataset}"
            )

    @staticmethod
    def _build_public_watermark_params(generator: WatermarkGenerator) -> dict:
        generator_config = getattr(generator, "config", None)
        if generator_config is None:
            raise ValueError(
                "Generator must expose watermark config via .config"
            )
        params = {
            "lsh_d": generator_config.lsh_d,
            "lsh_gamma": generator_config.lsh_gamma,
            "margin_base": generator_config.margin_base,
            "margin_alpha": generator_config.margin_alpha,
        }
        adaptive_gamma = WatermarkPipeline._build_public_adaptive_gamma_params(generator)
        if adaptive_gamma is not None:
            params["adaptive_gamma"] = adaptive_gamma
        token_channel = getattr(generator_config, "token_channel", None)
        if token_channel is not None and getattr(token_channel, "enabled", False):
            params["token_channel"] = WatermarkPipeline._build_public_token_channel_params(
                generator
            )
        return params

    @staticmethod
    def _build_public_token_channel_params(generator: WatermarkGenerator) -> dict:
        generator_config = getattr(generator, "config", None)
        token_channel = getattr(generator_config, "token_channel", None)
        return {
            "enabled": True,
            "mode": getattr(token_channel, "mode", "semantic-only"),
            "context_width": getattr(token_channel, "context_width", 0),
            "switch_threshold": getattr(token_channel, "switch_threshold", 0.0),
            "delta": getattr(token_channel, "delta", 0.0),
            "token_altering_postprocess": False,
        }

    @staticmethod
    def _build_token_channel_metadata(generator: WatermarkGenerator) -> dict | None:
        generator_config = getattr(generator, "config", None)
        token_channel = getattr(generator_config, "token_channel", None)
        if token_channel is None or not getattr(token_channel, "enabled", False):
            return None

        payload = {
            **WatermarkPipeline._build_public_token_channel_params(generator),
            "ignore_repeated_ngrams": getattr(token_channel, "ignore_repeated_ngrams", False),
            "ignore_repeated_prefixes": getattr(token_channel, "ignore_repeated_prefixes", False),
        }
        artifact = getattr(generator, "_token_channel_artifact", None)
        if artifact is None:
            return payload

        metadata = getattr(artifact, "metadata", None)
        metadata_dict = WatermarkPipeline._coerce_token_channel_artifact_metadata(metadata)
        if metadata_dict is not None:
            payload["artifact_metadata"] = metadata_dict

        model_path = getattr(artifact, "model_path", None)
        metadata_path = getattr(artifact, "metadata_path", None)
        if WatermarkPipeline._is_real_path(model_path) and WatermarkPipeline._is_real_path(metadata_path):
            payload["artifact_fingerprints"] = {
                "model_sha256": WatermarkPipeline._sha256_file(Path(model_path)),
                "metadata_sha256": WatermarkPipeline._sha256_file(Path(metadata_path)),
            }
        return payload

    @staticmethod
    def _coerce_token_channel_artifact_metadata(metadata: object) -> dict[str, object] | None:
        if metadata is None:
            return None
        to_dict = getattr(metadata, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                return payload
        training_config = getattr(metadata, "training_config", None)
        if not isinstance(training_config, dict):
            return None
        schema_version = getattr(metadata, "schema_version", None)
        tokenizer_name = getattr(metadata, "tokenizer_name", None)
        tokenizer_vocab_size = getattr(metadata, "tokenizer_vocab_size", None)
        context_width = getattr(metadata, "context_width", None)
        feature_version = getattr(metadata, "feature_version", None)
        if not isinstance(schema_version, str) or not isinstance(tokenizer_name, str):
            return None
        if not isinstance(tokenizer_vocab_size, int) or not isinstance(context_width, int):
            return None
        if not isinstance(feature_version, str):
            return None
        return {
            "schema_version": schema_version,
            "tokenizer_name": tokenizer_name,
            "tokenizer_vocab_size": tokenizer_vocab_size,
            "context_width": context_width,
            "feature_version": feature_version,
            "training_config": dict(training_config),
        }

    @staticmethod
    def _is_real_path(path: object) -> bool:
        return isinstance(path, (str, Path)) and Path(path).exists()

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            while True:
                chunk = handle.read(65536)
                if not chunk:
                    break
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _build_public_adaptive_gamma_params(
        generator: WatermarkGenerator,
    ) -> dict | None:
        generator_config = getattr(generator, "config", None)
        adaptive_config = getattr(generator_config, "adaptive_gamma", None)
        profile = getattr(generator, "_entropy_profile", None)
        if adaptive_config is None or not getattr(adaptive_config, "enabled", False):
            return None
        if profile is None:
            return None
        return {
            "strategy": getattr(adaptive_config, "strategy", "piecewise_quantile"),
            "profile_id": getattr(adaptive_config, "profile_id", None),
            "anchors": dict(getattr(adaptive_config, "anchors", {}) or {}),
            "profile": {
                "language": profile.language,
                "model_family": profile.model_family,
                "quantiles_units": dict(profile.quantiles_units_map),
                "strategy": profile.strategy,
            },
        }

    @staticmethod
    def _resolve_diagnostics_dir(output_path: Path) -> Path:
        """Resolve the diagnostics directory paired with a watermarked output dir."""
        output_parent = output_path.parent
        if output_parent.name == "watermarked":
            return output_parent.parent / "diagnostics"
        return output_parent / "diagnostics"

    @staticmethod
    def _build_block_ledger_path(output_path: Path, diagnostics_dir: Path) -> Path:
        """Build a deterministic block-ledger path from the watermarked artifact stem."""
        return diagnostics_dir / f"{output_path.stem}_block_ledger.jsonl"

    @classmethod
    def _merge_diagnostic_summary(
        cls,
        record: dict[str, object],
        diagnostic_summary: object,
    ) -> None:
        if not isinstance(diagnostic_summary, dict):
            return
        for key in cls._DIAGNOSTIC_SUMMARY_ALLOWLIST:
            if key in diagnostic_summary:
                record[key] = diagnostic_summary[key]

    @staticmethod
    def _sample_requires_ledger(payload: dict[str, object]) -> bool:
        diagnostics_ledger_rows = payload.get("diagnostics_ledger_rows")
        if isinstance(diagnostics_ledger_rows, int):
            return diagnostics_ledger_rows > 0
        total_blocks = payload.get("total_blocks")
        if isinstance(total_blocks, int):
            if total_blocks > 0:
                return True
            alignment_summary = payload.get("alignment_summary")
            if isinstance(alignment_summary, dict):
                generator_total_blocks = alignment_summary.get("generator_total_blocks")
                if isinstance(generator_total_blocks, int):
                    return generator_total_blocks > 0
            return False
        return True

    @classmethod
    def _resume_row_expects_sidecar(cls, payload: dict[str, object]) -> bool:
        return any(field in payload for field in cls._ROUTE_ONE_RESUME_SENTINEL_FIELDS)

    @staticmethod
    def _load_resume_records_by_id(resume_path: Path) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        with open(resume_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                sample_id = payload.get("id")
                if isinstance(sample_id, str) and sample_id:
                    records[sample_id] = payload
        return records

    @staticmethod
    def _load_diagnostics_sample_ids(diagnostics_path: Path) -> set[str]:
        sample_ids: set[str] = set()
        with open(diagnostics_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                sample_id = payload.get("sample_id")
                if isinstance(sample_id, str) and sample_id:
                    sample_ids.add(sample_id)
        return sample_ids

    @staticmethod
    def _load_diagnostics_ordinals(
        diagnostics_path: Path,
    ) -> dict[str, Counter[int]]:
        ordinals_by_sample: dict[str, Counter[int]] = defaultdict(Counter)
        with open(diagnostics_path, encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue
                sample_id = payload.get("sample_id")
                block_ordinal = payload.get("block_ordinal")
                if (
                    isinstance(sample_id, str)
                    and sample_id
                    and isinstance(block_ordinal, int)
                ):
                    ordinals_by_sample[sample_id][block_ordinal] += 1
        return ordinals_by_sample

    @staticmethod
    def _expected_block_count(payload: dict[str, object]) -> int | None:
        diagnostics_ledger_rows = payload.get("diagnostics_ledger_rows")
        if isinstance(diagnostics_ledger_rows, int) and diagnostics_ledger_rows >= 0:
            return diagnostics_ledger_rows
        alignment_summary = payload.get("alignment_summary")
        if isinstance(alignment_summary, dict):
            final_block_count = alignment_summary.get("final_block_count")
            if isinstance(final_block_count, int) and final_block_count > 0:
                return final_block_count
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            return len(blocks)
        total_blocks = payload.get("total_blocks")
        if isinstance(total_blocks, int):
            return total_blocks
        return None

    @staticmethod
    def _incomplete_diagnostics_message(
        sample_id: str,
        ordinal_counts: Counter[int],
        expected_count: int,
    ) -> str:
        observed_set = set(ordinal_counts.keys())
        missing_ordinals = sorted(
            set(range(expected_count)) - observed_set
        )
        out_of_range = sorted(
            ordinal for ordinal in observed_set if ordinal < 0 or ordinal >= expected_count
        )
        duplicate_ordinals = sorted(
            ordinal for ordinal, count in ordinal_counts.items() if count > 1
        )
        row_count = sum(ordinal_counts.values())
        details: list[str] = []
        if missing_ordinals:
            details.append(f"missing ordinals {missing_ordinals}")
        if out_of_range:
            details.append(f"out-of-range ordinals {out_of_range}")
        if duplicate_ordinals:
            details.append(f"duplicate ordinals {duplicate_ordinals}")
        if row_count != expected_count:
            details.append(f"expected {expected_count} rows but found {row_count}")
        detail_text = "; ".join(details) if details else "unexpected diagnostics layout"
        return (
            f"Resume diagnostics sidecar for {sample_id} is incomplete: {detail_text}"
        )

    def _validate_resume_diagnostics_alignment(
        self,
        resume_path: Path,
        diagnostics_path: Path,
    ) -> None:
        resume_records = self._load_resume_records_by_id(resume_path)
        expected_records = {
            sample_id: payload
            for sample_id, payload in resume_records.items()
            if self._resume_row_expects_sidecar(payload)
            and self._sample_requires_ledger(payload)
        }
        if not expected_records:
            return
        if not diagnostics_path.exists():
            raise ValueError(
                f"Resume diagnostics sidecar missing: {diagnostics_path}"
            )
        ordinals_by_sample = self._load_diagnostics_ordinals(diagnostics_path)
        observed_ids = set(ordinals_by_sample)
        missing_ids = sorted(set(expected_records) - observed_ids)
        if missing_ids:
            preview = ", ".join(missing_ids[:3])
            raise ValueError(
                "Resume diagnostics sidecar is not aligned with processed samples; "
                f"missing sample_id entries for: {preview}"
            )
        for sample_id, payload in expected_records.items():
            ordinal_counts = ordinals_by_sample.get(sample_id, Counter())
            expected_count = self._expected_block_count(payload)
            if expected_count is None:
                continue
            if (
                ordinal_counts
                and sum(ordinal_counts.values()) == expected_count
                and set(range(expected_count)) == set(ordinal_counts.keys())
                and all(
                    ordinal >= 0 and ordinal < expected_count
                    for ordinal in ordinal_counts
                )
            ):
                continue
            raise ValueError(
                self._incomplete_diagnostics_message(
                    sample_id=sample_id,
                    ordinal_counts=ordinal_counts,
                    expected_count=expected_count,
                )
            )

    @staticmethod
    def _iter_persisted_block_ledgers(
        sample_id: str,
        block_ledgers: object,
    ) -> list[dict[str, object]]:
        if not isinstance(block_ledgers, list):
            return []
        rows: list[dict[str, object]] = []
        for row in block_ledgers:
            if not isinstance(row, dict):
                continue
            payload = dict(row)
            payload["sample_id"] = sample_id
            rows.append(payload)
        return rows

    def run(self) -> str:
        """Run batch watermarking. Returns path to output JSONL file."""
        out_dir = Path(self._config.output_dir)
        resume_path, is_resume = resolve_resume_path(
            self._config.resume,
            out_dir,
            default_pattern=f"{self._config.dataset}_*.jsonl",
        )

        processed_ids: set[str] = set()
        if is_resume and resume_path is not None:
            self._validate_resume_path(resume_path)
            processed_ids = load_processed_ids(resume_path)

        out_dir.mkdir(parents=True, exist_ok=True)
        if is_resume and resume_path is not None:
            out_path = resume_path
            mode = "a"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"{self._config.dataset}_{timestamp}.jsonl"
            mode = "w"

        diagnostics_dir = self._resolve_diagnostics_dir(out_path)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        diagnostics_path = self._build_block_ledger_path(out_path, diagnostics_dir)

        if is_resume and resume_path is not None and processed_ids:
            self._validate_resume_diagnostics_alignment(
                resume_path=resume_path,
                diagnostics_path=diagnostics_path,
            )

        all_prompts = self._load_prompts()
        if self._config.sample_limit is not None:
            all_prompts = all_prompts[: self._config.sample_limit]
        prompts = [item for item in all_prompts if item["id"] not in processed_ids]
        if is_resume and resume_path is not None and not prompts:
            print("All samples already processed", file=sys.stderr)
            return str(resume_path)

        iterator = (
            tqdm(prompts, desc=f"Watermarking {self._config.dataset}", unit="prompt")
            if tqdm is not None
            else prompts
        )

        with open(out_path, mode, encoding="utf-8") as f:
            with open(diagnostics_path, mode, encoding="utf-8") as diagnostics_file:
                for item in iterator:
                    result = self._generator.generate(item["prompt"])
                    embed_rate = (
                        result.embedded_blocks / result.total_blocks
                        if result.total_blocks > 0
                        else 0.0
                    )
                    record = {
                        "id": item["id"],
                        "dataset": self._config.dataset,
                        "prompt": item["prompt"],
                        "generated_code": result.code,
                        "blocks": [asdict(contract) for contract in result.block_contracts],
                        "adaptive_mode": result.adaptive_mode,
                        "profile_id": result.profile_id,
                        "alignment_summary": result.alignment_summary,
                        "total_blocks": result.total_blocks,
                        "embedded_blocks": result.embedded_blocks,
                        "failed_blocks": result.failed_blocks,
                        "fallback_blocks": result.fallback_blocks,
                        "embed_rate": embed_rate,
                    }
                    persisted_block_ledgers = self._iter_persisted_block_ledgers(
                        sample_id=item["id"],
                        block_ledgers=result.block_ledgers,
                    )
                    record["diagnostics_ledger_rows"] = len(persisted_block_ledgers)
                    self._merge_diagnostic_summary(record, result.diagnostic_summary)
                    record["watermark_params"] = self._build_public_watermark_params(
                        self._generator
                    )
                    token_channel_metadata = self._build_token_channel_metadata(
                        self._generator
                    )
                    if token_channel_metadata is not None:
                        record["token_channel"] = token_channel_metadata
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    for ledger_row in persisted_block_ledgers:
                        diagnostics_file.write(
                            json.dumps(ledger_row, ensure_ascii=False) + "\n"
                        )

                    f.flush()
                    diagnostics_file.flush()

                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    summary = (
                        f"  ✓ {item['id']} | "
                        f"blocks: {result.embedded_blocks}/{result.total_blocks} | "
                        f"failed: {result.failed_blocks} | "
                        f"fallback: {result.fallback_blocks} | "
                        f"embed_rate: {embed_rate:.1%}"
                    )
                    print(summary, file=sys.stderr)

        return str(out_path)
