"""Runner for offline anchor effectiveness diagnostics."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from wfcllm.evaluation.anchor_validation.anchors import build_anchor_text
from wfcllm.evaluation.anchor_validation.embedding import (
    EmbeddingProvider,
    EncoderEmbeddingProvider,
    HashEmbeddingProvider,
)
from wfcllm.evaluation.anchor_validation.io import load_candidate_contexts, write_jsonl
from wfcllm.evaluation.anchor_validation.metrics import summarize_signature_metrics
from wfcllm.evaluation.anchor_validation.pool_diagnostics import (
    build_pool_quality_summary,
    enrich_pool_quality_with_embedding_diversity,
)
from wfcllm.evaluation.anchor_validation.schema import AnchorMethod
from wfcllm.evaluation.anchor_validation.selection import simulate_retry_selection
from wfcllm.evaluation.anchor_validation.summary import (
    build_anchor_validation_summary,
    validate_primary_method_rows,
)
from wfcllm.semantic.anchor_lsh import (
    anchored_signature,
    random_anchor,
    residual_signature,
    sign_with_planes,
)
from wfcllm.semantic.keying import WatermarkKeying
from wfcllm.semantic.lsh_space import LSHSpace
from wfcllm.semantic.rules import _quantize_gamma as quantize_gamma

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - tqdm is in requirements, fallback for minimal envs.
    tqdm = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnchorValidationConfig:
    pool_path: Path
    output_dir: Path
    secret_keys: tuple[str, ...]
    gammas: tuple[float, ...]
    methods: tuple[str, ...]
    retry_budgets: tuple[int, ...]
    primary_method: str = "role_aware_slot_context"
    lsh_d: int = 3
    embed_dim: int = 128
    embedding_mode: str = "hash"
    encoder_model_path: str = "data/models/codet5-base"
    encoder_checkpoint: Path | None = None
    encoder_device: str = "cpu"
    max_length: int = 256
    use_ordinal_keying: bool = True
    show_progress: bool = False


@dataclass(frozen=True)
class AnchorValidationResult:
    metrics_path: Path
    selection_path: Path
    summary_path: Path
    anchor_text_debug_path: Path
    anchor_diagnostics_path: Path


class AnchorValidationRunner:
    def __init__(self, config: AnchorValidationConfig) -> None:
        self._config = config

    def run(self) -> AnchorValidationResult:
        _validate_primary_method_config(self._config)
        logger.info("loading candidate pool: %s", self._config.pool_path)
        contexts = load_candidate_contexts(self._config.pool_path)
        logger.info("loaded %d candidate contexts", len(contexts))
        logger.info(
            "building %s embedding provider with embed_dim=%d",
            self._config.embedding_mode,
            self._config.embed_dim,
        )
        provider = _build_embedding_provider(self._config)
        metrics_rows = []
        selection_rows = []
        empirical_gamma_rows: list[dict] = []
        anchor_debug_rows: list[dict[str, Any]] = []
        agreement_counts: dict[str, list[bool]] = {}
        embeddings_by_context: dict[str, list[tuple[str, tuple[float, ...]]]] = {}

        context_iterable = _progress(
            contexts,
            enabled=self._config.show_progress,
            desc="Anchor diagnostics contexts",
            unit="context",
        )
        for context in context_iterable:
            anchor_embedding_cache: dict[tuple[str, str, str | None], torch.Tensor] = {}
            block_embeddings = {
                candidate.candidate_id: provider.embed(candidate.block_text)
                for candidate in context.candidates
            }
            embeddings_by_context[context.context_id] = [
                (candidate_id, tuple(float(value) for value in embedding.tolist()))
                for candidate_id, embedding in block_embeddings.items()
            ]
            oracle_anchor = _mean_embedding(tuple(block_embeddings.values()))
            for method_name in self._config.methods:
                method = AnchorMethod(method_name)
                for key_index, secret_key in enumerate(self._config.secret_keys):
                    key_id = f"key-{key_index:02d}"
                    lsh = LSHSpace(
                        secret_key=secret_key,
                        embed_dim=provider.embed_dim,
                        d=self._config.lsh_d,
                    )
                    region_count = 2 ** self._config.lsh_d
                    signatures: dict[str, tuple[int, ...]] | None = None
                    if method != AnchorMethod.CANDIDATE_CENTROID_ORACLE:
                        signatures = _signatures_for_method(
                            method=method,
                            context=context,
                            block_embeddings=block_embeddings,
                            provider=provider,
                            planes=lsh.planes,
                            secret_key=secret_key,
                            oracle_anchor=oracle_anchor,
                            valid_set=None,
                            anchor_embedding_cache=anchor_embedding_cache,
                        )
                    if method not in {
                        AnchorMethod.SEQMARK_ORACLE,
                        AnchorMethod.CANDIDATE_CENTROID_ORACLE,
                    }:
                        oracle_signatures = _signatures_for_method(
                            method=AnchorMethod.SEQMARK_ORACLE,
                            context=context,
                            block_embeddings=block_embeddings,
                            provider=provider,
                            planes=lsh.planes,
                            secret_key=secret_key,
                            oracle_anchor=oracle_anchor,
                            valid_set=None,
                            anchor_embedding_cache=anchor_embedding_cache,
                        )
                        if signatures is None:
                            raise ValueError("base signatures are required for agreement")
                        agreement_counts.setdefault(method.value, []).append(
                            _top_candidate_signature(signatures, context)
                            == _top_candidate_signature(oracle_signatures, context)
                        )
                    if signatures is not None:
                        signature_list = [
                            signatures[candidate.candidate_id]
                            for candidate in context.candidates
                        ]
                        metrics_rows.append(
                            summarize_signature_metrics(
                                context_id=context.context_id,
                                dataset=context.dataset,
                                task_id=context.task_id,
                                method=method.value,
                                signatures=signature_list,
                                region_count=region_count,
                                projection_key_id=key_id,
                                key_id=None,
                                gamma=None,
                                valid_set=None,
                                node_type=context.node_type,
                                block_ordinal=context.block_ordinal,
                            )
                        )
                    entropy_by_gamma: dict[str, float] = {}
                    valid_hit_balance_by_gamma: dict[str, dict[str, float | None]] = {}
                    anchor_embedding_by_gamma: dict[str, dict[str, Any]] = {}
                    cosine_distribution_by_gamma: dict[str, dict[str, float | int | None]] = {}
                    for gamma in self._config.gammas:
                        gamma_resolution = quantize_gamma(gamma, self._config.lsh_d)
                        keying = WatermarkKeying(secret_key, self._config.lsh_d)
                        ordinal = (
                            context.block_ordinal
                            if self._config.use_ordinal_keying
                            else None
                        )
                        valid_set = keying.derive(
                            context.parent_node_type,
                            k=gamma_resolution.k,
                            ordinal=ordinal,
                        )
                        if method == AnchorMethod.CANDIDATE_CENTROID_ORACLE:
                            gamma_anchor = _candidate_centroid_oracle_anchor(
                                block_embeddings,
                                lsh.planes,
                                oracle_anchor,
                                valid_set,
                            )
                            gamma_signatures = _signatures_for_method(
                                method=method,
                                context=context,
                                block_embeddings=block_embeddings,
                                provider=provider,
                                planes=lsh.planes,
                                secret_key=secret_key,
                                oracle_anchor=oracle_anchor,
                                valid_set=valid_set,
                                anchor_embedding_cache=anchor_embedding_cache,
                            )
                            anchor_embedding_by_gamma[f"{gamma_resolution.gamma_effective:g}"] = {
                                "embedding_norm": float(torch.linalg.vector_norm(gamma_anchor).item()),
                                "source": _candidate_centroid_oracle_anchor_source(
                                    block_embeddings,
                                    lsh.planes,
                                    oracle_anchor,
                                    valid_set,
                                ),
                            }
                            cosine_distribution_by_gamma[f"{gamma_resolution.gamma_effective:g}"] = (
                                _cosine_distribution(gamma_anchor, block_embeddings)
                            )
                        else:
                            if signatures is None:
                                raise ValueError("base signatures are required for keyed rows")
                            gamma_signatures = signatures
                        gamma_signature_list = [
                            gamma_signatures[candidate.candidate_id]
                            for candidate in context.candidates
                        ]
                        balance_row = summarize_signature_metrics(
                            context_id=context.context_id,
                            dataset=context.dataset,
                            task_id=context.task_id,
                            method=method.value,
                            signatures=gamma_signature_list,
                            region_count=region_count,
                            projection_key_id=key_id,
                            key_id=key_id,
                            gamma=gamma_resolution.gamma_effective,
                            valid_set=valid_set,
                            node_type=context.node_type,
                            block_ordinal=context.block_ordinal,
                        )
                        metrics_rows.append(balance_row)
                        gamma_key = f"{gamma_resolution.gamma_effective:g}"
                        entropy_by_gamma[gamma_key] = balance_row.normalized_entropy
                        valid_hit_balance_by_gamma[gamma_key] = {
                            "valid_hit_rate": balance_row.valid_hit_rate,
                            "gamma_deviation": balance_row.gamma_deviation,
                        }
                        empirical_gamma_rows.append(
                            {
                                "context_id": context.context_id,
                                "method": method.value,
                                "key_id": key_id,
                                "target_gamma": gamma_resolution.gamma_effective,
                                "empirical_gamma": balance_row.valid_hit_rate,
                                "delta": balance_row.gamma_deviation,
                            }
                        )
                        for budget in self._config.retry_budgets:
                            selection_rows.append(
                                simulate_retry_selection(
                                    context_id=context.context_id,
                                    method=method.value,
                                    key_id=key_id,
                                    gamma=gamma_resolution.gamma_effective,
                                    retry_budget=budget,
                                    candidates=context.candidates,
                                    signatures_by_candidate_id=gamma_signatures,
                                    valid_set=valid_set,
                                )
                            )
                    anchor_debug_rows.append(
                        _build_anchor_debug_row(
                            context=context,
                            method=method,
                            key_id=key_id,
                            provider=provider,
                            block_embeddings=block_embeddings,
                            secret_key=secret_key,
                            oracle_anchor=oracle_anchor,
                            entropy_by_gamma=entropy_by_gamma,
                            valid_hit_balance_by_gamma=valid_hit_balance_by_gamma,
                            anchor_embedding_by_gamma=anchor_embedding_by_gamma,
                            cosine_distribution_by_gamma=cosine_distribution_by_gamma,
                            anchor_embedding_cache=anchor_embedding_cache,
                        )
                    )

        validate_primary_method_rows(
            metrics_rows,
            tuple(self._config.methods),
            self._config.primary_method,
        )
        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            "writing %d metric rows and %d selection rows to %s",
            len(metrics_rows),
            len(selection_rows),
            self._config.output_dir,
        )
        metrics_path = write_jsonl(
            self._config.output_dir / "region_metrics.jsonl",
            metrics_rows,
        )
        selection_path = write_jsonl(
            self._config.output_dir / "selection_simulation.jsonl",
            selection_rows,
        )
        anchor_text_debug_path = write_jsonl(
            self._config.output_dir / "anchor_text_debug.jsonl",
            anchor_debug_rows,
        )
        pool_quality = enrich_pool_quality_with_embedding_diversity(
            build_pool_quality_summary(contexts),
            embeddings_by_context,
        )
        summary_payload = build_anchor_validation_summary(
            metrics_rows,
            selection_rows,
            context_count=len(contexts),
            methods=tuple(self._config.methods),
            primary_method=self._config.primary_method,
            pool_quality=pool_quality,
            empirical_gamma_rows=empirical_gamma_rows,
            method_oracle_agreement={
                method: {
                    "proxy": "top_rank_candidate_signature_match",
                    "agreement_rate": _mean_bool(values),
                    "comparison_count": len(values),
                }
                for method, values in sorted(agreement_counts.items())
            },
        )
        summary_payload["meta"].update(
            {
                "pool_path": str(self._config.pool_path),
                "gammas": list(self._config.gammas),
                "retry_budgets": list(self._config.retry_budgets),
                "embedding_mode": self._config.embedding_mode,
                "use_ordinal_keying": self._config.use_ordinal_keying,
                "diagnostic_method_notes": {
                    "candidate_centroid_oracle": (
                        "diagnostic-only; keyed gamma rows use a valid-vs-invalid "
                        "candidate embedding centroid direction when both sides are "
                        "non-empty, otherwise the context candidate centroid fallback"
                    ),
                    "context_centroid_oracle": (
                        "diagnostic-only; uses the context candidate centroid "
                        "embedding as an upper-bound anchor direction"
                    ),
                },
            }
        )
        summary_path = self._write_summary(summary_payload)
        anchor_diagnostics_path = self._write_anchor_diagnostics(
            summary_payload["anchor_diagnostics"]
        )
        logger.info("wrote anchor validation summary: %s", summary_path)
        return AnchorValidationResult(
            metrics_path,
            selection_path,
            summary_path,
            anchor_text_debug_path,
            anchor_diagnostics_path,
        )

    def _write_summary(self, payload: dict) -> Path:
        path = self._config.output_dir / "anchor_validation_summary.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def _write_anchor_diagnostics(self, payload: dict) -> Path:
        path = self._config.output_dir / "anchor_diagnostics.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


def _validate_primary_method_config(config: AnchorValidationConfig) -> None:
    if config.primary_method not in config.methods:
        raise ValueError(
            f"primary_method {config.primary_method!r} is not present in methods"
        )
    missing_baselines = [
        method
        for method in ("vanilla", "random", "seqmark_oracle")
        if method not in config.methods
    ]
    if missing_baselines:
        raise ValueError(
            "required baseline methods missing from config methods: "
            + ", ".join(missing_baselines)
        )


def _build_embedding_provider(config: AnchorValidationConfig) -> EmbeddingProvider:
    if config.embedding_mode == "hash":
        return HashEmbeddingProvider(embed_dim=config.embed_dim)
    if config.embedding_mode != "encoder":
        raise ValueError(f"unsupported embedding mode: {config.embedding_mode}")

    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder

    enc_config = EncoderConfig(
        model_name=config.encoder_model_path,
        embed_dim=config.embed_dim,
    )
    encoder = SemanticEncoder(config=enc_config)
    if config.encoder_checkpoint is not None:
        checkpoint = torch.load(config.encoder_checkpoint, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        encoder.load_state_dict(state_dict)
    encoder = encoder.to(config.encoder_device)
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_model_path)
    return EncoderEmbeddingProvider(
        encoder=encoder,
        tokenizer=tokenizer,
        device=config.encoder_device,
        max_length=config.max_length,
    )


def _mean_embedding(values: tuple[torch.Tensor, ...]) -> torch.Tensor:
    if not values:
        raise ValueError("at least one candidate embedding is required")
    return torch.stack(values).mean(dim=0)


def _top_candidate_signature(
    signatures: dict[str, tuple[int, ...]],
    context,
) -> tuple[int, ...]:
    top_candidate = min(context.candidates, key=lambda candidate: candidate.rank)
    return signatures[top_candidate.candidate_id]


def _mean_bool(values: list[bool]) -> float:
    if not values:
        return 0.0
    return sum(1.0 if value else 0.0 for value in values) / len(values)


def _progress(
    values,
    *,
    enabled: bool,
    desc: str,
    unit: str,
):
    if not enabled or tqdm is None:
        return values
    return tqdm(
        values,
        total=len(values),
        desc=desc,
        unit=unit,
        dynamic_ncols=True,
    )


def _signatures_for_method(
    method: AnchorMethod,
    context,
    block_embeddings: dict[str, torch.Tensor],
    provider,
    planes: torch.Tensor,
    secret_key: str,
    oracle_anchor: torch.Tensor,
    valid_set: frozenset[tuple[int, ...]] | None,
    anchor_embedding_cache: dict[tuple[str, str, str | None], torch.Tensor],
) -> dict[str, tuple[int, ...]]:
    if method == AnchorMethod.CANDIDATE_CENTROID_ORACLE:
        anchor = _candidate_centroid_oracle_anchor(
            block_embeddings,
            planes,
            oracle_anchor,
            valid_set,
        )
    else:
        anchor = None
    signatures: dict[str, tuple[int, ...]] = {}
    for candidate in context.candidates:
        u = block_embeddings[candidate.candidate_id]
        if method == AnchorMethod.VANILLA:
            signature = sign_with_planes(u, planes)
        elif method == AnchorMethod.RANDOM:
            anchor = random_anchor(
                secret_key,
                context.context_id,
                method.value,
                provider.embed_dim,
            )
            signature = anchored_signature(u, planes, anchor)
        elif method == AnchorMethod.SEQMARK_ORACLE:
            signature = residual_signature(u, center=oracle_anchor, planes=planes)
        elif method == AnchorMethod.CONTEXT_CENTROID_ORACLE:
            signature = anchored_signature(u, planes, oracle_anchor)
        elif method == AnchorMethod.CANDIDATE_CENTROID_ORACLE:
            signature = anchored_signature(u, planes, anchor)
        else:
            anchor = _cached_anchor_embedding(
                cache=anchor_embedding_cache,
                method=method,
                context=context,
                candidate=candidate,
                provider=provider,
            )
            signature = anchored_signature(u, planes, anchor)
        signatures[candidate.candidate_id] = signature
    return signatures


def _cached_anchor_embedding(
    *,
    cache: dict[tuple[str, str, str | None], torch.Tensor],
    method: AnchorMethod,
    context,
    candidate,
    provider,
) -> torch.Tensor:
    cache_key = (
        context.context_id,
        method.value,
        _anchor_cache_candidate_id(method, candidate),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    anchor_text = build_anchor_text(method, context, candidate)
    embedding = provider.embed(anchor_text)
    cache[cache_key] = embedding
    return embedding


def _anchor_cache_candidate_id(method: AnchorMethod, candidate) -> str | None:
    if method in {
        AnchorMethod.SKELETON,
        AnchorMethod.SLOT_CONTEXT_SKELETON,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT,
        AnchorMethod.ROLE_AWARE_SLOT_CONTEXT_SKELETON,
        AnchorMethod.CODET5_VALID_SKELETON,
        AnchorMethod.CODET5_COMMENT_ANCHOR,
        AnchorMethod.CODET5_COMMENT_MINIMAL,
        AnchorMethod.CODET5_COMMENT_CONTEXTUAL,
        AnchorMethod.CODET5_IDENTIFIER_ANCHOR,
    }:
        return candidate.candidate_id
    return None


def _candidate_centroid_oracle_anchor(
    block_embeddings: dict[str, torch.Tensor],
    planes: torch.Tensor,
    oracle_anchor: torch.Tensor,
    valid_set: frozenset[tuple[int, ...]] | None,
) -> torch.Tensor:
    if valid_set is None:
        return oracle_anchor
    valid_embeddings: list[torch.Tensor] = []
    invalid_embeddings: list[torch.Tensor] = []
    for embedding in block_embeddings.values():
        base_signature = anchored_signature(embedding, planes, oracle_anchor)
        if base_signature in valid_set:
            valid_embeddings.append(embedding)
        else:
            invalid_embeddings.append(embedding)
    if not valid_embeddings or not invalid_embeddings:
        return oracle_anchor
    valid_centroid = _mean_embedding(tuple(valid_embeddings))
    invalid_centroid = _mean_embedding(tuple(invalid_embeddings))
    direction = valid_centroid - invalid_centroid
    if torch.linalg.vector_norm(direction).item() <= 1e-8:
        return oracle_anchor
    return direction


def _candidate_centroid_oracle_anchor_source(
    block_embeddings: dict[str, torch.Tensor],
    planes: torch.Tensor,
    oracle_anchor: torch.Tensor,
    valid_set: frozenset[tuple[int, ...]],
) -> str:
    valid_count = 0
    invalid_count = 0
    for embedding in block_embeddings.values():
        base_signature = anchored_signature(embedding, planes, oracle_anchor)
        if base_signature in valid_set:
            valid_count += 1
        else:
            invalid_count += 1
    if valid_count and invalid_count:
        return "valid_minus_invalid_centroid"
    return "context_centroid_fallback"


def _build_anchor_debug_row(
    *,
    context,
    method: AnchorMethod,
    key_id: str,
    provider,
    block_embeddings: dict[str, torch.Tensor],
    secret_key: str,
    oracle_anchor: torch.Tensor,
    entropy_by_gamma: dict[str, float],
    valid_hit_balance_by_gamma: dict[str, dict[str, float | None]],
    anchor_embedding_by_gamma: dict[str, dict[str, Any]],
    cosine_distribution_by_gamma: dict[str, dict[str, float | int | None]],
    anchor_embedding_cache: dict[tuple[str, str, str | None], torch.Tensor],
) -> dict[str, Any]:
    anchor_text = _debug_anchor_text(method, context)
    anchor_embedding = _debug_anchor_embedding(
        method=method,
        context=context,
        provider=provider,
        secret_key=secret_key,
        oracle_anchor=oracle_anchor,
        anchor_text=anchor_text,
        anchor_embedding_cache=anchor_embedding_cache,
    )
    token_info = _token_debug(provider, anchor_text)
    cosine_distribution = (
        None
        if method == AnchorMethod.CANDIDATE_CENTROID_ORACLE
        else _cosine_distribution(anchor_embedding, block_embeddings)
    )
    return {
        "context_id": context.context_id,
        "method": method.value,
        "key_id": key_id,
        "anchor_text": anchor_text,
        "tokenized_length": token_info["tokenized_length"],
        "first_token_ids": token_info["first_token_ids"],
        "token_pieces": token_info["token_pieces"],
        "embedding_norm": (
            float(torch.linalg.vector_norm(anchor_embedding).item())
            if anchor_embedding is not None
            else None
        ),
        "cosine_distribution": cosine_distribution,
        "anchor_embedding_by_gamma": anchor_embedding_by_gamma,
        "cosine_distribution_by_gamma": cosine_distribution_by_gamma,
        "entropy_by_gamma": entropy_by_gamma,
        "valid_hit_balance_by_gamma": valid_hit_balance_by_gamma,
    }


def _debug_anchor_text(method: AnchorMethod, context) -> str | None:
    if method in {
        AnchorMethod.VANILLA,
        AnchorMethod.RANDOM,
        AnchorMethod.SEQMARK_ORACLE,
        AnchorMethod.CANDIDATE_CENTROID_ORACLE,
        AnchorMethod.CONTEXT_CENTROID_ORACLE,
    }:
        return None
    candidate = min(context.candidates, key=lambda item: item.rank)
    return build_anchor_text(method, context, candidate)


def _debug_anchor_embedding(
    *,
    method: AnchorMethod,
    context,
    provider,
    secret_key: str,
    oracle_anchor: torch.Tensor,
    anchor_text: str | None,
    anchor_embedding_cache: dict[tuple[str, str, str | None], torch.Tensor],
) -> torch.Tensor | None:
    if method == AnchorMethod.VANILLA:
        return None
    if method == AnchorMethod.RANDOM:
        return random_anchor(secret_key, context.context_id, method.value, provider.embed_dim)
    if method == AnchorMethod.CANDIDATE_CENTROID_ORACLE:
        return None
    if method in {AnchorMethod.SEQMARK_ORACLE, AnchorMethod.CONTEXT_CENTROID_ORACLE}:
        return oracle_anchor
    if anchor_text is None:
        return None
    return _cached_anchor_embedding(
        cache=anchor_embedding_cache,
        method=method,
        context=context,
        candidate=min(context.candidates, key=lambda item: item.rank),
        provider=provider,
    )


def _token_debug(provider, anchor_text: str | None) -> dict[str, Any]:
    tokenizer = getattr(provider, "_tokenizer", None)
    if tokenizer is None or anchor_text is None:
        return {
            "tokenized_length": None,
            "first_token_ids": [],
            "token_pieces": [],
        }
    encoded = tokenizer(
        anchor_text,
        add_special_tokens=True,
        truncation=True,
        max_length=getattr(provider, "_max_length", None),
    )
    input_ids = [int(value) for value in encoded.get("input_ids", [])]
    first_token_ids = input_ids[:8]
    converter = getattr(tokenizer, "convert_ids_to_tokens", None)
    token_pieces = converter(first_token_ids) if converter is not None else []
    return {
        "tokenized_length": len(input_ids),
        "first_token_ids": first_token_ids,
        "token_pieces": list(token_pieces),
    }


def _cosine_distribution(
    anchor_embedding: torch.Tensor | None,
    block_embeddings: dict[str, torch.Tensor],
) -> dict[str, float | int | None]:
    if anchor_embedding is None or not block_embeddings:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }
    anchor = F.normalize(anchor_embedding.float().flatten().unsqueeze(0), dim=1)
    values = []
    for embedding in block_embeddings.values():
        candidate = F.normalize(embedding.float().flatten().unsqueeze(0), dim=1)
        values.append(float((anchor @ candidate.T).squeeze().item()))
    values.sort()
    return {
        "count": len(values),
        "min": values[0],
        "max": values[-1],
        "mean": sum(values) / len(values),
        "median": values[len(values) // 2],
    }
