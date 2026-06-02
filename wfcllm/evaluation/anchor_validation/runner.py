"""Runner for offline anchor effectiveness diagnostics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch

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
from wfcllm.evaluation.anchor_validation.summary import build_anchor_validation_summary
from wfcllm.watermark.adaptive_gamma.schedule import quantize_gamma
from wfcllm.watermark.anchor_lsh import (
    anchored_signature,
    random_anchor,
    residual_signature,
    sign_with_planes,
)
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace


@dataclass(frozen=True)
class AnchorValidationConfig:
    pool_path: Path
    output_dir: Path
    secret_keys: tuple[str, ...]
    gammas: tuple[float, ...]
    methods: tuple[str, ...]
    retry_budgets: tuple[int, ...]
    lsh_d: int = 3
    embed_dim: int = 128
    embedding_mode: str = "hash"
    encoder_model_path: str = "data/models/codet5-base"
    encoder_checkpoint: Path | None = None
    encoder_device: str = "cpu"
    max_length: int = 256
    use_ordinal_keying: bool = True


@dataclass(frozen=True)
class AnchorValidationResult:
    metrics_path: Path
    selection_path: Path
    summary_path: Path


class AnchorValidationRunner:
    def __init__(self, config: AnchorValidationConfig) -> None:
        self._config = config

    def run(self) -> AnchorValidationResult:
        contexts = load_candidate_contexts(self._config.pool_path)
        provider = _build_embedding_provider(self._config)
        metrics_rows = []
        selection_rows = []
        empirical_gamma_rows: list[dict] = []
        agreement_counts: dict[str, list[bool]] = {}
        embeddings_by_context: dict[str, list[tuple[str, tuple[float, ...]]]] = {}

        for context in contexts:
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
                    signatures = _signatures_for_method(
                        method=method,
                        context=context,
                        block_embeddings=block_embeddings,
                        provider=provider,
                        planes=lsh.planes,
                        secret_key=secret_key,
                        oracle_anchor=oracle_anchor,
                    )
                    if method != AnchorMethod.SEQMARK_ORACLE:
                        oracle_signatures = _signatures_for_method(
                            method=AnchorMethod.SEQMARK_ORACLE,
                            context=context,
                            block_embeddings=block_embeddings,
                            provider=provider,
                            planes=lsh.planes,
                            secret_key=secret_key,
                            oracle_anchor=oracle_anchor,
                        )
                        agreement_counts.setdefault(method.value, []).append(
                            _top_candidate_signature(signatures, context)
                            == _top_candidate_signature(oracle_signatures, context)
                        )
                    signature_list = [
                        signatures[candidate.candidate_id]
                        for candidate in context.candidates
                    ]
                    region_count = 2 ** self._config.lsh_d
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
                        )
                    )
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
                        balance_row = summarize_signature_metrics(
                            context_id=context.context_id,
                            dataset=context.dataset,
                            task_id=context.task_id,
                            method=method.value,
                            signatures=signature_list,
                            region_count=region_count,
                            projection_key_id=key_id,
                            key_id=key_id,
                            gamma=gamma_resolution.gamma_effective,
                            valid_set=valid_set,
                            node_type=context.node_type,
                        )
                        metrics_rows.append(balance_row)
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
                                    signatures_by_candidate_id=signatures,
                                    valid_set=valid_set,
                                )
                            )

        self._config.output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = write_jsonl(
            self._config.output_dir / "region_metrics.jsonl",
            metrics_rows,
        )
        selection_path = write_jsonl(
            self._config.output_dir / "selection_simulation.jsonl",
            selection_rows,
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
            }
        )
        summary_path = self._write_summary(summary_payload)
        return AnchorValidationResult(metrics_path, selection_path, summary_path)

    def _write_summary(self, payload: dict) -> Path:
        path = self._config.output_dir / "anchor_validation_summary.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path


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


def _signatures_for_method(
    method: AnchorMethod,
    context,
    block_embeddings: dict[str, torch.Tensor],
    provider,
    planes: torch.Tensor,
    secret_key: str,
    oracle_anchor: torch.Tensor,
) -> dict[str, tuple[int, ...]]:
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
        else:
            anchor_text = build_anchor_text(method, context, candidate)
            anchor = provider.embed(anchor_text)
            signature = anchored_signature(u, planes, anchor)
        signatures[candidate.candidate_id] = signature
    return signatures
