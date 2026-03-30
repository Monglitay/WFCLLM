"""Watermark-embedded code generation using rejection sampling."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.block_contract import BlockContract, build_block_contracts
from wfcllm.watermark.cascade import CascadeManager
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.context import GenerationContext
from wfcllm.watermark.diagnostics import (
    BlockLifecycleRecord,
    FailureReason,
    hash_block_text,
    summarize_sample_diagnostics,
)
from wfcllm.watermark.entropy import ENTROPY_SCALE, NodeEntropyEstimator
from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import GammaResolution, PiecewiseQuantileSchedule, quantize_gamma
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.retry_loop import RetryLoop, RetryDiagnostics
from wfcllm.watermark.verifier import ProjectionVerifier

logger = logging.getLogger(__name__)


@dataclass
class EmbedStats:
    """Watermark embedding statistics."""

    total_blocks: int = 0
    embedded_blocks: int = 0
    failed_blocks: int = 0
    fallback_blocks: int = 0
    cascade_blocks: int = 0
    retry_diagnostics: list[RetryDiagnostics] = field(default_factory=list)


@dataclass
class GenerateResult:
    """Result of watermark-embedded generation."""

    code: str
    stats: EmbedStats
    block_contracts: list[BlockContract] = field(default_factory=list)
    adaptive_mode: str = "fixed"
    profile_id: str | None = None
    alignment_summary: dict[str, int | bool] = field(default_factory=dict)
    diagnostic_summary: dict[str, object] = field(default_factory=dict)
    block_ledgers: list[dict[str, object]] = field(default_factory=list)

    # Backward-compatible properties
    @property
    def total_blocks(self) -> int:
        return self.stats.total_blocks

    @property
    def embedded_blocks(self) -> int:
        return self.stats.embedded_blocks

    @property
    def failed_blocks(self) -> int:
        return self.stats.failed_blocks

    @property
    def fallback_blocks(self) -> int:
        return self.stats.fallback_blocks


class WatermarkGenerator:
    """Code generator with watermark embedding via rejection sampling."""

    def __init__(
        self,
        model,
        tokenizer,
        encoder,
        encoder_tokenizer,
        config: WatermarkConfig,
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._config = config

        self._entropy_est = NodeEntropyEstimator()
        self._lsh_space = LSHSpace(
            config.secret_key, config.encoder_embed_dim, config.lsh_d
        )
        self._keying = WatermarkKeying(
            config.secret_key, config.lsh_d, config.lsh_gamma
        )
        self._verifier = ProjectionVerifier(
            encoder, encoder_tokenizer,
            lsh_space=self._lsh_space,
            device=config.encoder_device,
        )
        self._entropy_profile: EntropyProfile | None = None
        self._gamma_schedule: PiecewiseQuantileSchedule | None = None
        self._initialize_adaptive_gamma()
        self._cascade_rollback_counter = 0

        _STRUCTURAL_KEYWORDS = [
            "import", "return", "def", "class", "if", "else", "elif",
            "for", "while", "with", "try", "except", "finally", "pass",
            "break", "continue", "raise", "yield", "lambda",
            "and", "or", "not", "in", "is", "from", "as", "assert",
            "del", "global", "nonlocal", "\n", " ", "\t",
        ]
        self._structural_token_ids: set[int] = {
            tid
            for kw in _STRUCTURAL_KEYWORDS
            for tid in self._tokenizer.encode(kw, add_special_tokens=False)
        }

    @property
    def config(self) -> WatermarkConfig:
        return self._config

    @torch.no_grad()
    def generate(self, prompt: str) -> GenerateResult:
        """Generate code with watermark embedding."""
        ctx = GenerationContext(
            model=self._model,
            tokenizer=self._tokenizer,
            config=self._config,
        )
        ctx.prefill(prompt)

        stats = EmbedStats()
        cascade_mgr = CascadeManager(self._config)
        retry_loop = RetryLoop(
            ctx=ctx,
            config=self._config,
            verifier=self._verifier,
            keying=self._keying,
            entropy_est=self._entropy_est,
            structural_token_ids=self._structural_token_ids,
            gamma_resolver=self._resolve_gamma_for_block_text,
        )
        pending_fallbacks: list[str] = []
        sample_id = self._sample_id_for_prompt(prompt)
        ledger_entries: list[dict[str, object]] = []
        ledger_by_ordinal: dict[int, dict[str, object]] = {}
        active_cascade_scope: dict[str, object] | None = None
        self._last_active_cascade_scope = None

        while not ctx.is_finished():
            next_id = ctx.forward_and_sample()

            if next_id == ctx.eos_id:
                break

            event = ctx.last_event
            if event is None:
                continue

            if event.block_type == "compound":
                active_cascade_scope = self._update_active_cascade_scope_for_compound(
                    active_cascade_scope,
                    event,
                )
                cascade_mgr.on_compound_block_start(
                    ctx,
                    event,
                    stats_snapshot=self._snapshot_runtime_stats(stats),
                )
                continue

            self._process_simple_block(
                event=event,
                ctx=ctx,
                stats=stats,
                cascade_mgr=cascade_mgr,
                retry_loop=retry_loop,
                pending_fallbacks=pending_fallbacks,
                sample_id=sample_id,
                ledger_entries=ledger_entries,
                ledger_by_ordinal=ledger_by_ordinal,
                active_cascade_scope=active_cascade_scope,
                allow_cascade=True,
            )
            active_cascade_scope = self._last_active_cascade_scope

        final_event = getattr(ctx, "flush_final_event", lambda: None)()
        if final_event is not None and final_event.block_type == "simple":
            self._process_simple_block(
                event=final_event,
                ctx=ctx,
                stats=stats,
                cascade_mgr=cascade_mgr,
                retry_loop=retry_loop,
                pending_fallbacks=pending_fallbacks,
                sample_id=sample_id,
                ledger_entries=ledger_entries,
                ledger_by_ordinal=ledger_by_ordinal,
                active_cascade_scope=active_cascade_scope,
                allow_cascade=False,
            )
            active_cascade_scope = self._last_active_cascade_scope

        final_code = ctx.generated_text
        gamma_resolver = (
            self._resolve_gamma_for_entropy_units
            if self._is_adaptive_runtime_enabled()
            else None
        )
        block_contracts = build_block_contracts(
            final_code,
            gamma_resolver=gamma_resolver,
        )
        runtime_total_blocks = stats.total_blocks
        final_total_blocks, final_embedded_blocks = self._finalize_stats(final_code)
        stats.total_blocks = final_total_blocks
        stats.embedded_blocks = final_embedded_blocks
        lifecycle_records = [
            entry["record"]
            for entry in ledger_entries
            if isinstance(entry.get("record"), BlockLifecycleRecord)
        ]

        return GenerateResult(
            code=final_code,
            stats=stats,
            block_contracts=block_contracts,
            adaptive_mode=self._adaptive_mode(),
            profile_id=self._profile_id(),
            alignment_summary=self._build_alignment_summary(
                runtime_total_blocks,
                block_contracts,
            ),
            diagnostic_summary=summarize_sample_diagnostics(lifecycle_records),
            block_ledgers=self._serialize_block_ledgers(ledger_entries),
        )

    def _verify_block(self, event):
        """Verify a single block against LSH criteria."""
        entropy_units = self._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin = self._entropy_est.compute_margin(block_entropy, self._config)
        gamma_resolution = self._resolve_gamma_for_entropy_units(entropy_units)
        valid_set = self._keying.derive(
            event.parent_node_type or "module",
            k=gamma_resolution.k,
        )
        result = self._verifier.verify(event.block_text, valid_set, margin)

        logger.debug(
            "[simple block] node=%s parent=%s entropy=%.4f margin_thresh=%.4f "
            "gamma_target=%.4f k=%d gamma_effective=%.4f\n"
            "  sig=%s in_valid=%s valid_set_size=%d min_margin=%.4f passed=%s\n"
            "  text=%r",
            event.node_type, event.parent_node_type,
            block_entropy, margin,
            gamma_resolution.gamma_target,
            gamma_resolution.k,
            gamma_resolution.gamma_effective,
            result.lsh_signature,
            result.lsh_signature in valid_set,
            len(valid_set), result.min_margin, result.passed,
            event.block_text[:80],
        )
        return result

    def _try_cascade(self, ctx, cascade_mgr, retry_loop, stats, pending_fallbacks):
        """Active cascade: rollback to compound block start, then resume main loop.

        We intentionally do NOT verify the first regenerated compound event here.
        Incremental parsing emits compound blocks at header-complete time, so
        verifying inside this helper would use intermediate text that does not
        match the final AST seen by extraction. After rollback, the outer
        generation loop continues normally and verifies the regenerated simple
        blocks against final-text boundaries.
        """
        cascade_cp = cascade_mgr.cascade(ctx)
        if cascade_cp is None:
            return None

        self._restore_runtime_stats(
            stats,
            getattr(cascade_cp, "stats_snapshot", None),
        )
        stats.cascade_blocks += 1
        pending_fallbacks.clear()
        restored_stats = self._snapshot_runtime_stats(stats)
        logger.debug(
            "[CASCADE OK] rolled back to compound block start; resuming main loop"
        )
        metadata_builder = getattr(cascade_cp, "build_diagnostic_metadata", None)
        scope_builder = getattr(cascade_cp, "build_replacement_scope", None)
        diagnostic_metadata = (
            metadata_builder(restored_stats=restored_stats)
            if callable(metadata_builder)
            else self._fallback_cascade_metadata(cascade_cp, restored_stats)
        )
        replacement_scope = (
            scope_builder()
            if callable(scope_builder)
            else self._fallback_replacement_scope(cascade_cp)
        )
        self._cascade_rollback_counter += 1
        rollback_id = f"cascade-{self._cascade_rollback_counter}"
        if isinstance(diagnostic_metadata, dict):
            diagnostic_metadata = dict(diagnostic_metadata)
            diagnostic_metadata["rollback_id"] = rollback_id
        if isinstance(replacement_scope, dict):
            replacement_scope = dict(replacement_scope)
            replacement_scope["rollback_id"] = rollback_id
        return {
            "diagnostic_metadata": diagnostic_metadata,
            "replacement_scope": replacement_scope,
        }

    def _adaptive_mode(self) -> str:
        """Return canonical adaptive metadata mode for the current config."""
        if self._is_adaptive_runtime_enabled():
            return self._config.adaptive_gamma.strategy
        return "fixed"

    def _profile_id(self) -> str | None:
        if not self._is_adaptive_runtime_enabled():
            return None
        return self._config.adaptive_gamma.profile_id

    def _is_adaptive_runtime_enabled(self) -> bool:
        return self._gamma_schedule is not None

    def _initialize_adaptive_gamma(self) -> None:
        adaptive_config = self._config.adaptive_gamma
        if not adaptive_config.enabled:
            return
        if adaptive_config.profile_path is None:
            return
        if adaptive_config.strategy != "piecewise_quantile":
            raise ValueError(
                f"unsupported adaptive gamma strategy: {adaptive_config.strategy}"
            )

        self._entropy_profile = EntropyProfile.load(adaptive_config.profile_path)
        anchor_quantiles = tuple(adaptive_config.anchors.keys())
        anchor_gammas = tuple(
            adaptive_config.anchors[quantile]
            for quantile in anchor_quantiles
        )
        self._gamma_schedule = PiecewiseQuantileSchedule(
            profile=self._entropy_profile,
            anchor_quantiles=anchor_quantiles,
            anchor_gammas=anchor_gammas,
        )

    def _resolve_gamma_for_block_text(self, block_text: str) -> GammaResolution:
        entropy_units = self._entropy_est.estimate_block_entropy_units(block_text)
        return self._resolve_gamma_for_entropy_units(entropy_units)

    def _resolve_gamma_for_entropy_units(self, entropy_units: int) -> GammaResolution:
        if self._gamma_schedule is not None:
            return self._gamma_schedule.resolve(entropy_units, self._config.lsh_d)
        return quantize_gamma(self._config.lsh_gamma, self._config.lsh_d)

    def _build_alignment_summary(
        self,
        runtime_total_blocks: int,
        block_contracts: list[BlockContract],
    ) -> dict[str, int | bool]:
        """Summarize final AST block metadata vs generation-time counters."""
        final_block_count = len(block_contracts)
        return {
            "final_block_count": final_block_count,
            "generator_total_blocks": runtime_total_blocks,
            "block_count_matches_total_blocks": final_block_count == runtime_total_blocks,
        }

    def _process_simple_block(
        self,
        event,
        ctx,
        stats: EmbedStats,
        cascade_mgr: CascadeManager,
        retry_loop: RetryLoop,
        pending_fallbacks: list[str],
        sample_id: str,
        ledger_entries: list[dict[str, object]],
        ledger_by_ordinal: dict[int, dict[str, object]],
        active_cascade_scope: dict[str, object] | None,
        allow_cascade: bool,
    ) -> None:
        stats.total_blocks += 1
        ledger_entry, reused_from_cascade, active_cascade_scope = self._acquire_block_ledger(
            sample_id,
            event,
            ledger_entries,
            ledger_by_ordinal,
            active_cascade_scope,
        )
        self._last_active_cascade_scope = active_cascade_scope
        record = ledger_entry["record"]
        self._capture_block_identity(ledger_entry, event)

        verify_result = self._verify_block(event)
        if not reused_from_cascade and not record.initial_verify:
            record.initial_verify = {"passed": verify_result.passed}
            if not verify_result.passed:
                record.initial_verify["failure_reason"] = self._classify_failure_reason(
                    event,
                    verify_result,
                )

        if verify_result.passed:
            stats.embedded_blocks += 1
            pending_fallbacks.clear()
            self._mark_block_success(
                record,
                rescued_by_cascade=reused_from_cascade,
            )
            return

        failure_reason = self._classify_failure_reason(event, verify_result)

        block_cp = ctx.last_block_checkpoint
        if block_cp is None:
            stats.failed_blocks += 1
            pending_fallbacks.append(event.block_text)
            self._mark_block_failure(
                record,
                failure_reason=failure_reason,
                exhausted_retries=False,
            )
            return

        retry_result = retry_loop.run(block_cp, event)
        stats.retry_diagnostics.append(retry_result.diagnostics)
        self._append_retry_attempts(record, retry_result.diagnostics)

        if retry_result.success:
            stats.embedded_blocks += 1
            pending_fallbacks.clear()
            self._mark_block_success(
                record,
                rescued_by_retry=True,
                rescued_by_cascade=reused_from_cascade,
            )
            logger.debug(
                "[RETRY OK] block #%d after %d attempts",
                stats.total_blocks, retry_result.attempts,
            )
            return

        stats.failed_blocks += 1
        pending_fallbacks.append(event.block_text)
        self._mark_block_failure(
            record,
            failure_reason=failure_reason,
            exhausted_retries=True,
        )
        cascade_mgr.on_simple_block_failed(
            event.block_text,
            block_ordinal=record.block_ordinal,
        )
        logger.debug(
            "[RETRY FAILED] block #%d exhausted %d retries",
            stats.total_blocks, retry_result.attempts,
        )

        if allow_cascade and cascade_mgr.should_cascade():
            cascade_result = self._try_cascade(
                ctx,
                cascade_mgr,
                retry_loop,
                stats,
                pending_fallbacks,
            )
            self._last_active_cascade_scope = self._activate_cascade_replacement_scope(
                ledger_by_ordinal,
                cascade_result,
            )

    @staticmethod
    def _sample_id_for_prompt(prompt: str) -> str:
        return f"prompt:{hash_block_text(prompt)[:16]}"

    def _acquire_block_ledger(
        self,
        sample_id: str,
        event,
        ledger_entries: list[dict[str, object]],
        ledger_by_ordinal: dict[int, dict[str, object]],
        active_cascade_scope: dict[str, object] | None,
    ) -> tuple[dict[str, object], bool, dict[str, object] | None]:
        active_cascade_scope, block_ordinal = self._resolve_cascade_replacement_ordinal(
            active_cascade_scope,
            event,
        )
        if block_ordinal is not None:
            entry = ledger_by_ordinal[block_ordinal]
            return entry, True, active_cascade_scope

        block_ordinal = len(ledger_entries)
        entry = {
            "record": BlockLifecycleRecord(
                sample_id=sample_id,
                block_ordinal=block_ordinal,
            ),
        }
        ledger_entries.append(entry)
        ledger_by_ordinal[block_ordinal] = entry
        return entry, False, active_cascade_scope

    @staticmethod
    def _capture_block_identity(
        ledger_entry: dict[str, object],
        event,
    ) -> None:
        ledger_entry["node_type"] = event.node_type
        ledger_entry["parent_node_type"] = event.parent_node_type or "module"
        ledger_entry["block_text_hash"] = hash_block_text(event.block_text)

    def _classify_failure_reason(self, event, verify_result) -> str:
        if verify_result.passed:
            return FailureReason.unknown.value
        entropy_units = self._entropy_est.estimate_block_entropy_units(event.block_text)
        block_entropy = entropy_units / ENTROPY_SCALE
        margin_threshold = self._entropy_est.compute_margin(block_entropy, self._config)
        in_valid_set = verify_result.in_valid_set
        if in_valid_set is None:
            return FailureReason.unknown.value
        margin_passed = verify_result.min_margin > margin_threshold
        if not in_valid_set and margin_passed:
            return FailureReason.signature_miss.value
        if in_valid_set and not margin_passed:
            return FailureReason.margin_miss.value
        if not in_valid_set and not margin_passed:
            return FailureReason.signature_and_margin_miss.value
        return FailureReason.unknown.value

    @staticmethod
    def _append_retry_attempts(
        record: BlockLifecycleRecord,
        diagnostics: RetryDiagnostics,
    ) -> None:
        for attempt_index, attempt in enumerate(diagnostics.per_attempt):
            retry_record = {
                "attempt_index": attempt_index,
                "produced_block": not attempt.no_block,
            }
            if attempt.failure_reason is not None:
                retry_record["failure_reason"] = attempt.failure_reason
            record.retry_attempts.append(retry_record)

    @staticmethod
    def _mark_block_success(
        record: BlockLifecycleRecord,
        rescued_by_retry: bool = False,
        rescued_by_cascade: bool = False,
    ) -> None:
        record.final_outcome["embedded"] = True
        record.final_outcome.pop("failure_reason", None)
        record.final_outcome.pop("exhausted_retries", None)
        record.final_outcome.setdefault("rescued_by_retry", False)
        record.final_outcome.setdefault("rescued_by_cascade", False)
        if rescued_by_retry:
            record.final_outcome["rescued_by_retry"] = True
        if rescued_by_cascade:
            record.final_outcome["rescued_by_cascade"] = True

    @staticmethod
    def _mark_block_failure(
        record: BlockLifecycleRecord,
        failure_reason: str,
        exhausted_retries: bool,
    ) -> None:
        record.final_outcome["embedded"] = False
        record.final_outcome.setdefault("rescued_by_retry", False)
        record.final_outcome.setdefault("rescued_by_cascade", False)
        record.final_outcome["exhausted_retries"] = exhausted_retries
        record.final_outcome["failure_reason"] = failure_reason

    def _activate_cascade_replacement_scope(
        self,
        ledger_by_ordinal: dict[int, dict[str, object]],
        cascade_result: dict[str, object] | None,
    ) -> dict[str, object] | None:
        if not isinstance(cascade_result, dict):
            return None
        cascade_metadata = cascade_result.get("diagnostic_metadata")
        replacement_scope = cascade_result.get("replacement_scope")
        if isinstance(cascade_metadata, dict):
            for block_ordinal in cascade_metadata.get("replaced_block_ordinals", []):
                if not isinstance(block_ordinal, int):
                    continue
                entry = ledger_by_ordinal.get(block_ordinal)
                if entry is None:
                    continue
                record = entry["record"]
                record.cascade_events.append(dict(cascade_metadata))
                record.final_outcome["failure_reason"] = FailureReason.cascade_replaced.value

        if not isinstance(replacement_scope, dict):
            return None
        pending_ordinals: list[int] = []
        for block_ordinal in replacement_scope.get("replaced_block_ordinals", []):
            if not isinstance(block_ordinal, int):
                continue
            if block_ordinal in ledger_by_ordinal:
                pending_ordinals.append(block_ordinal)
        if not pending_ordinals:
            return None
        return {
            "rollback_id": replacement_scope.get("rollback_id"),
            "compound_node_type": replacement_scope.get("compound_node_type"),
            "compound_parent_node_type": replacement_scope.get(
                "compound_parent_node_type",
                "module",
            ),
            "pending_block_ordinals": pending_ordinals,
            "descendant_parent_types": set(),
            "ambiguous_same_type_nesting": False,
        }

    def _resolve_cascade_replacement_ordinal(
        self,
        active_cascade_scope: dict[str, object] | None,
        event,
    ) -> tuple[dict[str, object] | None, int | None]:
        if not isinstance(active_cascade_scope, dict):
            return None, None

        compound_node_type = active_cascade_scope.get("compound_node_type")
        if not isinstance(compound_node_type, str):
            return None, None

        if active_cascade_scope.get("ambiguous_same_type_nesting"):
            return None, None

        parent_node_type = event.parent_node_type or "module"
        descendant_parent_types = active_cascade_scope.get("descendant_parent_types")
        if not isinstance(descendant_parent_types, set):
            descendant_parent_types = set()
            active_cascade_scope["descendant_parent_types"] = descendant_parent_types

        if parent_node_type == compound_node_type:
            pending_block_ordinals = active_cascade_scope.get("pending_block_ordinals")
            if not isinstance(pending_block_ordinals, list) or not pending_block_ordinals:
                return None, None
            block_ordinal = pending_block_ordinals.pop(0)
            if pending_block_ordinals:
                active_cascade_scope["pending_block_ordinals"] = pending_block_ordinals
                return active_cascade_scope, block_ordinal
            return None, block_ordinal

        if parent_node_type in descendant_parent_types:
            return active_cascade_scope, None

        return None, None

    def _update_active_cascade_scope_for_compound(
        self,
        active_cascade_scope: dict[str, object] | None,
        event,
    ) -> dict[str, object] | None:
        if not isinstance(active_cascade_scope, dict):
            return None

        compound_node_type = active_cascade_scope.get("compound_node_type")
        compound_parent_node_type = active_cascade_scope.get(
            "compound_parent_node_type",
            "module",
        )
        if not isinstance(compound_node_type, str):
            return None

        parent_node_type = event.parent_node_type or "module"
        descendant_parent_types = active_cascade_scope.get("descendant_parent_types")
        if not isinstance(descendant_parent_types, set):
            descendant_parent_types = set()
            active_cascade_scope["descendant_parent_types"] = descendant_parent_types

        if event.node_type == compound_node_type and parent_node_type == compound_parent_node_type:
            return active_cascade_scope
        if parent_node_type == compound_node_type or parent_node_type in descendant_parent_types:
            if event.node_type == compound_node_type:
                active_cascade_scope["ambiguous_same_type_nesting"] = True
            descendant_parent_types.add(event.node_type)
            return active_cascade_scope
        return None

    @staticmethod
    def _serialize_block_ledgers(
        ledger_entries: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        payloads: list[dict[str, object]] = []
        for entry in ledger_entries:
            record = entry["record"]
            payload = record.to_dict()
            if "node_type" in entry:
                payload["node_type"] = entry["node_type"]
            if "parent_node_type" in entry:
                payload["parent_node_type"] = entry["parent_node_type"]
            if "block_text_hash" in entry:
                payload["block_text_hash"] = entry["block_text_hash"]
            payloads.append(payload)
        return payloads

    @staticmethod
    def _fallback_cascade_metadata(
        cascade_cp,
        restored_stats: dict[str, object],
    ) -> dict[str, object]:
        replaced_block_ordinals: list[int] = []
        for item in getattr(cascade_cp, "failed_simple_blocks", []):
            if isinstance(item, dict) and isinstance(item.get("block_ordinal"), int):
                replaced_block_ordinals.append(item["block_ordinal"])
        return {
            "triggered": True,
            "compound_node_type": getattr(cascade_cp.compound_event, "node_type", None),
            "failed_simple_count_before_cascade": len(
                getattr(cascade_cp, "failed_simple_blocks", [])
            ),
            "replaced_block_ordinals": replaced_block_ordinals,
            "restored_total_blocks": int(restored_stats.get("total_blocks", 0)),
            "restored_embedded_blocks": int(restored_stats.get("embedded_blocks", 0)),
            "restored_failed_blocks": int(restored_stats.get("failed_blocks", 0)),
        }

    @staticmethod
    def _fallback_replacement_scope(cascade_cp) -> dict[str, object]:
        replaced_block_ordinals: list[int] = []
        for item in getattr(cascade_cp, "failed_simple_blocks", []):
            if isinstance(item, dict) and isinstance(item.get("block_ordinal"), int):
                replaced_block_ordinals.append(item["block_ordinal"])
        return {
            "compound_node_type": getattr(cascade_cp.compound_event, "node_type", None),
            "compound_parent_node_type": (
                getattr(cascade_cp.compound_event, "parent_node_type", None) or "module"
            ),
            "replaced_block_ordinals": replaced_block_ordinals,
        }

    @staticmethod
    def _snapshot_runtime_stats(stats: EmbedStats) -> dict[str, object]:
        return {
            "total_blocks": stats.total_blocks,
            "embedded_blocks": stats.embedded_blocks,
            "failed_blocks": stats.failed_blocks,
            "fallback_blocks": stats.fallback_blocks,
            "retry_diagnostics": list(stats.retry_diagnostics),
        }

    @staticmethod
    def _restore_runtime_stats(
        stats: EmbedStats,
        snapshot: object | None,
    ) -> None:
        if not isinstance(snapshot, dict):
            return
        stats.total_blocks = int(snapshot.get("total_blocks", stats.total_blocks))
        stats.embedded_blocks = int(snapshot.get("embedded_blocks", stats.embedded_blocks))
        stats.failed_blocks = int(snapshot.get("failed_blocks", stats.failed_blocks))
        stats.fallback_blocks = int(snapshot.get("fallback_blocks", stats.fallback_blocks))
        retry_diagnostics = snapshot.get("retry_diagnostics")
        if isinstance(retry_diagnostics, list):
            stats.retry_diagnostics = list(retry_diagnostics)

    def _finalize_stats(self, final_code: str) -> tuple[int, int]:
        """Recompute final simple-block totals from the emitted code."""
        all_blocks = extract_statement_blocks(final_code)
        simple_blocks = [block for block in all_blocks if block.block_type == "simple"]
        if not simple_blocks:
            return 0, 0

        block_by_id = {block.block_id: block for block in all_blocks}
        embedded_blocks = 0
        for block in simple_blocks:
            parent_node_type = (
                block_by_id[block.parent_id].node_type
                if block.parent_id is not None
                else "module"
            )
            event = type(
                "_FinalBlockEvent",
                (),
                {
                    "block_text": block.source,
                    "block_type": "simple",
                    "node_type": block.node_type,
                    "parent_node_type": parent_node_type,
                    "token_start_idx": 0,
                    "token_count": 0,
                },
            )()
            if self._verify_block(event).passed:
                embedded_blocks += 1

        return len(simple_blocks), embedded_blocks
