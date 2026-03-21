"""Extract-side helpers for block contract rebuilding and comparison."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass, field

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.watermark.config import AdaptiveGammaConfig
from wfcllm.watermark.entropy_profile import EntropyProfile
from wfcllm.watermark.gamma_schedule import PiecewiseQuantileSchedule

_STRUCTURAL_FIELDS = (
    "ordinal",
    "block_id",
    "node_type",
    "parent_node_type",
    "block_text_hash",
    "start_line",
    "end_line",
)
_NUMERIC_FIELDS = (
    "entropy_units",
    "gamma_target",
    "k",
    "gamma_effective",
)


@dataclass(frozen=True)
class AlignmentReport:
    """Comparison result between embedded and rebuilt block contracts."""

    embedded_block_count: int
    rebuilt_block_count: int
    block_count_mismatch: bool = False
    structure_mismatch: bool = False
    numeric_mismatch: bool = False
    structure_mismatches: list[dict[str, object]] = field(default_factory=list)
    numeric_mismatches: list[dict[str, object]] = field(default_factory=list)

    @property
    def contract_valid(self) -> bool:
        return not self.structure_mismatch

    @property
    def is_aligned(self) -> bool:
        return not self.structure_mismatch and not self.numeric_mismatch

    @property
    def status(self) -> str:
        if self.structure_mismatch and self.numeric_mismatch:
            return "structure_and_numeric_mismatch"
        if self.structure_mismatch:
            return "structure_mismatch"
        if self.numeric_mismatch:
            return "numeric_mismatch"
        return "aligned"

    def to_dict(self) -> dict[str, object]:
        return {
            "embedded_block_count": self.embedded_block_count,
            "rebuilt_block_count": self.rebuilt_block_count,
            "block_count_mismatch": self.block_count_mismatch,
            "structure_mismatch": self.structure_mismatch,
            "numeric_mismatch": self.numeric_mismatch,
            "contract_valid": self.contract_valid,
            "is_aligned": self.is_aligned,
            "status": self.status,
            "structure_mismatches": list(self.structure_mismatches),
            "numeric_mismatches": list(self.numeric_mismatches),
        }


def rebuild_block_contracts(
    code: str,
    *,
    watermark_metadata: dict | None = None,
    adaptive_gamma_config: AdaptiveGammaConfig | None = None,
    default_lsh_d: int | None = None,
) -> list[dict[str, object]]:
    """Rebuild canonical simple-block contracts from final code."""
    gamma_resolver = _resolve_gamma_resolver(
        watermark_metadata=watermark_metadata,
        adaptive_gamma_config=adaptive_gamma_config,
        default_lsh_d=default_lsh_d,
    )
    return [
        asdict(contract)
        for contract in build_block_contracts(code, gamma_resolver=gamma_resolver)
    ]


def compare_block_contracts(
    embedded_contracts: list[object],
    rebuilt_contracts: list[object],
) -> AlignmentReport:
    """Compare embedded block metadata against rebuilt final-code contracts."""
    embedded = [_normalize_contract(contract) for contract in embedded_contracts]
    rebuilt = [_normalize_contract(contract) for contract in rebuilt_contracts]

    structure_mismatches: list[dict[str, object]] = []
    numeric_mismatches: list[dict[str, object]] = []
    block_count_mismatch = len(embedded) != len(rebuilt)

    if block_count_mismatch:
        structure_mismatches.append(
            {
                "field": "block_count",
                "embedded": len(embedded),
                "rebuilt": len(rebuilt),
            }
        )

    for embedded_contract, rebuilt_contract in zip(embedded, rebuilt):
        ordinal = rebuilt_contract.get("ordinal", embedded_contract.get("ordinal"))
        for field_name in _STRUCTURAL_FIELDS:
            embedded_value = embedded_contract.get(field_name)
            rebuilt_value = rebuilt_contract.get(field_name)
            if embedded_value != rebuilt_value:
                structure_mismatches.append(
                    {
                        "ordinal": ordinal,
                        "field": field_name,
                        "embedded": embedded_value,
                        "rebuilt": rebuilt_value,
                    }
                )

        for field_name in _NUMERIC_FIELDS:
            embedded_value = embedded_contract.get(field_name)
            rebuilt_value = rebuilt_contract.get(field_name)
            if embedded_value != rebuilt_value:
                numeric_mismatches.append(
                    {
                        "ordinal": ordinal,
                        "field": field_name,
                        "embedded": embedded_value,
                        "rebuilt": rebuilt_value,
                    }
                )

    return AlignmentReport(
        embedded_block_count=len(embedded),
        rebuilt_block_count=len(rebuilt),
        block_count_mismatch=block_count_mismatch,
        structure_mismatch=bool(structure_mismatches),
        numeric_mismatch=bool(numeric_mismatches),
        structure_mismatches=structure_mismatches,
        numeric_mismatches=numeric_mismatches,
    )


def _normalize_contract(contract: object) -> dict[str, object]:
    if isinstance(contract, dict):
        normalized = dict(contract)
    else:
        normalized = {
            field_name: getattr(contract, field_name)
            for field_name in (
                *_STRUCTURAL_FIELDS,
                *_NUMERIC_FIELDS,
            )
            if hasattr(contract, field_name)
        }

    normalized.setdefault("gamma_target", 0.0)
    normalized.setdefault("k", 0)
    normalized.setdefault("gamma_effective", 0.0)
    return normalized


def _resolve_gamma_resolver(
    *,
    watermark_metadata: dict | None,
    adaptive_gamma_config: AdaptiveGammaConfig | None,
    default_lsh_d: int | None,
):
    if isinstance(watermark_metadata, dict):
        adaptive_mode = watermark_metadata.get("adaptive_mode")
        if adaptive_mode in (None, "fixed"):
            return None

        lsh_d = _resolve_lsh_d(watermark_metadata, default_lsh_d)
        if lsh_d is None:
            return None

        schedule = _build_embedded_schedule(watermark_metadata)
        if schedule is None:
            schedule = _build_config_schedule(adaptive_gamma_config)
        if schedule is None:
            return None

        return lambda entropy_units: schedule.resolve(entropy_units, lsh_d)

    if default_lsh_d is None:
        return None
    schedule = _build_config_schedule(adaptive_gamma_config)
    if schedule is None:
        return None
    return lambda entropy_units: schedule.resolve(entropy_units, default_lsh_d)


def _resolve_lsh_d(
    watermark_metadata: dict,
    default_lsh_d: int | None,
) -> int | None:
    watermark_params = watermark_metadata.get("watermark_params")
    if isinstance(watermark_params, dict) and watermark_params.get("lsh_d") is not None:
        return int(watermark_params["lsh_d"])
    if default_lsh_d is None:
        return None
    return int(default_lsh_d)


def _build_embedded_schedule(
    watermark_metadata: dict,
) -> PiecewiseQuantileSchedule | None:
    watermark_params = watermark_metadata.get("watermark_params")
    if not isinstance(watermark_params, dict):
        return None
    adaptive_gamma = watermark_params.get("adaptive_gamma")
    if not isinstance(adaptive_gamma, dict):
        return None
    if adaptive_gamma.get("strategy", "piecewise_quantile") != "piecewise_quantile":
        return None

    profile_payload = adaptive_gamma.get("profile")
    anchors = adaptive_gamma.get("anchors")
    if not isinstance(profile_payload, dict) or not isinstance(anchors, dict):
        return None

    profile = EntropyProfile(
        language=profile_payload.get("language"),
        model_family=profile_payload.get("model_family"),
        quantiles_units_map=profile_payload.get("quantiles_units"),
        strategy=profile_payload.get("strategy", "piecewise_quantile"),
    )
    anchor_quantiles = tuple(anchors.keys())
    anchor_gammas = tuple(float(anchors[key]) for key in anchor_quantiles)
    return PiecewiseQuantileSchedule(
        profile=profile,
        anchor_quantiles=anchor_quantiles,
        anchor_gammas=anchor_gammas,
    )


def _build_config_schedule(
    adaptive_gamma_config: AdaptiveGammaConfig | None,
) -> PiecewiseQuantileSchedule | None:
    if adaptive_gamma_config is None:
        return None
    if not adaptive_gamma_config.enabled:
        return None
    if adaptive_gamma_config.profile_path is None:
        return None
    if adaptive_gamma_config.strategy != "piecewise_quantile":
        return None

    profile = EntropyProfile.load(adaptive_gamma_config.profile_path)
    anchor_quantiles = tuple(adaptive_gamma_config.anchors.keys())
    anchor_gammas = tuple(
        float(adaptive_gamma_config.anchors[key])
        for key in anchor_quantiles
    )
    return PiecewiseQuantileSchedule(
        profile=profile,
        anchor_quantiles=anchor_quantiles,
        anchor_gammas=anchor_gammas,
    )
