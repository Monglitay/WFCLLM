"""Adaptive gamma schedule based on entropy profile quantiles."""

from __future__ import annotations

from dataclasses import dataclass

from wfcllm.watermark.entropy_profile import EntropyProfile


@dataclass(frozen=True)
class GammaResolution:
    """Resolved gamma parameters for a given block entropy and LSH dimension."""

    gamma_target: float
    k: int
    gamma_effective: float


def quantize_gamma(gamma_target: float, lsh_d: int) -> GammaResolution:
    """Quantize a target gamma into the effective LSH region count."""
    if lsh_d < 1:
        raise ValueError("lsh_d must be >= 1")

    clipped_gamma = min(max(gamma_target, 0.0), 1.0)
    universe_size = 2 ** lsh_d
    k_unclipped = round(clipped_gamma * universe_size)
    k = min(max(k_unclipped, 1), universe_size - 1)
    gamma_effective = k / universe_size

    return GammaResolution(
        gamma_target=round(clipped_gamma, 12),
        k=k,
        gamma_effective=gamma_effective,
    )


@dataclass(frozen=True)
class PiecewiseQuantileSchedule:
    """Piecewise-linear gamma schedule from profile quantile anchors."""

    profile: EntropyProfile
    anchor_quantiles: tuple[str, ...] = ("p10", "p50", "p75", "p90", "p95")
    anchor_gammas: tuple[float, ...] = (0.95, 0.75, 0.55, 0.35, 0.25)

    def __post_init__(self) -> None:
        if len(self.anchor_quantiles) != len(self.anchor_gammas):
            raise ValueError("anchor_quantiles and anchor_gammas must have equal length")
        if len(self.anchor_quantiles) < 2:
            raise ValueError("at least two anchors are required")

    def resolve(self, entropy_units: int, lsh_d: int) -> GammaResolution:
        """Resolve target and quantized gamma for a block."""
        gamma_target = self._interpolate_gamma(entropy_units)
        return quantize_gamma(gamma_target, lsh_d)

    def _interpolate_gamma(self, entropy_units: int) -> float:
        entropy_anchors = [self.profile.quantile_units(q) for q in self.anchor_quantiles]

        if entropy_units <= entropy_anchors[0]:
            return self.anchor_gammas[0]
        if entropy_units >= entropy_anchors[-1]:
            return self.anchor_gammas[-1]

        for index in range(len(entropy_anchors) - 1):
            lower_entropy = entropy_anchors[index]
            upper_entropy = entropy_anchors[index + 1]
            if lower_entropy <= entropy_units <= upper_entropy:
                lower_gamma = self.anchor_gammas[index]
                upper_gamma = self.anchor_gammas[index + 1]
                if upper_entropy == lower_entropy:
                    return upper_gamma
                t = (entropy_units - lower_entropy) / (upper_entropy - lower_entropy)
                return lower_gamma + t * (upper_gamma - lower_gamma)

        return self.anchor_gammas[-1]
