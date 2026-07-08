from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Protocol

from wfcllm.generation.boundary import Candidate


@dataclass(frozen=True)
class RuleRequest:
    sample_id: str
    position_id: str
    candidates: tuple[Candidate, ...]
    seed: int
    final_flush: bool


@dataclass(frozen=True)
class RuleDecision:
    hit: bool
    reason: str
    rule_name: str


@dataclass(frozen=True)
class _GammaResolution:
    gamma_target: float
    k: int
    gamma_effective: float


class EmbeddingRule(Protocol):
    def evaluate(self, request: RuleRequest) -> RuleDecision:
        ...


class SemanticLshVerifyResult(Protocol):
    passed: bool
    lsh_signature: tuple[int, ...] | None
    min_margin: float
    in_valid_set: bool | None


class SemanticLshVerifier(Protocol):
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> SemanticLshVerifyResult:
        ...


class SemanticLshKeying(Protocol):
    def derive(
        self,
        parent_node_type: str,
        *,
        k: int,
        ordinal: int | None,
    ) -> frozenset[tuple[int, ...]]:
        ...


class HashEmbeddingRule:
    """Deterministic local predicate for SAWR smoke embedding decisions."""

    rule_name = "hash"

    def __init__(self, target_accept_rate: float = 0.5) -> None:
        if not 0 <= target_accept_rate <= 1:
            raise ValueError("target_accept_rate must be in [0, 1]")
        self._target_accept_rate = target_accept_rate

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        fraction = self.score_fraction(request)
        hit = fraction < self._target_accept_rate
        return RuleDecision(
            hit=hit,
            reason=(
                f"hash_fraction={fraction:.12f};"
                f"target_accept_rate={self._target_accept_rate:.12f}"
            ),
            rule_name=self.rule_name,
        )

    def score_fraction(self, request: RuleRequest) -> float:
        if not request.candidates:
            raise ValueError("candidates must not be empty")
        digest = hashlib.sha256(self._payload(request)).digest()
        value = int.from_bytes(digest[:8], "big")
        return value / float(1 << 64)

    def _payload(self, request: RuleRequest) -> bytes:
        payload = {
            "seed": request.seed,
            "sample_id": request.sample_id,
            "position_id": request.position_id,
            "candidate_group": [
                {
                    "text": _normalize_candidate_text(candidate.text),
                    "candidate_type": candidate.candidate_type,
                    "node_type": candidate.node_type,
                    "position_id": candidate.position_id,
                }
                for candidate in request.candidates
            ],
            "final_flush": request.final_flush,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")


def _normalize_candidate_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())


def _quantize_gamma(gamma_target: float, lsh_d: int) -> _GammaResolution:
    if lsh_d < 1:
        raise ValueError("lsh_d must be >= 1")
    clipped_gamma = min(max(gamma_target, 0.0), 1.0)
    universe_size = 2 ** lsh_d
    k_unclipped = round(clipped_gamma * universe_size)
    k = min(max(k_unclipped, 1), universe_size - 1)
    return _GammaResolution(
        gamma_target=round(clipped_gamma, 12),
        k=k,
        gamma_effective=k / universe_size,
    )


class SemanticLshEmbeddingRule:
    """CodeT5/LSH semantic-space predicate for SAWR embedding decisions."""

    rule_name = "semantic_lsh"

    def __init__(
        self,
        *,
        verifier: SemanticLshVerifier,
        keying: SemanticLshKeying,
        lsh_d: int,
        lsh_gamma: float,
        margin: float = 0.0,
        use_ordinal_keying: bool = False,
    ) -> None:
        if lsh_d < 1:
            raise ValueError("lsh_d must be >= 1")
        if margin < 0:
            raise ValueError("margin must be non-negative")
        self._verifier = verifier
        self._keying = keying
        self._lsh_d = lsh_d
        self._lsh_gamma = lsh_gamma
        self._margin = margin
        self._use_ordinal_keying = use_ordinal_keying

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        if not request.candidates:
            raise ValueError("candidates must not be empty")

        gamma_resolution = _quantize_gamma(self._lsh_gamma, self._lsh_d)
        first_candidate = request.candidates[0]
        parent_node_type = first_candidate.parent_node_type or "module"
        key_ordinal = (
            first_candidate.ordinal
            if self._use_ordinal_keying
            else None
        )
        valid_set = self._keying.derive(
            parent_node_type,
            k=gamma_resolution.k,
            ordinal=key_ordinal,
        )
        code_text = "\n".join(
            _normalize_candidate_text(candidate.text)
            for candidate in request.candidates
        )
        result = self._verifier.verify(code_text, valid_set, self._margin)
        return RuleDecision(
            hit=bool(result.passed),
            reason=(
                f"lsh_signature={result.lsh_signature};"
                f"in_valid_set={result.in_valid_set};"
                f"min_margin={result.min_margin:.12f};"
                f"margin={self._margin:.12f};"
                f"parent_node_type={parent_node_type};"
                f"ordinal={key_ordinal};"
                f"k={gamma_resolution.k};"
                f"gamma_target={gamma_resolution.gamma_target:.12f};"
                f"gamma_effective={gamma_resolution.gamma_effective:.12f}"
            ),
            rule_name=self.rule_name,
        )
