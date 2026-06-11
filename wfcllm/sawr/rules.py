from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

from wfcllm.sawr.boundary import Candidate


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


class EmbeddingRule(Protocol):
    def evaluate(self, request: RuleRequest) -> RuleDecision:
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
        normalized_group = "\n".join(
            _normalize_candidate_text(candidate.text)
            for candidate in request.candidates
        )
        text = "\x1f".join(
            (
                str(request.seed),
                request.sample_id,
                request.position_id,
                normalized_group,
                "final_flush=1" if request.final_flush else "final_flush=0",
            )
        )
        return text.encode("utf-8")


def _normalize_candidate_text(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.strip().splitlines())
