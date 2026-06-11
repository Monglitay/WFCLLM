from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Candidate:
    text: str
    candidate_type: str
    node_type: str
    position_id: str
    token_start_idx: int
    token_count: int
