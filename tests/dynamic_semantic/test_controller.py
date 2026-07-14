from __future__ import annotations

from types import SimpleNamespace

from wfcllm.dynamic_semantic.channel import UnitEvidence
from wfcllm.dynamic_semantic.config import ContextConfig
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.controller import DynamicSemanticController
from wfcllm.generation.selection_v2 import RetryAttempt


class _Scheduler:
    def __init__(self) -> None:
        self.completed = []
        self.flushed = 0
        self.evidence = {
            0: (UnitEvidence.synthetic("a", matches=3),),
            1: (UnitEvidence.synthetic("b", matches=7),),
        }

    def attempt_completed(self, index: int) -> None:
        self.completed.append(index)

    def flush_final(self) -> None:
        self.flushed += 1

    def evidence_for_attempt(self, index: int):
        return self.evidence[index]

    def submit(self, **kwargs) -> None:
        pass

    def update_live_contexts(self, *args) -> None:
        pass


def _attempt(index: int) -> RetryAttempt:
    return RetryAttempt(
        attempt_index=index,
        seed=20260713 + index * 101,
        result=SimpleNamespace(
            final_code=f"def f(x):\n    return x + {index}\n",
            fallback_count=0,
            accepted_hit_count=0,
            closed_without_hit_count=0,
            candidate_count=1,
        ),
    )


def test_controller_selects_and_emits_private_audit_rows_without_secret_bits() -> None:
    scheduler = _Scheduler()
    controller = DynamicSemanticController(
        extractor=DynamicContextExtractor(
            ContextConfig(), token_counter=lambda text: len(text.split())
        ),
        scheduler=scheduler,
        minimum_independent_units=1,
    )
    controller.observer_for_attempt(0)
    controller.observer_for_attempt(1)
    controller.attempt_completed(0)
    controller.attempt_completed(1)

    selection = controller.select(
        sample_id="HumanEval/0",
        prompt='def f(x):\n    """Return a value."""\n',
        attempts=(_attempt(0), _attempt(1)),
    )

    assert selection.attempt_index == 1
    assert scheduler.completed == [0, 1]
    assert scheduler.flushed == 1
    assert len(selection.ledger_rows) == 2
    assert sum(row["selected"] for row in selection.ledger_rows) == 1
    assert all(row["audit_only"] for row in selection.ledger_rows)
    forbidden = {"target_bits", "signature_bits", "quantized_embedding", "key_hash"}
    assert all(not (forbidden & set(row)) for row in selection.ledger_rows)
    assert controller.selected_evidence == scheduler.evidence[1]
