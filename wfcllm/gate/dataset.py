"""Immutable gate examples, safe token collation, and group-aware batches."""

from __future__ import annotations

import random
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import InitVar, asdict, dataclass
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from wfcllm.gate.input import (
    GateInput,
    serialize_gate_input,
)
from wfcllm.method.contracts import reject_quality_proxy_fields

_ALLOWED_CONTEXT_LENGTHS = frozenset({1, 2, 3})
_ALLOWED_BUDGETS = frozenset({1, 3, 6})
_FORMAL_MAX_TOKENS = 512


@dataclass(frozen=True)
class GateExample:
    """One keyless gate-supervision variant.

    ``source_family`` and ``generation_model_id`` are analysis-only metadata.
    The collator sends only ``serialized_gate_input`` to the tokenizer.  A
    rewrite budget is likewise retained for stratified analysis, never as a
    model feature.
    """

    group_id: str
    window_start_unit_id: str
    context_length: int
    budget: int
    serialized_gate_input: str
    close_target: bool
    suitable_target: bool
    dangerous_negative: bool
    variant_id: str = ""
    source_family: str | None = None
    generation_model_id: str | None = None
    gate_input: InitVar[GateInput | None] = None

    def __post_init__(self, gate_input: GateInput | None) -> None:
        _require_nonempty_string("group_id", self.group_id)
        _require_nonempty_string("window_start_unit_id", self.window_start_unit_id)
        if (
            type(self.context_length) is not int
            or self.context_length not in _ALLOWED_CONTEXT_LENGTHS
        ):
            raise ValueError("context_length must be an integer in {1, 2, 3}")
        if type(self.budget) is not int or self.budget not in _ALLOWED_BUDGETS:
            raise ValueError("budget must be an integer in {1, 3, 6}")
        _require_nonempty_string("serialized_gate_input", self.serialized_gate_input)
        for name in ("close_target", "suitable_target", "dangerous_negative"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        for name in ("source_family", "generation_model_id"):
            value = getattr(self, name)
            if value is not None:
                _require_nonempty_string(name, value)

        variant_id = self.variant_id
        if not variant_id:
            variant_id = (
                f"{self.group_id}:{self.window_start_unit_id}:"
                f"context-{self.context_length}:budget-{self.budget}"
            )
            object.__setattr__(self, "variant_id", variant_id)
        _require_nonempty_string("variant_id", variant_id)

        if not isinstance(gate_input, GateInput):
            raise ValueError(
                "serialized_gate_input requires a structured GateInput attestation"
            )
        if serialize_gate_input(gate_input) != self.serialized_gate_input:
            raise ValueError(
                "serialized_gate_input must match the structured GateInput attestation"
            )

        # Audit the structured record, not the code-bearing serialized text.
        # The latter can legitimately contain arbitrary user source tokens.
        reject_quality_proxy_fields(self.to_dict())

    @classmethod
    def from_gate_input(
        cls,
        *,
        group_id: str,
        window_start_unit_id: str,
        context_length: int,
        budget: int,
        gate_input: GateInput,
        close_target: bool,
        suitable_target: bool,
        dangerous_negative: bool,
        variant_id: str = "",
        source_family: str | None = None,
        generation_model_id: str | None = None,
    ) -> GateExample:
        """Construct an example whose serialization is formally attested."""

        if not isinstance(gate_input, GateInput):
            raise ValueError("gate_input must be a GateInput")
        return cls(
            group_id=group_id,
            window_start_unit_id=window_start_unit_id,
            context_length=context_length,
            budget=budget,
            serialized_gate_input=serialize_gate_input(gate_input),
            close_target=close_target,
            suitable_target=suitable_target,
            dangerous_negative=dangerous_negative,
            variant_id=variant_id,
            source_family=source_family,
            generation_model_id=generation_model_id,
            gate_input=gate_input,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a strict-JSON-compatible, keyless snapshot."""

        row = asdict(self)
        reject_quality_proxy_fields(row)
        return row


class GateDataset(Dataset[GateExample]):
    """Tuple-backed immutable snapshot of gate examples."""

    def __init__(self, examples: Sequence[GateExample]) -> None:
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
            raise ValueError("examples must be a sequence of GateExample instances")
        snapshot = tuple(examples)
        if any(not isinstance(item, GateExample) for item in snapshot):
            raise ValueError("examples must contain only GateExample instances")
        _reject_duplicate_variants(snapshot)
        self._examples = snapshot

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, index: int) -> GateExample:
        return self._examples[index]


class GateCollator:
    """Tokenize complete inputs without truncating overlength windows.

    Overlength rows receive an all-masked placeholder.  Training code can
    still learn the forced-close target, but must honor ``suitable_loss_mask``
    and must not interpret placeholder logits as ordinary suitable examples.
    """

    def __init__(self, tokenizer: Any, *, max_tokens: int = _FORMAL_MAX_TOKENS) -> None:
        if tokenizer is None or not callable(tokenizer):
            raise ValueError("tokenizer must be callable")
        if (
            type(max_tokens) is not int
            or max_tokens <= 0
            or max_tokens > _FORMAL_MAX_TOKENS
        ):
            raise ValueError("max_tokens must be an integer from 1 through 512")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens

    def __call__(self, examples: Sequence[GateExample]) -> dict[str, Any]:
        if isinstance(examples, (str, bytes)) or not isinstance(examples, Sequence):
            raise ValueError("examples must be a sequence")
        snapshot = tuple(examples)
        if not snapshot:
            raise ValueError("examples must not be empty")
        if any(not isinstance(item, GateExample) for item in snapshot):
            raise ValueError("examples must contain only GateExample instances")

        tokenized = [
            self._tokenize_complete(item.serialized_gate_input) for item in snapshot
        ]
        overflow = [len(ids) > self.max_tokens for ids, _ in tokenized]
        retained_lengths = [
            len(ids) for (ids, _), is_overflow in zip(tokenized, overflow, strict=True)
            if not is_overflow
        ]
        padded_length = max(retained_lengths, default=1)
        pad_token_id = getattr(self.tokenizer, "pad_token_id", 0)
        if type(pad_token_id) is not int or pad_token_id < 0:
            pad_token_id = 0

        input_rows: list[list[int]] = []
        mask_rows: list[list[int]] = []
        for (ids, mask), is_overflow in zip(tokenized, overflow, strict=True):
            if is_overflow:
                input_rows.append([pad_token_id] * padded_length)
                mask_rows.append([0] * padded_length)
                continue
            padding = padded_length - len(ids)
            input_rows.append(ids + [pad_token_id] * padding)
            mask_rows.append(mask + [0] * padding)

        close_targets = [
            True if flag else item.close_target
            for item, flag in zip(snapshot, overflow, strict=True)
        ]
        suitable_targets = [
            False if flag else item.suitable_target
            for item, flag in zip(snapshot, overflow, strict=True)
        ]
        return {
            "input_ids": torch.tensor(input_rows, dtype=torch.long),
            "attention_mask": torch.tensor(mask_rows, dtype=torch.long),
            "group_ids": [item.group_id for item in snapshot],
            "window_start_unit_ids": [item.window_start_unit_id for item in snapshot],
            "context_lengths": [item.context_length for item in snapshot],
            "budgets": [item.budget for item in snapshot],
            "variant_ids": [item.variant_id for item in snapshot],
            "dangerous_negative": torch.tensor(
                [item.dangerous_negative for item in snapshot], dtype=torch.bool
            ),
            "close_targets": torch.tensor(close_targets, dtype=torch.float32),
            "suitable_targets": torch.tensor(suitable_targets, dtype=torch.float32),
            "overflow": torch.tensor(overflow, dtype=torch.bool),
            "close_loss_mask": torch.ones(len(snapshot), dtype=torch.bool),
            "suitable_loss_mask": torch.tensor(
                [not flag for flag in overflow], dtype=torch.bool
            ),
            "overflow_skip_reasons": [
                "max_tokens_exceeded" if flag else None for flag in overflow
            ],
        }

    def _tokenize_complete(self, text: str) -> tuple[list[int], list[int]]:
        encoded = self.tokenizer(
            text,
            add_special_tokens=True,
            truncation=False,
            return_attention_mask=True,
        )
        if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
            raise ValueError("tokenizer output must map input_ids")
        ids = _one_token_row("input_ids", encoded["input_ids"])
        if not ids:
            raise ValueError("tokenizer input_ids must not be empty")
        if "attention_mask" in encoded:
            mask = _one_token_row("attention_mask", encoded["attention_mask"])
        else:
            mask = [1] * len(ids)
        if len(mask) != len(ids):
            raise ValueError("tokenizer attention_mask must match input_ids length")
        if any(value not in {0, 1} for value in mask):
            raise ValueError("tokenizer attention_mask must be binary")
        return ids, mask


class GroupConsistencyBatchSampler(Sampler[list[int]]):
    """Pack complete per-budget cohorts without splitting their contexts."""

    def __init__(
        self,
        examples: Sequence[GateExample] | GateDataset,
        *,
        batch_size: int,
        seed: int = 0,
        shuffle: bool = True,
    ) -> None:
        if not isinstance(examples, (Sequence, GateDataset)) or isinstance(
            examples, (str, bytes)
        ):
            raise ValueError("examples must be a sequence or GateDataset")
        snapshot = tuple(examples[index] for index in range(len(examples)))
        if not snapshot:
            raise ValueError("examples must not be empty")
        if any(not isinstance(item, GateExample) for item in snapshot):
            raise ValueError("examples must contain only GateExample instances")
        _reject_duplicate_variants(snapshot)
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if type(seed) is not int:
            raise ValueError("seed must be an integer")
        if type(shuffle) is not bool:
            raise ValueError("shuffle must be a bool")

        group_window_starts: dict[str, set[str]] = {}
        cohorts: dict[tuple[str, str, int], list[int]] = {}
        for index, item in enumerate(snapshot):
            group_window_starts.setdefault(item.group_id, set()).add(
                item.window_start_unit_id
            )
            cohort_id = (item.group_id, item.window_start_unit_id, item.budget)
            cohorts.setdefault(cohort_id, []).append(index)
        for group_id, window_starts in group_window_starts.items():
            if len(window_starts) != 1:
                raise ValueError(
                    f"group {group_id!r} must use exactly one window_start_unit_id"
                )
        for (group_id, window_start, budget), indices in cohorts.items():
            contexts = {snapshot[index].context_length for index in indices}
            if contexts != _ALLOWED_CONTEXT_LENGTHS or len(indices) != 3:
                raise ValueError(
                    "each budget cohort must contain contexts 1, 2, and 3 "
                    f"exactly once; got group={group_id!r}, "
                    f"window_start={window_start!r}, budget={budget}"
                )
            if len(indices) > batch_size:
                raise ValueError(
                    "batch_size cannot fit group cohort "
                    f"{(group_id, window_start, budget)!r}"
                )

        self._cohorts = tuple(
            (cohort_id, tuple(indices))
            for cohort_id, indices in cohorts.items()
        )
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[list[int]]:
        cohorts = list(self._cohorts)
        if self.shuffle:
            random.Random(self.seed).shuffle(cohorts)
        batch: list[int] = []
        for _, indices in cohorts:
            if batch and len(batch) + len(indices) > self.batch_size:
                yield batch
                batch = []
            batch.extend(indices)
        if batch:
            yield batch

    def __len__(self) -> int:
        count = 0
        used = 0
        cohorts = list(self._cohorts)
        if self.shuffle:
            random.Random(self.seed).shuffle(cohorts)
        for _, indices in cohorts:
            if used and used + len(indices) > self.batch_size:
                count += 1
                used = 0
            used += len(indices)
        return count + int(used > 0)


def _one_token_row(name: str, value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"tokenizer {name} must be a sequence")
    row: object = value
    if (
        row
        and isinstance(row[0], (Sequence, torch.Tensor))  # type: ignore[index]
        and not isinstance(row[0], (str, bytes))  # type: ignore[index]
    ):
        if len(row) != 1:  # type: ignore[arg-type]
            raise ValueError(f"tokenizer {name} must contain exactly one row")
        row = row[0]  # type: ignore[index]
        if isinstance(row, torch.Tensor):
            row = row.detach().cpu().tolist()
    if not isinstance(row, Sequence) or isinstance(row, (str, bytes)):
        raise ValueError(f"tokenizer {name} row must be a sequence")
    result = list(row)
    if any(type(item) is not int or item < 0 for item in result):
        raise ValueError(f"tokenizer {name} must contain non-negative integers")
    return result


def _reject_duplicate_variants(examples: Sequence[GateExample]) -> None:
    variant_ids = [item.variant_id for item in examples]
    if len(set(variant_ids)) != len(variant_ids):
        raise ValueError("duplicate variant_id is forbidden")
    identities = [
        (
            item.group_id,
            item.window_start_unit_id,
            item.context_length,
            item.budget,
        )
        for item in examples
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate gate variant is forbidden")


def _require_nonempty_string(name: str, value: object) -> None:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{name} must be a non-empty NUL-free string")
