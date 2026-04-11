"""Tests for token-channel offline training corpus helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from wfcllm.watermark.token_channel.features import build_structure_masks
from wfcllm.watermark.token_channel.features import collect_excluded_token_spans
from wfcllm.watermark.token_channel.teacher import extract_teacher_rows
from wfcllm.watermark.token_channel.teacher import load_teacher_cache
from wfcllm.watermark.token_channel.teacher import save_teacher_cache
from wfcllm.watermark.token_channel.train import main
from wfcllm.watermark.token_channel.train_corpus import build_augmented_variants
from wfcllm.watermark.token_channel.train_corpus import build_training_rows
from wfcllm.watermark.token_channel.train_corpus import load_training_cache
from wfcllm.watermark.token_channel.train_corpus import save_training_cache


class CharacterTokenizer:
    def __init__(self) -> None:
        self._char_to_id = {"<pad>": 0}
        self._id_to_char = {0: ""}

    def register_text(self, text: str) -> None:
        for ch in text:
            self._ensure_char(ch)

    def _ensure_char(self, ch: str) -> int:
        token_id = self._char_to_id.get(ch)
        if token_id is not None:
            return token_id
        token_id = len(self._char_to_id)
        self._char_to_id[ch] = token_id
        self._id_to_char[token_id] = ch
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [self._ensure_char(ch) for ch in text]

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            return self._id_to_char[ids]
        return "".join(self._id_to_char[token_id] for token_id in ids)

    def __len__(self) -> int:
        return len(self._char_to_id)


class FakeTeacherModel:
    def __init__(self, tokenizer: CharacterTokenizer) -> None:
        self._tokenizer = tokenizer

    def score_next(self, prefix_ids: tuple[int, ...]) -> torch.Tensor:
        vocab_size = len(self._tokenizer)
        logits = torch.zeros(vocab_size, dtype=torch.float32)
        if prefix_ids and self._tokenizer.decode(prefix_ids[-1]) == "b":
            logits[0] = 6.0
        return logits


class RecordingTeacherModel:
    def __init__(self) -> None:
        self.prefixes: list[tuple[int, ...]] = []

    def score_next(self, prefix_ids: tuple[int, ...]) -> torch.Tensor:
        self.prefixes.append(prefix_ids)
        return torch.tensor([1.0, 0.0], dtype=torch.float32)


class RecordingForwardTeacherModel:
    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def __call__(self, input_ids: torch.Tensor):
        self.calls.append(input_ids.clone())
        return {"logits": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32)}


class PretokenizedTokenizer:
    def __init__(self) -> None:
        self._tokens = {
            "ab": [10, 11],
        }
        self._token_text = {
            10: "a",
            11: "b",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(self._tokens[text])

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(ids, int):
            ids = [ids]
        return "".join(f"<{self._token_text[token_id]}>" for token_id in ids)

    def convert_ids_to_tokens(self, ids):
        if isinstance(ids, int):
            return self._token_text[ids]
        return [self._token_text[token_id] for token_id in ids]

    def convert_tokens_to_string(self, tokens) -> str:
        if isinstance(tokens, str):
            tokens = [tokens]
        return "".join(tokens)

    def __len__(self) -> int:
        return len(self._token_text)


class FakeTransformEngine:
    def __init__(self, transformed_source: str) -> None:
        self._transformed_source = transformed_source

    def generate_variants(self, source: str) -> list[dict[str, object]]:
        return [
            {
                "variant_id": 0,
                "rules_applied": ["rename"],
                "transformed_source": self._transformed_source,
                "sample_type": "positive",
            }
        ]


def test_structure_mask_excludes_imports_signatures_and_decorators() -> None:
    source = (
        "import os\n"
        "from math import sqrt\n"
        "@decorator\n"
        "class Example(Base):\n"
        "    @staticmethod\n"
        "    def run(value: int) -> int:\n"
        "        return value + 1\n"
    )

    spans = collect_excluded_token_spans(source)
    masks = build_structure_masks(source)

    assert {span.label for span in spans} >= {
        "import_statement",
        "import_from_statement",
        "decorator",
        "function_signature",
        "class_header",
    }
    _assert_mask_for_snippet(source, masks, "import os", expected=False)
    _assert_mask_for_snippet(source, masks, "from math import sqrt", expected=False)
    _assert_mask_for_snippet(source, masks, "@decorator", expected=False)
    _assert_mask_for_snippet(source, masks, "class Example(Base):", expected=False)
    _assert_mask_for_snippet(source, masks, "def run(value: int) -> int:", expected=False)
    _assert_mask_for_snippet(source, masks, "return value + 1", expected=True)


def test_build_augmented_variants_includes_base_and_positive_transforms() -> None:
    source = "value = a\n"
    variants = build_augmented_variants(
        source,
        transform_engine=FakeTransformEngine("value = b\n"),
    )

    assert variants == ["value = a\n", "value = b\n"]


def test_build_training_rows_collects_prefix_entropy_and_next_token(tmp_path: Path) -> None:
    del tmp_path
    tokenizer = CharacterTokenizer()
    tokenizer.register_text("value = a\n")
    tokenizer.register_text("value = b\n")
    rows = build_training_rows(
        samples=[{"source_code": "value = a\n"}],
        tokenizer=tokenizer,
        teacher_model=FakeTeacherModel(tokenizer),
        context_width=4,
        transform_engine=FakeTransformEngine("value = b\n"),
        entropy_threshold=1.0,
        diversity_threshold=2,
    )

    target_row = next(
        row
        for row in rows
        if tokenizer.decode(row["prefix_tokens"]) == "e = " and tokenizer.decode(row["next_token"]) == "a"
    )
    assert target_row["prefix_tokens"]
    assert target_row["entropy"] > 1.0
    assert target_row["continuation_diversity"] == 2
    assert target_row["node_type"]
    assert target_row["parent_node_type"]
    assert target_row["block_relative_offset"] == 0
    assert target_row["structure_mask"] is True
    assert target_row["switch_target"] == 1


def test_training_and_teacher_cache_round_trip(tmp_path: Path) -> None:
    training_rows = [{"prefix_tokens": [1, 2], "next_token": 3, "switch_target": 1}]
    teacher_rows = [{"prefix_tokens": [1], "entropy": 1.2, "teacher_logits": [0.1, 0.9]}]
    training_path = tmp_path / "corpus.pkl"
    teacher_path = tmp_path / "teacher.pkl"

    save_training_cache(training_path, training_rows)
    save_teacher_cache(teacher_path, teacher_rows)

    assert load_training_cache(training_path) == training_rows
    assert load_teacher_cache(teacher_path) == teacher_rows


def test_extract_teacher_rows_collects_entropy_and_logits() -> None:
    tokenizer = CharacterTokenizer()
    tokenizer.register_text("abc")
    rows = extract_teacher_rows(
        tokenizer=tokenizer,
        model=FakeTeacherModel(tokenizer),
        text="abc",
        context_width=2,
    )

    assert len(rows) == 3
    assert rows[0]["prefix_tokens"] == []
    assert rows[1]["next_token"] == tokenizer.encode("b", add_special_tokens=False)[0]
    assert len(rows[2]["teacher_logits"]) == len(tokenizer)
    assert rows[2]["entropy"] >= 0.0


def test_extract_teacher_rows_preserves_explicit_empty_prefix() -> None:
    model = RecordingTeacherModel()
    tokenizer = CharacterTokenizer()
    tokenizer.register_text("ab")

    rows = extract_teacher_rows(
        tokenizer=tokenizer,
        model=model,
        text="ab",
        context_width=2,
    )

    assert rows[0]["prefix_tokens"] == []
    assert model.prefixes[0] == ()


def test_extract_teacher_rows_does_not_fabricate_prefix_token_for_forward_models() -> None:
    model = RecordingForwardTeacherModel()
    tokenizer = CharacterTokenizer()
    tokenizer.register_text("ab")

    extract_teacher_rows(
        tokenizer=tokenizer,
        model=model,
        text="ab",
        context_width=2,
    )

    assert model.calls[0].shape == (1, 0)


def test_extract_teacher_rows_aligns_with_tokenizer_token_strings() -> None:
    tokenizer = PretokenizedTokenizer()

    rows = extract_teacher_rows(
        tokenizer=tokenizer,
        model=RecordingTeacherModel(),
        text="ab",
        context_width=2,
    )

    assert [(row["token_start"], row["token_end"]) for row in rows] == [(0, 1), (1, 2)]


def test_train_entry_loads_cached_rows_without_running_training(tmp_path: Path, capsys) -> None:
    cache_path = tmp_path / "corpus.pkl"
    save_training_cache(cache_path, [{"prefix_tokens": [1], "next_token": 2, "switch_target": 0}])

    exit_code = main(["--corpus-cache", str(cache_path)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "Loaded 1 training rows" in captured.out


def _assert_mask_for_snippet(
    source: str,
    masks: list[bool],
    snippet: str,
    *,
    expected: bool,
) -> None:
    start = source.index(snippet)
    end = start + len(snippet)
    relevant_positions = [
        index for index in range(start, end) if not source[index].isspace()
    ]
    assert relevant_positions
    assert all(masks[index] is expected for index in relevant_positions)
