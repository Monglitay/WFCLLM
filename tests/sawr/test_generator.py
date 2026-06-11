from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from wfcllm.sawr.config import SawrGenerationConfig
from wfcllm.sawr.generator import (
    SawrGenerateResult,
    SawrGenerator,
    SawrModelContext,
    build_chat_prompt,
    load_sawr_model,
    resolve_torch_dtype,
    strip_repeated_prompt_function,
)
from wfcllm.sawr.rules import RuleDecision, RuleRequest


class AlwaysHitRule:
    rule_name = "always-hit"

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        return RuleDecision(hit=True, reason="always", rule_name=self.rule_name)


class AlwaysMissRule:
    rule_name = "always-miss"

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        return RuleDecision(hit=False, reason="never", rule_name=self.rule_name)


class SequenceRule:
    rule_name = "sequence"

    def __init__(self, hits: list[bool]) -> None:
        self._hits = hits

    def evaluate(self, request: RuleRequest) -> RuleDecision:
        hit = self._hits.pop(0)
        return RuleDecision(hit=hit, reason=f"hit={hit}", rule_name=self.rule_name)


class FakeTokenizer:
    eos_token_id = 99

    def __init__(self, token_texts: list[str]) -> None:
        self._token_texts = token_texts
        self.apply_chat_template_calls = 0

    def encode(self, text, return_tensors=None):
        if return_tensors == "pt":
            return torch.tensor([[1, 2]], dtype=torch.long)
        return [1, 2]

    def decode(self, token_ids, skip_special_tokens=True):
        token_id = token_ids[0]
        return self._token_texts[token_id]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.apply_chat_template_calls += 1
        return "CHAT:" + messages[0]["content"]


class FakeModel:
    def __init__(self, token_ids: list[int]) -> None:
        self._token_ids = token_ids
        self._cursor = 0
        self.prefill_calls = 0
        self.step_calls = 0

    def parameters(self):
        return iter([torch.zeros(1)])

    def eval(self):
        return self

    def __call__(self, input_ids, past_key_values=None, use_cache=True):
        vocab_size = max(self._token_ids) + 1
        token_id = self._token_ids[min(self._cursor, len(self._token_ids) - 1)]
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, token_id] = 10.0
        if past_key_values is None:
            self.prefill_calls += 1
        else:
            self.step_calls += 1
        self._cursor += 1
        seq_len = (
            input_ids.shape[1]
            if past_key_values is None
            else past_key_values[0][0].shape[2] + 1
        )
        past = (
            (
                torch.zeros(1, 1, seq_len, 1),
                torch.zeros(1, 1, seq_len, 1),
            ),
        )
        return SimpleNamespace(logits=logits, past_key_values=past)


def _generation_config(tmp_path: Path, **overrides) -> SawrGenerationConfig:
    model_path = tmp_path / "model"
    model_path.mkdir()
    values = {
        "model_path": str(model_path),
        "max_new_tokens": 32,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": 0,
        "torch_dtype": "auto",
        "device": "cpu",
        "seed": 3,
        "load_in_4bit": False,
    }
    values.update(overrides)
    return SawrGenerationConfig(**values)


def test_resolve_torch_dtype_maps_supported_values() -> None:
    assert resolve_torch_dtype("auto") is None
    assert resolve_torch_dtype("fp32") is torch.float32
    assert resolve_torch_dtype("fp16") is torch.float16
    assert resolve_torch_dtype("bf16") is torch.bfloat16


def test_build_chat_prompt_uses_template_when_available() -> None:
    tokenizer = FakeTokenizer([])

    prompt = build_chat_prompt("def foo():\n", tokenizer)

    assert prompt.startswith("CHAT:")
    assert tokenizer.apply_chat_template_calls == 1
    assert "Complete the following Python function" in prompt


def test_build_chat_prompt_falls_back_to_raw_prompt() -> None:
    tokenizer = object()

    assert build_chat_prompt("def foo():\n", tokenizer) == "def foo():\n"


def test_strip_repeated_prompt_function_removes_duplicate_def_line() -> None:
    prompt = "def foo(x):\n"
    generated = "def foo(x):\n    return x\n"

    assert strip_repeated_prompt_function(prompt, generated) == "    return x\n"


def test_model_context_checkpoint_and_rollback_restore_text(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(["", "a", "b"])
    model = FakeModel([1, 2])
    context = SawrModelContext(
        model=model,
        tokenizer=tokenizer,
        config=_generation_config(tmp_path),
    )
    context.prefill("prompt")
    checkpoint = context.checkpoint()

    context.forward_and_sample()
    context.rollback(checkpoint)

    assert context.generated_ids == []
    assert context.generated_text == ""


def test_generator_accepts_simple_statement_group(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(["", "    ", "return", " ", "1", "\n", ""])
    model = FakeModel([1, 2, 3, 4, 5, 6])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=6, eos_token_id=6),
        rule=AlwaysHitRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=2,
        retry_budget=1,
    )

    assert isinstance(result, SawrGenerateResult)
    assert result.final_code == "def foo():\n    return 1\n"
    assert result.accepted_hit_count == 1
    assert result.closed_without_hit_count == 0
    assert result.fallback_count == 0
    assert result.candidate_count == 1
    assert [event.event for event in result.audit_events] == [
        "candidate_observed",
        "accepted_generation_time_group",
    ]


def test_generator_falls_back_when_rule_misses_and_no_retry_budget(
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer(["", "    ", "return", " ", "1", "\n", ""])
    model = FakeModel([1, 2, 3, 4, 5, 6])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=6, eos_token_id=6),
        rule=AlwaysMissRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=0,
    )

    assert result.final_code == "def foo():\n    return 1\n"
    assert result.fallback_count == 1
    assert [event.event for event in result.audit_events][-1] == (
        "fallback_committed_without_hit"
    )


def test_generator_closes_final_flush_miss_without_rollback(tmp_path: Path) -> None:
    tokenizer = FakeTokenizer(["", "    ", "return", " ", "1"])
    model = FakeModel([1, 2, 3, 4])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=4),
        rule=AlwaysMissRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=1,
    )

    assert result.final_code == "def foo():\n    return 1"
    assert result.closed_without_hit_count == 1
    assert [event.event for event in result.audit_events] == [
        "candidate_observed",
        "closed_without_hit",
    ]
    assert result.audit_events[-1].final_flush is True
    assert "rollback_requested" not in [
        event.event for event in result.audit_events
    ]


def test_generator_accepts_final_flush_hit_without_truncating_code(
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer(["", "    ", "return", " ", "1"])
    model = FakeModel([1, 2, 3, 4])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=4),
        rule=AlwaysHitRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=1,
    )

    assert result.final_code == "def foo():\n    return 1"
    assert result.accepted_hit_count == 1
    assert [event.event for event in result.audit_events] == [
        "candidate_observed",
        "accepted_generation_time_group",
    ]
    assert result.audit_events[-1].final_flush is True


def test_generator_rolls_back_to_group_start_and_skips_empty_decoded_chunks(
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer(
        ["", "    ", "return", " ", "1", "\n", "", "    ", "return", " ", "2", "\n"]
    )
    model = FakeModel([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=12, eos_token_id=11),
        rule=SequenceRule([False, True]),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="HumanEval/0",
        prompt="def foo():\n",
        dataset="humaneval",
        max_group_statements=1,
        retry_budget=1,
    )

    assert result.final_code == "def foo():\n    return 2\n"
    assert result.accepted_hit_count == 1
    assert result.candidate_count == 2
    assert [event.event for event in result.audit_events] == [
        "candidate_observed",
        "group_rule_miss",
        "rollback_requested",
        "candidate_observed",
        "accepted_generation_time_group",
    ]


def test_generator_records_no_controlled_body_when_no_function_body(
    tmp_path: Path,
) -> None:
    tokenizer = FakeTokenizer(["", "p", "l", "a", "i", "n"])
    model = FakeModel([1, 2, 3, 4, 5])
    generator = SawrGenerator(
        config=_generation_config(tmp_path, max_new_tokens=5),
        rule=AlwaysHitRule(),
        model=model,
        tokenizer=tokenizer,
    )

    result = generator.generate(
        sample_id="mbpp/1",
        prompt="Write text.",
        dataset="mbpp",
        max_group_statements=2,
        retry_budget=1,
    )

    assert result.final_code == "Write text.plain"
    assert result.accepted_hit_count == 0
    assert result.audit_events[-1].event == "closed_without_hit"
    assert result.audit_events[-1].reason == "no_controlled_function_body"


def test_load_sawr_model_uses_4bit_config_when_requested(tmp_path: Path) -> None:
    cfg = _generation_config(
        tmp_path,
        load_in_4bit=True,
        device="cuda",
        torch_dtype="bf16",
    )
    tokenizer = object()
    model = MagicMock()

    with (
        patch(
            "wfcllm.sawr.generator.AutoTokenizer.from_pretrained",
            return_value=tokenizer,
        ) as tok,
        patch("wfcllm.sawr.generator.BitsAndBytesConfig") as bnb,
        patch(
            "wfcllm.sawr.generator.AutoModelForCausalLM.from_pretrained",
            return_value=model,
        ) as mdl,
    ):
        loaded_model, loaded_tokenizer = load_sawr_model(cfg)

    tok.assert_called_once_with(cfg.model_path)
    bnb.assert_called_once()
    mdl.assert_called_once()
    assert loaded_model is model
    assert loaded_tokenizer is tokenizer
