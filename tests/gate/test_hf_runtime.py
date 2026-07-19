from __future__ import annotations

from types import SimpleNamespace

import torch

from wfcllm.gate.production import HFCausalRewriteBackend, LocalSemanticRuntime


class _Tokenizer:
    def __init__(self):
        self.chat_messages = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_messages = messages
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        return "CHAT-FORMATTED"

    def __call__(self, text, **kwargs):
        assert text == "CHAT-FORMATTED"
        return {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        return "x = 1\ny = 2\n"


class _Model:
    config = SimpleNamespace(is_encoder_decoder=False)

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 64
        assert kwargs["temperature"] == 0.2
        assert kwargs["top_p"] == 0.7
        return torch.tensor([[10, 11, 21, 22]])


def test_causal_rewrite_backend_decodes_only_new_tokens() -> None:
    tokenizer = _Tokenizer()
    backend = HFCausalRewriteBackend(
        model=_Model(),
        tokenizer=tokenizer,
        device="cpu",
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.7,
    )

    result = backend.generate_window(
        prompt="complete code",
        completed_prefix="def f():\n",
        original_window="    x = 0\n",
        candidate_index=2,
        max_units=3,
    )

    assert result.text == "    x = 1\n    y = 2\n"
    assert result.token_ids == (21, 22)
    assert result.generation_seed_id.startswith("local-hf-v1:")
    assert tokenizer.chat_messages[0]["role"] == "user"
    instruction = tokenizer.chat_messages[0]["content"]
    assert "exactly 3 complete Python statements" in instruction
    assert "Preserve every referenced name" in instruction
    assert "do not output it" in instruction


def test_causal_rewrite_backend_restores_public_window_indentation() -> None:
    class FencedTokenizer(_Tokenizer):
        def decode(self, token_ids, **kwargs):
            return "```python\nx = 1\nif ready:\n    y = 2\n```"

    backend = HFCausalRewriteBackend(
        model=_Model(),
        tokenizer=FencedTokenizer(),
        device="cpu",
        max_new_tokens=64,
        temperature=0.2,
        top_p=0.7,
    )

    result = backend.generate_window(
        prompt="",
        completed_prefix="def f():\n",
        original_window="    x = 0\n    if ready:\n        y = 0\n",
        candidate_index=1,
        max_units=2,
    )

    assert result.text == "    x = 1\n    if ready:\n        y = 2\n"


class _Verifier:
    def __init__(self):
        self.calls = 0

    def semantic_reference_cosine(self, reference_text, candidate_text):
        assert reference_text and candidate_text
        return 1.0

    def signature_and_margin_modes(self, code_text):
        self.calls += 1
        assert code_text == "x = 1"
        return (1, 0, 1, 0), 0.75, True, True


def test_local_semantic_runtime_requires_measured_mode_stability() -> None:
    verifier = _Verifier()
    runtime = LocalSemanticRuntime(verifier)

    signature, margin, precision_stable, batch_stable = runtime.signature_and_margin("x = 1")

    assert signature == (1, 0, 1, 0)
    assert margin == 0.75
    assert precision_stable is True
    assert batch_stable is True
    assert verifier.calls == 1
