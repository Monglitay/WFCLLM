from __future__ import annotations

from types import SimpleNamespace

import torch

from wfcllm.gate.production import HFCausalRewriteBackend, LocalSemanticRuntime


class _Tokenizer:
    def __call__(self, text, **kwargs):
        assert "Original window" in text
        return {"input_ids": torch.tensor([[10, 11]]), "attention_mask": torch.ones(1, 2, dtype=torch.long)}

    def decode(self, token_ids, **kwargs):
        assert kwargs["skip_special_tokens"] is True
        return "x = 1\ny = 2\n"


class _Model:
    config = SimpleNamespace(is_encoder_decoder=False)

    def generate(self, **kwargs):
        assert kwargs["max_new_tokens"] == 64
        return torch.tensor([[10, 11, 21, 22]])


def test_causal_rewrite_backend_decodes_only_new_tokens() -> None:
    backend = HFCausalRewriteBackend(
        model=_Model(), tokenizer=_Tokenizer(), device="cpu", max_new_tokens=64
    )

    result = backend.generate_window(
        prompt="complete code",
        completed_prefix="def f():\n",
        original_window="    x = 0\n",
        candidate_index=2,
        max_units=3,
    )

    assert result.text == "x = 1\ny = 2\n"
    assert result.token_ids == (21, 22)
    assert result.generation_seed_id.startswith("local-hf-v1:")


class _Verifier:
    def __init__(self):
        self.calls = 0

    def verify(self, code_text, valid_set, margin):
        self.calls += 1
        return SimpleNamespace(lsh_signature=(1, 0, 1, 0), min_margin=0.75)


def test_local_semantic_runtime_requires_repeatable_signature() -> None:
    verifier = _Verifier()
    runtime = LocalSemanticRuntime(verifier)

    signature, margin, precision_stable, batch_stable = runtime.signature_and_margin("x = 1")

    assert signature == (1, 0, 1, 0)
    assert margin == 0.75
    assert precision_stable is True
    assert batch_stable is True
    assert verifier.calls == 2
