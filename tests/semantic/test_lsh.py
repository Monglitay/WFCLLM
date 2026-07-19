from __future__ import annotations

import pytest


def test_checkpoint_metadata_cannot_override_runtime_precision_mode() -> None:
    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.semantic.lsh import resolve_checkpoint_encoder_config

    resolved = resolve_checkpoint_encoder_config(
        EncoderConfig(use_bf16=False),
        {"config": {"use_bf16": True}},
    )

    assert resolved.use_bf16 is False


def test_codet5_verifier_reports_key_independent_reference_cosine() -> None:
    import torch

    from wfcllm.semantic.lsh import CodeT5LshVerifier

    class Tokenizer:
        def __call__(self, text, **_kwargs):
            texts = text if isinstance(text, list) else [text]
            values = [1 if item == "reference" else 2 for item in texts]
            return {
                "input_ids": torch.tensor([[value] for value in values]),
                "attention_mask": torch.tensor([[1]] * len(values)),
            }

    class Encoder:
        def eval(self):
            return self

        def __call__(self, *, input_ids, attention_mask):
            del attention_mask
            return torch.stack(
                [
                    torch.tensor([1.0, 0.0])
                    if int(row[0]) == 1
                    else torch.tensor([0.8, 0.6])
                    for row in input_ids
                ]
            )

    verifier = CodeT5LshVerifier(
        encoder=Encoder(),
        tokenizer=Tokenizer(),
        lsh_space=object(),
        device="cpu",
    )

    assert verifier.semantic_reference_cosine(
        "reference", "candidate"
    ) == pytest.approx(0.8)


def test_codet5_verifier_measures_real_precision_and_batch_modes() -> None:
    import torch

    from wfcllm.semantic.lsh import CodeT5LshVerifier

    class Tokenizer:
        def __call__(self, texts, **_kwargs):
            return {
                "input_ids": torch.ones((len(texts), 2), dtype=torch.long),
                "attention_mask": torch.ones((len(texts), 2), dtype=torch.long),
            }

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)
            self.linear.weight.data.copy_(torch.eye(2))
            self.calls = []

        def forward(self, *, input_ids, attention_mask):
            del attention_mask
            output = self.linear(input_ids.float())
            self.calls.append((input_ids.shape[0], output.dtype))
            return output

    class Space:
        @staticmethod
        def sign(embedding):
            return tuple(int(value >= 0) for value in embedding.tolist())

        @staticmethod
        def min_margin(embedding):
            return float(embedding.abs().min())

    encoder = Encoder()
    verifier = CodeT5LshVerifier(
        encoder=encoder,
        tokenizer=Tokenizer(),
        lsh_space=Space(),
        device="cpu",
    )

    signature, margin, precision_stable, batch_stable = (
        verifier.signature_and_margin_modes("x = 1")
    )

    assert signature == (1, 1)
    assert margin == pytest.approx(1.0)
    assert precision_stable is True
    assert batch_stable is True
    assert encoder.calls == [
        (1, torch.float32),
        (1, torch.bfloat16),
        (2, torch.float32),
        (2, torch.bfloat16),
    ]


def test_codet5_verifier_detects_bfloat16_batch_interaction() -> None:
    import torch

    from wfcllm.semantic.lsh import CodeT5LshVerifier

    class Encoder:
        def eval(self):
            return self

    class Space:
        @staticmethod
        def sign(embedding):
            return tuple(int(value >= 0) for value in embedding.tolist())

        @staticmethod
        def min_margin(embedding):
            return float(embedding.abs().min())

    verifier = CodeT5LshVerifier(
        encoder=Encoder(),
        tokenizer=object(),
        lsh_space=Space(),
        device="cpu",
    )

    def fake_embed_batch(code_texts, *, use_bfloat16=False):
        first = -1.0 if use_bfloat16 and len(code_texts) == 2 else 1.0
        return torch.tensor([[first, 1.0]] * len(code_texts))

    verifier.embed_batch = fake_embed_batch

    _signature, _margin, precision_stable, batch_stable = (
        verifier.signature_and_margin_modes("x = 1")
    )

    assert precision_stable is False
    assert batch_stable is False


def test_semantic_lsh_module_exports_loader() -> None:
    from wfcllm.semantic.lsh import load_semantic_lsh_rule

    assert callable(load_semantic_lsh_rule)
