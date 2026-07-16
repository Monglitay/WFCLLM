from __future__ import annotations

import importlib
import sys
import types
from dataclasses import FrozenInstanceError

import pytest
import torch
from torch import nn

from wfcllm.gate.config import GateTrainConfig
from wfcllm.gate.model import GateModel, GateModelOutput


class FakeEncoder(nn.Module):
    def __init__(self, hidden_size: int = 16) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, *, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> object:
        hidden = input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, self.hidden_size)
        return types.SimpleNamespace(last_hidden_state=hidden)


def test_gate_model_returns_two_finite_logits() -> None:
    model = GateModel(encoder=FakeEncoder(), hidden_size=16)
    output = model(
        input_ids=torch.ones((4, 12), dtype=torch.long),
        attention_mask=torch.ones((4, 12), dtype=torch.long),
    )
    assert isinstance(output, GateModelOutput)
    assert output.close_logits.shape == (4,)
    assert output.suitable_logits.shape == (4,)
    assert torch.isfinite(output.close_logits).all()
    assert torch.isfinite(output.suitable_logits).all()
    with pytest.raises(FrozenInstanceError):
        output.close_logits = torch.zeros(4)  # type: ignore[misc]


def test_gate_model_uses_attention_masked_mean_pooling() -> None:
    model = GateModel(encoder=FakeEncoder(hidden_size=2), hidden_size=2)
    with torch.no_grad():
        model.close_head.weight.fill_(1.0)
        model.close_head.bias.zero_()
    output = model(
        input_ids=torch.tensor([[2, 4, 100]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
    )
    assert output.close_logits.item() == pytest.approx(6.0)


@pytest.mark.parametrize(
    "input_ids,attention_mask,message",
    [
        (torch.ones(3, dtype=torch.long), torch.ones(3, dtype=torch.long), "2-D"),
        (torch.ones((2, 3), dtype=torch.float32), torch.ones((2, 3), dtype=torch.long), "integer dtype"),
        (torch.ones((2, 3), dtype=torch.int32), torch.ones((2, 3), dtype=torch.long), "torch.long"),
        (torch.ones((2, 3), dtype=torch.long), torch.ones((2, 2), dtype=torch.long), "same shape"),
        (torch.ones((2, 3), dtype=torch.long), torch.tensor([[1, 2, 1], [1, 1, 1]]), "binary"),
    ],
)
def test_gate_model_rejects_invalid_input_boundaries(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, message: str
) -> None:
    model = GateModel(encoder=FakeEncoder(), hidden_size=16)
    with pytest.raises(ValueError, match=message):
        model(input_ids=input_ids, attention_mask=attention_mask)


class BadEncoder(nn.Module):
    def __init__(self, hidden: torch.Tensor) -> None:
        super().__init__()
        self.hidden = hidden

    def forward(self, **_: object) -> object:
        return types.SimpleNamespace(last_hidden_state=self.hidden)


def test_gate_model_rejects_wrong_or_nonfinite_encoder_output() -> None:
    inputs = torch.ones((2, 3), dtype=torch.long)
    for hidden, message in [
        (torch.ones((2, 4, 16)), "shape"),
        (torch.ones((2, 3, 15)), "hidden size"),
        (torch.full((2, 3, 16), float("nan")), "finite"),
    ]:
        model = GateModel(encoder=BadEncoder(hidden), hidden_size=16)
        with pytest.raises(ValueError, match=message):
            model(input_ids=inputs, attention_mask=inputs)


def test_transformers_is_lazy_and_local_loader_forces_offline(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    encoder = FakeEncoder(hidden_size=7)
    encoder.config = types.SimpleNamespace(hidden_size=7)  # type: ignore[attr-defined]

    class AutoModel:
        @staticmethod
        def from_pretrained(path: str, **kwargs: object) -> nn.Module:
            calls.append((path, dict(kwargs)))
            return encoder

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(AutoModel=AutoModel))
    model = GateModel.from_local_pretrained(
        GateTrainConfig(base_model_path=tmp_path / "local-model")
    )
    assert model.encoder is encoder
    assert calls == [(str(tmp_path / "local-model"), {"local_files_only": True})]


def test_importing_model_module_does_not_import_transformers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    import wfcllm.gate.model as model_module

    importlib.reload(model_module)
    assert "transformers" not in sys.modules


def test_local_loader_rejects_non_config_before_transformers_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "transformers", raising=False)
    with pytest.raises(ValueError, match="GateTrainConfig"):
        GateModel.from_local_pretrained(object())  # type: ignore[arg-type]
    assert "transformers" not in sys.modules


def test_model_constructor_rejects_bool_hidden_size_and_nonmodule_encoder() -> None:
    with pytest.raises(ValueError, match="hidden_size"):
        GateModel(encoder=FakeEncoder(), hidden_size=True)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="encoder"):
        GateModel(encoder=object(), hidden_size=16)  # type: ignore[arg-type]
