from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.sawr.rules import SemanticLshEmbeddingRule
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace


@dataclass(frozen=True)
class SemanticLshResult:
    passed: bool
    lsh_signature: tuple[int, ...]
    min_margin: float
    in_valid_set: bool


@dataclass(frozen=True)
class SemanticLshComponents:
    verifier: CodeT5LshVerifier
    keying: WatermarkKeying


class CodeT5LshVerifier:
    """Verify candidate statement text in the CodeT5 semantic LSH space."""

    def __init__(
        self,
        *,
        encoder: SemanticEncoder,
        tokenizer,
        lsh_space: LSHSpace,
        device: str,
        max_length: int = 256,
    ) -> None:
        self._encoder = encoder
        self._tokenizer = tokenizer
        self._lsh_space = lsh_space
        self._device = device
        self._max_length = max_length
        self._encoder.eval()

    @torch.no_grad()
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> SemanticLshResult:
        encoded = self._tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=self._max_length,
        )
        encoded = {
            name: tensor.to(self._device)
            for name, tensor in encoded.items()
        }
        embedding = self._encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
        )[0]
        signature = self._lsh_space.sign(embedding)
        min_margin = self._lsh_space.min_margin(embedding)
        in_valid_set = signature in valid_set
        return SemanticLshResult(
            passed=in_valid_set and min_margin > margin,
            lsh_signature=signature,
            min_margin=min_margin,
            in_valid_set=in_valid_set,
        )


def load_semantic_lsh_components(
    *,
    encoder_model_path: str,
    encoder_checkpoint_path: str | None,
    embed_dim: int,
    device: str,
    use_lora: bool,
    use_bf16: bool,
    secret_key: str,
    lsh_d: int,
    whitening_path: str | None,
) -> SemanticLshComponents:
    """Load CodeT5 encoder, tokenizer, LSH space, and keying."""

    from transformers import AutoTokenizer

    model_path = Path(encoder_model_path)
    if not model_path.exists():
        raise ValueError(f"encoder_model_path does not exist: {encoder_model_path}")
    if encoder_checkpoint_path is not None and not Path(encoder_checkpoint_path).exists():
        raise ValueError(
            f"encoder_checkpoint_path does not exist: {encoder_checkpoint_path}"
        )
    if whitening_path is not None and not Path(whitening_path).exists():
        raise ValueError(f"lsh_whitening_path does not exist: {whitening_path}")

    encoder_config = EncoderConfig(
        model_name=encoder_model_path,
        embed_dim=embed_dim,
        use_lora=use_lora,
        use_bf16=use_bf16,
    )
    encoder = SemanticEncoder(config=encoder_config)
    if encoder_checkpoint_path is not None:
        checkpoint = torch.load(encoder_checkpoint_path, map_location="cpu")
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        encoder.load_state_dict(state_dict)
    encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(encoder_model_path)
    lsh_space = LSHSpace(
        secret_key=secret_key,
        embed_dim=embed_dim,
        d=lsh_d,
        whitening_path=whitening_path,
    )
    verifier = CodeT5LshVerifier(
        encoder=encoder,
        tokenizer=tokenizer,
        lsh_space=lsh_space,
        device=device,
        max_length=encoder_config.max_seq_length,
    )
    keying = WatermarkKeying(secret_key, lsh_d)
    return SemanticLshComponents(verifier=verifier, keying=keying)


def load_semantic_lsh_rule(
    *,
    encoder_model_path: str,
    encoder_checkpoint_path: str | None,
    embed_dim: int,
    device: str,
    use_lora: bool,
    use_bf16: bool,
    secret_key: str,
    lsh_d: int,
    lsh_gamma: float,
    margin: float,
    whitening_path: str | None,
    use_ordinal_keying: bool = False,
) -> SemanticLshEmbeddingRule:
    """Load CodeT5 encoder, LSH space, and keying for a semantic SAWR rule."""

    components = load_semantic_lsh_components(
        encoder_model_path=encoder_model_path,
        encoder_checkpoint_path=encoder_checkpoint_path,
        embed_dim=embed_dim,
        device=device,
        use_lora=use_lora,
        use_bf16=use_bf16,
        secret_key=secret_key,
        lsh_d=lsh_d,
        whitening_path=whitening_path,
    )
    return SemanticLshEmbeddingRule(
        verifier=components.verifier,
        keying=components.keying,
        lsh_d=lsh_d,
        lsh_gamma=lsh_gamma,
        margin=margin,
        use_ordinal_keying=use_ordinal_keying,
    )
