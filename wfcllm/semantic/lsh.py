from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from collections.abc import Mapping
from pathlib import Path
from contextlib import nullcontext

import torch

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.semantic.rules import SemanticLshEmbeddingRule
from wfcllm.semantic.keying import WatermarkKeying
from wfcllm.semantic.lsh_space import LSHSpace


@dataclass(frozen=True)
class SemanticLshResult:
    passed: bool
    lsh_signature: tuple[int, ...]
    min_margin: float
    in_valid_set: bool


@dataclass(frozen=True)
class SemanticLshModeResult(SemanticLshResult):
    stable_across_precision_modes: bool
    stable_across_batch_modes: bool


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
    def embed(self, code_text: str) -> torch.Tensor:
        """Return one public, key-independent normalized code embedding."""

        if not isinstance(code_text, str) or not code_text.strip():
            raise ValueError("code_text must be a non-empty string")
        return self.embed_batch((code_text,))[0]

    @torch.no_grad()
    def embed_batch(
        self, code_texts: tuple[str, ...], *, use_bfloat16: bool = False
    ) -> torch.Tensor:
        if (
            not isinstance(code_texts, tuple)
            or not code_texts
            or any(not isinstance(text, str) or not text.strip() for text in code_texts)
        ):
            raise ValueError("code_texts must contain non-empty strings")
        encoded = self._tokenizer(
            list(code_texts),
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self._max_length,
        )
        encoded = {
            name: tensor.to(self._device)
            for name, tensor in encoded.items()
        }
        device_type = str(self._device).split(":", 1)[0]
        precision_context = (
            torch.autocast(device_type=device_type, dtype=torch.bfloat16)
            if use_bfloat16
            else nullcontext()
        )
        with precision_context:
            embeddings = self._encoder(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            )
        embeddings = embeddings.float()
        if embeddings.ndim != 2 or not torch.isfinite(embeddings).all():
            raise ValueError("semantic encoder returned an invalid embedding")
        return embeddings

    @torch.no_grad()
    def signature_and_margin_modes(
        self, code_text: str
    ) -> tuple[tuple[int, ...], float, bool, bool]:
        """Measure signature stability across real precision and batch modes."""

        float_single = self.embed_batch((code_text,))[0]
        bfloat_single = self.embed_batch(
            (code_text,), use_bfloat16=True
        )[0]
        float_batched = self.embed_batch((code_text, code_text))[0]
        bfloat_batched = self.embed_batch(
            (code_text, code_text), use_bfloat16=True
        )[0]
        signatures = tuple(
            self._lsh_space.sign(embedding)
            for embedding in (
                float_single,
                bfloat_single,
                float_batched,
                bfloat_batched,
            )
        )
        margins = tuple(
            float(self._lsh_space.min_margin(embedding))
            for embedding in (
                float_single,
                bfloat_single,
                float_batched,
                bfloat_batched,
            )
        )
        return (
            signatures[0],
            min(margins),
            signatures[0] == signatures[1]
            and signatures[2] == signatures[3],
            signatures[0] == signatures[2]
            and signatures[1] == signatures[3],
        )

    @torch.no_grad()
    def verify_modes(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> SemanticLshModeResult:
        signature, min_margin, precision_stable, batch_stable = (
            self.signature_and_margin_modes(code_text)
        )
        stable = precision_stable and batch_stable
        in_valid_set = signature in valid_set
        return SemanticLshModeResult(
            passed=stable and in_valid_set and min_margin > margin,
            lsh_signature=signature,
            min_margin=min_margin,
            in_valid_set=in_valid_set,
            stable_across_precision_modes=precision_stable,
            stable_across_batch_modes=batch_stable,
        )

    @torch.no_grad()
    def semantic_reference_cosine(
        self, reference_text: str, candidate_text: str
    ) -> float:
        """Compare a candidate to its public original window without any key."""

        reference = self.embed(reference_text)
        candidate = self.embed(candidate_text)
        if reference.shape != candidate.shape:
            raise ValueError("semantic embeddings must have matching shapes")
        cosine = torch.nn.functional.cosine_similarity(
            reference.unsqueeze(0), candidate.unsqueeze(0), dim=1
        )[0]
        return float(cosine.clamp(-1.0, 1.0).item())

    @torch.no_grad()
    def verify(
        self,
        code_text: str,
        valid_set: frozenset[tuple[int, ...]],
        margin: float,
    ) -> SemanticLshResult:
        embedding = self.embed(code_text)
        signature = self._lsh_space.sign(embedding)
        min_margin = self._lsh_space.min_margin(embedding)
        in_valid_set = signature in valid_set
        return SemanticLshResult(
            passed=in_valid_set and min_margin > margin,
            lsh_signature=signature,
            min_margin=min_margin,
            in_valid_set=in_valid_set,
        )


def resolve_checkpoint_encoder_config(
    base: EncoderConfig,
    checkpoint: object,
) -> EncoderConfig:
    """Bind architecture metadata while preserving the deployed model path."""

    if not isinstance(base, EncoderConfig):
        raise ValueError("base must be an EncoderConfig")
    if not isinstance(checkpoint, Mapping):
        return base
    metadata = checkpoint.get("config")
    if not isinstance(metadata, Mapping):
        return base
    embed_dim = metadata.get("embed_dim", base.embed_dim)
    if type(embed_dim) is not int or embed_dim != base.embed_dim:
        raise ValueError("checkpoint embed_dim does not match runtime embed_dim")
    pooling = metadata.get("pooling", base.pooling)
    if pooling not in {"first", "masked_mean"}:
        raise ValueError("checkpoint pooling must be first or masked_mean")
    updates: dict[str, object] = {"pooling": pooling}
    # LoRA changes the checkpoint architecture and must follow its metadata.
    # Precision is a deployment execution mode: preserving ``base.use_bf16``
    # lets formal verification load FP32 weights and measure a real BF16
    # autocast path against them.
    for name in ("use_lora",):
        value = metadata.get(name, getattr(base, name))
        if not isinstance(value, bool):
            raise ValueError(f"checkpoint {name} must be a bool")
        updates[name] = value
    updates["use_bf16"] = base.use_bf16
    for name in ("lora_r", "lora_alpha"):
        value = metadata.get(name, getattr(base, name))
        if type(value) is not int or value <= 0:
            raise ValueError(f"checkpoint {name} must be a positive integer")
        updates[name] = value
    dropout = metadata.get("lora_dropout", base.lora_dropout)
    if isinstance(dropout, bool) or not isinstance(dropout, (int, float)) or not 0 <= dropout < 1:
        raise ValueError("checkpoint lora_dropout must be in [0, 1)")
    updates["lora_dropout"] = float(dropout)
    targets = metadata.get("lora_target_modules", base.lora_target_modules)
    if not isinstance(targets, (list, tuple)) or not targets or any(
        not isinstance(value, str) or not value for value in targets
    ):
        raise ValueError("checkpoint lora_target_modules must be non-empty strings")
    updates["lora_target_modules"] = list(targets)
    return replace(base, **updates)


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
    checkpoint = None
    if encoder_checkpoint_path is not None:
        checkpoint = torch.load(
            encoder_checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        encoder_config = resolve_checkpoint_encoder_config(
            encoder_config,
            checkpoint,
        )
    encoder = SemanticEncoder(config=encoder_config)
    if checkpoint is not None:
        state_dict = (
            checkpoint["model_state_dict"]
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
            else checkpoint
        )
        encoder.load_state_dict(state_dict)
    encoder.to(device)

    tokenizer = AutoTokenizer.from_pretrained(
        encoder_model_path, local_files_only=True
    )
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
