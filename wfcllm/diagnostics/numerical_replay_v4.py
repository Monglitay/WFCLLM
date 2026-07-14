"""Pure helpers for the V4 batch-invariance numerical diagnosis.

The tracked experiment driver uses these helpers while raw numerical records
remain under the Git-ignored V4 experiment directory.  No function in this
module loads a model, reads a private key, or writes an artifact.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import random
import re
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
import torch


_DIAGNOSTIC_HEADER = b"WFCLLM_BATCH_INVARIANT_SEMANTIC_V4_DIAGNOSTIC\0"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_NUMERICAL_STAGE_ORDER = (
    "input_ids",
    "attention_mask",
    "t5_cls_hidden",
    "projection_pre_norm",
    "model_post_norm",
    "runtime_post_norm",
    "cpu_centered",
    "whitening_pre_norm",
    "whitening_post_norm",
    "quantized",
    "projection_dots",
    "signature_bits",
)


@dataclass(frozen=True)
class ContextCase:
    """One public serialized canonical context used by the diagnosis."""

    case_id: str
    serialized: str
    context_sha256: str
    role: str
    token_count: int
    category: str

    def __post_init__(self) -> None:
        if not self.case_id or not self.serialized or not self.role:
            raise ValueError("context identity, serialization, and role are required")
        if not _SHA256_PATTERN.fullmatch(self.context_sha256):
            raise ValueError("context SHA-256 must be lowercase hexadecimal")
        actual = hashlib.sha256(self.serialized.encode("utf-8")).hexdigest()
        if actual != self.context_sha256:
            raise ValueError("context SHA-256 does not match serialized content")
        if (
            isinstance(self.token_count, bool)
            or not isinstance(self.token_count, int)
            or self.token_count <= 0
        ):
            raise ValueError("token_count must be a positive integer")
        if self.category not in {"failure", "control", "synthetic", "boundary"}:
            raise ValueError("unsupported context category")


@dataclass(frozen=True)
class ComposedBatch:
    """A physical encoder batch plus every row occupied by the target."""

    contexts: tuple[str, ...]
    target_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.contexts:
            raise ValueError("batch must not be empty")
        if not self.target_indices:
            raise ValueError("batch must contain the target context")
        if tuple(sorted(set(self.target_indices))) != self.target_indices:
            raise ValueError("target_indices must be unique and sorted")
        if any(not 0 <= index < len(self.contexts) for index in self.target_indices):
            raise ValueError("target index is outside the batch")

    @property
    def target_index(self) -> int:
        """Compatibility accessor for conditions with one target row."""

        return self.target_indices[0]


@dataclass(frozen=True)
class ConditionSpec:
    """One orthogonal numerical condition applied to every context case."""

    condition_id: str
    axis: str
    batch_size: int
    composition: str
    order: str
    padding: str
    grad_mode: str
    deterministic_algorithms: bool
    tf32: bool
    matmul_precision: str
    warmup_count: int
    repeats: int = 1
    reference: bool = False
    masked_tail_variant: str = "pad_token"

    def __post_init__(self) -> None:
        if not self.condition_id or not self.axis:
            raise ValueError("condition identity and axis are required")
        if self.batch_size not in {1, 2, 4, 8, 16, 32}:
            raise ValueError("unsupported batch size")
        if self.composition not in {
            "self_repeat",
            "short_mix",
            "long_mix",
            "failure_mix",
        }:
            raise ValueError("unsupported batch composition")
        if self.order not in {"forward", "reverse", "random"}:
            raise ValueError("unsupported batch order")
        if self.padding not in {"dynamic", "fixed_256"}:
            raise ValueError("unsupported padding mode")
        if self.grad_mode not in {"no_grad", "inference_mode"}:
            raise ValueError("unsupported grad mode")
        if self.matmul_precision not in {"highest", "high", "medium"}:
            raise ValueError("unsupported matmul precision")
        if self.warmup_count not in {0, 1, 5}:
            raise ValueError("unsupported warm-up count")
        if self.repeats <= 0:
            raise ValueError("repeats must be positive")
        if self.masked_tail_variant not in {"pad_token", "alternate_token"}:
            raise ValueError("unsupported masked-tail variant")


@dataclass(frozen=True)
class BoundaryCandidate:
    """Observed boundary margins for deterministic context selection."""

    case: ContextCase
    minimum_quantization_margin: float
    minimum_projection_margin: float

    def __post_init__(self) -> None:
        margins = (
            self.minimum_quantization_margin,
            self.minimum_projection_margin,
        )
        if any(not math.isfinite(value) or value < 0 for value in margins):
            raise ValueError("boundary margins must be finite and non-negative")


@dataclass(frozen=True)
class TensorDelta:
    """Exact and numerical difference summary for two equal-schema tensors."""

    reference_sha256: str
    candidate_sha256: str
    mismatch_count: int
    mismatch_indices: tuple[int, ...]
    max_abs: float
    max_relative: float
    max_ulp: int
    cosine_similarity: float


@dataclass(frozen=True)
class BatchCapture:
    """Measured target rows and effective runtime metadata for one batch."""

    sequence_length: int
    target_layers: tuple[dict[str, torch.Tensor], ...]
    batch_member_sha256: tuple[str, ...]
    model_eval: bool
    effective_flags: dict[str, Any]


def quantization_boundary_margins(
    values: torch.Tensor | Sequence[float],
    *,
    scale: int,
) -> tuple[float, ...]:
    """Return distance to the nearest half-integer in scaled coordinates."""

    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    tensor = torch.as_tensor(values, dtype=torch.float64).flatten()
    if not torch.isfinite(tensor).all():
        raise ValueError("values must be finite")
    scaled = tensor * scale
    fraction = scaled - torch.floor(scaled)
    margins = torch.abs(fraction - 0.5)
    return tuple(float(value) for value in margins.tolist())


def quantize_half_away_from_zero(
    values: torch.Tensor | Sequence[float],
    *,
    scale: int,
) -> torch.Tensor:
    """Apply V3's exact half-away-from-zero discretization rule."""

    if isinstance(scale, bool) or not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be a positive integer")
    tensor = torch.as_tensor(values, dtype=torch.float64)
    if not torch.isfinite(tensor).all():
        raise ValueError("values must be finite")
    scaled = tensor * scale
    return torch.where(
        scaled >= 0,
        torch.floor(scaled + 0.5),
        torch.ceil(scaled - 0.5),
    ).to(torch.int64)


def _cyclic_fill(candidates: Sequence[ContextCase], count: int) -> list[ContextCase]:
    if count == 0:
        return []
    if not candidates:
        raise ValueError("composition has no eligible filler contexts")
    return [candidates[index % len(candidates)] for index in range(count)]


def compose_batch(
    target: ContextCase,
    pool: Sequence[ContextCase],
    *,
    batch_size: int,
    composition: str,
    order: str,
    seed: int,
) -> ComposedBatch:
    """Construct one deterministic target batch for a controlled condition."""

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    if order not in {"forward", "reverse", "random"}:
        raise ValueError("unsupported batch order")
    if composition == "self_repeat":
        items = [target] * batch_size
    else:
        others = [item for item in pool if item.case_id != target.case_id]
        if composition == "short_mix":
            candidates = sorted(others, key=lambda item: (item.token_count, item.case_id))[:2]
        elif composition == "long_mix":
            candidates = sorted(
                others,
                key=lambda item: (-item.token_count, item.case_id),
            )[:2]
        elif composition == "failure_mix":
            candidates = sorted(
                (item for item in others if item.category == "failure"),
                key=lambda item: item.case_id,
            )
        else:
            raise ValueError("unsupported batch composition")
        items = [target, *_cyclic_fill(candidates, batch_size - 1)]
    if order == "reverse":
        items.reverse()
    elif order == "random":
        random.Random(seed).shuffle(items)
    target_indices = tuple(
        index for index, item in enumerate(items) if item.case_id == target.case_id
    )
    contexts = tuple(item.serialized for item in items)
    if any(contexts[index] != target.serialized for index in target_indices):
        raise AssertionError("composed batch lost target context")
    return ComposedBatch(contexts=contexts, target_indices=target_indices)


def _condition(
    conditions: list[ConditionSpec],
    *,
    axis: str,
    **changes: object,
) -> None:
    base = ConditionSpec(
        condition_id="pending",
        axis=axis,
        batch_size=1,
        composition="self_repeat",
        order="forward",
        padding="fixed_256",
        grad_mode="inference_mode",
        deterministic_algorithms=True,
        tf32=False,
        matmul_precision="highest",
        warmup_count=1,
    )
    item = replace(base, **changes)
    conditions.append(
        replace(item, condition_id=f"{axis}-{len(conditions):03d}")
    )


def build_condition_specs(*, repeats: int = 20) -> tuple[ConditionSpec, ...]:
    """Build a prereviewed orthogonal matrix, avoiding a confounded factorial."""

    if repeats < 20:
        raise ValueError("same-process reference requires at least 20 repeats")
    conditions: list[ConditionSpec] = []
    _condition(
        conditions,
        axis="reference",
        repeats=repeats,
        reference=True,
    )
    for batch_size in (2, 4, 8, 16, 32):
        _condition(conditions, axis="batch_size", batch_size=batch_size)
    for batch_size in (2, 4, 8, 16, 32):
        for composition in ("short_mix", "long_mix", "failure_mix"):
            _condition(
                conditions,
                axis="composition",
                batch_size=batch_size,
                composition=composition,
            )
    for order in ("forward", "reverse", "random"):
        _condition(
            conditions,
            axis="order",
            batch_size=32,
            composition="failure_mix",
            order=order,
        )
    for padding in ("fixed_256", "dynamic"):
        _condition(
            conditions,
            axis="padding",
            batch_size=8,
            composition="short_mix",
            padding=padding,
        )
    for grad_mode in ("inference_mode", "no_grad"):
        _condition(
            conditions,
            axis="grad_mode",
            batch_size=8,
            composition="short_mix",
            grad_mode=grad_mode,
        )
    for deterministic in (True, False):
        _condition(
            conditions,
            axis="deterministic",
            batch_size=8,
            composition="short_mix",
            deterministic_algorithms=deterministic,
        )
    for tf32 in (False, True):
        _condition(
            conditions,
            axis="tf32",
            batch_size=8,
            composition="short_mix",
            tf32=tf32,
        )
    for precision in ("highest", "high", "medium"):
        _condition(
            conditions,
            axis="matmul_precision",
            batch_size=8,
            composition="short_mix",
            matmul_precision=precision,
        )
    for warmup_count in (0, 1, 5):
        _condition(
            conditions,
            axis="warmup",
            batch_size=8,
            composition="short_mix",
            warmup_count=warmup_count,
        )
    for masked_tail in ("pad_token", "alternate_token"):
        _condition(
            conditions,
            axis="masked_tail",
            batch_size=8,
            composition="short_mix",
            masked_tail_variant=masked_tail,
        )
    return tuple(conditions)


def _ordered_stage_names(names: set[str]) -> tuple[str, ...]:
    unknown = names - set(_NUMERICAL_STAGE_ORDER)
    unknown -= {name for name in unknown if name.startswith("t5_block_")}
    if unknown:
        raise ValueError(f"unknown numerical stages: {sorted(unknown)}")
    blocks = sorted(name for name in names if name.startswith("t5_block_"))
    result: list[str] = []
    for name in _NUMERICAL_STAGE_ORDER:
        if name == "t5_cls_hidden":
            result.extend(blocks)
        if name in names:
            result.append(name)
    return tuple(result)


def first_divergent_layer(
    reference: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
) -> str | None:
    """Return the earliest stage that is not bitwise equal, fail-fast on schema."""

    if set(reference) != set(candidate):
        raise ValueError("numerical stage schema differs between reference and candidate")
    for layer in _ordered_stage_names(set(reference)):
        left = torch.as_tensor(reference[layer]).detach().cpu()
        right = torch.as_tensor(candidate[layer]).detach().cpu()
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"numerical stage schema differs at {layer}")
        if not torch.equal(left, right):
            return layer
    return None


def _validate_downstream_inputs(
    projected: torch.Tensor,
    target_indices: tuple[int, ...],
    whitening_mean: torch.Tensor,
    whitening_projection: torch.Tensor,
    projection_rows: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    values = torch.as_tensor(projected, dtype=torch.float32)
    mean = torch.as_tensor(whitening_mean, dtype=torch.float32, device="cpu")
    whitening = torch.as_tensor(
        whitening_projection,
        dtype=torch.float32,
        device="cpu",
    )
    signs = torch.as_tensor(projection_rows, dtype=torch.int64, device="cpu")
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("projected rows must be a non-empty rank-2 tensor")
    if mean.shape != (values.shape[1],):
        raise ValueError("whitening mean dimensions do not match projected rows")
    if whitening.ndim != 2 or whitening.shape[1] != values.shape[1]:
        raise ValueError("whitening projection dimensions do not match")
    if signs.ndim != 2 or signs.shape[1] != whitening.shape[0]:
        raise ValueError("diagnostic projection dimensions do not match whitening")
    if not target_indices or any(
        not 0 <= index < values.shape[0] for index in target_indices
    ):
        raise ValueError("target indices do not address projected rows")
    return values, mean, whitening, signs


def capture_downstream_from_saved(
    projected: torch.Tensor,
    *,
    target_indices: tuple[int, ...],
    normalization_device: str | torch.device = "cpu",
    whitening_mean: torch.Tensor,
    whitening_projection: torch.Tensor,
    projection_rows: torch.Tensor,
    quantization_scale: int,
    model_post_normalized: torch.Tensor | None = None,
    runtime_post_normalized: torch.Tensor | None = None,
) -> tuple[dict[str, torch.Tensor], ...]:
    """Replay normalization/whitening from an identical saved pre-norm tensor."""

    values, mean, whitening, signs = _validate_downstream_inputs(
        projected,
        target_indices,
        whitening_mean,
        whitening_projection,
        projection_rows,
    )
    normalization_values = values.to(normalization_device)
    if model_post_normalized is None:
        model_post = torch.nn.functional.normalize(normalization_values, p=2, dim=1)
    else:
        model_post = torch.as_tensor(
            model_post_normalized,
            dtype=torch.float32,
            device=normalization_device,
        )
    if runtime_post_normalized is None:
        runtime_post = torch.nn.functional.normalize(model_post, p=2, dim=1)
    else:
        runtime_post = torch.as_tensor(
            runtime_post_normalized,
            dtype=torch.float32,
            device=normalization_device,
        )
    if model_post.shape != normalization_values.shape or runtime_post.shape != normalization_values.shape:
        raise ValueError("provided normalized stages do not match projected rows")
    cpu_values = runtime_post.detach().to(dtype=torch.float32, device="cpu")
    centered = cpu_values - mean
    whitening_pre = centered @ whitening.T
    whitening_post = torch.nn.functional.normalize(whitening_pre, p=2, dim=1)
    quantized = quantize_half_away_from_zero(
        whitening_post,
        scale=quantization_scale,
    ).to(torch.int64)
    dots = quantized @ signs.T
    signature = (dots >= 0).to(torch.int8)
    result: list[dict[str, torch.Tensor]] = []
    projected_cpu = values.detach().to(dtype=torch.float32, device="cpu")
    model_cpu = model_post.detach().to(dtype=torch.float32, device="cpu")
    runtime_cpu = runtime_post.detach().to(dtype=torch.float32, device="cpu")
    for index in target_indices:
        result.append(
            {
                "projection_pre_norm": projected_cpu[index].clone(),
                "model_post_norm": model_cpu[index].clone(),
                "runtime_post_norm": runtime_cpu[index].clone(),
                "cpu_centered": centered[index].clone(),
                "whitening_pre_norm": whitening_pre[index].clone(),
                "whitening_post_norm": whitening_post[index].clone(),
                "quantized": quantized[index].clone(),
                "projection_dots": dots[index].clone(),
                "signature_bits": signature[index].clone(),
            }
        )
    return tuple(result)


def _first_tensor(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _first_tensor(output[0])
    return None


class LayerCaptureRuntime:
    """Instrument a frozen semantic model without changing its evidence path."""

    def __init__(
        self,
        *,
        model: Any,
        tokenizer: Any,
        whitening_mean: torch.Tensor,
        whitening_projection: torch.Tensor,
        projection_rows: torch.Tensor,
        quantization_scale: int,
        device: str | torch.device,
        max_tokens: int,
    ) -> None:
        if max_tokens != 256:
            raise ValueError("V4 diagnosis requires the frozen 256-token budget")
        self._device = torch.device(device)
        self._model = model.to(self._device).eval()
        self._tokenizer = tokenizer
        self._mean = torch.as_tensor(whitening_mean, dtype=torch.float32, device="cpu")
        self._whitening = torch.as_tensor(
            whitening_projection,
            dtype=torch.float32,
            device="cpu",
        )
        self._projection_rows = torch.as_tensor(
            projection_rows,
            dtype=torch.int8,
            device="cpu",
        )
        self._quantization_scale = quantization_scale
        self._max_tokens = max_tokens

    def _configure(self, condition: ConditionSpec) -> dict[str, Any]:
        torch.use_deterministic_algorithms(condition.deterministic_algorithms)
        effective_precision = condition.matmul_precision
        if condition.axis == "tf32":
            effective_precision = "high" if condition.tf32 else "highest"
        torch.set_float32_matmul_precision(effective_precision)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = condition.deterministic_algorithms
            torch.backends.cudnn.benchmark = False
        observed_precision = torch.get_float32_matmul_precision()
        return {
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "float32_matmul_precision": observed_precision,
            "tf32_requested": condition.tf32,
            "tf32_effective_inferred": observed_precision != "highest",
            "precision_api_family": "legacy",
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "grad_enabled": torch.is_grad_enabled(),
        }

    def _tokenize(
        self,
        batch: ComposedBatch,
        condition: ConditionSpec,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fixed = condition.padding == "fixed_256"
        tokenized = self._tokenizer(
            list(batch.contexts),
            padding="max_length" if fixed else True,
            max_length=self._max_tokens if fixed else None,
            truncation=False,
            return_tensors="pt",
        )
        input_ids = tokenized["input_ids"].to(self._device)
        attention_mask = tokenized["attention_mask"].to(self._device)
        if input_ids.shape[0] != len(batch.contexts):
            raise ValueError("tokenizer row count does not match physical batch")
        if input_ids.shape != attention_mask.shape:
            raise ValueError("input_ids and attention_mask shapes differ")
        if input_ids.shape[1] > self._max_tokens:
            raise ValueError("context exceeds frozen token budget; truncation is forbidden")
        if condition.masked_tail_variant == "alternate_token":
            pad_token_id = int(getattr(self._tokenizer, "pad_token_id", 0) or 0)
            vocab_size = int(getattr(self._tokenizer, "vocab_size", pad_token_id + 2))
            alternate = (pad_token_id + 1) % max(vocab_size, 2)
            if alternate == pad_token_id:
                alternate = (alternate + 1) % max(vocab_size, 2)
            input_ids = input_ids.clone()
            input_ids[attention_mask == 0] = alternate
        return input_ids, attention_mask

    @staticmethod
    def _hook_name(module_name: str) -> str | None:
        block_match = re.search(r"block\.(\d+)$", module_name)
        if block_match:
            return f"t5_block_{int(block_match.group(1)):02d}_output"
        sublayer_match = re.search(
            r"block\.(\d+)\.layer\.(\d+)\.(layer_norm|SelfAttention|DenseReluDense)$",
            module_name,
        )
        if sublayer_match:
            block, layer, name = sublayer_match.groups()
            return f"t5_block_{int(block):02d}_layer_{int(layer):02d}_{name}"
        return None

    def _forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *,
        target_indices: tuple[int, ...],
        grad_mode: str,
        capture_hooks: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        dict[str, torch.Tensor],
    ]:
        hook_rows: dict[str, torch.Tensor] = {}
        handles = []
        if capture_hooks:
            for module_name, module in self._model.encoder.named_modules():
                stage_name = self._hook_name(module_name)
                if stage_name is None:
                    continue

                def record(_module, _inputs, output, *, name=stage_name):
                    tensor = _first_tensor(output)
                    if tensor is None or tensor.ndim < 2:
                        return
                    if tensor.ndim >= 3:
                        rows = tensor[list(target_indices), 0, :]
                    else:
                        rows = tensor[list(target_indices), :]
                    hook_rows[name] = rows.detach().to(dtype=torch.float32, device="cpu")

                handles.append(module.register_forward_hook(record))
        context = torch.inference_mode if grad_mode == "inference_mode" else torch.no_grad
        try:
            with context():
                encoder_output = self._model.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                hidden = encoder_output.last_hidden_state[:, 0, :].float()
                projected = self._model.projection(hidden)
                model_post = torch.nn.functional.normalize(projected, p=2, dim=1)
                runtime_post = torch.nn.functional.normalize(model_post, p=2, dim=1)
        finally:
            for handle in handles:
                handle.remove()
        return hidden, projected, model_post, runtime_post, hook_rows

    def capture(
        self,
        batch: ComposedBatch,
        condition: ConditionSpec,
    ) -> BatchCapture:
        """Capture one physical batch after the condition's requested warm-up."""

        if len(batch.contexts) != condition.batch_size:
            raise ValueError("physical batch size does not match condition")
        if self._model.training:
            raise ValueError("semantic model must be in eval mode")
        effective_flags = self._configure(condition)
        input_ids, attention_mask = self._tokenize(batch, condition)
        for _ in range(condition.warmup_count):
            self._forward(
                input_ids,
                attention_mask,
                target_indices=batch.target_indices,
                grad_mode=condition.grad_mode,
                capture_hooks=False,
            )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        hidden, projected, model_post, runtime_post, hook_rows = self._forward(
            input_ids,
            attention_mask,
            target_indices=batch.target_indices,
            grad_mode=condition.grad_mode,
            capture_hooks=True,
        )
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        downstream = capture_downstream_from_saved(
            projected,
            target_indices=batch.target_indices,
            normalization_device=self._device,
            whitening_mean=self._mean,
            whitening_projection=self._whitening,
            projection_rows=self._projection_rows,
            quantization_scale=self._quantization_scale,
            model_post_normalized=model_post,
            runtime_post_normalized=runtime_post,
        )
        hidden_cpu = hidden.detach().to(dtype=torch.float32, device="cpu")
        input_cpu = input_ids.detach().to(device="cpu")
        mask_cpu = attention_mask.detach().to(device="cpu")
        pad_token_id = int(getattr(self._tokenizer, "pad_token_id", 0) or 0)
        canonical_fill = pad_token_id
        if condition.masked_tail_variant == "alternate_token":
            vocab_size = int(getattr(self._tokenizer, "vocab_size", pad_token_id + 2))
            canonical_fill = (pad_token_id + 1) % max(vocab_size, 2)
            if canonical_fill == pad_token_id:
                canonical_fill = (canonical_fill + 1) % max(vocab_size, 2)
        canonical_input = torch.full(
            (input_cpu.shape[0], self._max_tokens),
            canonical_fill,
            dtype=input_cpu.dtype,
        )
        canonical_mask = torch.zeros(
            (mask_cpu.shape[0], self._max_tokens),
            dtype=mask_cpu.dtype,
        )
        physical_width = input_cpu.shape[1]
        canonical_input[:, :physical_width] = input_cpu
        canonical_mask[:, :physical_width] = mask_cpu
        target_layers: list[dict[str, torch.Tensor]] = []
        for offset, index in enumerate(batch.target_indices):
            layers = {
                "input_ids": canonical_input[index].clone(),
                "attention_mask": canonical_mask[index].clone(),
                **{
                    name: rows[offset].clone()
                    for name, rows in sorted(hook_rows.items())
                },
                "t5_cls_hidden": hidden_cpu[index].clone(),
                **downstream[offset],
            }
            target_layers.append(layers)
        return BatchCapture(
            sequence_length=int(input_ids.shape[1]),
            target_layers=tuple(target_layers),
            batch_member_sha256=tuple(
                hashlib.sha256(text.encode("utf-8")).hexdigest()
                for text in batch.contexts
            ),
            model_eval=not self._model.training,
            effective_flags=effective_flags,
        )


def stable_tensor_sha256(tensor: torch.Tensor | Sequence[float]) -> str:
    """Hash canonical dtype/shape metadata plus little-endian contiguous bytes."""

    value = torch.as_tensor(tensor).detach().cpu().contiguous()
    array = value.numpy()
    little = array.astype(array.dtype.newbyteorder("<"), copy=False)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(b"\0")
    digest.update(",".join(str(item) for item in value.shape).encode("ascii"))
    digest.update(b"\0")
    digest.update(little.tobytes(order="C"))
    return digest.hexdigest()


def _maximum_float32_ulp(left: np.ndarray, right: np.ndarray) -> int:
    if left.dtype != np.float32 or right.dtype != np.float32:
        return 0
    left_bits = left.reshape(-1).view(np.uint32)
    right_bits = right.reshape(-1).view(np.uint32)

    def ordered(bits: np.ndarray) -> np.ndarray:
        negative = (bits & np.uint32(0x80000000)) != 0
        return np.where(
            negative,
            np.bitwise_not(bits),
            bits | np.uint32(0x80000000),
        ).astype(np.uint64)

    left_ordered = ordered(left_bits)
    right_ordered = ordered(right_bits)
    distance = np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )
    return int(distance.max(initial=0))


def tensor_delta(
    reference: torch.Tensor | Sequence[float],
    candidate: torch.Tensor | Sequence[float],
) -> TensorDelta:
    """Compute exact mismatch locations and bounded numerical summaries."""

    left = torch.as_tensor(reference).detach().cpu().contiguous()
    right = torch.as_tensor(candidate).detach().cpu().contiguous()
    if left.shape != right.shape or left.dtype != right.dtype:
        raise ValueError("tensor schema differs")
    flat_left = left.flatten()
    flat_right = right.flatten()
    mismatch = torch.ne(flat_left, flat_right)
    mismatch_indices = tuple(
        int(index) for index in torch.nonzero(mismatch, as_tuple=False).flatten().tolist()
    )
    numeric_left = flat_left.to(torch.float64)
    numeric_right = flat_right.to(torch.float64)
    absolute = torch.abs(numeric_left - numeric_right)
    denominator = torch.maximum(
        torch.abs(numeric_left),
        torch.full_like(numeric_left, torch.finfo(torch.float64).tiny),
    )
    relative = absolute / denominator
    if flat_left.numel() == 0:
        cosine = 1.0
    elif torch.linalg.vector_norm(numeric_left) == 0 or torch.linalg.vector_norm(numeric_right) == 0:
        cosine = 1.0 if torch.equal(flat_left, flat_right) else 0.0
    else:
        cosine = float(
            torch.nn.functional.cosine_similarity(
                numeric_left.unsqueeze(0),
                numeric_right.unsqueeze(0),
            ).item()
        )
        cosine = min(1.0, max(-1.0, cosine))
    left_np = left.numpy()
    right_np = right.numpy()
    return TensorDelta(
        reference_sha256=stable_tensor_sha256(left),
        candidate_sha256=stable_tensor_sha256(right),
        mismatch_count=len(mismatch_indices),
        mismatch_indices=mismatch_indices,
        max_abs=float(absolute.max().item()) if absolute.numel() else 0.0,
        max_relative=float(relative.max().item()) if relative.numel() else 0.0,
        max_ulp=_maximum_float32_ulp(left_np, right_np),
        cosine_similarity=cosine,
    )


def select_boundary_cases(
    candidates: Sequence[BoundaryCandidate],
    *,
    count: int,
    excluded_case_ids: set[str] | frozenset[str],
) -> tuple[BoundaryCandidate, ...]:
    """Interleave nearest quantization and projection boundaries deterministically."""

    if count <= 0:
        raise ValueError("count must be positive")
    eligible_by_id: dict[str, BoundaryCandidate] = {}
    for item in candidates:
        if item.case.case_id not in excluded_case_ids:
            eligible_by_id[item.case.case_id] = item
    eligible = tuple(eligible_by_id.values())
    if len(eligible) < count:
        raise ValueError("not enough distinct boundary candidates")
    quant_ranked = sorted(
        eligible,
        key=lambda item: (
            item.minimum_quantization_margin,
            item.minimum_projection_margin,
            item.case.case_id,
        ),
    )
    projection_ranked = sorted(
        eligible,
        key=lambda item: (
            item.minimum_projection_margin,
            item.minimum_quantization_margin,
            item.case.case_id,
        ),
    )
    selected: dict[str, BoundaryCandidate] = {}
    index = 0
    while len(selected) < count:
        for ranked in (quant_ranked, projection_ranked):
            item = ranked[index]
            selected.setdefault(item.case.case_id, item)
            if len(selected) == count:
                break
        index += 1
    return tuple(sorted(selected.values(), key=lambda item: item.case.case_id))


def _expand_hmac(material: bytes, message: bytes, length: int) -> bytes:
    output = bytearray()
    counter = 0
    while len(output) < length:
        counter += 1
        output.extend(
            hmac.new(
                material,
                message + counter.to_bytes(4, "big"),
                hashlib.sha256,
            ).digest()
        )
    return bytes(output[:length])


def projection_sign_rows(
    material: bytes,
    *,
    rows: int,
    dimensions: int,
    domain: str,
) -> torch.Tensor:
    """Derive diagnostic-only random-hyperplane rows under a V4-only domain."""

    if not isinstance(material, bytes) or len(material) < 32:
        raise ValueError("diagnostic key material must contain at least 32 bytes")
    if not domain.startswith("v4-diagnostic/") or "\0" in domain:
        raise ValueError("projection requires V4 diagnostic domain separation")
    if rows <= 0 or dimensions <= 0:
        raise ValueError("rows and dimensions must be positive")
    values: list[list[int]] = []
    for row_index in range(rows):
        message = (
            _DIAGNOSTIC_HEADER
            + domain.encode("utf-8")
            + b"\0"
            + row_index.to_bytes(4, "big")
        )
        raw = _expand_hmac(material, message, dimensions)
        values.append([1 if byte & 1 else -1 for byte in raw])
    return torch.tensor(values, dtype=torch.int8)
