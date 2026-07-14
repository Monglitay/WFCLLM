#!/usr/bin/env python
"""Controlled numerical diagnosis for batch-dependent semantic replay V4."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import stat
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from wfcllm.diagnostics.numerical_replay_v4 import (  # noqa: E402
    BatchCapture,
    BoundaryCandidate,
    ConditionSpec,
    ContextCase,
    LayerCaptureRuntime,
    TensorDelta,
    build_condition_specs,
    capture_downstream_from_saved,
    compose_batch,
    first_divergent_layer,
    projection_sign_rows,
    quantization_boundary_margins,
    select_boundary_cases,
    stable_tensor_sha256,
    tensor_delta,
)
from wfcllm.dynamic_semantic.config import load_public_config  # noqa: E402
from wfcllm.dynamic_semantic.context import (  # noqa: E402
    CanonicalContext,
    DynamicContextExtractor,
)
from wfcllm.dynamic_semantic.encoder import SemanticEncoderRuntime  # noqa: E402
from wfcllm.dynamic_semantic.whitening import load_whitening  # noqa: E402


SCHEMA_VERSION = "wfcllm-v4-numerical-replay-diagnosis/v1"
FAILURE_UNITS = {
    "HumanEval/34": 3,
    "HumanEval/39": 6,
    "HumanEval/71": 2,
}


@dataclass(frozen=True)
class SyntheticContextSpec:
    case_id: str
    code: str
    expected_role_prefix: str


def synthetic_context_specs() -> tuple[SyntheticContextSpec, ...]:
    """Return public synthetic programs spanning AST types and token lengths."""

    return (
        SyntheticContextSpec("synthetic-return-short", "def f(x):\n    return x + 1\n", "Return"),
        SyntheticContextSpec("synthetic-assign-list", "def f(xs):\n    result = [value * 2 for value in xs if value > 0]\n", "Assign"),
        SyntheticContextSpec("synthetic-annassign", "def f():\n    total: int = 0\n", "AnnAssign"),
        SyntheticContextSpec("synthetic-augassign", "def f(x):\n    x = int(x)\n    x += 7\n", "AugAssign"),
        SyntheticContextSpec("synthetic-if", "def f(x):\n    if x > 0:\n        return x\n    else:\n        return -x\n", "If"),
        SyntheticContextSpec("synthetic-for", "def f(xs):\n    for index, value in enumerate(xs):\n        xs[index] = value + index\n", "For"),
        SyntheticContextSpec("synthetic-while", "def f(n):\n    while n > 1:\n        n = n // 2 if n % 2 == 0 else 3 * n + 1\n", "While"),
        SyntheticContextSpec("synthetic-try", "def f(text):\n    try:\n        return int(text)\n    except (TypeError, ValueError):\n        return 0\n", "Try"),
        SyntheticContextSpec("synthetic-with", "def f(path):\n    with open(path, encoding='utf-8') as handle:\n        return handle.read().strip()\n", "With"),
        SyntheticContextSpec("synthetic-match", "def f(value):\n    match value:\n        case {'kind': 'point', 'x': x, 'y': y}:\n            return x + y\n        case _:\n            return None\n", "Match"),
        SyntheticContextSpec("synthetic-assert", "def f(items):\n    assert all(isinstance(item, int) for item in items), 'integers required'\n", "Assert"),
        SyntheticContextSpec("synthetic-raise", "def f(message):\n    raise RuntimeError(f'operation failed: {message!r}')\n", "Raise"),
        SyntheticContextSpec("synthetic-import", "def f():\n    import collections.abc as collections_abc\n", "Import"),
        SyntheticContextSpec("synthetic-importfrom", "def f():\n    from itertools import combinations, permutations, product\n", "ImportFrom"),
        SyntheticContextSpec("synthetic-delete", "def f(mapping):\n    key = next(iter(mapping))\n    del mapping[key]\n", "Delete"),
        SyntheticContextSpec("synthetic-expr-call", "def f(logger, payload):\n    logger.info('payload=%s size=%d', payload, len(payload))\n", "Expr"),
        SyntheticContextSpec("synthetic-asyncfor", "async def f(stream):\n    async for item in stream:\n        await item.process()\n", "AsyncFor"),
        SyntheticContextSpec("synthetic-global", "def f():\n    global GLOBAL_DIAGNOSTIC_COUNTER\n", "Global"),
        SyntheticContextSpec("synthetic-nested-function", "def f(offset):\n    def inner(value: int = 3) -> int:\n        return value + offset\n", "FunctionDef"),
        SyntheticContextSpec("synthetic-pass", "def f():\n    pass\n", "Pass"),
        SyntheticContextSpec("synthetic-long-return", "def f(records):\n    return {str(item['id']): tuple(sorted(value for value in item['values'] if value is not None)) for item in records if item.get('enabled', True)}\n", "Return"),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="freeze diagnostic context cases")
    prepare.add_argument("--public-config", required=True)
    prepare.add_argument("--whitening", required=True)
    prepare.add_argument("--v3-experiment-dir", required=True)
    prepare.add_argument("--diagnostic-key-file", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--device", default="cuda")

    run = subparsers.add_parser("run", help="run one fresh-process numerical matrix")
    run.add_argument("--public-config", required=True)
    run.add_argument("--whitening", required=True)
    run.add_argument("--contexts", required=True)
    run.add_argument("--diagnostic-key-file", required=True)
    run.add_argument("--output-jsonl", required=True)
    run.add_argument("--restart-id", required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--profile",
        choices=("full", "cpu_reference", "runtime_flags"),
        default="full",
    )
    run.add_argument("--cpu-threads", type=int, default=1)
    run.add_argument("--limit-contexts", type=int)

    summarize = subparsers.add_parser("summarize", help="aggregate raw matrix records")
    summarize.add_argument("--contexts", required=True)
    summarize.add_argument("--input-jsonl", action="append", required=True)
    summarize.add_argument("--output-json", required=True)
    return parser


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        if raw:
            payload = json.loads(raw)
            if not isinstance(payload, dict):
                raise ValueError(f"JSONL row in {path} must be an object")
            rows.append(payload)
    return rows


def _load_diagnostic_key(path: str | Path) -> bytes:
    key_path = Path(path)
    info = key_path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("diagnostic key must be a regular file")
    if stat.S_IMODE(info.st_mode) != 0o600:
        raise ValueError("diagnostic key mode must be exactly 0600")
    material = key_path.read_bytes()
    if len(material) < 32:
        raise ValueError("diagnostic key must contain at least 32 bytes")
    return material


def _make_case(
    case_id: str,
    context: CanonicalContext,
    *,
    token_count: int,
    category: str,
) -> ContextCase:
    return ContextCase(
        case_id=case_id,
        serialized=context.serialized,
        context_sha256=context.context_sha256,
        role=context.role,
        token_count=token_count,
        category=category,
    )


def select_varied_controls(
    candidates: Iterable[ContextCase],
    *,
    count: int,
) -> tuple[ContextCase, ...]:
    """Select role-diverse shortest/longest controls deterministically."""

    unique = {item.case_id: item for item in candidates}
    ordered = sorted(unique.values(), key=lambda item: (item.token_count, item.role, item.case_id))
    if len(ordered) < count:
        raise ValueError("not enough distinct control contexts")
    selected: dict[str, ContextCase] = {}

    def add(item: ContextCase) -> None:
        if len(selected) < count:
            selected.setdefault(item.case_id, item)

    add(ordered[0])
    add(ordered[-1])
    by_role: dict[str, list[ContextCase]] = defaultdict(list)
    for item in ordered:
        by_role[item.role].append(item)
    for role in sorted(by_role):
        add(by_role[role][0])
        add(by_role[role][-1])
    if len(selected) < count:
        denominator = max(count - 1, 1)
        for index in range(count):
            position = round(index * (len(ordered) - 1) / denominator)
            add(ordered[position])
    for item in ordered:
        add(item)
    if len(selected) != count:
        raise AssertionError("control selection cardinality mismatch")
    return tuple(sorted(selected.values(), key=lambda item: item.case_id))


def serialize_delta(
    delta: TensorDelta,
    *,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    mismatch_limit: int = 32,
) -> dict[str, Any]:
    left = torch.as_tensor(reference).detach().cpu().flatten()
    right = torch.as_tensor(candidate).detach().cpu().flatten()
    coordinates = []
    for index in delta.mismatch_indices[:mismatch_limit]:
        coordinates.append(
            {
                "index": index,
                "reference": left[index].item(),
                "candidate": right[index].item(),
            }
        )
    return {
        "reference_sha256": delta.reference_sha256,
        "candidate_sha256": delta.candidate_sha256,
        "mismatch_count": delta.mismatch_count,
        "mismatch_indices_truncated": delta.mismatch_count > mismatch_limit,
        "mismatch_coordinates": coordinates,
        "max_abs": delta.max_abs,
        "max_relative": delta.max_relative,
        "max_ulp": delta.max_ulp,
        "cosine_similarity": delta.cosine_similarity,
    }


def _load_model_components(
    public_config: str,
    whitening_path: str,
    *,
    device: str,
):
    config = load_public_config(public_config)
    runtime = SemanticEncoderRuntime.load(
        config.encoder,
        max_tokens=config.context.max_context_tokens,
        device=device,
        fixed_batch_size=None,
        fixed_sequence_length=True,
    )
    whitening = load_whitening(whitening_path)
    extractor = DynamicContextExtractor(config.context, token_counter=runtime.token_count)
    return config, runtime, whitening, extractor


def _build_capture_runtime(
    config,
    runtime,
    whitening,
    diagnostic_material: bytes,
    *,
    device: str,
) -> LayerCaptureRuntime:
    signs = projection_sign_rows(
        diagnostic_material,
        rows=config.channel.projection_rows,
        dimensions=config.channel.whitening_dimensions,
        domain="v4-diagnostic/numerical-boundary-projection",
    )
    return LayerCaptureRuntime(
        model=runtime._model,
        tokenizer=runtime._tokenizer,
        whitening_mean=torch.tensor(whitening.mean, dtype=torch.float32),
        whitening_projection=torch.tensor(whitening.projection, dtype=torch.float32),
        projection_rows=signs,
        quantization_scale=config.channel.quantization_scale,
        device=device,
        max_tokens=config.context.max_context_tokens,
    )


def _extract_cases_from_selected(
    rows: list[dict[str, Any]],
    extractor: DynamicContextExtractor,
    token_count,
) -> dict[str, tuple[CanonicalContext, ...]]:
    result: dict[str, tuple[CanonicalContext, ...]] = {}
    for row in rows:
        sample_id = str(row["id"])
        extraction = extractor.extract(str(row["final_code"]))
        if not extraction.parse_ok:
            raise ValueError(f"selected V3 final code does not parse: {sample_id}")
        for context in extraction.contexts:
            if token_count(context.serialized) > 256:
                raise ValueError("extracted context exceeds frozen token budget")
        result[sample_id] = extraction.contexts
    return result


def _prepare(args: argparse.Namespace) -> int:
    config, runtime, whitening, extractor = _load_model_components(
        args.public_config,
        args.whitening,
        device=args.device,
    )
    diagnostic_material = _load_diagnostic_key(args.diagnostic_key_file)
    experiment_dir = Path(args.v3_experiment_dir)
    pilot_dir = experiment_dir / "pilot30_finalrepair"
    selected_rows = _read_jsonl(pilot_dir / "v3_final_code.jsonl")
    replay_rows = _read_jsonl(pilot_dir / "r3_replay.jsonl")
    replay_by_id = {str(row["id"]): row for row in replay_rows}
    contexts_by_id = _extract_cases_from_selected(
        selected_rows,
        extractor,
        runtime.token_count,
    )

    failure_cases: list[ContextCase] = []
    reserved_hashes: set[str] = set()
    for sample_id, unit_index in FAILURE_UNITS.items():
        contexts = contexts_by_id[sample_id]
        if unit_index >= len(contexts):
            raise ValueError(f"missing frozen mismatch context: {sample_id} unit {unit_index}")
        context = contexts[unit_index]
        case = _make_case(
            f"failure-{sample_id.replace('/', '-')}-unit-{unit_index}",
            context,
            token_count=runtime.token_count(context.serialized),
            category="failure",
        )
        failure_cases.append(case)
        reserved_hashes.add(case.context_sha256)

    control_candidates: list[ContextCase] = []
    for sample_id in sorted(contexts_by_id):
        if not bool(replay_by_id[sample_id]["exact_replay"]):
            continue
        for index, context in enumerate(contexts_by_id[sample_id]):
            if context.context_sha256 in reserved_hashes:
                continue
            control_candidates.append(
                _make_case(
                    f"control-{sample_id.replace('/', '-')}-unit-{index}",
                    context,
                    token_count=runtime.token_count(context.serialized),
                    category="control",
                )
            )
    controls = select_varied_controls(control_candidates, count=20)
    reserved_hashes.update(item.context_sha256 for item in controls)

    synthetic_cases: list[ContextCase] = []
    for spec in synthetic_context_specs()[:20]:
        extraction = extractor.extract(spec.code)
        matching = [
            context
            for context in extraction.contexts
            if context.role.startswith(spec.expected_role_prefix + "|")
        ]
        if not matching:
            raise ValueError(f"synthetic context role missing: {spec.case_id}")
        context = matching[-1]
        case = _make_case(
            spec.case_id,
            context,
            token_count=runtime.token_count(context.serialized),
            category="synthetic",
        )
        if case.context_sha256 in reserved_hashes:
            raise ValueError(f"synthetic context duplicates a reserved context: {spec.case_id}")
        synthetic_cases.append(case)
        reserved_hashes.add(case.context_sha256)

    boundary_sources: dict[str, ContextCase] = {}
    for sample_id, contexts in contexts_by_id.items():
        for index, context in enumerate(contexts):
            if context.context_sha256 in reserved_hashes:
                continue
            case = _make_case(
                f"boundary-source-{sample_id.replace('/', '-')}-unit-{index}",
                context,
                token_count=runtime.token_count(context.serialized),
                category="control",
            )
            boundary_sources.setdefault(case.context_sha256, case)
    if len(boundary_sources) < 20:
        raise ValueError("fewer than 20 independent boundary-source contexts")

    capture_runtime = _build_capture_runtime(
        config,
        runtime,
        whitening,
        diagnostic_material,
        device=args.device,
    )
    base_condition = replace(
        build_condition_specs(repeats=20)[0],
        repeats=1,
        warmup_count=0,
    )
    source_pool = tuple(boundary_sources.values())
    first_source = source_pool[0]
    capture_runtime.capture(
        compose_batch(
            first_source,
            source_pool,
            batch_size=1,
            composition="self_repeat",
            order="forward",
            seed=0,
        ),
        replace(base_condition, warmup_count=1),
    )
    boundary_candidates: list[BoundaryCandidate] = []
    for case in source_pool:
        batch = compose_batch(
            case,
            source_pool,
            batch_size=1,
            composition="self_repeat",
            order="forward",
            seed=0,
        )
        layers = capture_runtime.capture(batch, base_condition).target_layers[0]
        quant_margin = min(
            quantization_boundary_margins(
                layers["whitening_post_norm"],
                scale=config.channel.quantization_scale,
            )
        )
        projection_margin = float(
            torch.abs(layers["projection_dots"].to(torch.float64)).min().item()
        )
        boundary_candidates.append(
            BoundaryCandidate(
                case=case,
                minimum_quantization_margin=quant_margin,
                minimum_projection_margin=projection_margin,
            )
        )
    selected_boundaries = select_boundary_cases(
        boundary_candidates,
        count=20,
        excluded_case_ids=frozenset(),
    )
    boundaries = tuple(
        replace(item.case, category="boundary") for item in selected_boundaries
    )
    all_cases = tuple([*failure_cases, *controls, *synthetic_cases, *boundaries])
    counts = Counter(item.category for item in all_cases)
    expected_counts = {"failure": 3, "control": 20, "synthetic": 20, "boundary": 20}
    if dict(counts) != expected_counts:
        raise AssertionError(f"diagnostic context counts differ: {dict(counts)}")
    if len({item.context_sha256 for item in all_cases}) != len(all_cases):
        raise ValueError("diagnostic context set contains duplicate serialized contexts")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    context_path = output_dir / "contexts.jsonl"
    lines = [_canonical_json(asdict(item)) for item in all_cases]
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    context_sha256 = hashlib.sha256(context_path.read_bytes()).hexdigest()
    boundary_by_id = {item.case.case_id: item for item in selected_boundaries}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "context_manifest",
        "counts": expected_counts,
        "total_contexts": len(all_cases),
        "contexts_jsonl_sha256": context_sha256,
        "v3_failure_units": FAILURE_UNITS,
        "control_source_contract": "V3 final-repair exact-replay selected final codes",
        "synthetic_contract": "first 20 frozen synthetic_context_specs",
        "boundary_selection": {
            "method": "interleaved nearest quantization/projection diagnostic margins",
            "quantization_scale": config.channel.quantization_scale,
            "projection_rows": config.channel.projection_rows,
            "selected": [
                {
                    "case_id": case.case_id,
                    "minimum_quantization_margin_scaled": boundary_by_id[case.case_id].minimum_quantization_margin,
                    "minimum_integer_projection_margin": boundary_by_id[case.case_id].minimum_projection_margin,
                }
                for case in boundaries
            ],
        },
        "secret_metadata_included": False,
    }
    (output_dir / "context_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"prepared {len(all_cases)} contexts at {context_path}")
    return 0


def _load_context_cases(path: str | Path) -> tuple[ContextCase, ...]:
    cases = tuple(ContextCase(**row) for row in _read_jsonl(path))
    if len({item.case_id for item in cases}) != len(cases):
        raise ValueError("duplicate context case IDs")
    return cases


def _profile_conditions(profile: str) -> tuple[ConditionSpec, ...]:
    all_conditions = build_condition_specs(repeats=20)
    if profile == "full":
        return all_conditions
    if profile == "cpu_reference":
        return (replace(all_conditions[0], repeats=1),)
    if profile == "runtime_flags":
        allowed = {"reference", "deterministic", "tf32", "matmul_precision"}
        return tuple(
            replace(item, repeats=1) if item.reference else item
            for item in all_conditions
            if item.axis in allowed
        )
    raise ValueError(f"unsupported profile: {profile}")


def _validate_order_contracts(
    cases: tuple[ContextCase, ...],
    conditions: tuple[ConditionSpec, ...],
) -> None:
    order_conditions = [item for item in conditions if item.axis == "order"]
    if not order_conditions:
        return
    for case in cases:
        multisets = []
        for condition in order_conditions:
            batch = compose_batch(
                case,
                cases,
                batch_size=condition.batch_size,
                composition=condition.composition,
                order=condition.order,
                seed=20260714,
            )
            multisets.append(tuple(sorted(batch.contexts)))
        if len(set(multisets)) != 1:
            raise AssertionError("order conditions changed the batch multiset")


def _environment_metadata(
    *,
    args: argparse.Namespace,
    config,
    conditions: tuple[ConditionSpec, ...],
) -> dict[str, Any]:
    cuda_version = torch.version.cuda
    cudnn_version = torch.backends.cudnn.version()
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "process_metadata",
        "restart_id": args.restart_id,
        "profile": args.profile,
        "device": args.device,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_version": cuda_version,
        "cudnn_version": cudnn_version,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cpu_threads": torch.get_num_threads(),
        "interop_threads": torch.get_num_interop_threads(),
        "condition_count": len(conditions),
        "encoder_checkpoint_sha256": config.encoder.checkpoint_sha256,
        "secret_metadata_included": False,
    }


def _stage_deltas(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
) -> dict[str, dict[str, Any]]:
    if set(reference) != set(candidate):
        raise ValueError("stage schema differs before delta serialization")
    return {
        name: serialize_delta(
            tensor_delta(reference[name], candidate[name]),
            reference=reference[name],
            candidate=candidate[name],
        )
        for name in sorted(reference)
    }


def _discrete_payload(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    *,
    quantization_scale: int,
) -> dict[str, Any]:
    quantized = candidate["quantized"].to(torch.int64)
    reference_quantized = reference["quantized"].to(torch.int64)
    dots = candidate["projection_dots"].to(torch.int64)
    reference_dots = reference["projection_dots"].to(torch.int64)
    bits = candidate["signature_bits"].to(torch.int8)
    reference_bits = reference["signature_bits"].to(torch.int8)
    return {
        "quantization_scale": quantization_scale,
        "reference_quantized": reference_quantized.tolist(),
        "candidate_quantized": quantized.tolist(),
        "quantized_mismatch_count": int(torch.ne(reference_quantized, quantized).sum().item()),
        "quantization_boundary_margins_scaled": list(
            quantization_boundary_margins(
                candidate["whitening_post_norm"],
                scale=quantization_scale,
            )
        ),
        "reference_projection_dots": reference_dots.tolist(),
        "candidate_projection_dots": dots.tolist(),
        "projection_margins": torch.abs(dots).tolist(),
        "reference_signature_bits": reference_bits.tolist(),
        "candidate_signature_bits": bits.tolist(),
        "signature_mismatch_count": int(torch.ne(reference_bits, bits).sum().item()),
    }


def _trial_record(
    *,
    args: argparse.Namespace,
    case: ContextCase,
    condition: ConditionSpec,
    repeat_index: int,
    copy_index: int,
    target_row: int,
    elapsed_seconds: float,
    capture: BatchCapture,
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    isolated_reference: dict[str, torch.Tensor] | None,
    isolated_candidate: dict[str, torch.Tensor] | None,
    quantization_scale: int,
) -> dict[str, Any]:
    deltas = _stage_deltas(reference, candidate)
    isolated_deltas = None
    isolated_first = None
    if isolated_reference is not None and isolated_candidate is not None:
        isolated_deltas = _stage_deltas(isolated_reference, isolated_candidate)
        isolated_first = first_divergent_layer(isolated_reference, isolated_candidate)
    return {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "numerical_trial",
        "restart_id": args.restart_id,
        "profile": args.profile,
        "device": args.device,
        "case_id": case.case_id,
        "category": case.category,
        "context_sha256": case.context_sha256,
        "role": case.role,
        "token_count": case.token_count,
        "condition": asdict(condition),
        "repeat_index": repeat_index,
        "copy_index": copy_index,
        "target_row": target_row,
        "physical_batch_size": len(capture.batch_member_sha256),
        "sequence_length": capture.sequence_length,
        "batch_member_context_sha256": list(capture.batch_member_sha256),
        "batch_multiset_sha256": hashlib.sha256(
            "\n".join(sorted(capture.batch_member_sha256)).encode("ascii")
        ).hexdigest(),
        "model_eval": capture.model_eval,
        "effective_flags": capture.effective_flags,
        "elapsed_seconds": elapsed_seconds,
        "first_divergent_layer": first_divergent_layer(reference, candidate),
        "stage_deltas": deltas,
        "downstream_isolated_first_divergent_layer": isolated_first,
        "downstream_isolated_stage_deltas": isolated_deltas,
        "discrete": _discrete_payload(
            reference,
            candidate,
            quantization_scale=quantization_scale,
        ),
    }


def _downstream_subset(layers: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    names = {
        "projection_pre_norm",
        "model_post_norm",
        "runtime_post_norm",
        "cpu_centered",
        "whitening_pre_norm",
        "whitening_post_norm",
        "quantized",
        "projection_dots",
        "signature_bits",
    }
    return {name: layers[name] for name in names}


def _run(args: argparse.Namespace) -> int:
    if args.cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cases = _load_context_cases(args.contexts)
    if args.limit_contexts is not None:
        if args.limit_contexts <= 0:
            raise ValueError("limit_contexts must be positive")
        cases = cases[: args.limit_contexts]
    conditions = _profile_conditions(args.profile)
    _validate_order_contracts(cases, conditions)
    config, runtime, whitening, _extractor = _load_model_components(
        args.public_config,
        args.whitening,
        device=args.device,
    )
    diagnostic_material = _load_diagnostic_key(args.diagnostic_key_file)
    capture_runtime = _build_capture_runtime(
        config,
        runtime,
        whitening,
        diagnostic_material,
        device=args.device,
    )
    signs = projection_sign_rows(
        diagnostic_material,
        rows=config.channel.projection_rows,
        dimensions=config.channel.whitening_dimensions,
        domain="v4-diagnostic/numerical-boundary-projection",
    )
    mean = torch.tensor(whitening.mean, dtype=torch.float32)
    whitening_projection = torch.tensor(whitening.projection, dtype=torch.float32)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        raise ValueError("output JSONL already exists; raw trials are immutable")

    reference_condition = next(item for item in conditions if item.reference)
    references: dict[str, dict[str, torch.Tensor]] = {}
    case_by_serialized = {item.serialized: item for item in cases}
    with output_path.open("x", encoding="utf-8", newline="\n") as output:
        output.write(_canonical_json(_environment_metadata(args=args, config=config, conditions=conditions)) + "\n")
        for case_number, case in enumerate(cases, start=1):
            batch = compose_batch(
                case,
                cases,
                batch_size=1,
                composition="self_repeat",
                order="forward",
                seed=20260714,
            )
            baseline: dict[str, torch.Tensor] | None = None
            for repeat_index in range(reference_condition.repeats):
                effective_condition = (
                    reference_condition
                    if repeat_index == 0
                    else replace(reference_condition, warmup_count=0)
                )
                started = time.perf_counter()
                capture = capture_runtime.capture(batch, effective_condition)
                elapsed = time.perf_counter() - started
                candidate = capture.target_layers[0]
                if baseline is None:
                    baseline = {name: value.clone() for name, value in candidate.items()}
                    references[case.case_id] = baseline
                record = _trial_record(
                    args=args,
                    case=case,
                    condition=reference_condition,
                    repeat_index=repeat_index,
                    copy_index=0,
                    target_row=0,
                    elapsed_seconds=elapsed,
                    capture=capture,
                    reference=baseline,
                    candidate=candidate,
                    isolated_reference=None,
                    isolated_candidate=None,
                    quantization_scale=config.channel.quantization_scale,
                )
                output.write(_canonical_json(record) + "\n")
            print(f"reference {case_number}/{len(cases)} {case.case_id}", flush=True)

        nonreference = [item for item in conditions if not item.reference]
        for case_number, case in enumerate(cases, start=1):
            reference = references[case.case_id]
            isolated_reference = _downstream_subset(reference)
            for condition in nonreference:
                seed = int.from_bytes(
                    hashlib.sha256(
                        f"{case.case_id}\0{condition.condition_id}".encode("utf-8")
                    ).digest()[:8],
                    "big",
                )
                batch = compose_batch(
                    case,
                    cases,
                    batch_size=condition.batch_size,
                    composition=condition.composition,
                    order=condition.order,
                    seed=seed,
                )
                started = time.perf_counter()
                capture = capture_runtime.capture(batch, condition)
                elapsed = time.perf_counter() - started
                saved_projected = torch.stack(
                    [
                        references[case_by_serialized[text].case_id]["projection_pre_norm"]
                        for text in batch.contexts
                    ]
                )
                isolated_layers = capture_downstream_from_saved(
                    saved_projected,
                    target_indices=batch.target_indices,
                    normalization_device=args.device,
                    whitening_mean=mean,
                    whitening_projection=whitening_projection,
                    projection_rows=signs,
                    quantization_scale=config.channel.quantization_scale,
                )
                for copy_index, (target_row, candidate, isolated_candidate) in enumerate(
                    zip(
                        batch.target_indices,
                        capture.target_layers,
                        isolated_layers,
                        strict=True,
                    )
                ):
                    record = _trial_record(
                        args=args,
                        case=case,
                        condition=condition,
                        repeat_index=0,
                        copy_index=copy_index,
                        target_row=target_row,
                        elapsed_seconds=elapsed,
                        capture=capture,
                        reference=reference,
                        candidate=candidate,
                        isolated_reference=isolated_reference,
                        isolated_candidate=isolated_candidate,
                        quantization_scale=config.channel.quantization_scale,
                    )
                    output.write(_canonical_json(record) + "\n")
            print(f"matrix {case_number}/{len(cases)} {case.case_id}", flush=True)
    print(f"wrote immutable raw matrix {output_path}")
    return 0


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def condition_value_label(condition: dict[str, Any]) -> str:
    """Return an explicit label for the one controlled value on a matrix axis."""

    axis = str(condition.get("axis"))
    fields_by_axis = {
        "reference": ("reference",),
        "batch_size": ("batch_size",),
        "composition": ("batch_size", "composition"),
        "order": ("order",),
        "padding": ("padding",),
        "grad_mode": ("grad_mode",),
        "deterministic": ("deterministic_algorithms",),
        "tf32": ("tf32",),
        "matmul_precision": ("matmul_precision",),
        "warmup": ("warmup_count",),
        "masked_tail": ("masked_tail_variant",),
    }
    if axis not in fields_by_axis:
        raise ValueError(f"unknown controlled matrix axis: {axis}")
    parts = []
    for field in fields_by_axis[axis]:
        if field not in condition:
            raise ValueError(f"condition axis {axis} is missing {field}")
        value = condition[field]
        if isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        parts.append(f"{field}={rendered}")
    return "|".join(parts)


def strict_candidate_fingerprint(row: dict[str, Any]) -> tuple[Any, ...]:
    """Fingerprint every stored continuous-stage hash and final discrete field."""

    stage_hashes = tuple(
        (str(stage), str(delta["candidate_sha256"]))
        for stage, delta in sorted(row["stage_deltas"].items())
    )
    isolated = row.get("downstream_isolated_stage_deltas")
    isolated_hashes = (
        tuple(
            (str(stage), str(delta["candidate_sha256"]))
            for stage, delta in sorted(isolated.items())
        )
        if isinstance(isolated, dict)
        else ()
    )
    discrete = row["discrete"]
    return (
        stage_hashes,
        isolated_hashes,
        tuple(int(value) for value in discrete["candidate_quantized"]),
        tuple(int(value) for value in discrete["candidate_projection_dots"]),
        tuple(int(value) for value in discrete["candidate_signature_bits"]),
    )


def _stats_payload(counts: Counter) -> dict[str, int | float | None]:
    trials = int(counts["trials"])
    return {
        **{key: int(value) for key, value in counts.items()},
        "exact_rate": _rate(int(counts["exact"]), trials),
        "quantized_mismatch_rate": _rate(
            int(counts["quantized_mismatch"]), trials
        ),
        "signature_mismatch_rate": _rate(
            int(counts["signature_mismatch"]), trials
        ),
    }


def classify_root_cause_trials(trials: Iterable[dict[str, Any]]) -> str:
    """Classify first-divergence and saved-tensor downstream isolation evidence."""

    unexpected_input = 0
    encoder_first = 0
    downstream_only = 0
    isolated_downstream = 0
    for row in trials:
        first = row.get("first_divergent_layer")
        axis = row.get("condition", {}).get("axis")
        if isinstance(row.get("downstream_isolated_first_divergent_layer"), str):
            isolated_downstream += 1
        if first in {"input_ids", "attention_mask"}:
            if axis != "masked_tail":
                unexpected_input += 1
            continue
        if not isinstance(first, str):
            continue
        if first.startswith("t5_block_") or first in {
            "t5_cls_hidden",
            "projection_pre_norm",
        }:
            encoder_first += 1
        elif first in {
            "model_post_norm",
            "runtime_post_norm",
            "cpu_centered",
            "whitening_pre_norm",
            "whitening_post_norm",
            "quantized",
            "projection_dots",
            "signature_bits",
        }:
            downstream_only += 1
    if unexpected_input:
        return "A3-input-construction"
    if encoder_first and isolated_downstream:
        return "A2-encoder-and-downstream-batch-dependence"
    if encoder_first:
        return "A1-batch-dependent-encoder-supported"
    if downstream_only:
        return "A2-downstream-numerics-primary"
    return "A2-or-A3-no-observed-numeric-divergence"


def _summarize(args: argparse.Namespace) -> int:
    cases = _load_context_cases(args.contexts)
    metadata: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    for path in args.input_jsonl:
        for row in _read_jsonl(path):
            if row.get("artifact_type") == "process_metadata":
                metadata.append(row)
            elif row.get("artifact_type") == "numerical_trial":
                trials.append(row)
            else:
                raise ValueError(f"unknown raw artifact row in {path}")
    metadata_by_restart = {str(row["restart_id"]): row for row in metadata}
    if len(metadata_by_restart) != len(metadata):
        raise ValueError("duplicate process restart_id in raw metadata")
    unique_keys = set()
    for row in trials:
        key = (
            row["restart_id"],
            row["case_id"],
            row["condition"]["condition_id"],
            row["repeat_index"],
            row["copy_index"],
        )
        if key in unique_keys:
            raise ValueError(f"duplicate numerical trial key: {key}")
        unique_keys.add(key)

    axis_totals: dict[str, Counter] = defaultdict(Counter)
    condition_value_totals: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    cuda_full_condition_value_totals: dict[str, dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    process_totals: dict[str, Counter] = defaultdict(Counter)
    first_layers = Counter()
    downstream_first = Counter()
    maximum_abs_by_stage: dict[str, float] = defaultdict(float)
    failure_details: dict[str, Counter] = defaultdict(Counter)
    same_process_total = 0
    same_process_exact = 0
    for row in trials:
        axis = str(row["condition"]["axis"])
        first = row["first_divergent_layer"]
        quant_mismatch = int(row["discrete"]["quantized_mismatch_count"])
        signature_mismatch = int(row["discrete"]["signature_mismatch_count"])
        value_label = condition_value_label(row["condition"])
        counters = [
            axis_totals[axis],
            condition_value_totals[axis][value_label],
            process_totals[str(row["restart_id"])],
        ]
        if row["device"] == "cuda" and row["profile"] == "full":
            counters.append(cuda_full_condition_value_totals[axis][value_label])
        for counts in counters:
            counts["trials"] += 1
            counts["exact"] += int(first is None)
            counts["quantized_mismatch"] += int(quant_mismatch > 0)
            counts["signature_mismatch"] += int(signature_mismatch > 0)
        first_layers[str(first)] += 1
        downstream_first[str(row["downstream_isolated_first_divergent_layer"])] += 1
        for stage, delta in row["stage_deltas"].items():
            maximum_abs_by_stage[stage] = max(
                maximum_abs_by_stage[stage],
                float(delta["max_abs"]),
            )
        if row["category"] == "failure":
            failure_details[row["case_id"]]["trials"] += 1
            failure_details[row["case_id"]]["exact"] += int(first is None)
            failure_details[row["case_id"]]["quantized_mismatch"] += int(quant_mismatch > 0)
        if axis == "reference" and int(row["repeat_index"]) > 0:
            same_process_total += 1
            same_process_exact += int(first is None)

    reference_rows = [
        row
        for row in trials
        if row["condition"]["axis"] == "reference"
        and int(row["repeat_index"]) == 0
        and int(row["copy_index"]) == 0
    ]
    cross_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        if row["device"] == "cuda" and row["profile"] == "full":
            cross_groups[row["case_id"]].append(row)
    cross_total = 0
    cross_exact = 0
    for rows in cross_groups.values():
        if len(rows) < 2:
            continue
        baseline = rows[0]
        baseline_hashes = {
            stage: delta["candidate_sha256"]
            for stage, delta in baseline["stage_deltas"].items()
        }
        for row in rows[1:]:
            hashes = {
                stage: delta["candidate_sha256"]
                for stage, delta in row["stage_deltas"].items()
            }
            cross_total += 1
            cross_exact += int(hashes == baseline_hashes)

    within_copy_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    cold_condition_groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in trials:
        within_copy_groups[
            (
                row["restart_id"],
                row["case_id"],
                row["condition"]["condition_id"],
                row["repeat_index"],
            )
        ].append(row)
        if row["device"] == "cuda" and row["profile"] == "full":
            cold_condition_groups[
                (
                    row["case_id"],
                    row["condition"]["condition_id"],
                    row["repeat_index"],
                    row["copy_index"],
                )
            ].append(row)

    within_copy_total = 0
    within_copy_exact = 0
    for rows in within_copy_groups.values():
        if len(rows) < 2:
            continue
        ordered = sorted(rows, key=lambda item: int(item["copy_index"]))
        baseline = strict_candidate_fingerprint(ordered[0])
        for row in ordered[1:]:
            within_copy_total += 1
            within_copy_exact += int(strict_candidate_fingerprint(row) == baseline)

    cold_restart_count = len(
        {
            row["restart_id"]
            for row in trials
            if row["device"] == "cuda" and row["profile"] == "full"
        }
    )
    cold_condition_total = 0
    cold_condition_exact = 0
    for rows in cold_condition_groups.values():
        if len({row["restart_id"] for row in rows}) != cold_restart_count:
            continue
        cold_condition_total += 1
        fingerprints = {strict_candidate_fingerprint(row) for row in rows}
        cold_condition_exact += int(len(fingerprints) == 1)

    device_reference_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reference_rows:
        device_reference_groups[str(row["case_id"])].append(row)
    cpu_gpu_total = 0
    cpu_gpu_quantized_exact = 0
    cpu_gpu_signature_exact = 0
    for rows in device_reference_groups.values():
        cpu_rows = [row for row in rows if row["device"] == "cpu"]
        gpu_rows = [
            row
            for row in rows
            if row["device"] == "cuda" and row["profile"] == "full"
        ]
        if not cpu_rows or not gpu_rows:
            continue
        cpu_discrete = cpu_rows[0]["discrete"]
        gpu_discrete = gpu_rows[0]["discrete"]
        cpu_gpu_total += 1
        cpu_gpu_quantized_exact += int(
            cpu_discrete["candidate_quantized"] == gpu_discrete["candidate_quantized"]
        )
        cpu_gpu_signature_exact += int(
            cpu_discrete["candidate_signature_bits"]
            == gpu_discrete["candidate_signature_bits"]
        )

    classification = classify_root_cause_trials(trials)

    axes_present = sorted(axis_totals)
    cuda_full_restarts = sorted(
        {
            str(item["restart_id"])
            for item in metadata
            if item["device"] == "cuda" and item["profile"] == "full"
        }
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "artifact_type": "root_cause_matrix_summary",
        "context_counts": dict(Counter(item.category for item in cases)),
        "total_contexts": len(cases),
        "raw_processes": metadata,
        "trial_count": len(trials),
        "axes_present": axes_present,
        "axis_summary": {
            axis: _stats_payload(counts)
            for axis, counts in sorted(axis_totals.items())
        },
        "condition_value_summary": {
            axis: {
                label: _stats_payload(counts)
                for label, counts in sorted(value_counts.items())
            }
            for axis, value_counts in sorted(condition_value_totals.items())
        },
        "cuda_full_condition_value_summary": {
            axis: {
                label: _stats_payload(counts)
                for label, counts in sorted(value_counts.items())
            }
            for axis, value_counts in sorted(cuda_full_condition_value_totals.items())
        },
        "process_summary": {
            restart_id: {
                **_stats_payload(counts),
                "device": str(metadata_by_restart[restart_id]["device"]),
                "profile": str(metadata_by_restart[restart_id]["profile"]),
                "cublas_workspace_config": metadata_by_restart[restart_id][
                    "cublas_workspace_config"
                ],
            }
            for restart_id, counts in sorted(process_totals.items())
        },
        "first_divergent_layer_counts": dict(sorted(first_layers.items())),
        "downstream_isolated_first_divergent_counts": dict(
            sorted(downstream_first.items())
        ),
        "maximum_observed_absolute_difference_by_stage": dict(
            sorted(maximum_abs_by_stage.items())
        ),
        "same_process_repeat_exact": {
            "exact": same_process_exact,
            "total": same_process_total,
            "rate": _rate(same_process_exact, same_process_total),
        },
        "cross_cold_process_reference_exact": {
            "exact": cross_exact,
            "total": cross_total,
            "rate": _rate(cross_exact, cross_total),
            "cuda_full_restart_ids": cuda_full_restarts,
        },
        "within_batch_copy_strict_exact": {
            "exact": within_copy_exact,
            "total": within_copy_total,
            "rate": _rate(within_copy_exact, within_copy_total),
        },
        "cross_cold_process_all_conditions_strict_exact": {
            "exact": cold_condition_exact,
            "total": cold_condition_total,
            "rate": _rate(cold_condition_exact, cold_condition_total),
            "required_restart_count": cold_restart_count,
        },
        "cpu_gpu_reference_discrete_exact": {
            "total": cpu_gpu_total,
            "quantized_exact": cpu_gpu_quantized_exact,
            "quantized_rate": _rate(cpu_gpu_quantized_exact, cpu_gpu_total),
            "signature_exact": cpu_gpu_signature_exact,
            "signature_rate": _rate(cpu_gpu_signature_exact, cpu_gpu_total),
        },
        "failure_context_summary": {
            case_id: dict(counts) for case_id, counts in sorted(failure_details.items())
        },
        "coverage": {
            "batch_sizes": sorted(
                {int(row["condition"]["batch_size"]) for row in trials}
            ),
            "compositions": sorted(
                {str(row["condition"]["composition"]) for row in trials}
            ),
            "orders": sorted({str(row["condition"]["order"]) for row in trials}),
            "padding": sorted({str(row["condition"]["padding"]) for row in trials}),
            "grad_modes": sorted(
                {str(row["condition"]["grad_mode"]) for row in trials}
            ),
            "deterministic_values": sorted(
                {bool(row["condition"]["deterministic_algorithms"]) for row in trials}
            ),
            "tf32_values": sorted(
                {bool(row["condition"]["tf32"]) for row in trials}
            ),
            "matmul_precisions": sorted(
                {str(row["condition"]["matmul_precision"]) for row in trials}
            ),
            "warmup_counts": sorted(
                {int(row["condition"]["warmup_count"]) for row in trials}
            ),
            "cold_cuda_full_restart_count": len(cuda_full_restarts),
            "cpu_reference_present": any(item["device"] == "cpu" for item in metadata),
        },
        "provisional_classification": classification,
        "bound_scope": "empirical observed matrix only; not a certified universal bound",
        "secret_metadata_included": False,
    }
    output_path = Path(args.output_json)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"summarized {len(trials)} trials into {output_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        return _prepare(args)
    if args.command == "run":
        return _run(args)
    if args.command == "summarize":
        return _summarize(args)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
