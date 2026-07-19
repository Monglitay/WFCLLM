"""Reproducible, grouped hard validation for formal gate bundles."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import pickle
import random
from pathlib import Path
import re
from typing import Any, Protocol

import torch

from wfcllm.gate.config import GateValidateConfig

GATE_VALIDATION_CONTRACT_VERSION = "wfcllm-gate-validation/v1"
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_WORKER_POLL_TIMEOUT_SECONDS = 30.0
_WORKER_JOIN_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True)
class GateValidationArtifacts:
    float_model_path: Path
    int8_model_path: Path
    float_model_sha256: str
    int8_model_sha256: str

    def __post_init__(self) -> None:
        for name in ("float_model_path", "int8_model_path"):
            path = getattr(self, name)
            if (
                not isinstance(path, Path)
                or _has_symlink_component(path)
                or not path.is_file()
            ):
                raise ValueError(f"{name} must be a non-symlink file")
        for name in ("float_model_sha256", "int8_model_sha256"):
            value = getattr(self, name)
            if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
                raise ValueError(f"{name} must be a SHA-256 digest")
        if self.float_model_path == self.int8_model_path:
            raise ValueError("float and int8 artifacts must be distinct files")

    def for_precision(self, precision: str) -> tuple[Path, str]:
        if precision == "float":
            return self.float_model_path, self.float_model_sha256
        if precision == "int8":
            return self.int8_model_path, self.int8_model_sha256
        raise ValueError("precision must be float or int8")


@dataclass(frozen=True)
class GateValidationExample:
    example_id: str
    group_id: str
    serialized_input: str
    span: tuple[int, int]
    close_target: bool
    suitable_target: bool
    validation_role: str

    def __post_init__(self) -> None:
        for name in ("example_id", "group_id", "serialized_input"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        if (
            not isinstance(self.span, tuple)
            or len(self.span) != 2
            or any(type(item) is not int for item in self.span)
            or self.span[0] < 0
            or self.span[1] <= self.span[0]
        ):
            raise ValueError("span must be an increasing non-negative integer pair")
        for name in ("close_target", "suitable_target"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be a bool")
        if self.validation_role not in {"threshold_fit", "agreement"}:
            raise ValueError(
                "validation_role must be threshold_fit or agreement; training, "
                "detector, and calibration rows are forbidden"
            )


@dataclass(frozen=True)
class GateValidationThresholds:
    close_low: float
    close_high: float
    suitable_accept: float
    max_probability_delta: float
    max_tokens: int
    threshold_fit_group_digest: str

    def validate(self) -> None:
        for name in (
            "close_low",
            "close_high",
            "suitable_accept",
            "max_probability_delta",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or not 0.0 <= value <= 1.0
            ):
                raise ValueError(f"{name} must be a probability in [0, 1]")
        if self.close_low >= self.close_high:
            raise ValueError("close thresholds must leave a non-empty uncertain region")
        if type(self.max_tokens) is not int or not 1 <= self.max_tokens <= 512:
            raise ValueError("max_tokens must be an integer from 1 through 512")
        if (
            not isinstance(self.threshold_fit_group_digest, str)
            or len(self.threshold_fit_group_digest) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.threshold_fit_group_digest
            )
        ):
            raise ValueError("threshold_fit_group_digest must be a SHA-256 digest")


@dataclass(frozen=True)
class GateValidationMode:
    device: str
    precision: str
    batch_size: int
    order: str

    @property
    def name(self) -> str:
        return f"{self.device}-{self.precision}-batch-{self.batch_size}-{self.order}"


@dataclass(frozen=True)
class GateValidationSummary:
    contract_version: str
    validated: bool
    decision_agreement: float
    float_quantized_accepted_set_agreement: float
    formal_accepted_span_consensus: float
    suitable_false_positive_rate: float
    gpu_status: str
    validation_scope: str
    mode_names: tuple[str, ...]
    threshold_fit_group_digest: str
    agreement_group_digest: str
    thresholds: GateValidationThresholds
    forced_uncertain_count: int
    formal_accepted_spans: tuple[tuple[str, str, int, int], ...]
    process_attestations: tuple[GateProcessAttestation, ...]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode_names"] = list(self.mode_names)
        value["formal_accepted_spans"] = [
            list(span) for span in self.formal_accepted_spans
        ]
        value["process_attestations"] = [
            {
                **asdict(attestation),
                "pids": list(attestation.pids),
            }
            for attestation in self.process_attestations
        ]
        return value

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(), allow_nan=False, sort_keys=True, separators=(",", ":")
        ) + "\n"


class ValidationPredictor(Protocol):
    model: torch.nn.Module

    def encode_batch(
        self, examples: Sequence[GateValidationExample]
    ) -> Mapping[str, torch.Tensor]: ...


@dataclass(frozen=True)
class GateProcessAttestation:
    mode_name: str
    pids: tuple[int, ...]
    device: str
    precision: str
    artifact_name: str
    artifact_sha256: str


PredictorFactory = Callable[
    [GateValidationMode, Mapping[str, Any]], ValidationPredictor
]


class GateValidator:
    """Run the configured minimal float/int8 validation matrix."""

    def __init__(self, config: GateValidateConfig) -> None:
        if not isinstance(config, GateValidateConfig):
            raise ValueError("config must be a GateValidateConfig")
        self.config = config

    def validate(
        self,
        *,
        threshold_fit_examples: Sequence[GateValidationExample],
        agreement_examples: Sequence[GateValidationExample],
        max_tokens: int,
        predictor_factory: PredictorFactory,
        gpu_available: bool,
    ) -> GateValidationSummary:
        if type(max_tokens) is not int or not 1 <= max_tokens <= 512:
            raise ValueError("max_tokens must be an integer from 1 through 512")
        if not callable(predictor_factory):
            raise ValueError("predictor_factory must be callable")
        artifacts = getattr(predictor_factory, "artifacts", None)
        if not isinstance(artifacts, GateValidationArtifacts):
            raise ValueError(
                "predictor_factory must bind verified GateValidationArtifacts"
            )
        if type(gpu_available) is not bool:
            raise ValueError("gpu_available must be a bool")
        if gpu_available != torch.cuda.is_available():
            raise ValueError("gpu_available must match the actual CUDA runtime")
        fit = _snapshot("threshold_fit_examples", threshold_fit_examples)
        agreement = _snapshot("agreement_examples", agreement_examples)
        if any(item.validation_role != "threshold_fit" for item in fit):
            raise ValueError("threshold-fit subset contains a non-fit validation role")
        if any(item.validation_role != "agreement" for item in agreement):
            raise ValueError("agreement subset contains a non-agreement validation role")
        _validate_grouped("threshold-fit", fit)
        _validate_grouped("agreement", agreement)
        fit_groups = {item.group_id for item in fit}
        agreement_groups = {item.group_id for item in agreement}
        if fit_groups & agreement_groups:
            raise ValueError("threshold-fit and agreement groups overlap")
        _validate_unique_ids((*fit, *agreement))

        thresholds = _fit_thresholds(
            fit,
            fit_groups=fit_groups,
            max_tokens=max_tokens,
            predictor_factory=predictor_factory,
            artifacts=artifacts,
        )

        modes = _modes(self.config, gpu_available=gpu_available)
        runs: dict[tuple[str, int], dict[str, tuple[float, float]]] = {}
        process_attestations: list[GateProcessAttestation] = []
        for mode in modes:
            mode_runs, attestation = _run_mode_reloads(
                predictor_factory,
                artifacts,
                agreement,
                mode,
            )
            process_attestations.append(attestation)
            for reload_index, predictions in enumerate(mode_runs):
                runs[(mode.name, reload_index)] = predictions

        baseline_key = (modes[0].name, 0)
        baseline = runs[baseline_key]
        baseline_decisions = {
            key: _decision(value, thresholds) for key, value in baseline.items()
        }
        total = 0
        agreeing = 0
        accepted_sets: dict[tuple[str, int], frozenset[str]] = {}
        examples_by_id = {item.example_id: item for item in agreement}
        for run_key, predictions in runs.items():
            decisions = {
                key: _decision(value, thresholds) for key, value in predictions.items()
            }
            for example_id in baseline_decisions:
                total += 1
                agreeing += decisions[example_id] == baseline_decisions[example_id]
            accepted = frozenset(
                example_id
                for example_id, decision in decisions.items()
                if decision[0] == "close" and decision[1]
            )
            accepted_sets[run_key] = accepted
        decision_agreement = agreeing / total

        paired_total = 0
        paired_equal = 0
        for mode in modes:
            if mode.device != "cpu" or mode.precision != "float":
                continue
            int8_name = GateValidationMode(
                "cpu", "int8", mode.batch_size, mode.order
            ).name
            for reload_index in range(self.config.independent_reloads):
                left = accepted_sets[(mode.name, reload_index)]
                right = accepted_sets[(int8_name, reload_index)]
                paired_total += 1
                paired_equal += left == right
        accepted_agreement = paired_equal / paired_total

        # Each float/int8 runtime pair independently applies the exact
        # StableGatePredictor contract. Only then are the per-pair formal span
        # sets compared. No global pre-filter is copied across runs.
        formal_span_sets: list[frozenset[tuple[str, str, int, int]]] = []
        forced_ids: set[str] = set()
        for mode in modes:
            if mode.precision != "float":
                continue
            int8_name = GateValidationMode(
                "cpu", "int8", mode.batch_size, mode.order
            ).name
            for reload_index in range(self.config.independent_reloads):
                float_run = runs[(mode.name, reload_index)]
                int8_run = runs[(int8_name, reload_index)]
                accepted_spans: set[tuple[str, str, int, int]] = set()
                for example in agreement:
                    float_value = float_run[example.example_id]
                    int8_value = int8_run[example.example_id]
                    stable = (
                        _decision(float_value, thresholds)
                        == _decision(int8_value, thresholds)
                        and max(
                            abs(float_value[0] - int8_value[0]),
                            abs(float_value[1] - int8_value[1]),
                        )
                        <= thresholds.max_probability_delta
                    )
                    if not stable:
                        forced_ids.add(example.example_id)
                        continue
                    decision = _decision(int8_value, thresholds)
                    if decision[0] == "close" and decision[1]:
                        accepted_spans.add(
                            (
                                example.group_id,
                                example.example_id,
                                example.span[0],
                                example.span[1],
                            )
                        )
                formal_span_sets.append(frozenset(accepted_spans))
        if not formal_span_sets:
            raise ValueError("validation matrix produced no float/int8 runtime pairs")
        union = frozenset().union(*formal_span_sets)
        intersection = frozenset.intersection(*formal_span_sets)
        span_consensus = 0.0 if not union else len(intersection) / len(union)
        formal_spans = tuple(sorted(intersection))
        negatives = [item for item in agreement if not item.suitable_target]
        if not negatives:
            raise ValueError("agreement subset needs suitable negatives")
        suitable_fpr = max(
            sum(
                predictions[item.example_id][1] >= thresholds.suitable_accept
                for item in negatives
            )
            / len(negatives)
            for predictions in runs.values()
        )

        validated = (
            decision_agreement >= self.config.decision_agreement_min
            and accepted_agreement
            >= self.config.float_quantized_accepted_set_agreement_min
            and span_consensus >= self.config.formal_accepted_span_consensus_min
            and bool(formal_spans)
            and suitable_fpr <= self.config.suitable_false_positive_rate_max
        )
        return GateValidationSummary(
            contract_version=GATE_VALIDATION_CONTRACT_VERSION,
            validated=validated,
            decision_agreement=decision_agreement,
            float_quantized_accepted_set_agreement=accepted_agreement,
            formal_accepted_span_consensus=span_consensus,
            suitable_false_positive_rate=suitable_fpr,
            gpu_status=(
                "validated"
                if gpu_available and self.config.gpu_float_if_available
                else "skipped_fast_contract"
                if gpu_available
                else "not_available"
            ),
            validation_scope="cpu+gpu-float" if gpu_available else "cpu",
            mode_names=tuple(mode.name for mode in modes),
            threshold_fit_group_digest=_group_digest(fit_groups),
            agreement_group_digest=_group_digest(agreement_groups),
            thresholds=thresholds,
            forced_uncertain_count=len(forced_ids),
            formal_accepted_spans=formal_spans,
            process_attestations=tuple(process_attestations),
        )


def _snapshot(
    name: str, values: Sequence[GateValidationExample]
) -> tuple[GateValidationExample, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError(f"{name} must be a sequence")
    result = tuple(values)
    if not result or any(not isinstance(item, GateValidationExample) for item in result):
        raise ValueError(f"{name} must contain GateValidationExample values")
    return result


def _validate_unique_ids(values: Sequence[GateValidationExample]) -> None:
    ids = [item.example_id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("example_id values must be globally unique")


def _validate_grouped(name: str, values: Sequence[GateValidationExample]) -> None:
    counts: dict[str, int] = {}
    for item in values:
        counts[item.group_id] = counts.get(item.group_id, 0) + 1
    if any(count < 2 for count in counts.values()):
        raise ValueError(f"{name} subset must preserve grouped examples")


def _modes(config: GateValidateConfig, *, gpu_available: bool) -> tuple[GateValidationMode, ...]:
    result = [
        GateValidationMode("cpu", precision, batch, order)
        for precision in ("float", "int8")
        for batch in config.batch_sizes
        for order in config.orders
    ]
    if gpu_available and config.gpu_float_if_available:
        result.extend(
            GateValidationMode("gpu", "float", batch, order)
            for batch in config.batch_sizes
            for order in config.orders
        )
    return tuple(result)


def _fit_thresholds(
    examples: tuple[GateValidationExample, ...],
    *,
    fit_groups: set[str],
    max_tokens: int,
    predictor_factory: PredictorFactory,
    artifacts: GateValidationArtifacts,
) -> GateValidationThresholds:
    """Fit all deployment thresholds on the isolated grouped fit subset."""

    modes = tuple(
        GateValidationMode("threshold-fit-cpu", precision, batch_size, order)
        for precision in ("float", "int8")
        for batch_size in (1,)
        for order in ("original",)
    )
    runs: list[dict[str, tuple[float, float]]] = []
    for mode in modes:
        mode_runs, _attestation = _run_mode_reloads(
            predictor_factory, artifacts, examples, mode
        )
        runs.extend(mode_runs)

    averages: dict[str, tuple[float, float]] = {}
    max_delta = 0.0
    for example in examples:
        values = [run[example.example_id] for run in runs]
        averages[example.example_id] = (
            sum(value[0] for value in values) / len(values),
            sum(value[1] for value in values) / len(values),
        )
        max_delta = max(
            max_delta,
            max(value[0] for value in values) - min(value[0] for value in values),
            max(value[1] for value in values) - min(value[1] for value in values),
        )

    close_negative = [
        averages[item.example_id][0] for item in examples if not item.close_target
    ]
    close_positive = [
        averages[item.example_id][0] for item in examples if item.close_target
    ]
    suitable_negative = [
        averages[item.example_id][1] for item in examples if not item.suitable_target
    ]
    suitable_positive = [
        averages[item.example_id][1] for item in examples if item.suitable_target
    ]
    if not all(
        (close_negative, close_positive, suitable_negative, suitable_positive)
    ):
        raise ValueError("threshold-fit subset must contain both labels for both heads")
    close_negative_max = max(close_negative)
    close_positive_min = min(close_positive)
    if close_negative_max >= close_positive_min:
        raise ValueError("threshold-fit close classes are not separable")
    suitable_negative_max = max(suitable_negative)
    suitable_positive_min = min(suitable_positive)
    if suitable_negative_max >= suitable_positive_min:
        raise ValueError("threshold-fit suitable classes are not separable")
    close_gap = close_positive_min - close_negative_max
    thresholds = GateValidationThresholds(
        close_low=close_negative_max + close_gap / 3.0,
        close_high=close_positive_min - close_gap / 3.0,
        suitable_accept=(suitable_negative_max + suitable_positive_min) / 2.0,
        max_probability_delta=min(1.0, max(1e-6, max_delta + 1e-6)),
        max_tokens=max_tokens,
        threshold_fit_group_digest=_group_digest(fit_groups),
    )
    thresholds.validate()
    return thresholds


def _run_mode_reloads(
    predictor_factory: PredictorFactory,
    artifacts: GateValidationArtifacts,
    examples: tuple[GateValidationExample, ...],
    mode: GateValidationMode,
    reload_count: int = 1,
) -> tuple[
    tuple[dict[str, tuple[float, float]], ...],
    GateProcessAttestation,
]:
    """Run the configured number of isolated predictor loads."""

    try:
        pickle.dumps((predictor_factory, artifacts, examples, mode))
    except Exception as exc:
        raise ValueError(
            "validation predictor factory and inputs must be serializable"
        ) from exc
    try:
        context = _worker_context()
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(
            "formal gate validation requires spawn worker support"
        ) from exc
    workers: list[tuple[Any, Any]] = []
    open_connections: list[Any] = []
    payloads: list[Mapping[str, Any]] = []
    try:
        for _reload_index in range(reload_count):
            receive = None
            send = None
            process = None
            try:
                receive, send = context.Pipe(duplex=False)
                open_connections.extend((receive, send))
                process = context.Process(
                    target=_validation_worker,
                    args=(send, predictor_factory, artifacts, mode, examples),
                )
                process.start()
                send.close()
                open_connections.remove(send)
                workers.append((process, receive))
            except Exception as exc:
                if send is not None:
                    _close_connection(send)
                if receive is not None:
                    _close_connection(receive)
                if process is not None:
                    _reap_process(process)
                raise RuntimeError("validation worker failed to start") from exc
        for process, receive in workers:
            if not receive.poll(_WORKER_POLL_TIMEOUT_SECONDS):
                raise RuntimeError("validation worker timed out")
            try:
                payload = receive.recv()
            except EOFError as exc:
                raise RuntimeError("validation worker exited without attestation") from exc
            process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
            if process.is_alive():
                raise RuntimeError("validation worker did not exit cleanly")
            if process.exitcode != 0:
                raise RuntimeError(
                    f"validation worker exited with status {process.exitcode}"
                )
            if not isinstance(payload, Mapping) or set(payload) != {
                "ok",
                "pid",
                "device",
                "precision",
                "artifact_name",
                "artifact_sha256",
                "predictions",
                "error",
            }:
                raise ValueError("validation worker payload schema mismatch")
            if payload["ok"] is not True:
                raise ValueError(f"validation worker failed: {payload['error']}")
            payloads.append(payload)
    finally:
        for connection in open_connections:
            _close_connection(connection)
        for process, _receive in workers:
            _reap_process(process)

    expected_device = "gpu" if mode.device == "gpu" else "cpu"
    pids = tuple(payload["pid"] for payload in payloads)
    if (
        len(set(pids)) != reload_count
        or any(type(pid) is not int or pid <= 0 or pid == os.getpid() for pid in pids)
    ):
        raise ValueError("validation reloads require distinct worker PIDs")
    if any(payload["device"] != expected_device for payload in payloads):
        raise ValueError("validation worker device attestation mismatch")
    if any(payload["precision"] != mode.precision for payload in payloads):
        raise ValueError("validation worker precision attestation mismatch")
    artifact_path, artifact_sha256 = artifacts.for_precision(mode.precision)
    if any(
        payload["artifact_name"] != artifact_path.name
        or payload["artifact_sha256"] != artifact_sha256
        for payload in payloads
    ):
        raise ValueError("validation worker artifact attestation mismatch")
    predictions = tuple(dict(payload["predictions"]) for payload in payloads)
    return (
        predictions,
        GateProcessAttestation(
            mode_name=mode.name,
            pids=pids,
            device=expected_device,
            precision=mode.precision,
            artifact_name=artifact_path.name,
            artifact_sha256=artifact_sha256,
        ),
    )


def _worker_context() -> Any:
    return mp.get_context("spawn")


def _close_connection(connection: Any) -> None:
    try:
        connection.close()
    except Exception:
        pass


def _reap_process(process: Any) -> None:
    """Reap a worker, escalating terminate to kill and confirming exit."""

    try:
        process.join(timeout=0)
    except (AssertionError, RuntimeError, ValueError):
        return
    if not process.is_alive():
        return
    process.terminate()
    process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
    if process.is_alive():
        kill = getattr(process, "kill", None)
        if not callable(kill):
            raise RuntimeError("validation worker survived terminate without kill support")
        kill()
        process.join(timeout=_WORKER_JOIN_TIMEOUT_SECONDS)
    if process.is_alive():
        raise RuntimeError("validation worker could not be reaped")


def _validation_worker(
    connection: Any,
    predictor_factory: PredictorFactory,
    artifacts: GateValidationArtifacts,
    mode: GateValidationMode,
    examples: tuple[GateValidationExample, ...],
) -> None:
    """Child-only predictor load/run with runtime attestation."""

    payload: dict[str, Any] = {
        "ok": False,
        "pid": os.getpid(),
        "device": None,
        "precision": None,
        "artifact_name": None,
        "artifact_sha256": None,
        "predictions": None,
        "error": None,
    }
    try:
        expected_device = "gpu" if mode.device == "gpu" else "cpu"
        torch.set_num_threads(1)
        if expected_device == "gpu":
            if not torch.cuda.is_available():
                raise ValueError("GPU mode requested without an available CUDA runtime")
            # A real CUDA allocation is mandatory; monkeypatching availability
            # cannot fabricate successful GPU validation.
            probe = torch.empty(1, device="cuda")
            if probe.device.type != "cuda":
                raise ValueError("GPU runtime probe did not execute on CUDA")
            torch.cuda.synchronize(probe.device)
        else:
            probe = torch.empty(1, device="cpu")
            if probe.device.type != "cpu":
                raise ValueError("CPU runtime probe did not execute on CPU")

        artifact_path, expected_sha256 = artifacts.for_precision(mode.precision)
        artifact_bytes = artifact_path.read_bytes()
        actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError("candidate model artifact hash mismatch")
        artifact_state = _safe_load_artifact_state(artifact_bytes)
        predictor = predictor_factory(mode, artifact_state)
        if not callable(getattr(predictor, "encode_batch", None)):
            raise ValueError("predictor must implement the controlled batch encoder")
        model = getattr(predictor, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("predictor must expose the actual inspected model")
        _validate_bound_model(
            model,
            precision=mode.precision,
            device=expected_device,
            artifact_state=artifact_state,
        )
        observed_devices: list[str] = []

        def observe_execution(
            _module: torch.nn.Module, inputs: tuple[Any, ...], output: Any
        ) -> None:
            tensors = tuple(_walk_tensors((inputs, output)))
            if not tensors:
                raise ValueError("inspected model execution produced no tensors")
            observed_devices.extend(tensor.device.type for tensor in tensors)

        inspected_modules = _precision_modules(model, mode.precision)
        hooks = [module.register_forward_hook(observe_execution) for module in inspected_modules]
        try:
            predictions = _run_mode(predictor, examples, mode)
        finally:
            for hook in hooks:
                hook.remove()
        runtime_device = "cuda" if expected_device == "gpu" else "cpu"
        if not observed_devices or any(
            device != runtime_device for device in observed_devices
        ):
            raise ValueError("predictor model input/output device attestation mismatch")
        if not _states_equal(model.state_dict(), artifact_state):
            raise ValueError("predictor model state diverged from candidate artifact")
        payload.update(
            {
                "ok": True,
                "device": expected_device,
                "precision": mode.precision,
                "artifact_name": artifact_path.name,
                "artifact_sha256": actual_sha256,
                "predictions": predictions,
            }
        )
    except Exception as exc:
        payload["error"] = f"{type(exc).__name__}: {exc}"
    try:
        connection.send(payload)
    finally:
        connection.close()


def _validate_bound_model(
    model: torch.nn.Module,
    *,
    precision: str,
    device: str,
    artifact_state: Mapping[str, Any],
) -> None:
    _validate_bound_model_structure(model, precision=precision, device=device)
    if not _states_equal(model.state_dict(), artifact_state):
        raise ValueError("predictor model state does not match candidate artifact")


def _validate_bound_model_structure(
    model: torch.nn.Module, *, precision: str, device: str
) -> None:
    modules = _precision_modules(model, precision)
    float_linears = tuple(
        module for module in model.modules() if isinstance(module, torch.nn.Linear)
    )
    quantized_linears = tuple(
        module
        for module in model.modules()
        if "quantized.dynamic" in type(module).__module__
        and type(module).__name__ == "Linear"
    )
    if precision == "float" and (not float_linears or quantized_linears):
        raise ValueError("float artifact predictor precision structure mismatch")
    if precision == "int8" and (not quantized_linears or float_linears):
        raise ValueError("int8 artifact predictor precision structure mismatch")
    runtime_device = "cuda" if device == "gpu" else "cpu"
    state_tensors = tuple(_walk_tensors(model.state_dict()))
    if not state_tensors or any(
        tensor.device.type != runtime_device for tensor in state_tensors
    ):
        raise ValueError("predictor model parameter device mismatch")
    if precision == "int8" and not any(tensor.is_quantized for tensor in state_tensors):
        raise ValueError("int8 predictor contains no qint8 artifact weights")
    if not modules:
        raise ValueError("predictor has no inspectable precision modules")


def _precision_modules(
    model: torch.nn.Module, precision: str
) -> tuple[torch.nn.Module, ...]:
    if precision == "float":
        return tuple(
            module for module in model.modules() if isinstance(module, torch.nn.Linear)
        )
    return tuple(
        module
        for module in model.modules()
        if "quantized.dynamic" in type(module).__module__
        and type(module).__name__ == "Linear"
    )


def _safe_load_artifact_state(raw_bytes: bytes) -> Mapping[str, Any]:
    try:
        value = torch.load(
            io.BytesIO(raw_bytes), map_location="cpu", weights_only=True
        )
    except Exception as exc:
        raise ValueError("candidate artifact safe load failed") from exc
    if not isinstance(value, Mapping) or not value:
        raise ValueError("candidate artifact state must be a non-empty mapping")
    if not tuple(_walk_tensors(value)):
        raise ValueError("candidate artifact state must contain tensors")
    return value


def _walk_tensors(value: Any):
    if isinstance(value, Mapping):
        for child in value.values():
            yield from _walk_tensors(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            yield from _walk_tensors(child)
    elif isinstance(value, torch.Tensor):
        yield value


def _states_equal(left: Any, right: Any) -> bool:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _states_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return len(left) == len(right) and all(
            _states_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and torch.equal(left.detach().cpu(), right.detach().cpu())
        )
    return type(left) is type(right) and left == right


def _has_symlink_component(path: Path) -> bool:
    return any(candidate.is_symlink() for candidate in (path, *path.parents))


def _run_mode(
    predictor: ValidationPredictor,
    examples: tuple[GateValidationExample, ...],
    mode: GateValidationMode,
) -> dict[str, tuple[float, float]]:
    ordered = list(examples)
    if mode.order == "reverse":
        ordered.reverse()
    elif mode.order == "random":
        random.Random(0).shuffle(ordered)
    elif mode.order != "original":
        raise ValueError("unknown validation order")
    result: dict[str, tuple[float, float]] = {}
    for start in range(0, len(ordered), mode.batch_size):
        batch = ordered[start : start + mode.batch_size]
        model_inputs = predictor.encode_batch(batch)
        if not isinstance(model_inputs, Mapping) or not model_inputs:
            raise ValueError("controlled batch encoder must return tensor inputs")
        if any(
            not isinstance(name, str) or not isinstance(value, torch.Tensor)
            for name, value in model_inputs.items()
        ):
            raise ValueError("controlled batch encoder inputs must be named tensors")
        if any(value.ndim == 0 or value.shape[0] != len(batch) for value in model_inputs.values()):
            raise ValueError("controlled batch encoder batch dimension mismatch")
        output = predictor.model(**dict(model_inputs))
        close_logits = getattr(output, "close_logits", None)
        suitable_logits = getattr(output, "suitable_logits", None)
        if (
            not isinstance(close_logits, torch.Tensor)
            or not isinstance(suitable_logits, torch.Tensor)
            or close_logits.shape != (len(batch),)
            or suitable_logits.shape != (len(batch),)
        ):
            raise ValueError("bound model must return close/suitable batch logits")
        probabilities = torch.stack(
            (torch.sigmoid(close_logits), torch.sigmoid(suitable_logits)), dim=1
        )
        if not torch.isfinite(probabilities).all():
            raise ValueError("bound model probabilities must be finite")
        rows = probabilities.detach().cpu().tolist()
        for example, row in zip(batch, rows, strict=True):
            value = (float(row[0]), float(row[1]))
            result[example.example_id] = _probability_pair(value)
    return result


def _probability_pair(value: object) -> tuple[float, float]:
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError("prediction must be a probability pair")
    result = []
    for item in value:
        if (
            isinstance(item, bool)
            or not isinstance(item, (int, float))
            or not math.isfinite(item)
            or not 0.0 <= item <= 1.0
        ):
            raise ValueError("prediction probability must be finite and in [0, 1]")
        result.append(float(item))
    return result[0], result[1]


def _decision(
    value: tuple[float, float], thresholds: GateValidationThresholds
) -> tuple[str, bool]:
    close = (
        "continue"
        if value[0] <= thresholds.close_low
        else "close"
        if value[0] >= thresholds.close_high
        else "uncertain"
    )
    return close, value[1] >= thresholds.suitable_accept


def _group_digest(groups: set[str]) -> str:
    payload = json.dumps(sorted(groups), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
