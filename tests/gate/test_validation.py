from __future__ import annotations

from dataclasses import replace
import hashlib
import os
from pathlib import Path
import types
from typing import Sequence
import warnings

import pytest
import torch
import wfcllm.gate.validation as validation_module

from wfcllm.gate.config import GateValidateConfig
from wfcllm.gate.validation import (
    GateValidationArtifacts,
    GateValidationExample,
    GateValidationMode,
    GateProcessAttestation,
    GateValidationThresholds,
    GateValidator,
)


_TEST_ARTIFACTS: GateValidationArtifacts | None = None


class TinyDualHeadModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(2, 2)

    def forward(self, *, input_features: torch.Tensor) -> object:
        logits = self.projection(input_features)
        return types.SimpleNamespace(
            close_logits=logits[:, 0], suitable_logits=logits[:, 1]
        )


@pytest.fixture(autouse=True)
def _tiny_candidate_artifacts(tmp_path: Path) -> None:
    global _TEST_ARTIFACTS
    float_model = TinyDualHeadModel().eval()
    with torch.no_grad():
        float_model.projection.weight.copy_(torch.eye(2))
        float_model.projection.bias.zero_()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=DeprecationWarning)
        int8_model = torch.ao.quantization.quantize_dynamic(
            float_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=False
        )
    float_path = tmp_path / "gate_float.pt"
    int8_path = tmp_path / "gate_int8.pt"
    torch.save(float_model.state_dict(), float_path)
    torch.save(int8_model.state_dict(), int8_path)
    _TEST_ARTIFACTS = GateValidationArtifacts(
        float_model_path=float_path,
        int8_model_path=int8_path,
        float_model_sha256=_file_sha256(float_path),
        int8_model_sha256=_file_sha256(int8_path),
    )


@pytest.fixture(autouse=True)
def _cpu_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


_PRODUCTION_WORKER_TESTS = frozenset(
    {
        "test_validation_matrix_covers_all_cpu_modes_orders_batches_and_reloads",
        "test_worker_rejects_wrong_artifact_hash",
        "test_second_worker_start_failure_reaps_first_and_closes_all_pipes",
        "test_worker_timeout_escalates_to_kill_and_confirms_reaping",
    }
)


@pytest.fixture(autouse=True)
def _fast_metric_runner(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Keep pure metric tests fast while designated tests exercise real spawn."""

    if request.node.name in _PRODUCTION_WORKER_TESTS:
        return
    monkeypatch.setattr(
        validation_module, "_run_mode_reloads", _run_mode_reloads_in_process
    )


def _run_mode_reloads_in_process(
    predictor_factory: object,
    artifacts: GateValidationArtifacts,
    examples: tuple[GateValidationExample, ...],
    mode: GateValidationMode,
) -> tuple[
    tuple[
        dict[str, tuple[float, float]],
        dict[str, tuple[float, float]],
    ],
    GateProcessAttestation,
]:
    artifact_path, expected_sha256 = artifacts.for_precision(mode.precision)
    artifact_bytes = artifact_path.read_bytes()
    actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("candidate model artifact hash mismatch")
    state = validation_module._safe_load_artifact_state(artifact_bytes)
    runs: list[dict[str, tuple[float, float]]] = []
    for _reload_index in range(2):
        try:
            predictor = predictor_factory(mode, state)
        except Exception as exc:
            raise ValueError(f"validation worker failed: {exc}") from exc
        if not callable(getattr(predictor, "encode_batch", None)):
            raise ValueError("predictor must implement the controlled batch encoder")
        model = getattr(predictor, "model", None)
        if not isinstance(model, torch.nn.Module):
            raise ValueError("predictor must expose the actual inspected model")
        expected_device = "gpu" if mode.device == "gpu" else "cpu"
        validation_module._validate_bound_model(
            model,
            precision=mode.precision,
            device=expected_device,
            artifact_state=state,
        )
        runs.append(validation_module._run_mode(predictor, examples, mode))
        if not validation_module._states_equal(model.state_dict(), state):
            raise ValueError("predictor model state diverged from candidate artifact")
    return (
        (runs[0], runs[1]),
        GateProcessAttestation(
            mode_name=mode.name,
            pids=(900_001, 900_002),
            device="gpu" if mode.device == "gpu" else "cpu",
            precision=mode.precision,
            artifact_name=artifact_path.name,
            artifact_sha256=actual_sha256,
        ),
    )


def _examples() -> tuple[GateValidationExample, ...]:
    return tuple(
        GateValidationExample(
            example_id=f"e-{index}",
            group_id=f"g-{index // 2}",
            serialized_input=f"input-{index}",
            span=(index * 2, index * 2 + 2),
            close_target=index % 2 == 1,
            suitable_target=index % 4 != 0,
            validation_role="threshold_fit" if index < 2 else "agreement",
        )
        for index in range(8)
    )


class DeterministicFactory:
    def __init__(
        self,
        *,
        drift_mode: str | None = None,
        drift_example: str | None = None,
        drift_close: float = 0.8,
        suitable_bias: float = 0.0,
        agreement_close_override: float | None = None,
        agreement_suitable_override: float | None = None,
        wrong_precision_model: bool = False,
        wrong_device: bool = False,
        nonfinite: bool = False,
    ) -> None:
        if _TEST_ARTIFACTS is None:
            raise AssertionError("tiny artifact fixture was not initialized")
        self.artifacts = _TEST_ARTIFACTS
        self.drift_mode = drift_mode
        self.drift_example = drift_example
        self.drift_close = drift_close
        self.suitable_bias = suitable_bias
        self.agreement_close_override = agreement_close_override
        self.agreement_suitable_override = agreement_suitable_override
        self.wrong_precision_model = wrong_precision_model
        self.wrong_device = wrong_device
        self.nonfinite = nonfinite
        self.reloads: dict[str, int] = {}

    def __call__(self, mode: GateValidationMode, state: object):
        self.reloads[mode.name] = self.reloads.get(mode.name, 0) + 1
        drift_mode = self.drift_mode
        drift_example = self.drift_example
        drift_close = self.drift_close
        suitable_bias = self.suitable_bias
        agreement_close_override = self.agreement_close_override
        agreement_suitable_override = self.agreement_suitable_override
        wrong_precision_model = self.wrong_precision_model
        wrong_device = self.wrong_device
        nonfinite = self.nonfinite

        wrong_for_mode = wrong_precision_model and mode.precision == "int8"
        if mode.precision == "float" or wrong_for_mode:
            model = TinyDualHeadModel().eval()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=DeprecationWarning)
                model = torch.ao.quantization.quantize_dynamic(
                    TinyDualHeadModel().eval(),
                    {torch.nn.Linear},
                    dtype=torch.qint8,
                )
        if not wrong_for_mode:
            model.load_state_dict(state, strict=True)
        if mode.device == "gpu":
            model = model.to("cuda")
        elif wrong_device and mode.precision == "float":
            model = model.to("meta")

        class Predictor:
            def __init__(self) -> None:
                self.model = model

            def encode_batch(self, examples: Sequence[GateValidationExample]):
                rows: list[tuple[float, float]] = []
                for example in examples:
                    index = int(example.example_id.split("-")[-1])
                    close = 0.75 if index % 2 else 0.25
                    suitable = 0.25 if index % 4 == 0 else 0.75
                    if not mode.device.startswith("threshold-fit"):
                        suitable += suitable_bias
                    if (
                        agreement_close_override is not None
                        and not mode.device.startswith("threshold-fit")
                    ):
                        close = agreement_close_override
                    if (
                        agreement_suitable_override is not None
                        and not mode.device.startswith("threshold-fit")
                    ):
                        suitable = agreement_suitable_override
                    if mode.name == drift_mode and example.example_id == drift_example:
                        close = drift_close
                    if nonfinite:
                        close = float("nan")
                    rows.append((close, min(1.0, suitable)))
                probabilities = torch.tensor(rows, dtype=torch.float32).clamp(
                    min=1e-6, max=1.0 - 1e-6
                )
                if nonfinite:
                    probabilities[0, 0] = float("nan")
                runtime_device = "cuda" if mode.device == "gpu" else "cpu"
                return {
                    "input_features": torch.logit(probabilities).to(runtime_device)
                }

        return Predictor()


class LyingFactory:
    def __init__(self) -> None:
        if _TEST_ARTIFACTS is None:
            raise AssertionError
        self.artifacts = _TEST_ARTIFACTS

    def __call__(self, mode: GateValidationMode, state: object):
        class Predictor:
            def predict_batch(self, examples: Sequence[GateValidationExample]):
                return [(0.2, 0.9) for _ in examples]

        return Predictor()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_validation_matrix_covers_all_cpu_modes_orders_batches_and_reloads() -> None:
    factory = DeterministicFactory()
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=factory,
        gpu_available=False,
    )
    assert summary.decision_agreement == 1.0
    assert summary.float_quantized_accepted_set_agreement == 1.0
    assert summary.formal_accepted_span_consensus == 1.0
    assert summary.suitable_false_positive_rate == 0.0
    assert summary.formal_confusion.tp == 3
    assert summary.formal_confusion.tn == 1
    assert summary.formal_confusion.fp == 0
    assert summary.formal_confusion.fn == 2
    assert summary.validated is True, summary.to_dict()
    assert summary.gpu_status == "not_available"
    assert summary.validation_scope == "cpu"
    assert set(summary.mode_names) == {
        f"cpu-{precision}-batch-{batch}-{order}"
        for precision in ("float", "int8")
        for batch in (1,)
        for order in ("original",)
    }
    assert len(summary.process_attestations) == len(summary.mode_names)
    for attestation in summary.process_attestations:
        assert len(set(attestation.pids)) == 1
        assert all(pid != os.getpid() for pid in attestation.pids)
        expected_path, expected_hash = factory.artifacts.for_precision(
            attestation.precision
        )
        assert attestation.artifact_name == expected_path.name
        assert attestation.artifact_sha256 == expected_hash
    assert summary.thresholds.threshold_fit_group_digest == summary.threshold_fit_group_digest


def test_legacy_agreement_metric_is_reported_without_overriding_nondegeneracy() -> None:
    validator = GateValidator(GateValidateConfig())
    summary = validator.validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(
            drift_mode="cpu-int8-batch-1-original", drift_example="e-2"
        ),
        gpu_available=False,
    )
    assert summary.validated is True
    assert summary.decision_agreement < 0.999
    assert summary.forced_uncertain_count >= 1
    assert ("g-1", "e-2", 4, 6) not in summary.formal_accepted_spans


def test_formal_span_consensus_independently_fails_on_stable_pair_delta() -> None:
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(
            drift_mode="cpu-int8-batch-1-original",
            drift_example="e-3",
            drift_close=0.7,
        ),
        gpu_available=False,
    )
    assert summary.decision_agreement == 1.0
    assert summary.float_quantized_accepted_set_agreement == 1.0
    assert summary.formal_accepted_span_consensus == 1.0
    assert summary.forced_uncertain_count >= 1
    assert summary.validated is True


def test_formal_consensus_distinguishes_groups_that_share_a_span() -> None:
    examples = _examples()
    shared_span = examples[3].span
    agreement = tuple(
        replace(example, span=shared_span) if example.example_id == "e-5" else example
        for example in examples[2:]
    )
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=examples[:2],
        agreement_examples=agreement,
        max_tokens=512,
        predictor_factory=DeterministicFactory(
            drift_mode="cpu-int8-batch-1-original",
            drift_example="e-3",
            drift_close=0.7,
        ),
        gpu_available=False,
    )
    assert summary.decision_agreement == 1.0
    assert summary.float_quantized_accepted_set_agreement == 1.0
    assert ("g-1", "e-3", *shared_span) not in summary.formal_accepted_spans
    assert ("g-2", "e-5", *shared_span) in summary.formal_accepted_spans
    assert summary.formal_accepted_span_consensus == 1.0
    assert summary.forced_uncertain_count >= 1
    assert summary.validated is True


def test_empty_formal_span_coverage_cannot_pass_consensus() -> None:
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(
            agreement_suitable_override=0.1
        ),
        gpu_available=False,
    )
    assert summary.formal_accepted_spans == ()
    assert summary.formal_accepted_span_consensus == 0.0
    assert summary.validated is False


def test_worker_rejects_wrong_precision_model() -> None:
    with pytest.raises(ValueError, match="precision structure mismatch"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=DeterministicFactory(wrong_precision_model=True),
            gpu_available=False,
        )


def test_worker_rejects_wrong_artifact_hash() -> None:
    factory = DeterministicFactory()
    factory.artifacts = replace(factory.artifacts, float_model_sha256="0" * 64)
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=factory,
            gpu_available=False,
        )


def test_worker_rejects_wrong_actual_model_device() -> None:
    with pytest.raises(ValueError, match="parameter device mismatch"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=DeterministicFactory(wrong_device=True),
            gpu_available=False,
        )


def test_worker_rejects_lying_factory_without_bound_model() -> None:
    with pytest.raises(ValueError, match="controlled batch encoder"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=LyingFactory(),
            gpu_available=False,
        )


class _FakeConnection:
    def __init__(self, *, poll_result: bool = False) -> None:
        self.poll_result = poll_result
        self.closed = False

    def poll(self, timeout: float) -> bool:
        assert timeout >= 0
        return self.poll_result

    def recv(self) -> object:
        raise AssertionError("fake connection has no payload")

    def close(self) -> None:
        self.closed = True


class _FakeProcess:
    def __init__(
        self, *, start_fails: bool = False, survive_terminate: bool = False
    ) -> None:
        self.start_fails = start_fails
        self.survive_terminate = survive_terminate
        self.started = False
        self.alive = False
        self.terminated = False
        self.killed = False
        self.joins: list[float] = []
        self.exitcode = None

    def start(self) -> None:
        if self.start_fails:
            raise OSError("start failed")
        self.started = True
        self.alive = True

    def join(self, timeout: float) -> None:
        if not self.started:
            raise AssertionError("not started")
        self.joins.append(timeout)

    def is_alive(self) -> bool:
        return self.alive

    def terminate(self) -> None:
        self.terminated = True
        if not self.survive_terminate:
            self.alive = False
            self.exitcode = -15

    def kill(self) -> None:
        self.killed = True
        self.alive = False
        self.exitcode = -9


class _FakeContext:
    def __init__(self, processes: list[_FakeProcess]) -> None:
        self.processes = processes
        self.connections: list[_FakeConnection] = []

    def Pipe(self, *, duplex: bool) -> tuple[_FakeConnection, _FakeConnection]:
        assert duplex is False
        receive = _FakeConnection()
        send = _FakeConnection()
        self.connections.extend((receive, send))
        return receive, send

    def Process(self, **_: object) -> _FakeProcess:
        return self.processes.pop(0)


def test_second_worker_start_failure_reaps_first_and_closes_all_pipes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakeProcess()
    second = _FakeProcess(start_fails=True)
    context = _FakeContext([first, second])
    monkeypatch.setattr(validation_module, "_worker_context", lambda: context)
    with pytest.raises(RuntimeError, match="failed to start"):
        validation_module._run_mode_reloads(
            DeterministicFactory(),
            _TEST_ARTIFACTS,
            _examples()[:2],
            GateValidationMode("cpu", "float", 1, "original"),
            2,
        )
    assert first.terminated is True
    assert first.is_alive() is False
    assert all(connection.closed for connection in context.connections)


def test_worker_timeout_escalates_to_kill_and_confirms_reaping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workers = [
        _FakeProcess(survive_terminate=True),
        _FakeProcess(survive_terminate=True),
    ]
    context = _FakeContext(list(workers))
    monkeypatch.setattr(validation_module, "_worker_context", lambda: context)
    monkeypatch.setattr(validation_module, "_WORKER_POLL_TIMEOUT_SECONDS", 0.0)
    with pytest.raises(RuntimeError, match="timed out"):
        validation_module._run_mode_reloads(
            DeterministicFactory(),
            _TEST_ARTIFACTS,
            _examples()[:2],
            GateValidationMode("cpu", "float", 1, "original"),
            2,
        )
    assert all(worker.terminated and worker.killed for worker in workers)
    assert all(not worker.is_alive() and worker.joins for worker in workers)
    assert all(connection.closed for connection in context.connections)


def test_suitable_false_positive_rate_is_reported_not_a_hard_gate() -> None:
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(agreement_suitable_override=0.75),
        gpu_available=False,
    )
    assert summary.suitable_false_positive_rate == 1.0
    assert summary.validated is True


@pytest.mark.parametrize(
    ("factory_kwargs", "expected_accepted"),
    (
        (
            {
                "agreement_close_override": 0.9,
                "agreement_suitable_override": 0.9,
            },
            6,
        ),
        ({"agreement_close_override": 0.1}, 0),
    ),
)
def test_accept_all_and_reject_all_formal_decisions_fail_closed(
    factory_kwargs: dict[str, float],
    expected_accepted: int,
) -> None:
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(**factory_kwargs),
        gpu_available=False,
    )

    assert summary.formal_confusion.accepted == expected_accepted
    assert summary.validated is False


def test_binary_threshold_fit_allows_overlapping_classes() -> None:
    """A real gate need not make the fit subset perfectly separable."""

    threshold = validation_module._fit_binary_threshold(
        negatives=[0.10, 0.40, 0.80],
        positives=[0.30, 0.70, 0.90],
    )

    predictions = [score >= threshold for score in (0.10, 0.40, 0.80, 0.30, 0.70, 0.90)]
    assert any(predictions)
    assert not all(predictions)
    assert 0.40 < threshold <= 0.70


def test_threshold_fit_and_agreement_groups_must_be_disjoint_and_group_complete() -> None:
    validator = GateValidator(GateValidateConfig())
    overlap_fit = (
        *_examples()[:2],
        replace(
            _examples()[2],
            example_id="fit-overlap-2",
            validation_role="threshold_fit",
        ),
        replace(
            _examples()[3],
            example_id="fit-overlap-3",
            validation_role="threshold_fit",
        ),
    )
    with pytest.raises(ValueError, match="overlap"):
        validator.validate(
            threshold_fit_examples=overlap_fit,
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=DeterministicFactory(),
            gpu_available=False,
        )
    split_group = replace(_examples()[1], group_id="other")
    with pytest.raises(ValueError, match="grouped"):
        validator.validate(
            threshold_fit_examples=(_examples()[0], split_group),
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=DeterministicFactory(),
            gpu_available=False,
        )


def test_training_detector_and_calibration_roles_cannot_tune_thresholds() -> None:
    for role in ("train", "final_detector", "negative_calibration"):
        with pytest.raises(ValueError, match="forbidden"):
            replace(_examples()[0], validation_role=role)


@pytest.mark.parametrize(
    "thresholds",
    [
        GateValidationThresholds(0.5, 0.5, 0.8, 0.02, 512, "2" * 64),
        GateValidationThresholds(0.6, 0.4, 0.8, 0.02, 512, "2" * 64),
        GateValidationThresholds(0.4, 0.6, 1.1, 0.02, 512, "2" * 64),
        GateValidationThresholds(0.4, 0.6, 0.8, -0.1, 512, "2" * 64),
        GateValidationThresholds(0.4, 0.6, 0.8, 0.02, 513, "2" * 64),
    ],
)
def test_threshold_contract_fails_closed(thresholds: GateValidationThresholds) -> None:
    with pytest.raises(ValueError):
        thresholds.validate()


def test_missing_gpu_is_reported_without_fake_gpu_validation() -> None:
    factory = DeterministicFactory()
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=factory,
        gpu_available=False,
    )
    assert summary.gpu_status == "not_available"
    assert summary.validation_scope == "cpu"
    assert not any(name.startswith("gpu-float-") for name in summary.mode_names)
    assert not any(name.startswith("gpu-int8-") for name in summary.mode_names)


def test_monkeypatched_gpu_availability_cannot_fake_cuda_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    summary = GateValidator(GateValidateConfig()).validate(
        threshold_fit_examples=_examples()[:2],
        agreement_examples=_examples()[2:],
        max_tokens=512,
        predictor_factory=DeterministicFactory(),
        gpu_available=True,
    )
    assert summary.gpu_status == "skipped_fast_contract"
    assert not any(name.startswith("gpu-") for name in summary.mode_names)


def test_duplicate_ids_and_nonfinite_predictions_are_rejected() -> None:
    duplicate = replace(_examples()[3], example_id=_examples()[2].example_id)
    with pytest.raises(ValueError, match="example_id"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=(_examples()[2], duplicate),
            max_tokens=512,
            predictor_factory=DeterministicFactory(),
            gpu_available=False,
        )

    with pytest.raises(ValueError, match="probabil"):
        GateValidator(GateValidateConfig()).validate(
            threshold_fit_examples=_examples()[:2],
            agreement_examples=_examples()[2:],
            max_tokens=512,
            predictor_factory=DeterministicFactory(nonfinite=True),
            gpu_available=False,
        )
