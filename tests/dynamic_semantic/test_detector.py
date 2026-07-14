from __future__ import annotations

import torch

from wfcllm.dynamic_semantic.calibration import fit_calibration
from wfcllm.dynamic_semantic.channel import SemanticChannel
from wfcllm.dynamic_semantic.config import ChannelConfig, ContextConfig
from wfcllm.dynamic_semantic.context import DynamicContextExtractor
from wfcllm.dynamic_semantic.detector import R3Detector, compare_exact_replay
from wfcllm.dynamic_semantic.keying import SecretKey
from wfcllm.dynamic_semantic.observer import AttemptContextObserver
from wfcllm.dynamic_semantic.scheduler import DynamicSemanticScheduler
from wfcllm.dynamic_semantic.config import SchedulerConfig


class _Encoder:
    def __init__(self) -> None:
        self.encoder_calls = 0
        self.encoded_contexts = 0

    def encode(self, contexts) -> torch.Tensor:
        self.encoder_calls += 1
        self.encoded_contexts += len(contexts)
        rows = []
        for text in contexts:
            value = float(sum(text.encode("utf-8")) % 19 + 1)
            rows.append((value, value + 1, value + 2, value + 3))
        return torch.nn.functional.normalize(
            torch.tensor(rows, dtype=torch.float32), p=2, dim=1
        )


class _Whitening:
    def transform(self, embeddings: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(embeddings.float(), p=2, dim=1)


def _components():
    extractor = DynamicContextExtractor(
        ContextConfig(), token_counter=lambda text: len(text.split())
    )
    channel = SemanticChannel(
        SecretKey.from_material_for_test(b"r" * 32),
        ChannelConfig(
            whitening_dimensions=4,
            quantization_scale=64,
            projection_rows=7,
            target_data_bits=4,
            minimum_independent_units=1,
        ),
    )
    return extractor, channel


def test_r3_uses_only_final_code_and_one_encoder_batch() -> None:
    extractor, channel = _components()
    encoder = _Encoder()
    calibration = fit_calibration(
        [0.0] * 250,
        source_role="calibration_negative",
        source_manifest_sha256="a" * 64,
        target_fpr=0.05,
    )
    detector = R3Detector(
        extractor=extractor,
        encoder=encoder,
        whitening=_Whitening(),
        channel=channel,
        calibration=calibration,
        minimum_independent_units=1,
    )
    code = "def f(x):\n    y = x + 1\n    return y\n"

    result = detector.detect_payload({"final_code": code})

    assert encoder.encoder_calls == 1
    assert encoder.encoded_contexts == 2
    assert result.aggregate.independent_units == 2
    assert result.public_dict()["final_code_sha256"]
    assert not ({"target_bits", "signature_bits", "quantized_embedding"} & set(result.public_dict()))


def test_r3_rejects_prompt_id_trace_or_candidate_inputs() -> None:
    extractor, channel = _components()
    detector = R3Detector(
        extractor=extractor,
        encoder=_Encoder(),
        whitening=_Whitening(),
        channel=channel,
        calibration=fit_calibration(
            [0.0],
            source_role="calibration_negative",
            source_manifest_sha256="b" * 64,
            target_fpr=0.05,
        ),
        minimum_independent_units=1,
    )
    for forbidden in ("prompt", "id", "trace", "candidates", "ledger"):
        payload = {"final_code": "def f():\n    return 1\n", forbidden: "x"}
        try:
            detector.detect_payload(payload)
        except ValueError as exc:
            assert "final_code" in str(exc)
        else:
            raise AssertionError(f"accepted forbidden detector input: {forbidden}")


def test_generation_observer_and_r3_replay_are_exact() -> None:
    extractor, channel = _components()
    generation_encoder = _Encoder()
    scheduler = DynamicSemanticScheduler(
        encoder=generation_encoder,
        whitening=_Whitening(),
        channel=channel,
        config=SchedulerConfig(
            target_batch_contexts=32,
            max_batch_contexts=128,
            max_queue_completed_attempts=4,
        ),
    )
    observer = AttemptContextObserver(
        attempt_index=0,
        extractor=extractor,
        scheduler=scheduler,
    )
    code = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
    observer.observe_prefix("def f(x):\n    y = x + 1\n    z = y * 2\n")
    observer.flush(code)
    scheduler.flush_final()
    generation_evidence = scheduler.evidence_for_attempt(0)

    detector = R3Detector(
        extractor=extractor,
        encoder=_Encoder(),
        whitening=_Whitening(),
        channel=channel,
        calibration=fit_calibration(
            [0.0],
            source_role="calibration_negative",
            source_manifest_sha256="c" * 64,
            target_fpr=0.05,
        ),
        minimum_independent_units=1,
    )
    replay = detector.detect(code)
    audit = compare_exact_replay(generation_evidence, replay.evidence)

    assert audit.exact is True
    assert audit.mismatches == ()
    assert len(generation_evidence) == 3


def test_parse_failure_is_ineligible_without_encoder_call() -> None:
    extractor, channel = _components()
    encoder = _Encoder()
    detector = R3Detector(
        extractor=extractor,
        encoder=encoder,
        whitening=_Whitening(),
        channel=channel,
        calibration=fit_calibration(
            [0.0],
            source_role="calibration_negative",
            source_manifest_sha256="d" * 64,
            target_fpr=0.05,
        ),
        minimum_independent_units=1,
    )

    result = detector.detect("def f(:\n")

    assert result.decision is False
    assert result.aggregate.eligible is False
    assert encoder.encoder_calls == 0
