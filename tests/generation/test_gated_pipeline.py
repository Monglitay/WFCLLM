from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from wfcllm.generation.completion_finalizer import ProgramFinalizationResult
from wfcllm.generation.gated_generator import (
    ByteSpan,
    GatedGenerationResult,
    GatedWindowAudit,
)
from wfcllm.generation.gated_pipeline import (
    GatedGenerationPipeline,
    GatedGenerationPipelineConfig,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _bundle() -> SimpleNamespace:
    return SimpleNamespace(
        manifest=SimpleNamespace(
            window_contract_version="python-statement-window/v1",
            gate_input_contract_version="wfcllm-gate-input/v1",
            tokenizer_sha256=_digest("tokenizer"),
        ),
        validation_summary={"validated": True},
    )


class _BaseModel:
    def __init__(self, code: str = "x = 1\n") -> None:
        self.code = code
        self.calls = 0

    def generate_program(self, *, prompt: str, sample_id: str) -> str:
        self.calls += 1
        return self.code


class _Generator:
    def generate(self, **_kwargs) -> GatedGenerationResult:
        return GatedGenerationResult(
            "x = 2\n",
            (
                GatedWindowAudit(
                    original_span=ByteSpan(0, 5),
                    selected_span=ByteSpan(0, 5),
                    rollback_anchor=0,
                    original_unit_count=1,
                    selected_unit_count=1,
                    selected_candidate_index=1,
                    original_token_ids=(),
                    selected_token_ids=(2,),
                    parent_descriptor="module|body|0|simple",
                    previous_statement_text_unchanged=True,
                    close_reason="end_of_input",
                ),
            ),
        )


def _config(tmp_path: Path) -> GatedGenerationPipelineConfig:
    return GatedGenerationPipelineConfig(
        output_dir=tmp_path,
        dataset="humaneval",
        bundle_path=tmp_path / "bundle",
        bundle_sha256=_digest("bundle"),
        parser_contract="python-statement-window/v1",
        gate_input_contract="wfcllm-gate-input/v1",
        tokenizer_sha256=_digest("tokenizer"),
        semantic_encoder_sha256=_digest("encoder"),
        lsh_config_sha256=_digest("lsh"),
        generation_config_sha256=_digest("generation"),
        secret_source_type="file",
    )


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_gated_pipeline_keeps_official_final_code_strict(tmp_path: Path) -> None:
    pipeline = GatedGenerationPipeline(
        config=_config(tmp_path),
        bundle_loader=lambda _path: _bundle(),
        bundle_hasher=lambda _path: _digest("bundle"),
        base_model=_BaseModel(),
        generator=_Generator(),
        data_adapter=lambda: [{"id": "HumanEval/0", "prompt": "def f():\n"}],
        deployment_key=b"never-persist-this",
    )
    pipeline.run()
    row = _read_jsonl(tmp_path / "inputs" / "final_code.jsonl")[0]
    assert set(row) == {"id", "dataset", "prompt", "final_code"}
    assert "gate" not in row
    assert "candidate" not in row
    all_bytes = b"".join(p.read_bytes() for p in tmp_path.rglob("*") if p.is_file())
    assert b"never-persist-this" not in all_bytes


def test_generation_audit_is_sidecar_only(tmp_path: Path) -> None:
    GatedGenerationPipeline(
        config=_config(tmp_path), bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"), base_model=_BaseModel(),
        generator=_Generator(), data_adapter=lambda: [{"id": "1", "prompt": "p"}],
        deployment_key=b"key",
    ).run()
    audit = _read_jsonl(tmp_path / "generation" / "audit.jsonl")
    assert audit[0]["audit_only"] is True
    assert audit[0]["not_detector_input"] is True
    assert audit[0]["selected_candidate_index"] == 1


def test_experimental_generation_is_marked_non_formal(tmp_path: Path) -> None:
    bundle = _bundle()
    bundle.experimental_only = True
    bundle.validation_summary = {
        "validated": False,
        "experimental_only": True,
        "diagnostic_only": True,
        "not_official_method": True,
    }
    pipeline = GatedGenerationPipeline(
        config=replace(_config(tmp_path), experimental_only=True),
        bundle_loader=lambda _path: bundle,
        bundle_hasher=lambda _path: _digest("bundle"),
        base_model=_BaseModel(),
        generator=_Generator(),
        data_adapter=lambda: [{"id": "1", "prompt": "p"}],
        deployment_key=b"key",
    )

    pipeline.run()

    manifest = json.loads(
        (tmp_path / "generation" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["formal"] is False
    assert manifest["experimental_only"] is True
    assert manifest["diagnostic_only"] is True
    assert manifest["not_official_method"] is True


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"validated": False}, "validated"),
        ({"parser": "wrong"}, "parser"),
        ({"gate": "wrong"}, "gate input"),
        ({"tokenizer": "0" * 64}, "tokenizer"),
    ],
)
def test_pipeline_rejects_invalid_or_mismatched_bundle(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    bundle = _bundle()
    if "validated" in change:
        bundle.validation_summary["validated"] = change["validated"]
    if "parser" in change:
        bundle.manifest.window_contract_version = change["parser"]
    if "gate" in change:
        bundle.manifest.gate_input_contract_version = change["gate"]
    if "tokenizer" in change:
        bundle.manifest.tokenizer_sha256 = change["tokenizer"]
    with pytest.raises(ValueError, match=message):
        GatedGenerationPipeline(
            config=_config(tmp_path), bundle_loader=lambda _p: bundle,
            bundle_hasher=lambda _p: _digest("bundle"), base_model=_BaseModel(),
            generator=_Generator(), data_adapter=lambda: [], deployment_key=b"key",
        )


def test_sample_failure_never_enters_detector_input(tmp_path: Path) -> None:
    class Broken:
        def generate(self, **_kwargs):
            raise SyntaxError("unrecoverable parser context")

    GatedGenerationPipeline(
        config=_config(tmp_path), bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"), base_model=_BaseModel(),
        generator=Broken(), data_adapter=lambda: [{"id": "bad", "prompt": "p"}],
        deployment_key=b"key",
    ).run()
    assert (tmp_path / "inputs" / "final_code.jsonl").read_text() == ""
    row = _read_jsonl(tmp_path / "generation" / "audit.jsonl")[0]
    assert row["sample_generation_failed"] is True


def test_each_sample_generates_exactly_one_base_program(tmp_path: Path) -> None:
    base = _BaseModel()
    GatedGenerationPipeline(
        config=_config(tmp_path), bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"), base_model=base,
        generator=_Generator(), data_adapter=lambda: [{"id": "1", "prompt": "p"}],
        deployment_key=b"key",
    ).run()
    assert base.calls == 1
    assert len(_read_jsonl(tmp_path / "inputs" / "final_code.jsonl")) == 1


def test_finalizer_runs_before_watermark_generator(tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def finalizer(_prompt: str, _source: str) -> ProgramFinalizationResult:
        return ProgramFinalizationResult(
            code="def f():\n    return 1\n",
            applied=True,
            reason="target_function_complete",
        )

    class CapturingGenerator(_Generator):
        def generate(self, **kwargs) -> GatedGenerationResult:
            observed.update(kwargs)
            return super().generate(**kwargs)

    GatedGenerationPipeline(
        config=_config(tmp_path),
        bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"),
        base_model=_BaseModel("def f():\n    return 1\nprint("),
        generator=CapturingGenerator(),
        data_adapter=[{"id": "HumanEval/0", "prompt": "def f():\n"}],
        deployment_key=b"key",
        program_finalizer=finalizer,
        program_finalizer_name="humaneval_target_function_v1",
    ).run()

    assert observed["original"] == "def f():\n    return 1\n"


def test_finalizer_provenance_and_manifest_counts(tmp_path: Path) -> None:
    calls = 0

    def finalizer(_prompt: str, source: str) -> ProgramFinalizationResult:
        nonlocal calls
        calls += 1
        return ProgramFinalizationResult(
            code=source.rstrip() + "\n",
            applied=calls == 1,
            reason=(
                "target_function_complete"
                if calls == 1
                else "no_complete_target_implementation"
            ),
        )

    GatedGenerationPipeline(
        config=_config(tmp_path),
        bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"),
        base_model=_BaseModel("x = 1\n"),
        generator=_Generator(),
        data_adapter=[
            {"id": "HumanEval/0", "prompt": "p0"},
            {"id": "HumanEval/1", "prompt": "p1"},
        ],
        deployment_key=b"key",
        program_finalizer=finalizer,
        program_finalizer_name="humaneval_target_function_v1",
    ).run()

    rows = _read_jsonl(tmp_path / "generation" / "finalizer.jsonl")
    assert [row["id"] for row in rows] == ["HumanEval/0", "HumanEval/1"]
    assert [row["applied"] for row in rows] == [True, False]
    assert all(len(str(row["before_sha256"])) == 64 for row in rows)
    assert all(len(str(row["after_sha256"])) == 64 for row in rows)
    assert all(row["audit_only"] is True for row in rows)
    assert all(row["not_detector_input"] is True for row in rows)
    manifest = json.loads(
        (tmp_path / "generation" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["program_finalizer"] == "humaneval_target_function_v1"
    assert manifest["finalizer_applied_count"] == 1
    assert manifest["finalizer_fallback_count"] == 1


def test_finalizer_fallback_keeps_sample_and_strict_schema(tmp_path: Path) -> None:
    original = "def f():\n    return ("

    def finalizer(_prompt: str, source: str) -> ProgramFinalizationResult:
        return ProgramFinalizationResult(
            code=source,
            applied=False,
            reason="no_complete_target_implementation",
        )

    class EchoGenerator(_Generator):
        def generate(self, **kwargs) -> GatedGenerationResult:
            return GatedGenerationResult(kwargs["original"], ())

    GatedGenerationPipeline(
        config=_config(tmp_path),
        bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"),
        base_model=_BaseModel(original),
        generator=EchoGenerator(),
        data_adapter=[{"id": "HumanEval/0", "prompt": "def f():\n"}],
        deployment_key=b"key",
        program_finalizer=finalizer,
        program_finalizer_name="humaneval_target_function_v1",
    ).run()

    row = _read_jsonl(tmp_path / "inputs" / "final_code.jsonl")[0]
    assert set(row) == {"id", "dataset", "prompt", "final_code"}
    assert row["final_code"] == original
