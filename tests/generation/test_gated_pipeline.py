from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
