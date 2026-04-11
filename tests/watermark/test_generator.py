"""Tests for wfcllm.watermark.generator — thin orchestrator."""

import hashlib
from dataclasses import asdict
import json
import logging
from types import SimpleNamespace
import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.common.block_contract import build_block_contracts
from wfcllm.watermark.diagnostics import hash_block_text
from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.retry_loop import AttemptInfo, RetryDiagnostics, RetryResult
from wfcllm.watermark.token_channel.config import TokenChannelConfig
from wfcllm.watermark.token_channel.features import TokenChannelFeatures
from wfcllm.watermark.verifier import VerifyResult


class TestEmbedStats:
    def test_default_values(self):
        s = EmbedStats()
        assert s.total_blocks == 0
        assert s.embedded_blocks == 0
        assert s.failed_blocks == 0
        assert s.fallback_blocks == 0
        assert s.cascade_blocks == 0
        assert s.retry_diagnostics == []


class TestGenerateResult:
    def test_result_with_stats(self):
        stats = EmbedStats(total_blocks=3, embedded_blocks=2, failed_blocks=1)
        r = GenerateResult(code="x = 1", stats=stats)
        assert r.code == "x = 1"
        assert r.stats.total_blocks == 3

    def test_exposes_adaptive_metadata_defaults(self):
        stats = EmbedStats()
        r = GenerateResult(code="x = 1\n", stats=stats)
        assert r.block_contracts == []
        assert r.adaptive_mode == "fixed"
        assert r.profile_id is None
        assert r.alignment_summary == {}
        assert r.diagnostic_summary == {}
        assert r.block_ledgers == []

    def test_backward_compat_properties(self):
        stats = EmbedStats(total_blocks=5, embedded_blocks=3, failed_blocks=1, fallback_blocks=1)
        r = GenerateResult(code="", stats=stats)
        assert r.total_blocks == 5
        assert r.embedded_blocks == 3
        assert r.failed_blocks == 1
        assert r.fallback_blocks == 1


class TestWatermarkGeneratorInit:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    @pytest.fixture
    def mock_components(self):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        encoder_tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        return model, tokenizer, encoder, encoder_tokenizer

    def test_generator_init(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert gen._config == config
        assert hasattr(gen, "_structural_token_ids")
        assert isinstance(gen._structural_token_ids, set)

    def test_sample_token_method_removed(self, config, mock_components):
        """_sample_token moved to GenerationContext._sample."""
        model, tokenizer, encoder, enc_tok = mock_components
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert not hasattr(gen, "_sample_token")

    def test_generate_returns_generate_result(self, config, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        vocab_size = 100
        logits = torch.zeros(1, 1, vocab_size)
        logits[0, 0, tokenizer.eos_token_id] = 10.0
        past_kv = tuple(
            (torch.randn(1, 4, 3, 32), torch.randn(1, 4, 3, 32))
            for _ in range(2)
        )
        mock_output = MagicMock()
        mock_output.logits = logits
        mock_output.past_key_values = past_kv
        model.return_value = mock_output
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        result = gen.generate("test")
        assert isinstance(result, GenerateResult)
        assert isinstance(result.stats, EmbedStats)
        assert isinstance(result.code, str)

    def test_generate_attaches_canonical_block_metadata(self, config, mock_components, monkeypatch):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")
        expected_contracts = build_block_contracts("x = 1\n")

        assert [asdict(contract) for contract in result.block_contracts] == [
            asdict(contract) for contract in expected_contracts
        ]
        assert result.adaptive_mode == "fixed"
        assert result.profile_id is None
        assert result.alignment_summary == {
            "final_block_count": 1,
            "generator_total_blocks": 1,
            "block_count_matches_total_blocks": True,
        }

    def test_generate_keeps_fixed_metadata_when_adaptive_config_enabled_but_runtime_is_not(self, mock_components, monkeypatch):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
        )
        config.adaptive_gamma.enabled = True

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert result.adaptive_mode == "fixed"
        assert result.profile_id is None

    def test_alignment_summary_marks_mismatch_when_final_blocks_differ(self, config, mock_components, monkeypatch):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.build_block_contracts",
            lambda code, gamma_resolver=None: [],
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert result.alignment_summary == {
            "final_block_count": 0,
            "generator_total_blocks": 1,
            "block_count_matches_total_blocks": False,
        }

    def test_generate_uses_final_ast_stats_when_no_runtime_simple_event_emitted(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "pass"
                    self.generated_ids.append(101)
                    self.last_event = None
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert result.total_blocks == 1
        assert result.embedded_blocks == 1
        assert result.alignment_summary == {
            "final_block_count": 1,
            "generator_total_blocks": 0,
            "block_count_matches_total_blocks": False,
        }

    def test_generate_final_stats_ignore_duplicate_runtime_simple_events(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                if self._steps == 2:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(102)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 102
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 3

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert result.total_blocks == 1
        assert result.embedded_blocks == 1
        assert result.alignment_summary == {
            "final_block_count": 1,
            "generator_total_blocks": 2,
            "block_count_matches_total_blocks": False,
        }

    def test_generate_verifies_flushed_final_simple_block_at_eof(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "return x"
                    self.generated_ids.append(101)
                    self.last_event = None
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

            def flush_final_event(self):
                event = InterceptEvent(
                    block_text="return x",
                    block_type="simple",
                    node_type="return_statement",
                    parent_node_type="module",
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_event = event
                self.last_block_checkpoint = SimpleNamespace(
                    generated_ids=[],
                    generated_text="",
                )
                return event

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert gen._verify_block.call_count == 2
        first_call = gen._verify_block.call_args_list[0]
        assert first_call.args[0].block_text == "return x"
        assert result.alignment_summary == {
            "final_block_count": 1,
            "generator_total_blocks": 1,
            "block_count_matches_total_blocks": True,
        }

    def test_generate_attaches_route_one_diagnostic_summary(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=True,
                    attempts=1,
                    final_event=original_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=True,
                                no_block=False,
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            side_effect=[
                VerifyResult(
                    passed=False,
                    min_margin=0.0,
                    lsh_signature=(1, 1, 1),
                    in_valid_set=False,
                ),
                VerifyResult(
                    passed=True,
                    min_margin=0.2,
                    lsh_signature=(1, 1, 1),
                    in_valid_set=True,
                ),
            ]
        )

        result = gen.generate("prompt")

        assert result.diagnostic_summary["diagnostics_version"] == 1
        assert "retry_summary" in result.diagnostic_summary
        assert "cascade_summary" in result.diagnostic_summary
        assert result.diagnostic_summary["retry_summary"]["retry_rescued_blocks"] == 1

    def test_generate_marks_block_rescued_by_retry(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                replacement_event = InterceptEvent(
                    block_text="return x",
                    block_type="simple",
                    node_type="return_statement",
                    parent_node_type="function_definition",
                    token_start_idx=0,
                    token_count=1,
                )
                return RetryResult(
                    success=True,
                    attempts=1,
                    final_event=replacement_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=True,
                                no_block=False,
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            side_effect=[
                VerifyResult(
                    passed=False,
                    min_margin=0.0,
                    lsh_signature=(1, 1, 1),
                    in_valid_set=False,
                ),
                VerifyResult(
                    passed=True,
                    min_margin=0.2,
                    lsh_signature=(1, 1, 1),
                    in_valid_set=True,
                ),
            ]
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["initial_verify"]["passed"] is False
        assert result.block_ledgers[0]["retry_attempts"][0]["produced_block"] is True
        assert result.block_ledgers[0]["final_outcome"]["embedded"] is True
        assert result.block_ledgers[0]["final_outcome"]["rescued_by_retry"] is True
        assert result.block_ledgers[0]["node_type"] == "return_statement"
        assert result.block_ledgers[0]["parent_node_type"] == "function_definition"
        assert result.block_ledgers[0]["block_text_hash"] == hashlib.sha256(
            "return x".encode("utf-8")
        ).hexdigest()

    def test_generate_retry_failure_uses_last_produced_retry_identity(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                replacement_event = InterceptEvent(
                    block_text="return fallback",
                    block_type="simple",
                    node_type="return_statement",
                    parent_node_type="function_definition",
                    token_start_idx=0,
                    token_count=1,
                )
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=replacement_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=False,
                min_margin=0.0,
                lsh_signature=(1, 1, 1),
                in_valid_set=False,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["final_outcome"]["embedded"] is False
        assert result.block_ledgers[0]["final_outcome"]["exhausted_retries"] is True
        assert result.block_ledgers[0]["retry_attempts"][0]["produced_block"] is True
        assert result.block_ledgers[0]["node_type"] == "return_statement"
        assert result.block_ledgers[0]["parent_node_type"] == "function_definition"
        assert result.block_ledgers[0]["block_text_hash"] == hashlib.sha256(
            "return fallback".encode("utf-8")
        ).hexdigest()

    def test_generate_retry_no_block_keeps_original_identity(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=True,
                                failure_reason="no_block_generated",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=False,
                min_margin=0.0,
                lsh_signature=(1, 1, 1),
                in_valid_set=False,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["retry_attempts"][0]["produced_block"] is False
        assert result.block_ledgers[0]["node_type"] == "expression_statement"
        assert result.block_ledgers[0]["parent_node_type"] == "module"
        assert result.block_ledgers[0]["block_text_hash"] == hashlib.sha256(
            "x = 1".encode("utf-8")
        ).hexdigest()

    def test_generate_failure_uses_terminal_retry_identity_when_final_attempt_produces_block(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        retry_event = InterceptEvent(
            block_text="return x",
            block_type="simple",
            node_type="return_statement",
            parent_node_type="module",
            token_start_idx=0,
            token_count=1,
        )

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=retry_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                                block_text_hash=hash_block_text(retry_event.block_text),
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=False,
                min_margin=0.0,
                lsh_signature=(1, 1, 1),
                in_valid_set=False,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["node_type"] == "return_statement"
        assert result.block_ledgers[0]["parent_node_type"] == "module"
        assert result.block_ledgers[0]["block_text_hash"] == hash_block_text("return x")
        assert result.block_ledgers[0]["final_outcome"]["embedded"] is False

    def test_generate_failure_clears_stale_retry_identity_when_terminal_attempt_has_no_block(
        self,
        config,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                stale_retry_event = InterceptEvent(
                    block_text="return x",
                    block_type="simple",
                    node_type="return_statement",
                    parent_node_type="module",
                    token_start_idx=0,
                    token_count=1,
                )
                return RetryResult(
                    success=False,
                    attempts=2,
                    final_event=stale_retry_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                                block_text_hash=hash_block_text("return x"),
                            ),
                            AttemptInfo(
                                no_block=True,
                                failure_reason="no_block_generated",
                            ),
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=False,
                min_margin=0.0,
                lsh_signature=(1, 1, 1),
                in_valid_set=False,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["node_type"] == "expression_statement"
        assert result.block_ledgers[0]["parent_node_type"] == "module"
        assert result.block_ledgers[0]["block_text_hash"] == hash_block_text("x = 1")
        assert result.block_ledgers[0]["retry_attempts"][0]["produced_block"] is True
        assert result.block_ledgers[0]["retry_attempts"][1]["produced_block"] is False

    def test_generate_marks_pending_failed_block_as_cascade_replaced(
        self,
        mock_components,
        monkeypatch,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="if flag:\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    y = bad\n"),
                    ("eos", ""),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    y = fixed\n"),
                    ("eos", ""),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text = self._sequence[self._cursor]
                self._cursor += 1
                self.generated_ids.append(self._steps)
                self.generated_text += text

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                if kind == "compound":
                    self.last_event = InterceptEvent(
                        block_text=text,
                        block_type="compound",
                        node_type="if_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = compound_cp
                    return 1

                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="if_statement",
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = simple_cp
                return 1

            def is_finished(self):
                return self._steps >= 6

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )

        def fake_verify_block(event):
            return VerifyResult(
                passed=event.block_text.strip() != "y = bad",
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=event.block_text.strip() != "y = bad",
            )

        gen._verify_block = MagicMock(side_effect=fake_verify_block)

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["cascade_events"][0]["triggered"] is True
        assert result.block_ledgers[0]["cascade_events"][0]["compound_node_type"] == "if_statement"
        assert result.block_ledgers[0]["final_outcome"]["rescued_by_cascade"] is True
        assert result.diagnostic_summary["cascade_summary"]["cascade_triggers"] == 1


class TestStructuralTokenFiltering:
    @pytest.fixture
    def config(self):
        return WatermarkConfig(secret_key="test-key", max_new_tokens=50, encoder_device="cpu")

    def test_generator_has_structural_token_ids(self, config):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        call_map = {"import": [10], "return": [11], "def": [12]}
        def encode_side_effect(text, **kw):
            return call_map.get(text, [99])
        tokenizer.encode = MagicMock(side_effect=encode_side_effect)
        gen = WatermarkGenerator(
            model=model, tokenizer=tokenizer,
            encoder=encoder, encoder_tokenizer=enc_tok, config=config,
        )
        assert 10 in gen._structural_token_ids
        assert 11 in gen._structural_token_ids


class TestBlockGammaRuntime:
    def test_verify_block_debug_log_includes_gamma_fields(self, caplog):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            lsh_d=4,
            lsh_gamma=0.75,
        )
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._verifier.verify = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1, 1),
            )
        )

        event = InterceptEvent(
            block_text="x = 1",
            block_type="simple",
            node_type="expression_statement",
            parent_node_type="module",
            token_start_idx=0,
            token_count=1,
        )

        with caplog.at_level(logging.DEBUG, logger="wfcllm.watermark.generator"):
            gen._verify_block(event)

        assert "gamma_target=" in caplog.text
        assert "gamma_effective=" in caplog.text
        assert "k=" in caplog.text

    def test_generate_with_adaptive_gamma_uses_resolved_k_and_emits_metadata(
        self,
        monkeypatch,
        tmp_path,
    ):
        profile_path = tmp_path / "profile.json"
        profile_path.write_text(
            json.dumps(
                {
                    "language": "python",
                    "model_family": "demo-model",
                    "quantiles_units": {
                        "p10": 1,
                        "p50": 2,
                        "p75": 3,
                        "p90": 4,
                        "p95": 5,
                    },
                }
            ),
            encoding="utf-8",
        )

        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=50,
            encoder_device="cpu",
            lsh_d=4,
        )
        config.adaptive_gamma.enabled = True
        config.adaptive_gamma.profile_path = str(profile_path)
        config.adaptive_gamma.profile_id = "entropy-profile-v1"

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )

        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        enc_tok = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._keying.derive = MagicMock(return_value=frozenset())
        gen._verifier.verify = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.1,
                lsh_signature=(1, 1, 1, 1),
            )
        )

        result = gen.generate("prompt")

        assert result.adaptive_mode == "piecewise_quantile"
        assert result.profile_id == "entropy-profile-v1"
        assert result.block_contracts[0].k == 4
        assert result.block_contracts[0].gamma_effective == pytest.approx(4 / 16)
        derive_call = gen._keying.derive.call_args_list[0]
        assert derive_call.args[0] == "module"
        assert derive_call.kwargs["k"] == 4


class TestFallbackDeprecated:
    """passive fallback 已废弃：generator 不应再有 _try_passive_fallback。"""

    def test_generator_has_no_try_passive_fallback(self):
        """废弃后 WatermarkGenerator 实例不应存在 _try_passive_fallback 方法。"""
        import torch
        from unittest.mock import MagicMock
        from wfcllm.watermark.generator import WatermarkGenerator
        from wfcllm.watermark.config import WatermarkConfig
        config = WatermarkConfig(secret_key="k", encoder_device="cpu")
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        encoder = MagicMock()
        enc_tok = MagicMock()
        gen = WatermarkGenerator(model, tokenizer, encoder, enc_tok, config)
        assert not hasattr(gen, "_try_passive_fallback"), (
            "WatermarkGenerator._try_passive_fallback 应已删除"
        )

    def test_enable_fallback_field_removed_from_config(self):
        """WatermarkConfig 不再有 enable_fallback 字段。"""
        from wfcllm.watermark.config import WatermarkConfig
        cfg = WatermarkConfig(secret_key="k")
        assert not hasattr(cfg, "enable_fallback"), (
            "WatermarkConfig.enable_fallback 已废弃，应已删除"
        )

    def test_enable_cascade_defaults_true(self):
        """enable_cascade 默认值改为 True。"""
        from wfcllm.watermark.config import WatermarkConfig
        cfg = WatermarkConfig(secret_key="k")
        assert cfg.enable_cascade is True


class TestCascadeRegression:
    def test_same_type_nested_compound_does_not_steal_outer_replacement_ordinal(
        self,
        monkeypatch,
    ):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        outer_compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        failed_simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="if outer:\n",
        )
        nested_compound_cp = SimpleNamespace(
            generated_ids=[102],
            generated_text="if outer:\n    if inner:\n",
        )
        recovered_outer_simple_cp = SimpleNamespace(
            generated_ids=[103],
            generated_text="if outer:\n    if inner:\n        keep = 1\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "if outer:\n", "if_statement", "module", outer_compound_cp),
                    ("simple", "    x = bad\n", "expression_statement", "if_statement", failed_simple_cp),
                    ("eos", "", None, None, None),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "if outer:\n", "if_statement", "module", outer_compound_cp),
                    ("compound", "    if inner:\n", "if_statement", "if_statement", nested_compound_cp),
                    ("simple", "        keep = 1\n", "expression_statement", "if_statement", nested_compound_cp),
                    ("simple", "    x = fixed\n", "expression_statement", "if_statement", recovered_outer_simple_cp),
                    ("eos", "", None, None, None),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text, node_type, parent_node_type, checkpoint = self._sequence[self._cursor]
                self._cursor += 1

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                self.generated_ids.append(self._steps)
                self.generated_text += text
                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type=kind,
                    node_type=node_type,
                    parent_node_type=parent_node_type,
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = checkpoint
                return 1

            def is_finished(self):
                return self._steps >= 8

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            stripped = event.block_text.strip()
            return VerifyResult(
                passed=stripped != "x = bad",
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=stripped != "x = bad",
            )

        gen._verify_block = MagicMock(side_effect=fake_verify_block)

        result = gen.generate("prompt")

        replaced = result.block_ledgers[0]
        regenerated_ledgers = [
            item for item in result.block_ledgers
            if item["block_text_hash"] != replaced["block_text_hash"]
        ]

        assert replaced["final_outcome"]["failure_reason"] == "cascade_replaced"
        assert replaced["final_outcome"].get("rescued_by_cascade") is False
        assert all(
            item["final_outcome"].get("rescued_by_cascade") is not True
            for item in regenerated_ledgers
        )
        assert result.diagnostic_summary["cascade_summary"]["cascade_rescued_blocks"] == 0

    def test_regenerated_compound_with_fewer_direct_simple_blocks_does_not_reuse_stale_replacement(
        self,
        monkeypatch,
    ):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="for i in items:\n",
        )
        nested_compound_cp = SimpleNamespace(
            generated_ids=[102],
            generated_text="for i in items:\n    if guard:\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "for i in items:\n", "for_statement", "module", compound_cp),
                    ("simple", "    x = bad\n", "expression_statement", "for_statement", simple_cp),
                    ("eos", "", None, None, None),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "for i in items:\n", "for_statement", "module", compound_cp),
                    ("compound", "    if guard:\n", "if_statement", "for_statement", nested_compound_cp),
                    ("simple", "        keep = i\n", "expression_statement", "if_statement", nested_compound_cp),
                    ("simple", "result = keep\n", "expression_statement", "module", simple_cp),
                    ("eos", "", None, None, None),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text, node_type, parent_node_type, checkpoint = self._sequence[self._cursor]
                self._cursor += 1

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                self.generated_ids.append(self._steps)
                self.generated_text += text
                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type=kind,
                    node_type=node_type,
                    parent_node_type=parent_node_type,
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = checkpoint
                return 1

            def is_finished(self):
                return self._steps >= 8

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            stripped = event.block_text.strip()
            return VerifyResult(
                passed=stripped != "x = bad",
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=stripped != "x = bad",
            )

        gen._verify_block = MagicMock(side_effect=fake_verify_block)

        result = gen.generate("prompt")

        replaced = result.block_ledgers[0]
        later_top_level = next(
            item for item in result.block_ledgers
            if item["parent_node_type"] == "module"
        )

        assert replaced["final_outcome"]["failure_reason"] == "cascade_replaced"
        assert replaced["final_outcome"].get("rescued_by_cascade") is False
        assert later_top_level["parent_node_type"] == "module"
        assert later_top_level["final_outcome"]["embedded"] is True
        assert later_top_level["final_outcome"].get("rescued_by_cascade") is False

    def test_later_top_level_simple_does_not_inherit_stale_cascade_replacement_state(
        self,
        monkeypatch,
    ):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        failed_simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="for i in items:\n",
        )
        nested_compound_cp = SimpleNamespace(
            generated_ids=[102],
            generated_text="for i in items:\n    if guard:\n",
        )
        top_level_cp = SimpleNamespace(
            generated_ids=[201],
            generated_text="for i in items:\n    if guard:\n        keep = i\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "for i in items:\n", "for_statement", "module", compound_cp),
                    ("simple", "    x = bad\n", "expression_statement", "for_statement", failed_simple_cp),
                    ("eos", "", None, None, None),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "for i in items:\n", "for_statement", "module", compound_cp),
                    ("compound", "    if guard:\n", "if_statement", "for_statement", nested_compound_cp),
                    ("simple", "        keep = i\n", "expression_statement", "if_statement", nested_compound_cp),
                    ("simple", "summary = keep\n", "expression_statement", "module", top_level_cp),
                    ("eos", "", None, None, None),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text, node_type, parent_node_type, checkpoint = self._sequence[self._cursor]
                self._cursor += 1

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                self.generated_ids.append(self._steps)
                self.generated_text += text
                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type=kind,
                    node_type=node_type,
                    parent_node_type=parent_node_type,
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = checkpoint
                return 1

            def is_finished(self):
                return self._steps >= 8

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            stripped = event.block_text.strip()
            return VerifyResult(
                passed=stripped != "x = bad",
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=stripped != "x = bad",
            )

        gen._verify_block = MagicMock(side_effect=fake_verify_block)

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["final_outcome"]["failure_reason"] == "cascade_replaced"
        top_level_ledger = next(
            item for item in result.block_ledgers
            if item["parent_node_type"] == "module"
        )
        assert top_level_ledger["final_outcome"].get("rescued_by_cascade") is False
        assert result.diagnostic_summary["cascade_summary"]["cascade_rescued_blocks"] == 0

    def test_cascade_rescued_block_clears_stale_terminal_failure_fields(
        self,
        monkeypatch,
    ):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        failed_simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="if flag:\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    y = bad\n"),
                    ("eos", ""),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    y = fixed\n"),
                    ("eos", ""),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text = self._sequence[self._cursor]
                self._cursor += 1
                self.generated_ids.append(self._steps)
                self.generated_text += text

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                if kind == "compound":
                    self.last_event = InterceptEvent(
                        block_text=text,
                        block_type="compound",
                        node_type="if_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = compound_cp
                    return 1

                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="if_statement",
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = failed_simple_cp
                return 1

            def is_finished(self):
                return self._steps >= 6

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event):
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[
                            AttemptInfo(
                                passed=False,
                                no_block=False,
                                failure_reason="signature_miss",
                            )
                        ]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            return VerifyResult(
                passed=event.block_text.strip() != "y = bad",
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=event.block_text.strip() != "y = bad",
            )

        gen._verify_block = MagicMock(side_effect=fake_verify_block)

        result = gen.generate("prompt")

        final_outcome = result.block_ledgers[0]["final_outcome"]
        assert final_outcome["embedded"] is True
        assert final_outcome["rescued_by_cascade"] is True
        assert "failure_reason" not in final_outcome
        assert "exhausted_retries" not in final_outcome

    def test_same_compound_cascades_at_most_once(self, monkeypatch):
        """同一个 compound block 成功 cascade 后，不应再次回滚到同一起点。"""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )

        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="for i in range(n):\n",
        )

        class FakeContext:
            last_instance = None

            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self.rollback_calls = []
                self._sequence = self._first_pass_sequence()
                FakeContext.last_instance = self

            def _first_pass_sequence(self):
                return [
                    ("compound", "for i in range(n):"),
                    ("simple", "my_list.append(4)"),
                    ("simple", "my_list.append(2)"),
                    ("simple", "my_list.append(1)"),
                    ("eos", ""),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "for i in range(n):"),
                    ("compound", "for i in range(n):\n    if x > 0:"),
                    ("simple", "my_list.append(4)"),
                    ("simple", "my_list.append(2)"),
                    ("simple", "my_list.append(1)"),
                    ("eos", ""),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.rollback_calls.append(cp)
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text = self._sequence[self._cursor]
                self._cursor += 1
                self.generated_ids.append(self._steps)
                self.generated_text += text

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                if kind == "compound":
                    self.last_event = InterceptEvent(
                        block_text=text,
                        block_type="compound",
                        node_type="for_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = compound_cp
                    return 1

                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="if_statement",
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = simple_cp
                return 1

            def is_finished(self):
                return self._steps >= 14

        class FakeRetryLoop:
            last_instance = None

            def __init__(self, **kwargs):
                self.calls = []
                FakeRetryLoop.last_instance = self

            def run(self, checkpoint, original_event):
                self.calls.append((checkpoint, original_event.block_text))
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            return VerifyResult(
                passed=event.block_text != "my_list.append(1)",
                min_margin=0.01,
                lsh_signature=(0, 0, 0, 0),
            )

        gen._verify_block = fake_verify_block


class TestTokenChannelGeneration:
    @pytest.fixture
    def mock_components(self):
        model = MagicMock()
        tokenizer = MagicMock()
        encoder = MagicMock()
        encoder_tokenizer = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        tokenizer.name_or_path = "demo-tokenizer"
        tokenizer.__len__ = MagicMock(return_value=8)
        return model, tokenizer, encoder, encoder_tokenizer

    def test_generator_loads_token_channel_runtime_from_artifact_when_enabled(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                model_path="/tmp/token-channel-artifact",
            ),
        )
        artifact = SimpleNamespace(model=MagicMock(), metadata=MagicMock())
        captured: dict[str, object] = {}

        class FakeRuntime:
            def __init__(self, **kwargs):
                captured.update(kwargs)

        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: artifact,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            FakeRuntime,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        assert isinstance(gen._token_channel_runtime, FakeRuntime)
        assert captured["model"] is artifact.model
        assert captured["config"] is config.token_channel
        assert captured["artifact_metadata"] is artifact.metadata
        assert captured["tokenizer"] is tokenizer
        assert captured["secret_key"] == "test-key"

    def test_apply_token_channel_bias_skips_scoring_when_structure_mask_is_false(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(enabled=True, mode="dual-channel"),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._token_channel_runtime = MagicMock()
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=False,
                structure_mask=False,
            )
        )
        ctx = SimpleNamespace(
            generated_ids=[1, 2, 3],
            generated_text="x = 1",
            _next_logits=torch.zeros(1, 8),
        )
        lexical_state = gen._create_token_channel_state()

        applied = gen._apply_token_channel_bias(ctx, lexical_state)

        assert applied is False
        gen._token_channel_runtime.score_prefix.assert_not_called()

    def test_apply_token_channel_bias_scores_first_scorable_token_without_min_block_delay(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                lexical_min_block_tokens=8,
                delta=1.5,
            ),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        partition = SimpleNamespace(green_token_ids=[1, 3])
        gen._token_channel_runtime = MagicMock()
        gen._token_channel_runtime.score_prefix.return_value = SimpleNamespace(
            should_switch=True,
            partition=partition,
        )
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=True,
                structure_mask=True,
            )
        )
        ctx = SimpleNamespace(
            generated_ids=[1, 2, 3],
            generated_text="x = ",
            _next_logits=torch.zeros(1, 8),
        )
        lexical_state = gen._create_token_channel_state()

        applied = gen._apply_token_channel_bias(ctx, lexical_state)

        assert applied is True
        gen._token_channel_runtime.score_prefix.assert_called_once_with(
            ctx.generated_ids,
            features=gen._build_runtime_token_features.return_value,
        )
        assert ctx._next_logits[0, 1].item() == pytest.approx(1.5)
        assert ctx._next_logits[0, 3].item() == pytest.approx(1.5)

    def test_token_channel_low_gate_fraction_probe_starts_from_first_scorable_token(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                lexical_min_block_tokens=8,
                lexical_gate_probe_tokens=2,
                lexical_gate_min_fraction=0.75,
            ),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._token_channel_runtime = MagicMock()
        gen._token_channel_runtime.score_prefix.return_value = SimpleNamespace(
            should_switch=False,
            partition=SimpleNamespace(green_token_ids=[1]),
        )
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=True,
                structure_mask=True,
            )
        )
        ctx = SimpleNamespace(
            generated_ids=[1, 2, 3],
            generated_text="x = ",
            _next_logits=torch.zeros(1, 8),
        )
        lexical_state = gen._create_token_channel_state()

        assert gen._apply_token_channel_bias(ctx, lexical_state) is False
        assert lexical_state.disabled_for_block is False

        lexical_state.current_block_tokens += 1
        assert gen._apply_token_channel_bias(ctx, lexical_state) is False
        assert lexical_state.disabled_for_block is True
        assert lexical_state.low_gate_fraction_shutdown is True
        assert lexical_state.scorable_tokens == 2
        assert gen._token_channel_runtime.score_prefix.call_count == 2

    def test_lexical_only_generation_skips_semantic_verification_and_retry(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=16,
            encoder_device="cpu",
            token_channel=TokenChannelConfig(enabled=True, mode="lexical-only"),
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._verify_block = MagicMock(side_effect=AssertionError("should not verify"))

        result = gen.generate("prompt")

        assert result.total_blocks == 1
        assert result.embedded_blocks == 0
        assert result.failed_blocks == 0
        assert result.diagnostic_summary["token_channel_enabled"] is True
        assert result.diagnostic_summary["generation_mode"] == "lexical-only"

    def test_lexical_only_short_block_is_regenerated_without_token_channel_bias(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=16,
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="lexical-only",
                lexical_min_block_tokens=2,
            ),
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._phase = "initial"
                self.rollback_calls = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, checkpoint):
                self.rollback_calls += 1
                self.generated_ids = list(checkpoint.generated_ids)
                self.generated_text = checkpoint.generated_text
                self.last_event = None
                self.last_block_checkpoint = None
                self._phase = "retry"

            def forward_and_sample(self, penalty_ids=None):
                if self._phase == "initial":
                    self._phase = "initial-eos"
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                if self._phase == "retry":
                    self._phase = "retry-eos"
                    self.generated_text = "y = 2\n"
                    self.generated_ids.append(202)
                    self.last_event = InterceptEvent(
                        block_text="y = 2",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 202
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._phase in {"initial-eos", "retry-eos"}

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._token_channel_runtime = MagicMock()
        gen._token_channel_runtime.score_prefix.return_value = SimpleNamespace(
            should_switch=True,
            partition=SimpleNamespace(green_token_ids=[1]),
        )
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=True,
                structure_mask=True,
            )
        )
        gen._verify_block = MagicMock(side_effect=AssertionError("should not verify"))

        result = gen.generate("prompt")

        assert result.code == "y = 2\n"
        assert result.total_blocks == 1
        assert result.diagnostic_summary["generation_mode"] == "lexical-only"
        assert any(call.args[0] == [101] for call in gen._token_channel_runtime.score_prefix.call_args_list)
        assert all(call.args[0] != [202] for call in gen._token_channel_runtime.score_prefix.call_args_list)

    def test_token_channel_retry_rules_decay_then_disable_bias(self, monkeypatch, mock_components):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                delta=2.0,
                lexical_retry_decay_start=2,
                lexical_retry_disable_after=4,
            ),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        lexical_state = gen._create_token_channel_state()

        lexical_state.semantic_failure_count = 1
        assert gen._resolve_token_channel_delta(lexical_state) == pytest.approx(2.0)

        lexical_state.semantic_failure_count = 2
        assert gen._resolve_token_channel_delta(lexical_state) == pytest.approx(1.0)

        lexical_state.semantic_failure_count = 4
        assert gen._resolve_token_channel_delta(lexical_state) == 0.0

    def test_token_channel_state_resets_block_failure_count_after_simple_block(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(enabled=True, mode="dual-channel"),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        lexical_state = gen._create_token_channel_state()
        lexical_state.semantic_failure_count = 3
        stats = EmbedStats(failed_blocks=9)

        gen._update_token_channel_state_after_simple_block(lexical_state, stats)

        assert lexical_state.semantic_failure_count == 0

    def test_retry_generation_uses_token_channel_bias_when_lexical_mode_is_enabled(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=16,
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                lexical_min_block_tokens=1,
            ),
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event, attempt_pre_sample_hook_factory=None):
                retry_ctx = SimpleNamespace(
                    generated_ids=[11, 12],
                    generated_text="retry = ",
                    _next_logits=torch.zeros(1, 8),
                )
                hook = attempt_pre_sample_hook_factory(1)
                hook(retry_ctx)
                replacement_event = InterceptEvent(
                    block_text="retry = 1",
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="module",
                    token_start_idx=0,
                    token_count=1,
                )
                return RetryResult(
                    success=True,
                    attempts=1,
                    final_event=replacement_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[AttemptInfo(passed=True, no_block=False)]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._token_channel_runtime = MagicMock()
        gen._token_channel_runtime.score_prefix.return_value = SimpleNamespace(
            should_switch=True,
            partition=SimpleNamespace(green_token_ids=[1]),
        )
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=True,
                structure_mask=True,
            )
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=False,
                min_margin=0.0,
                lsh_signature=(1, 1, 1),
                in_valid_set=False,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["final_outcome"]["rescued_by_retry"] is True
        assert any(
            call.args[0] == [11, 12]
            for call in gen._token_channel_runtime.score_prefix.call_args_list
        )

    def test_short_block_with_early_bias_is_regenerated_without_token_channel_bias(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=16,
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                lexical_min_block_tokens=2,
            ),
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_text = ""
                self.generated_ids = []
                self.last_event = None
                self.last_block_checkpoint = None
                self._steps = 0

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._steps == 1:
                    self.generated_text = "x = 1\n"
                    self.generated_ids.append(101)
                    self.last_event = InterceptEvent(
                        block_text="x = 1",
                        block_type="simple",
                        node_type="expression_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = SimpleNamespace(
                        generated_ids=[],
                        generated_text="",
                    )
                    return 101
                self.last_event = None
                return self.eos_id

            def is_finished(self):
                return self._steps >= 2

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                return None

            def run(self, checkpoint, original_event, attempt_pre_sample_hook_factory=None):
                retry_ctx = SimpleNamespace(
                    generated_ids=[11, 12],
                    generated_text="retry = ",
                    _next_logits=torch.zeros(1, 8),
                )
                hook = None
                if attempt_pre_sample_hook_factory is not None:
                    hook = attempt_pre_sample_hook_factory(1)
                assert hook is None
                replacement_event = InterceptEvent(
                    block_text="retry = 1",
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="module",
                    token_start_idx=0,
                    token_count=1,
                )
                return RetryResult(
                    success=True,
                    attempts=1,
                    final_event=replacement_event,
                    diagnostics=RetryDiagnostics(
                        per_attempt=[AttemptInfo(passed=True, no_block=False)]
                    ),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        gen._token_channel_runtime = MagicMock()
        gen._token_channel_runtime.score_prefix.return_value = SimpleNamespace(
            should_switch=True,
            partition=SimpleNamespace(green_token_ids=[1]),
        )
        gen._build_runtime_token_features = MagicMock(
            return_value=TokenChannelFeatures(
                node_type="expression_statement",
                parent_node_type="module",
                block_relative_offset=0,
                in_code_body=True,
                structure_mask=True,
            )
        )
        gen._verify_block = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.2,
                lsh_signature=(1, 1, 1),
                in_valid_set=True,
            )
        )

        result = gen.generate("prompt")

        assert result.block_ledgers[0]["final_outcome"]["rescued_by_retry"] is True
        assert result.block_ledgers[0]["block_text_hash"] == hashlib.sha256(
            "retry = 1".encode("utf-8")
        ).hexdigest()
        assert all(
            call.args[0] != [11, 12]
            for call in gen._token_channel_runtime.score_prefix.call_args_list
        )

    def test_token_channel_low_gate_fraction_shutdown_uses_early_scorable_window(
        self,
        monkeypatch,
        mock_components,
    ):
        model, tokenizer, encoder, enc_tok = mock_components
        config = WatermarkConfig(
            secret_key="test-key",
            encoder_device="cpu",
            token_channel=TokenChannelConfig(
                enabled=True,
                mode="dual-channel",
                lexical_gate_probe_tokens=16,
                lexical_gate_min_fraction=0.25,
            ),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.load_token_channel_artifact",
            lambda path: SimpleNamespace(model=MagicMock(), metadata=MagicMock()),
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.TokenChannelRuntime",
            lambda **kwargs: MagicMock(),
        )
        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )
        lexical_state = gen._create_token_channel_state()
        lexical_state.scorable_tokens = 16
        lexical_state.gated_tokens = 3

        assert gen._should_disable_token_channel_for_low_gate_fraction(lexical_state) is True

    def test_try_cascade_resumes_main_loop_without_compound_verification(self):
        """_try_cascade 不应在内部验证 compound 中间态文本。"""
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        ctx = MagicMock()
        retry_loop = MagicMock()
        stats = EmbedStats()
        pending_fallbacks = ["my_list.append(1)"]
        cascade_cp = SimpleNamespace(
            checkpoint=SimpleNamespace(generated_ids=[], generated_text=""),
            compound_event=InterceptEvent(
                block_text="for i in range(n):",
                block_type="compound",
                node_type="for_statement",
                parent_node_type="module",
                token_start_idx=0,
                token_count=1,
            ),
            checkpoint_key=("k",),
            failed_simple_blocks=["my_list.append(1)"],
        )
        cascade_mgr = MagicMock()
        cascade_mgr.cascade.return_value = cascade_cp

        gen._verifier.verify = MagicMock()
        ctx.forward_and_sample = MagicMock()

        gen._try_cascade(ctx, cascade_mgr, retry_loop, stats, pending_fallbacks)

        gen._verifier.verify.assert_not_called()
        ctx.forward_and_sample.assert_not_called()
        assert stats.cascade_blocks == 1
        assert pending_fallbacks == []

    def test_cascade_restores_runtime_counts_to_compound_checkpoint(self, monkeypatch):
        config = WatermarkConfig(
            secret_key="test-key",
            max_new_tokens=64,
            encoder_device="cpu",
            enable_cascade=True,
        )
        model = MagicMock()
        model.parameters = MagicMock(return_value=iter([torch.zeros(1)]))
        tokenizer = MagicMock()
        tokenizer.encode = MagicMock(return_value=[1])
        tokenizer.decode = MagicMock(return_value="")
        tokenizer.eos_token_id = 2
        encoder = MagicMock()
        enc_tok = MagicMock()

        compound_cp = SimpleNamespace(
            generated_ids=[],
            generated_text="",
        )
        simple_cp = SimpleNamespace(
            generated_ids=[101],
            generated_text="if flag:\n",
        )

        class FakeContext:
            def __init__(self, model, tokenizer, config):
                self.generated_ids = []
                self.generated_text = ""
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._steps = 0
                self._sequence = self._first_pass_sequence()

            def _first_pass_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    x = old\n"),
                    ("simple", "    y = bad\n"),
                    ("eos", ""),
                ]

            def _post_cascade_sequence(self):
                return [
                    ("compound", "if flag:\n"),
                    ("simple", "    x = final\n"),
                    ("eos", ""),
                ]

            @property
            def eos_id(self):
                return -1

            def prefill(self, prompt):
                return None

            def rollback(self, cp):
                self.generated_ids = list(getattr(cp, "generated_ids", []))
                self.generated_text = getattr(cp, "generated_text", "")
                self.last_event = None
                self.last_block_checkpoint = None
                self._cursor = 0
                self._sequence = self._post_cascade_sequence()

            def forward_and_sample(self, penalty_ids=None):
                self._steps += 1
                if self._cursor >= len(self._sequence):
                    self.last_event = None
                    return self.eos_id

                kind, text = self._sequence[self._cursor]
                self._cursor += 1
                self.generated_ids.append(self._steps)
                self.generated_text += text

                if kind == "eos":
                    self.last_event = None
                    return self.eos_id

                if kind == "compound":
                    self.last_event = InterceptEvent(
                        block_text=text,
                        block_type="compound",
                        node_type="if_statement",
                        parent_node_type="module",
                        token_start_idx=0,
                        token_count=1,
                    )
                    self.last_block_checkpoint = compound_cp
                    return 1

                self.last_event = InterceptEvent(
                    block_text=text,
                    block_type="simple",
                    node_type="expression_statement",
                    parent_node_type="if_statement",
                    token_start_idx=0,
                    token_count=1,
                )
                self.last_block_checkpoint = simple_cp
                return 1

            def is_finished(self):
                return self._steps >= 8

        class FakeRetryLoop:
            def __init__(self, **kwargs):
                self.calls = []

            def run(self, checkpoint, original_event):
                self.calls.append((checkpoint, original_event.block_text))
                return RetryResult(
                    success=False,
                    attempts=1,
                    final_event=None,
                    diagnostics=RetryDiagnostics(),
                )

        monkeypatch.setattr(
            "wfcllm.watermark.generator.GenerationContext",
            FakeContext,
        )
        monkeypatch.setattr(
            "wfcllm.watermark.generator.RetryLoop",
            FakeRetryLoop,
        )

        gen = WatermarkGenerator(
            model=model,
            tokenizer=tokenizer,
            encoder=encoder,
            encoder_tokenizer=enc_tok,
            config=config,
        )

        def fake_verify_block(event):
            return VerifyResult(
                passed=event.block_text != "    y = bad\n",
                min_margin=0.01,
                lsh_signature=(0, 0, 0, 0),
            )

        gen._verify_block = fake_verify_block

        result = gen.generate("prompt")

        assert result.total_blocks == 1
        assert result.embedded_blocks == 1
        assert result.failed_blocks == 0
        assert result.alignment_summary == {
            "final_block_count": 1,
            "generator_total_blocks": 1,
            "block_count_matches_total_blocks": True,
        }
