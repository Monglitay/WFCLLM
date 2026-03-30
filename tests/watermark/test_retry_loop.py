"""Tests for wfcllm.watermark.retry_loop — rejection sampling retry logic."""

import hashlib
import logging
import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

from wfcllm.watermark.gamma_schedule import GammaResolution
from wfcllm.watermark.retry_loop import RetryLoop, RetryResult, RetryDiagnostics, AttemptInfo
from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.verifier import VerifyResult


class TestRetryResult:
    def test_success_result(self):
        r = RetryResult(
            success=True, attempts=2,
            final_event=MagicMock(spec=InterceptEvent),
            diagnostics=RetryDiagnostics(per_attempt=[], unique_signatures=1, unique_texts=1),
        )
        assert r.success is True
        assert r.attempts == 2

    def test_failure_result(self):
        r = RetryResult(
            success=False, attempts=5, final_event=None,
            diagnostics=RetryDiagnostics(per_attempt=[], unique_signatures=0, unique_texts=0),
        )
        assert r.success is False


class TestRetryLoopUnit:
    """Unit tests with mock GenerationContext."""

    @pytest.fixture
    def config(self):
        return WatermarkConfig(
            secret_key="test-key", max_retries=3, max_new_tokens=50,
            encoder_device="cpu", temperature=0.0,
        )

    @pytest.fixture
    def mock_ctx(self):
        ctx = MagicMock(spec=GenerationContext)
        ctx.generated_ids = []
        ctx.eos_id = 2
        ctx.last_event = None
        return ctx

    @pytest.fixture
    def mock_verifier(self):
        return MagicMock()

    @pytest.fixture
    def mock_keying(self):
        return MagicMock()

    @pytest.fixture
    def mock_entropy(self):
        return MagicMock()

    def test_retry_succeeds_on_first_attempt(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """First retry produces a passing block."""
        # Setup: forward_and_sample triggers event, verifier passes
        event = InterceptEvent(
            block_text="return result", block_type="simple",
            node_type="return_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False

        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.1)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert result.success is True
        assert result.attempts == 1

    def test_retry_exhausts_max_retries(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """All retries fail, returns success=False."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5, 6]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert result.success is False
        assert result.attempts == config.max_retries

    def test_each_retry_calls_rollback(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """Each retry attempt starts with ctx.rollback(checkpoint)."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        loop.run(cp, original_event)
        assert mock_ctx.rollback.call_count == config.max_retries

    def test_diagnostics_records_all_attempts(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """RetryDiagnostics.per_attempt has one entry per attempt."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.return_value = VerifyResult(passed=False, min_margin=0.001)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        assert len(result.diagnostics.per_attempt) == config.max_retries

    def test_no_penalty_on_first_retry(self, config, mock_ctx, mock_verifier, mock_keying, mock_entropy):
        """First retry should not pass penalty_ids to forward_and_sample."""
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )

        # Make first retry succeed
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.forward_and_sample.return_value = 5

        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.1)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        loop.run(cp, original_event)
        # First call to forward_and_sample should have penalty_ids=None
        first_call = mock_ctx.forward_and_sample.call_args_list[0]
        assert first_call.kwargs.get("penalty_ids") is None or first_call.args == ()

    def test_retry_debug_log_includes_gamma_fields(
        self,
        config,
        mock_ctx,
        mock_verifier,
        mock_keying,
        mock_entropy,
        caplog,
    ):
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )

        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.is_finished.return_value = False
        mock_ctx.forward_and_sample.return_value = 5

        mock_verifier.verify.return_value = VerifyResult(passed=True, min_margin=0.1)
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.001

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
            gamma_resolver=lambda block_text: GammaResolution(
                gamma_target=0.75,
                k=6,
                gamma_effective=0.75,
            ),
        )
        cp = MagicMock(spec=Checkpoint)
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        with caplog.at_level(logging.DEBUG, logger="wfcllm.watermark.retry_loop"):
            loop.run(cp, original_event)

        assert "gamma_target=" in caplog.text
        assert "gamma_effective=" in caplog.text
        assert "k=" in caplog.text

    def test_retry_records_signature_miss_with_hash_and_in_valid_set(
        self,
        config,
        mock_ctx,
        mock_verifier,
        mock_keying,
        mock_entropy,
    ):
        config.max_retries = 1
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.return_value = VerifyResult(
            passed=False,
            min_margin=0.2,
            lsh_signature=(1, 1),
            in_valid_set=False,
        )
        mock_keying.derive.return_value = frozenset({(9, 9)})
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.1

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        attempt = result.diagnostics.per_attempt[0]
        assert attempt.failure_reason == "signature_miss"
        assert attempt.in_valid_set is False
        assert attempt.block_text_hash == hashlib.sha256(
            event.block_text.encode("utf-8")
        ).hexdigest()

    def test_retry_records_no_block_generated_attempts(
        self,
        config,
        mock_ctx,
        mock_verifier,
        mock_keying,
        mock_entropy,
    ):
        config.max_retries = 2
        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        with patch.object(loop, "_generate_until_block", return_value=None):
            result = loop.run(cp, original_event)

        assert len(result.diagnostics.per_attempt) == 2
        for attempt in result.diagnostics.per_attempt:
            assert attempt.no_block is True
            assert attempt.failure_reason == "no_block_generated"

    def test_retry_classifies_margin_and_dual_miss_reasons(
        self,
        config,
        mock_ctx,
        mock_verifier,
        mock_keying,
        mock_entropy,
    ):
        config.max_retries = 2
        event = InterceptEvent(
            block_text="x = 1", block_type="simple",
            node_type="expression_statement", parent_node_type="module",
            token_start_idx=0, token_count=2,
        )
        mock_ctx.last_event = event
        mock_ctx.eos_id = 2
        mock_ctx.forward_and_sample.return_value = 5
        mock_ctx.generated_ids = [5]

        mock_verifier.verify.side_effect = [
            VerifyResult(
                passed=False,
                min_margin=0.01,
                lsh_signature=(1, 1),
                in_valid_set=True,
            ),
            VerifyResult(
                passed=False,
                min_margin=0.01,
                lsh_signature=(0, 0),
                in_valid_set=False,
            ),
        ]
        mock_keying.derive.return_value = frozenset()
        mock_entropy.estimate_block_entropy.return_value = 1.0
        mock_entropy.compute_margin.return_value = 0.1

        loop = RetryLoop(
            ctx=mock_ctx, config=config,
            verifier=mock_verifier, keying=mock_keying,
            entropy_est=mock_entropy, structural_token_ids=set(),
        )
        cp = MagicMock(spec=Checkpoint)
        cp.generated_ids = []
        original_event = MagicMock(spec=InterceptEvent)
        original_event.parent_node_type = "module"

        result = loop.run(cp, original_event)
        reasons = [attempt.failure_reason for attempt in result.diagnostics.per_attempt]
        in_valid_values = [attempt.in_valid_set for attempt in result.diagnostics.per_attempt]
        assert reasons == ["margin_miss", "signature_and_margin_miss"]
        assert in_valid_values == [True, False]
