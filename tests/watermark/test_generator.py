"""Tests for wfcllm.watermark.generator — thin orchestrator."""

from types import SimpleNamespace
import pytest
import torch
from unittest.mock import MagicMock

from wfcllm.watermark.generator import WatermarkGenerator, GenerateResult, EmbedStats
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import InterceptEvent
from wfcllm.watermark.retry_loop import RetryDiagnostics, RetryResult
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
        gen._entropy_est.estimate_block_entropy = MagicMock(return_value=1.0)
        gen._entropy_est.compute_margin = MagicMock(return_value=0.001)
        gen._keying.derive = MagicMock(return_value=frozenset({(1, 1, 1, 1)}))
        gen._verifier.verify = MagicMock(
            return_value=VerifyResult(
                passed=True,
                min_margin=0.5,
                lsh_signature=(1, 1, 1, 1),
            )
        )

        gen.generate("prompt")

        ctx = FakeContext.last_instance
        assert ctx is not None
        assert ctx.rollback_calls == [compound_cp], (
            "同一个 compound block 不应重复 cascade；"
            f"实际 rollback 次数={len(ctx.rollback_calls)}"
        )

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
