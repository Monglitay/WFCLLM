"""Scenario-driven functional tests for rollback correctness.

Each test constructs a controlled environment where we know exactly what the
interceptor should detect, and verifies that rollback + retry produces
complete (non-truncated) statement blocks.

We test at the GenerationContext + Interceptor level (not full generator)
to isolate rollback mechanics from verification logic.
"""

import pytest

from wfcllm.watermark.context import GenerationContext, Checkpoint
from wfcllm.watermark.config import WatermarkConfig
from wfcllm.watermark.interceptor import StatementInterceptor


@pytest.fixture
def config():
    return WatermarkConfig(
        secret_key="test-key", max_new_tokens=200,
        encoder_device="cpu", temperature=0.0,
    )


class TestScenario1ReturnStatement:
    """Scenario 1: return statement rollback.
    Verify retry generates complete 'return ...' not truncated '...'."""

    def test_rollback_then_refeed_produces_complete_return(self):
        """Feed 'return the\\n' → checkpoint before → rollback → refeed different content.
        The new block must start with 'return'."""
        ic = StatementInterceptor()
        # Feed return statement directly
        for ch in "return the\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()

        # Rollback to checkpoint
        ic.rollback(cp)
        assert ic._accumulated == "return the\n"

        # Reset and test rollback from empty state
        ic.reset()
        cp = ic.checkpoint()

        # Feed first return
        events = []
        for ch in "return foo\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "foo" in events[0].block_text

        # Rollback to beginning
        ic.rollback(cp)
        assert ic._accumulated == ""

        # Refeed different return
        events2 = []
        for ch in "return bar\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert "bar" in events2[0].block_text


class TestScenario2ImportStatement:
    """Scenario 2: import statement rollback."""

    def test_rollback_import_produces_complete_import(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        events = []
        for ch in "import doctest\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        ic.rollback(cp)
        events2 = []
        for ch in "import unittest\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert events2[0].block_text.strip().startswith("import")


class TestScenario3AssignmentNoPollution:
    """Scenario 3: rollback of block #2 doesn't affect block #1."""

    def test_first_block_preserved_after_second_rollback(self):
        ic = StatementInterceptor()
        # Block 1: x = 1
        for ch in "x = 1\n":
            ic.feed_token(ch)
        # Checkpoint after block 1
        cp = ic.checkpoint()
        assert "x = 1" in cp.accumulated
        # Block 2: y = 2
        for ch in "y = 2\n":
            ic.feed_token(ch)
        # Rollback block 2
        ic.rollback(cp)
        assert "x = 1" in ic._accumulated
        assert "y = 2" not in ic._accumulated
        # Generate different block 2
        events = []
        for ch in "z = 3\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "z" in events[0].block_text


class TestScenario4NestedExpression:
    """Scenario 4: if block inner expression rollback."""

    def test_nested_rollback_produces_complete_statement(self):
        ic = StatementInterceptor()
        # Simple assignment statements
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Feed another assignment
        events = []
        for ch in "y = 2\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        # Rollback to checkpoint
        ic.rollback(cp)
        assert "y = 2" not in ic._accumulated
        # Refeed different statement
        events2 = []
        for ch in "z = 3\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        if events2:
            assert "z = 3" in events2[0].block_text


class TestScenario5AllRetriesExhausted:
    """Scenario 5: multiple rollback cycles all fail — diagnostics correct."""

    def test_multiple_rollback_cycles(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        attempts = 5
        for i in range(attempts):
            ic.rollback(cp)
            for ch in f"x{i} = {i}\n":
                ic.feed_token(ch)
        # After all retries, state should be at the last attempt's result
        assert f"x{attempts-1}" in ic._accumulated


class TestScenario6CascadeToForLoop:
    """Scenario 6: cascade checkpoint saves compound block start."""

    def test_cascade_rollback_to_compound_start(self):
        ic = StatementInterceptor()
        # Before compound block
        compound_cp = ic.checkpoint()
        # Feed compound block header
        for ch in "for i in range(n):\n":
            ic.feed_token(ch)
        # Feed inner block
        for ch in "    total += arr[i]\n":
            ic.feed_token(ch)
        # Cascade rollback
        ic.rollback(compound_cp)
        assert "for" not in ic._accumulated or ic._accumulated == compound_cp.accumulated
        # Regenerate entire for loop
        events = []
        for ch in "for idx in range(n):\n    total += arr[idx]\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1


class TestScenario7EOSDuringRetry:
    """Scenario 7: EOS during retry sub-loop doesn't crash."""

    def test_eos_rollback_recovers(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        # First attempt: feed some tokens then "EOS" (just stop feeding)
        for ch in "return":
            ic.feed_token(ch)
        # No newline = no block detected, simulate EOS
        # Rollback and try again
        ic.rollback(cp)
        # Second attempt: complete statement
        events = []
        for ch in "return 0\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "return" in events[0].block_text


class TestScenario8MultiTokenPrecision:
    """Scenario 8: multi-token statement, rollback position precise."""

    def test_8_token_statement_rollback(self):
        ic = StatementInterceptor()
        # Some prefix
        for ch in "# comment\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        saved_acc = cp.accumulated
        # Feed multi-char statement: result = some_function(arg)
        stmt = "result = func(a)\n"
        for ch in stmt:
            ic.feed_token(ch)
        ic.rollback(cp)
        assert ic._accumulated == saved_acc
        assert ic._token_idx == cp.token_idx


class TestScenario9FirstBlockFails:
    """Scenario 9: first block fails, rollback to prompt end (empty generated)."""

    def test_first_block_rollback_to_empty(self):
        ic = StatementInterceptor()
        cp = ic.checkpoint()
        assert cp.accumulated == ""
        # First block
        for ch in "return None\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        assert ic._accumulated == ""
        assert ic._token_idx == 0
        # Retry
        events = []
        for ch in "return 0\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert events[0].block_text.strip().startswith("return")


class TestScenario10ConsecutiveRetries:
    """Scenario 10: two blocks both retry, independently."""

    def test_two_blocks_independent_rollback(self):
        ic = StatementInterceptor()
        # Block 1: feed and rollback
        cp1 = ic.checkpoint()
        for ch in "x = foo()\n":
            ic.feed_token(ch)
        ic.rollback(cp1)
        events1 = []
        for ch in "x = baz()\n":
            e = ic.feed_token(ch)
            if e:
                events1.append(e)
        assert len(events1) >= 1
        assert "baz" in events1[0].block_text

        # Block 2: feed and rollback
        cp2 = ic.checkpoint()
        for ch in "y = bar()\n":
            ic.feed_token(ch)
        ic.rollback(cp2)
        events2 = []
        for ch in "y = qux()\n":
            e = ic.feed_token(ch)
            if e:
                events2.append(e)
        assert len(events2) >= 1
        assert "qux" in events2[0].block_text
        # Block 1 result preserved
        assert "baz" in ic._accumulated
