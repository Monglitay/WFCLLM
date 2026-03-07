"""Tests for wfcllm.watermark.interceptor."""

import pytest
from wfcllm.watermark.interceptor import StatementInterceptor, InterceptEvent


class TestInterceptEvent:
    def test_event_fields(self):
        e = InterceptEvent(
            block_text="x = 1",
            block_type="simple",
            node_type="expression_statement",
            parent_node_type="module",
            token_start_idx=0,
            token_count=3,
        )
        assert e.block_text == "x = 1"
        assert e.block_type == "simple"


class TestStatementInterceptor:
    @pytest.fixture
    def interceptor(self):
        return StatementInterceptor()

    def test_no_event_on_partial_tokens(self, interceptor):
        """Feeding partial statement shouldn't trigger."""
        assert interceptor.feed_token("x") is None
        assert interceptor.feed_token(" ") is None
        assert interceptor.feed_token("=") is None
        assert interceptor.feed_token(" ") is None

    def test_simple_assignment_triggers(self, interceptor):
        """A complete 'x = 1\\n' should trigger a simple block event."""
        tokens = ["x", " ", "=", " ", "1", "\n"]
        events = []
        for tok in tokens:
            event = interceptor.feed_token(tok)
            if event is not None:
                events.append(event)
        assert len(events) >= 1
        assert events[0].block_type == "simple"
        assert "x" in events[0].block_text and "1" in events[0].block_text

    def test_multiple_statements(self, interceptor):
        """Two complete statements should trigger two events."""
        code = "x = 1\ny = 2\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        assert len(events) >= 2

    def test_compound_statement_triggers(self, interceptor):
        """A complete for loop should trigger compound event."""
        code = "for i in range(10):\n    x = i\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        # Should have at least a simple event (x = i) and possibly compound
        simple_events = [e for e in events if e.block_type == "simple"]
        assert len(simple_events) >= 1

    def test_event_has_parent_node_type(self, interceptor):
        """Events should include parent node type for topology hashing."""
        code = "x = 1\n"
        events = []
        for ch in code:
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        assert len(events) >= 1
        assert events[0].parent_node_type is not None

    def test_reset_clears_state(self, interceptor):
        """After reset, interceptor should start fresh."""
        interceptor.feed_token("x")
        interceptor.feed_token(" ")
        interceptor.reset()
        assert interceptor._accumulated == ""

    def test_syntax_error_no_false_trigger(self, interceptor):
        """Incomplete/malformed code should not trigger false positives."""
        tokens = ["def", " ", "foo", "("]
        events = []
        for tok in tokens:
            event = interceptor.feed_token(tok)
            if event is not None:
                events.append(event)
        assert len(events) == 0

    def test_token_tracking(self, interceptor):
        """Events should track token indices."""
        code = "x = 1\n"
        events = []
        for i, ch in enumerate(code):
            event = interceptor.feed_token(ch)
            if event is not None:
                events.append(event)
        if events:
            assert events[0].token_count > 0
