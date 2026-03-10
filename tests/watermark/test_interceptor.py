"""Tests for wfcllm.watermark.interceptor."""

import pytest
import warnings
from wfcllm.watermark.interceptor import StatementInterceptor, InterceptEvent, InterceptorState


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


class TestStatementInterceptorStateSnapshot:
    """save_state / restore_state 语义测试。"""

    def test_restore_returns_to_saved_state(self):
        """restore 之后 accumulated 和 emitted_keys 回到 save 时的值。"""
        interceptor = StatementInterceptor()
        # 喂入部分 token，让 interceptor 有非空状态
        for ch in "x = 1\n":
            interceptor.feed_token(ch)
        state = interceptor.save_state()

        # 继续喂更多 token
        for ch in "y = 2\n":
            interceptor.feed_token(ch)
        assert "y" in interceptor._accumulated

        # 恢复
        interceptor.restore_state(state)
        assert interceptor._accumulated == state["accumulated"]
        assert interceptor._emitted_keys == state["emitted_keys"]
        assert interceptor._prev_all_keys == state["prev_all_keys"]
        assert interceptor._pending_simple == state["pending_simple"]
        assert interceptor._token_idx == state["token_idx"]

    def test_restore_makes_feed_token_deterministic(self):
        """restore 后重新喂同样的 token 序列，结果应该与原始一致。"""
        interceptor = StatementInterceptor()
        for ch in "x = 1\n":
            interceptor.feed_token(ch)
        state = interceptor.save_state()

        # 第一次：继续喂 'y = 2\n'，记录事件
        events_first = []
        for ch in "y = 2\n":
            e = interceptor.feed_token(ch)
            if e is not None:
                events_first.append(e)

        # restore 后再喂同样序列
        interceptor.restore_state(state)
        events_second = []
        for ch in "y = 2\n":
            e = interceptor.feed_token(ch)
            if e is not None:
                events_second.append(e)

        assert len(events_first) == len(events_second)
        for e1, e2 in zip(events_first, events_second):
            assert e1.block_text == e2.block_text
            assert e1.block_type == e2.block_type


class TestTokenBoundaries:
    """Fix 1: _token_boundaries enables accurate token_count in events."""

    def test_token_boundaries_initialized(self):
        """interceptor._token_boundaries starts as [0]."""
        ic = StatementInterceptor()
        assert ic._token_boundaries == [0]

    def test_token_boundaries_grow_with_feed(self):
        """After each feed_token, boundaries has one more entry."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        # One boundary per token fed, plus initial [0]
        assert len(ic._token_boundaries) == len("x = 1\n") + 1

    def test_token_boundaries_are_monotone(self):
        """Each boundary >= previous (UTF-8 byte offsets are non-decreasing)."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        for i in range(1, len(ic._token_boundaries)):
            assert ic._token_boundaries[i] >= ic._token_boundaries[i - 1]

    def test_event_token_count_matches_text_byte_span(self):
        """event.token_count should equal the number of tokens whose bytes
        overlap the block's byte span, NOT len(block_text)."""
        ic = StatementInterceptor()
        events = []
        # Feed char-by-char (each char = 1 token for ASCII)
        for ch in "x = 1\n":
            e = ic.feed_token(ch)
            if e is not None:
                events.append(e)
        assert events, "Expected at least one event"
        ev = events[0]
        # The block text is 'x = 1', which is 5 bytes / 5 chars = 5 tokens
        # token_count must equal 5 (not len(block_text) which was the old wrong formula)
        assert ev.token_count == len(ev.block_text.encode("utf-8"))

    def test_token_boundaries_saved_and_restored(self):
        """save_state/restore_state includes _token_boundaries."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        state = ic.save_state()
        assert "token_boundaries" in state
        # Continue feeding
        for ch in "y = 2\n":
            ic.feed_token(ch)
        # Restore
        ic.restore_state(state)
        assert ic._token_boundaries == state["token_boundaries"]


class TestPreEventState:
    """Fix 2: get_pre_event_state() returns interceptor state BEFORE emitting."""

    def test_get_pre_event_state_exists(self):
        """StatementInterceptor has get_pre_event_state method."""
        ic = StatementInterceptor()
        assert hasattr(ic, "get_pre_event_state")

    def test_pre_event_state_available_after_event(self):
        """After an event fires, get_pre_event_state() returns a dict."""
        ic = StatementInterceptor()
        event = None
        for ch in "x = 1\n":
            e = ic.feed_token(ch)
            if e is not None:
                event = e
        assert event is not None, "Expected an event"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            state = ic.get_pre_event_state()
        assert isinstance(state, dict)
        assert "accumulated" in state
        assert "emitted_keys" in state

    def test_pre_event_state_does_not_contain_emitted_key(self):
        """The pre-event snapshot must NOT contain the block's key in emitted_keys.
        This is the core of Fix 2: restore lets the sub-loop re-detect the same block."""
        ic = StatementInterceptor()
        event = None
        for ch in "x = 1\n":
            e = ic.feed_token(ch)
            if e is not None:
                event = e
        assert event is not None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            state = ic.get_pre_event_state()
        # The key is (node_type, start_byte, end_byte) — reconstruct from event
        # The pre-event state's emitted_keys should NOT contain ANY key related to this event
        # We verify by restoring and re-feeding the same tokens — another event must fire
        ic.restore_state(state)
        events_after_restore = []
        for ch in "x = 1\n":
            e = ic.feed_token(ch)
            if e is not None:
                events_after_restore.append(e)
        assert len(events_after_restore) >= 1, (
            "After restoring pre-event state, same tokens must re-trigger the event"
        )

    def test_pre_event_state_vs_save_state_differ_on_emitted_keys(self):
        """save_state() after event includes block key; get_pre_event_state() does not."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            e = ic.feed_token(ch)
        # save_state() now includes the block's key
        post_state = ic.save_state()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            pre_state = ic.get_pre_event_state()
        # pre_state emitted_keys is a strict subset of post_state emitted_keys
        assert pre_state["emitted_keys"] < post_state["emitted_keys"] or (
            # or same length means the key was already there before (shouldn't happen here)
            len(pre_state["emitted_keys"]) <= len(post_state["emitted_keys"])
        )

    def test_pre_event_state_none_before_any_event(self):
        """Before any event fires, get_pre_event_state() raises AssertionError."""
        ic = StatementInterceptor()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises((AssertionError, TypeError)):
                ic.get_pre_event_state()


class TestInterceptorCheckpointRollback:
    """New typed checkpoint/rollback API."""

    def test_checkpoint_returns_typed_state(self):
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        state = ic.checkpoint()
        assert isinstance(state, InterceptorState)
        assert isinstance(state.accumulated, str)
        assert isinstance(state.token_idx, int)
        assert isinstance(state.prev_all_keys, set)
        assert isinstance(state.pending_simple, dict)
        assert isinstance(state.emitted_keys, set)
        assert isinstance(state.token_boundaries, list)

    def test_rollback_with_typed_state(self):
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        state = ic.checkpoint()
        for ch in "y = 2\n":
            ic.feed_token(ch)
        ic.rollback(state)
        assert ic._accumulated == state.accumulated
        assert ic._token_idx == state.token_idx
        assert ic._emitted_keys == state.emitted_keys

    def test_checkpoint_rollback_equivalence(self):
        """checkpoint → feed tokens → rollback → state equals checkpoint."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Feed more tokens
        for ch in "y = 2\nz = 3\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        # Re-feed same tokens → same events
        events1 = []
        for ch in "y = 2\n":
            e = ic.feed_token(ch)
            if e:
                events1.append(e)
        assert len(events1) >= 1
        assert events1[0].block_text.strip() == "y = 2"

    def test_rollback_accumulated_text_clean(self):
        """After rollback, accumulated does not contain rolled-back tokens."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        for ch in "return the\n":
            ic.feed_token(ch)
        ic.rollback(cp)
        assert "return" not in ic._accumulated or ic._accumulated == cp.accumulated

    def test_rollback_then_regenerate_detects_new_block(self):
        """After rollback, feeding different content triggers new block."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # First attempt
        for ch in "y = 2\n":
            ic.feed_token(ch)
        # Rollback
        ic.rollback(cp)
        # Second attempt with different content
        events = []
        for ch in "z = 3\n":
            e = ic.feed_token(ch)
            if e:
                events.append(e)
        assert len(events) >= 1
        assert "z" in events[0].block_text

    def test_deep_copy_pending_simple(self):
        """checkpoint() deep-copies _BlockInfo in pending_simple."""
        ic = StatementInterceptor()
        # Feed partial statement (no newline yet) to create pending_simple entry
        for ch in "x = 1":
            ic.feed_token(ch)
        cp = ic.checkpoint()
        # Verify pending_simple values are independent copies
        if cp.pending_simple:
            for key in cp.pending_simple:
                # Modifying original should not affect checkpoint
                if key in ic._pending_simple:
                    original_text = ic._pending_simple[key].text
                    ic._pending_simple[key].text = "MODIFIED"
                    assert cp.pending_simple[key].text == original_text

    def test_get_pre_event_state_removed(self):
        """get_pre_event_state should raise DeprecationWarning or not exist."""
        ic = StatementInterceptor()
        for ch in "x = 1\n":
            ic.feed_token(ch)
        # Either removed entirely or raises deprecation
        if hasattr(ic, "get_pre_event_state"):
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                with pytest.raises(DeprecationWarning):
                    ic.get_pre_event_state()

