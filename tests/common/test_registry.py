"""Tests for the generic Registry[T] container."""
import pytest

from wfcllm.common.registry import Registry


class FakeAdapter:
    name = "fake"


class AnotherAdapter:
    name = "another"


def test_register_and_get_by_name():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    assert isinstance(reg.get("fake"), FakeAdapter)


def test_get_unknown_name_raises_with_listing():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    with pytest.raises(KeyError) as excinfo:
        reg.get("missing")
    assert "missing" in str(excinfo.value)
    assert "fake" in str(excinfo.value)
    assert "test" in str(excinfo.value)


def test_duplicate_registration_raises():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("fake", FakeAdapter)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("fake", AnotherAdapter)


def test_names_returns_sorted_list():
    reg: Registry[FakeAdapter] = Registry("test")
    reg.register("zebra", FakeAdapter)
    reg.register("alpha", FakeAdapter)
    assert reg.names() == ["alpha", "zebra"]
