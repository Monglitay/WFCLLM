"""Contract tests for PythonAdapter (spec §5.1 "契约测试")."""
from __future__ import annotations


def test_python_adapter_registered():
    from wfcllm import lang
    assert "python" in lang.names()


def test_python_adapter_name_field():
    from wfcllm import lang
    assert lang.get("python").name == "python"
