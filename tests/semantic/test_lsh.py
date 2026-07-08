from __future__ import annotations


def test_semantic_lsh_module_exports_loader() -> None:
    from wfcllm.semantic.lsh import load_semantic_lsh_rule

    assert callable(load_semantic_lsh_rule)
