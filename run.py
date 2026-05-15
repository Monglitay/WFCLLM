"""WFCLLM unified entry script (thin shim around wfcllm.cli.entry)."""
from wfcllm.cli.entry import main
from wfcllm.cli.config_resolver import resolve_extract_lsh_params  # noqa: F401  re-export for tools/debug_extract_alignment.py

if __name__ == "__main__":
    raise SystemExit(main())
