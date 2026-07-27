"""WFCLLM unified entry script (thin shim around wfcllm.cli.entry)."""

from wfcllm.cli.entry import main


if __name__ == "__main__":
    raise SystemExit(main())
