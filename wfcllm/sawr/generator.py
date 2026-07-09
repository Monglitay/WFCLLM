from __future__ import annotations

import warnings
from typing import Any

from wfcllm.generation import generator as _generation_generator

warnings.warn(
    "wfcllm.sawr.generator is deprecated; import wfcllm.generation.generator instead",
    DeprecationWarning,
    stacklevel=2,
)

from wfcllm.generation.generator import *  # noqa: F401,F403,E402


def load_sawr_model(config: SawrGenerationConfig) -> tuple[Any, Any]:
    """Compatibility wrapper preserving monkeypatches against this shim."""
    previous_tokenizer = _generation_generator.AutoTokenizer
    previous_model = _generation_generator.AutoModelForCausalLM
    previous_bnb = _generation_generator.BitsAndBytesConfig
    try:
        _generation_generator.AutoTokenizer = AutoTokenizer
        _generation_generator.AutoModelForCausalLM = AutoModelForCausalLM
        _generation_generator.BitsAndBytesConfig = BitsAndBytesConfig
        return _generation_generator.load_sawr_model(config)
    finally:
        _generation_generator.AutoTokenizer = previous_tokenizer
        _generation_generator.AutoModelForCausalLM = previous_model
        _generation_generator.BitsAndBytesConfig = previous_bnb


load_wfcllm_model = load_sawr_model
