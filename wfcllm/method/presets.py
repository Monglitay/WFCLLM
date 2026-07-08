from __future__ import annotations

from copy import deepcopy

from wfcllm.method.config import WFCLLMMethodPreset

EVIDENCE_RETRY_SEED7X3_NAME = "evidence_retry_seed7x3"

_EVIDENCE_RETRY_SEED7X3 = WFCLLMMethodPreset(
    method={
        "name": EVIDENCE_RETRY_SEED7X3_NAME,
        "strict_no_quality_gate": True,
        "strict_code_only_detector": True,
    },
    generation={
        "dataset": "humaneval",
        "max_new_tokens": 256,
        "temperature": 0.25,
        "top_p": 0.95,
        "top_k": 0,
        "retry_repetition_penalty": 4.0,
        "torch_dtype": "bf16",
        "device": "cuda",
        "seed": 7,
        "load_in_4bit": True,
        "prompt_mode": "completion",
        "max_group_statements": 2,
        "retry_budget": 2,
        "statement_retry_budget": 4,
        "window_retry_budget": 3,
        "compound_retry_budget": 2,
        "global_rollback_budget": 180,
        "max_total_sampled_tokens": 32768,
        "evidence_retry_attempts": 3,
        "evidence_retry_seed_stride": 101,
    },
    semantic_lsh={
        "rule_name": "semantic_lsh",
        "lsh_d": 4,
        "lsh_gamma": 0.25,
        "semantic_margin": 0.0,
        "use_ordinal_keying": False,
    },
    detector={
        "lsh_d": 4,
        "gamma": 0.25,
        "semantic_margin": 0.0,
        "max_group_statements": 2,
        "min_scoreable_contexts": 1,
        "min_proxy_windows": 2,
        "target_fpr": 0.05,
        "use_ordinal_keying": False,
        "evidence_mode": "hit_plus_margin",
        "statistic": "calibrated_context_mean_proxy_penalized",
        "proxy_penalty_alpha": 0.4,
    },
    calibration={},
    artifacts={"run_root": "data/runs"},
    runtime={"default_phases": ["generate", "calibrate", "detect", "report", "audit"]},
)


def load_method_preset(name: str) -> WFCLLMMethodPreset:
    if name != EVIDENCE_RETRY_SEED7X3_NAME:
        raise ValueError(f"unknown WFCLLM method preset: {name}")
    payload = deepcopy(_EVIDENCE_RETRY_SEED7X3.to_dict())
    return WFCLLMMethodPreset(**payload)
