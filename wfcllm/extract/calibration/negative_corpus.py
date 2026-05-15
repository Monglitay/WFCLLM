"""Negative corpus generation for FPR threshold calibration.

Generates unwatermarked code using a code LLM directly, writing JSONL output
compatible with ThresholdCalibrator's corpus format.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from wfcllm.common.dataset_loader import (
    SUPPORTED_DATASETS,
    load_prompts,
    load_reference_solutions,
)


@dataclass
class NegativeCorpusConfig:
    """Configuration for negative corpus generation."""

    lm_model_path: str
    """Path to the code-generation LLM (same as --lm-model-path in run.py)."""

    output_path: str
    """Output JSONL file path (e.g. data/negative_corpus.jsonl)."""

    dataset: str = "humaneval"
    """Dataset to use for prompts ("humaneval" or "mbpp")."""

    dataset_path: str = "data/datasets"
    """Local dataset root directory."""

    max_new_tokens: int = 512
    temperature: float = 0.8
    top_p: float = 0.95
    top_k: int = 50
    device: str = "cuda"
    limit: int | None = None
    source_mode: Literal["reference", "llm"] = "reference"
    """Process only first N prompts (for debugging). None = all."""

    def __post_init__(self):
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(
                f"dataset must be one of {SUPPORTED_DATASETS}, got '{self.dataset}'"
            )
        if self.source_mode not in {"reference", "llm"}:
            raise ValueError("source_mode must be 'reference' or 'llm'")


class NegativeCorpusGenerator:
    """Generate unwatermarked code samples for negative corpus.

    Loads a code LLM and generates code for each prompt in the dataset
    without any watermarking, writing results to JSONL for use with
    ThresholdCalibrator.
    """

    def __init__(self, config: NegativeCorpusConfig):
        self._config = config
        self._device = config.device
        self._model = None
        self._tokenizer = None

        if config.source_mode == "reference":
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        device = config.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[警告] CUDA 不可用，回退到 CPU", file=sys.stderr)
            device = "cpu"
        self._device = device

        print(f"加载模型 {config.lm_model_path} ...", file=sys.stderr)
        self._tokenizer = AutoTokenizer.from_pretrained(config.lm_model_path)

        if device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            self._model = AutoModelForCausalLM.from_pretrained(
                config.lm_model_path,
                quantization_config=bnb_config,
                device_map="auto",
            )
        else:
            self._model = AutoModelForCausalLM.from_pretrained(
                config.lm_model_path,
                torch_dtype=torch.float32,
            ).to(device)
        self._model.eval()

    def _generate(self, prompt: str) -> str:
        """Generate code for a single prompt without watermarking."""
        import torch

        cfg = self._config
        raw = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) if hasattr(v, "to") else v for k, v in raw.items()}
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=cfg.max_new_tokens,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated_ids = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated_ids, skip_special_tokens=True)

    def run(self) -> str:
        """Generate negative corpus and write to JSONL.

        Returns:
            Path to output JSONL file.
        """
        import torch

        cfg = self._config
        out_path = Path(cfg.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if cfg.source_mode == "reference":
            rows = load_reference_solutions(cfg.dataset, cfg.dataset_path)
            if cfg.limit is not None:
                rows = rows[: cfg.limit]
            print(f"  共 {len(rows)} 条 reference 样本", file=sys.stderr)

            with open(out_path, "w", encoding="utf-8") as f:
                for row in rows:
                    record = {
                        "id": row["id"],
                        "dataset": cfg.dataset,
                        "prompt": row["prompt"],
                        "generated_code": row["generated_code"],
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")

            print(f"\n完成，输出至: {out_path}（{len(rows)} 条）", file=sys.stderr)
            return str(out_path)

        prompts = load_prompts(cfg.dataset, cfg.dataset_path)
        if cfg.limit is not None:
            prompts = prompts[: cfg.limit]
        print(f"  共 {len(prompts)} 条 prompt", file=sys.stderr)

        with open(out_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(prompts):
                try:
                    code = self._generate(item["prompt"])
                except Exception as e:
                    print(f"[警告] {item['id']} 生成失败：{e}", file=sys.stderr)
                    code = ""

                record = {
                    "id": item["id"],
                    "dataset": cfg.dataset,
                    "prompt": item["prompt"],
                    "generated_code": code,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                print(
                    f"  [{i + 1}/{len(prompts)}] {item['id']} | "
                    f"words: {len(code.split())}",
                    file=sys.stderr,
                )

        print(f"\n完成，输出至: {out_path}（{len(prompts)} 条）", file=sys.stderr)
        return str(out_path)
