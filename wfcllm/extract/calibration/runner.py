"""Calibrate FPR-based watermark detection threshold from a negative corpus.

Logic moved from scripts/calibrate.py:_calibrate_threshold (Phase 4 refactor).
The CLI shell that wraps this lives in scripts/calibrate_threshold.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from transformers import AutoModel, AutoTokenizer

from wfcllm.extract.calibration.threshold import ThresholdCalibrator
from wfcllm.extract.scorer import BlockScorer
from wfcllm.watermark.keying import WatermarkKeying
from wfcllm.watermark.lsh_space import LSHSpace
from wfcllm.watermark.verifier import ProjectionVerifier


def _load_jsonl(path: str | Path) -> list[dict]:
    records: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def calibrate_threshold_from_corpus(
    *,
    input: str | Path,
    output: str | Path,
    secret_key: str,
    model: str,
    device: str = "cuda",
    fpr: float = 0.01,
    embed_dim: int = 128,
    lsh_d: int = 3,
    gamma: float = 0.5,
) -> Path:
    """Calibrate the FPR detection threshold and persist the result JSON.

    Returns the resolved output path.
    """
    print(f"Loading model from {model} ...", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(model)
    encoder = AutoModel.from_pretrained(model).to(device)
    encoder.eval()

    lsh_space = LSHSpace(secret_key, embed_dim, lsh_d)
    keying = WatermarkKeying(secret_key, lsh_d, gamma)
    verifier = ProjectionVerifier(encoder, tokenizer, lsh_space=lsh_space, device=device)
    scorer = BlockScorer(keying, verifier)

    print(f"Loading corpus from {input} ...", file=sys.stderr)
    corpus = _load_jsonl(input)
    print(f"  {len(corpus)} samples loaded.", file=sys.stderr)

    calibrator = ThresholdCalibrator(scorer, gamma=gamma)
    result = calibrator.calibrate(corpus, fpr=fpr)

    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"Calibration complete:\n"
        f"  FPR target    : {result['fpr']}\n"
        f"  M_r threshold : {result['fpr_threshold']:.4f}\n"
        f"  Samples used  : {result['n_samples']}\n"
        f"  Output        : {output}",
        file=sys.stderr,
    )
    return out_path
