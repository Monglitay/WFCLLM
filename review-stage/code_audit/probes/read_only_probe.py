from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
from transformers import AutoTokenizer

from wfcllm.cli.config_resolver import (
    resolve_adaptive_detection_config,
    resolve_extract_adaptive_gamma_config,
    resolve_extract_lsh_params,
    resolve_token_channel_config,
)
from wfcllm.datasets.loaders.local import load_prompts, load_reference_solutions
from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.extract.config import ExtractConfig
from wfcllm.extract.detector import WatermarkDetector
from wfcllm.lang.python.parser import extract_statement_blocks
from wfcllm.common.block_contract import build_block_contracts

ROOT = Path.cwd()
OUT_DIR = ROOT / "review-stage" / "code_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PROBE_DIR = OUT_DIR / "probes"
PROBE_DIR.mkdir(parents=True, exist_ok=True)


def read_json(path: str | Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        return rows
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit is not None and len(rows) >= limit:
                break
    return rows


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_tensor(tensor: torch.Tensor) -> str:
    arr = tensor.detach().cpu().float().numpy()
    return hashlib.sha256(arr.tobytes()).hexdigest()


def float_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0}
    return {
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(sum(values) / len(values)),
        "std": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def summarize_repeats(vectors: list[torch.Tensor]) -> dict[str, Any]:
    hashes = [sha256_tensor(v) for v in vectors]
    base = vectors[0].detach().cpu().float()
    l2 = [float(torch.linalg.vector_norm(v.detach().cpu().float() - base).item()) for v in vectors[1:]]
    cos_d = [float(1.0 - torch.nn.functional.cosine_similarity(v.detach().cpu().float(), base, dim=0).item()) for v in vectors[1:]]
    return {
        "unique_hashes": len(set(hashes)),
        "hashes": hashes,
        "max_l2_from_first": max(l2) if l2 else 0.0,
        "max_cosine_distance_from_first": max(cos_d) if cos_d else 0.0,
        "l2_stats_from_first": float_stats(l2),
        "cosine_distance_stats_from_first": float_stats(cos_d),
    }


def make_args_stub() -> SimpleNamespace:
    names = [
        "gamma_strategy", "entropy_profile", "profile_id", "adaptive_detection_mode", "strict_contract",
        "token_channel_enabled", "token_channel_mode", "token_channel_model_path", "token_channel_context_width",
        "token_channel_switch_threshold", "token_channel_delta", "token_channel_ignore_repeated_ngrams",
        "token_channel_ignore_repeated_prefixes", "token_channel_debug_mode", "token_channel_lexical_min_block_tokens",
        "token_channel_lexical_retry_decay_start", "token_channel_lexical_retry_disable_after",
        "token_channel_lexical_gate_probe_tokens", "token_channel_lexical_gate_min_fraction",
        "token_channel_joint_semantic_weight", "token_channel_joint_lexical_weight",
        "token_channel_lexical_full_weight_min_positions", "token_channel_joint_threshold",
    ]
    ns = SimpleNamespace()
    for name in names:
        setattr(ns, name, None)
    ns.strict_contract = False
    return ns


def load_encoder_and_tokenizer(embed_dim: int) -> tuple[SemanticEncoder, Any, str, str | None, dict[str, Any]]:
    enc_config = EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
    model = SemanticEncoder(config=enc_config)
    ckpt_path = Path("data/models/encoder/best_model.pt")
    fallback_path = Path("data/checkpoints/encoder/encoder_epoch9.pt")
    loaded = None
    ckpt_info: dict[str, Any] = {}
    for candidate in (ckpt_path, fallback_path):
        if candidate.exists():
            ckpt = torch.load(candidate, map_location="cpu")
            missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
            loaded = str(candidate)
            ckpt_info = {
                "loaded_checkpoint": loaded,
                "strict_false_missing_keys": list(missing),
                "strict_false_unexpected_keys": list(unexpected),
                "checkpoint_top_keys": sorted(ckpt.keys()),
                "checkpoint_epoch": ckpt.get("epoch"),
                "checkpoint_config": ckpt.get("config"),
            }
            break
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)
    return model, tokenizer, enc_config.model_name, loaded, ckpt_info


def encode_once(model: SemanticEncoder, tokenizer: Any, code: str, device: str) -> torch.Tensor:
    inputs = tokenizer(code, return_tensors="pt", padding=True, truncation=True, max_length=256)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    with torch.no_grad():
        return model(input_ids, attention_mask).squeeze(0).detach().cpu()


def run_encoder_probe(config: dict[str, Any], generated_samples: list[dict[str, Any]]) -> dict[str, Any]:
    embed_dim = int((config.get("extract") or {}).get("embed_dim") or (config.get("watermark") or {}).get("encoder_embed_dim") or 128)
    model, tokenizer, model_name, ckpt_path, ckpt_info = load_encoder_and_tokenizer(embed_dim)
    static_codes = [
        "def add(a, b):\n    return a + b\n",
        "for i in range(3):\n    print(i)\n",
        "if x > 0:\n    y = x\nelse:\n    y = -x\n",
    ]
    generated_codes = [str(row.get("generated_code", "")) for row in generated_samples[:3]]
    samples = [{"label": f"short_{i}", "code": code} for i, code in enumerate(static_codes)]
    samples += [{"label": f"generated_{i}", "id": generated_samples[i].get("id"), "code": code} for i, code in enumerate(generated_codes) if code.strip()]
    result: dict[str, Any] = {
        "model_name": model_name,
        "checkpoint_path": ckpt_path,
        "checkpoint_info": ckpt_info,
        "samples": [],
    }
    devices = ["cuda"] if torch.cuda.is_available() else ["cpu"]
    for device in devices:
        try:
            model.to(device)
        except Exception as exc:
            result.setdefault("device_errors", {})[device] = repr(exc)
            continue
        for mode in ("train", "eval"):
            if mode == "train":
                model.train()
            else:
                model.eval()
            for sample in samples:
                vectors = []
                started = time.perf_counter()
                try:
                    for _ in range(10):
                        vectors.append(encode_once(model, tokenizer, sample["code"], device))
                    summary = summarize_repeats(vectors)
                    error = None
                except Exception as exc:
                    summary = {}
                    error = repr(exc)
                result["samples"].append({
                    "device": device,
                    "mode": mode,
                    "label": sample.get("label"),
                    "id": sample.get("id"),
                    "code_sha256": sha256_text(sample["code"]),
                    "elapsed_seconds": round(time.perf_counter() - started, 4),
                    "error": error,
                    **summary,
                })
    return result


def hit_vector_hash(result: Any) -> str:
    payload = [
        {
            "block_id": score.block_id,
            "score": score.score,
            "gamma_effective": score.gamma_effective,
            "source_hash": score.source_hash,
        }
        for score in result.block_details
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def build_extract_config(config: dict[str, Any], first_record: dict[str, Any], mode: str, enabled: bool) -> ExtractConfig:
    args = make_args_stub()
    ext_cfg = config.get("extract") or {}
    lsh_d, lsh_gamma = resolve_extract_lsh_params(first_record, ext_cfg)
    tc_section = copy.deepcopy(ext_cfg.get("token_channel") or {})
    tc_section["enabled"] = enabled
    tc_section["channel_mode"] = mode
    token_channel = resolve_token_channel_config(tc_section, None)
    return ExtractConfig(
        secret_key=str(ext_cfg.get("secret_key", "")),
        embed_dim=int(ext_cfg.get("embed_dim", 128)),
        fpr_threshold=float(ext_cfg.get("fpr_threshold", 3.0)),
        lsh_d=int(lsh_d),
        lsh_gamma=float(lsh_gamma),
        min_blocks=int(ext_cfg.get("min_blocks", 2)),
        adaptive_detection=resolve_adaptive_detection_config(args, ext_cfg),
        adaptive_gamma=resolve_extract_adaptive_gamma_config(args, config),
        token_channel=token_channel,
    )


def make_detector(config: dict[str, Any], first_record: dict[str, Any], mode: str, enabled: bool, encoder: SemanticEncoder, tokenizer: Any, device: str) -> WatermarkDetector:
    extract_config = build_extract_config(config, first_record, mode=mode, enabled=enabled)
    lm_tokenizer = None
    if enabled and mode != "semantic-only":
        metadata_path = Path(extract_config.token_channel.model_path) / "metadata.json"
        if metadata_path.exists():
            metadata = read_json(metadata_path)
            tok_name = metadata.get("tokenizer_name")
            if tok_name:
                lm_tokenizer = AutoTokenizer.from_pretrained(tok_name)
    return WatermarkDetector(extract_config, encoder, tokenizer, device=device, lm_tokenizer=lm_tokenizer)


def summarize_detection_repeats(detector: WatermarkDetector, row: dict[str, Any], with_metadata: bool, repeats: int = 5) -> dict[str, Any]:
    z_values: list[float] = []
    pred_values: list[bool] = []
    hit_hashes: list[str] = []
    hits: list[int] = []
    blocks: list[int] = []
    lexical_z: list[float] = []
    joint_scores: list[float] = []
    errors: list[str] = []
    code = str(row.get("generated_code", ""))
    for _ in range(repeats):
        try:
            result = detector.detect(
                code,
                watermark_metadata=row if with_metadata else None,
                prompt=str(row.get("prompt", "")),
            )
            z_values.append(float(result.z_score))
            pred_values.append(bool(result.is_watermarked))
            hit_hashes.append(hit_vector_hash(result))
            hits.append(int(result.hit_blocks))
            blocks.append(int(result.independent_blocks))
            if result.lexical_result is not None:
                lexical_z.append(float(result.lexical_result.lexical_z_score))
            if result.joint_result is not None:
                joint_scores.append(float(result.joint_result.joint_score))
        except Exception as exc:
            errors.append(repr(exc))
    return {
        "z_values": z_values,
        "z_stats": float_stats(z_values),
        "unique_z_values": len(set(z_values)),
        "pred_values": pred_values,
        "unique_predictions": len(set(pred_values)),
        "hit_hashes": hit_hashes,
        "unique_hit_hashes": len(set(hit_hashes)),
        "hits": hits,
        "independent_blocks": blocks,
        "lexical_z_values": lexical_z,
        "joint_scores": joint_scores,
        "errors": errors,
    }


def run_detector_probe(config: dict[str, Any], watermarked_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]], persisted_details: dict[str, dict[str, Any]]) -> dict[str, Any]:
    first_record = watermarked_rows[0] if watermarked_rows else {}
    embed_dim = int((config.get("extract") or {}).get("embed_dim") or 128)
    encoder, tokenizer, model_name, ckpt_path, ckpt_info = load_encoder_and_tokenizer(embed_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    result: dict[str, Any] = {
        "device": device,
        "model_name": model_name,
        "checkpoint_path": ckpt_path,
        "checkpoint_info": ckpt_info,
        "watermarked_sample_ids": [r.get("id") for r in watermarked_rows],
        "negative_sample_ids": [r.get("id") for r in negative_rows],
        "records": [],
    }
    for mode_name in ("train", "eval"):
        if mode_name == "train":
            encoder.train()
        else:
            encoder.eval()
        try:
            semantic_detector = make_detector(config, first_record, "semantic-only", False, encoder, tokenizer, device)
            dual_detector = make_detector(config, first_record, "dual-channel", True, encoder, tokenizer, device)
            detector_error = None
        except Exception as exc:
            semantic_detector = None
            dual_detector = None
            detector_error = repr(exc)
        if detector_error:
            result["records"].append({"mode": mode_name, "detector_init_error": detector_error})
            continue
        for label, rows, with_metadata in (("watermarked", watermarked_rows, True), ("negative", negative_rows, False)):
            for row in rows:
                sid = row.get("id")
                sem = summarize_detection_repeats(semantic_detector, row, with_metadata=with_metadata)
                dual = summarize_detection_repeats(dual_detector, row, with_metadata=with_metadata)
                persisted = persisted_details.get(str(sid), {})
                result["records"].append({
                    "mode": mode_name,
                    "set": label,
                    "id": sid,
                    "code_sha256": sha256_text(str(row.get("generated_code", ""))),
                    "with_metadata": with_metadata,
                    "semantic_only": sem,
                    "dual_channel": dual,
                    "semantic_z_delta_dual_minus_semantic_first": (
                        (dual["z_values"][0] - sem["z_values"][0])
                        if dual.get("z_values") and sem.get("z_values") else None
                    ),
                    "persisted_detail": {
                        key: persisted.get(key)
                        for key in ("z_score", "is_watermarked", "hits", "independent_blocks", "joint_score", "joint_prediction", "lexical_z_score")
                        if key in persisted
                    },
                })
    unstable = []
    for rec in result["records"]:
        if not isinstance(rec, dict) or "semantic_only" not in rec:
            continue
        for channel in ("semantic_only", "dual_channel"):
            ch = rec[channel]
            if ch.get("unique_z_values", 0) > 1 or ch.get("unique_hit_hashes", 0) > 1 or ch.get("unique_predictions", 0) > 1:
                unstable.append({"mode": rec["mode"], "set": rec["set"], "id": rec["id"], "channel": channel})
    result["unstable_records"] = unstable
    return result


def run_dataset_probe() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for dataset in ("humaneval", "mbpp"):
        entry: dict[str, Any] = {}
        for loader_name, loader in (("prompts", load_prompts), ("references", load_reference_solutions)):
            try:
                rows = loader(dataset, "data/datasets")
                ids = [str(r.get("id", "")) for r in rows]
                entry[loader_name] = {
                    "count": len(rows),
                    "first_ids": ids[:5],
                    "duplicate_ids": sorted({sid for sid in ids if ids.count(sid) > 1})[:20],
                    "empty_ids": sum(1 for sid in ids if not sid),
                    "empty_prompt": sum(1 for r in rows if not str(r.get("prompt", "")).strip()),
                    "empty_generated_code": sum(1 for r in rows if loader_name == "references" and not str(r.get("generated_code", "")).strip()),
                    "keys_first": sorted(rows[0].keys()) if rows else [],
                }
            except Exception as exc:
                entry[loader_name] = {"error": repr(exc)}
        out[dataset] = entry
    return out


def run_block_probe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    samples = [
        {"label": "simple_function", "code": "def f(x):\n    y = x + 1\n    return y\n"},
        {"label": "compound", "code": "for i in range(3):\n    if i % 2:\n        print(i)\n"},
        {"label": "bad_syntax", "code": "def broken(:\n    return 1\n"},
    ]
    for row in rows[:3]:
        samples.append({"label": "generated", "id": row.get("id"), "code": str(row.get("generated_code", ""))})
    out = []
    for sample in samples:
        code = sample["code"]
        try:
            blocks = extract_statement_blocks(code)
            contracts = build_block_contracts(code)
            out.append({
                "label": sample.get("label"),
                "id": sample.get("id"),
                "code_sha256": sha256_text(code),
                "block_count": len(blocks),
                "simple_count": sum(1 for b in blocks if b.block_type == "simple"),
                "compound_count": sum(1 for b in blocks if b.block_type == "compound"),
                "blocks": [
                    {
                        "block_id": b.block_id,
                        "block_type": b.block_type,
                        "node_type": b.node_type,
                        "parent_id": b.parent_id,
                        "start_line": b.start_line,
                        "end_line": b.end_line,
                        "source_hash": sha256_text(b.source),
                    }
                    for b in blocks[:20]
                ],
                "contract_count": len(contracts),
                "contracts": [c.__dict__ for c in contracts[:20]],
            })
        except Exception as exc:
            out.append({"label": sample.get("label"), "id": sample.get("id"), "error": repr(exc)})
    return {"samples": out}


def artifact_pairing_probe(config: dict[str, Any], run_state: dict[str, Any], watermarked_rows: list[dict[str, Any]], negative_rows: list[dict[str, Any]], details_rows: list[dict[str, Any]], summary: dict[str, Any] | None) -> dict[str, Any]:
    wm_ids = [str(r.get("id")) for r in watermarked_rows]
    neg_ids = [str(r.get("id")) for r in negative_rows]
    det_ids = [str(r.get("id")) for r in details_rows]
    first = watermarked_rows[0] if watermarked_rows else {}
    secrets_in_wm = []
    for row in watermarked_rows[:50]:
        payload = json.dumps(row, ensure_ascii=False)
        if "secret_key" in payload:
            secrets_in_wm.append(row.get("id"))
    return {
        "config_calibration_corpus": (config.get("extract") or {}).get("calibration_corpus"),
        "run_state_negative_output": (run_state.get("generate-negative") or {}).get("output_file"),
        "run_state_watermark_output": (run_state.get("watermark") or {}).get("output_file"),
        "run_state_extract_details": (run_state.get("extract") or {}).get("details_file"),
        "run_state_extract_summary": (run_state.get("extract") or {}).get("summary_file"),
        "watermarked_count_loaded": len(watermarked_rows),
        "negative_count_loaded": len(negative_rows),
        "details_count_loaded": len(details_rows),
        "watermarked_first_ids": wm_ids[:10],
        "negative_first_ids": neg_ids[:10],
        "details_first_ids": det_ids[:10],
        "details_missing_from_watermarked": sorted(set(det_ids) - set(wm_ids))[:50],
        "watermarked_missing_from_details": sorted(set(wm_ids) - set(det_ids))[:50],
        "negative_overlap_with_watermarked": sorted(set(neg_ids) & set(wm_ids))[:50],
        "first_watermarked_keys": sorted(first.keys()) if isinstance(first, dict) else [],
        "first_watermark_params": first.get("watermark_params") if isinstance(first, dict) else None,
        "secret_key_literal_seen_in_first_50_watermarked_rows": secrets_in_wm,
        "summary_meta": summary.get("meta") if isinstance(summary, dict) else None,
        "summary_summary": summary.get("summary") if isinstance(summary, dict) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-heavy", action="store_true")
    args = parser.parse_args()
    config = read_json("configs/base_config.json")
    run_state = read_json("data/run_state.json") if Path("data/run_state.json").exists() else {}
    wm_path = Path((run_state.get("watermark") or {}).get("output_file") or "data/watermarked/humaneval_20260523_130234.jsonl")
    neg_path = Path((config.get("extract") or {}).get("calibration_corpus") or "data/negative_corpus.jsonl")
    details_path = Path((run_state.get("extract") or {}).get("details_file") or "data/results/humaneval_full_cap9_details.jsonl")
    summary_path = Path((run_state.get("extract") or {}).get("summary_file") or "data/results/humaneval_full_cap9_summary.json")

    watermarked_all = read_jsonl(wm_path)
    negative_all = read_jsonl(neg_path)
    details_all = read_jsonl(details_path)
    summary = read_json(summary_path) if summary_path.exists() else None
    persisted_details = {str(r.get("id")): r for r in details_all}
    watermarked_probe_rows = [r for r in watermarked_all if str(r.get("generated_code", "")).strip()][:10]
    negative_probe_rows = [r for r in negative_all if str(r.get("generated_code", "")).strip()][:10]

    env = {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "env": {k: os.environ.get(k) for k in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE")},
        "torch": {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }

    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": env,
        "paths": {
            "watermarked": str(wm_path),
            "negative": str(neg_path),
            "details": str(details_path),
            "summary": str(summary_path),
        },
        "artifact_pairing": artifact_pairing_probe(config, run_state, watermarked_all, negative_all, details_all, summary),
        "dataset_probe": run_dataset_probe(),
        "block_probe": run_block_probe(watermarked_probe_rows),
        "selected_samples": {
            "watermarked": [r.get("id") for r in watermarked_probe_rows],
            "negative": [r.get("id") for r in negative_probe_rows],
            "encoder_generated": [r.get("id") for r in watermarked_probe_rows[:3]],
        },
    }
    if not args.skip_heavy:
        results["encoder_repro_probe"] = run_encoder_probe(config, watermarked_probe_rows[:3])
        results["detector_repro_probe"] = run_detector_probe(config, watermarked_probe_rows, negative_probe_rows, persisted_details)

    output = PROBE_DIR / "read_only_probe_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
