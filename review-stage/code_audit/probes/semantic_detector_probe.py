from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import torch
import read_only_probe as p

OUT = Path("review-stage/code_audit/probes/semantic_detector_probe_results.json")


def save(obj):
    OUT.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    config = p.read_json("configs/base_config.json")
    run_state = p.read_json("data/run_state.json") if Path("data/run_state.json").exists() else {}
    wm_path = Path((run_state.get("watermark") or {}).get("output_file") or "data/watermarked/humaneval_20260523_130234.jsonl")
    neg_path = Path((config.get("extract") or {}).get("calibration_corpus") or "data/negative_corpus.jsonl")
    details_path = Path((run_state.get("extract") or {}).get("details_file") or "data/results/humaneval_full_cap9_details.jsonl")
    summary_path = Path((run_state.get("extract") or {}).get("summary_file") or "data/results/humaneval_full_cap9_summary.json")
    watermarked_all = p.read_jsonl(wm_path)
    negative_all = p.read_jsonl(neg_path)
    details_all = p.read_jsonl(details_path)
    summary = p.read_json(summary_path) if summary_path.exists() else None
    persisted_details = {str(r.get("id")): r for r in details_all}
    watermarked_rows = [r for r in watermarked_all if str(r.get("generated_code", "")).strip()][:10]
    negative_rows = [r for r in negative_all if str(r.get("generated_code", "")).strip()][:10]

    result = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "paths": {"watermarked": str(wm_path), "negative": str(neg_path), "details": str(details_path), "summary": str(summary_path)},
        "selected_samples": {"watermarked": [r.get("id") for r in watermarked_rows], "negative": [r.get("id") for r in negative_rows]},
        "artifact_pairing": p.artifact_pairing_probe(config, run_state, watermarked_all, negative_all, details_all, summary),
        "status": "artifact/dataset/block pending",
    }
    save(result)
    result["dataset_probe"] = p.run_dataset_probe()
    result["block_probe"] = p.run_block_probe(watermarked_rows)
    result["status"] = "encoder pending"
    save(result)
    result["encoder_repro_probe_gpu_only"] = p.run_encoder_probe(config, watermarked_rows[:3])
    result["status"] = "semantic detector pending"
    save(result)

    first_record = watermarked_rows[0] if watermarked_rows else {}
    embed_dim = int((config.get("extract") or {}).get("embed_dim") or 128)
    encoder, tokenizer, model_name, ckpt_path, ckpt_info = p.load_encoder_and_tokenizer(embed_dim)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder.to(device)
    sem_out = {"scope": "semantic-only 10 watermarked + 10 negative, repeated 5, train/eval", "device": device, "model_name": model_name, "checkpoint_path": ckpt_path, "checkpoint_info": ckpt_info, "records": []}
    save({**result, "semantic_detector_probe": sem_out, "status": "semantic detector running"})
    for mode_name in ("train", "eval"):
        encoder.train() if mode_name == "train" else encoder.eval()
        detector = p.make_detector(config, first_record, "semantic-only", False, encoder, tokenizer, device)
        for label, rows, with_metadata in (("watermarked", watermarked_rows, True), ("negative", negative_rows, False)):
            for row in rows:
                sid = str(row.get("id"))
                rep = p.summarize_detection_repeats(detector, row, with_metadata=with_metadata, repeats=5)
                persisted = persisted_details.get(sid, {})
                sem_out["records"].append({
                    "mode": mode_name,
                    "set": label,
                    "id": sid,
                    "code_sha256": p.sha256_text(str(row.get("generated_code", ""))),
                    "repeats": rep,
                    "persisted_detail": {k: persisted.get(k) for k in ("z_score", "is_watermarked", "hits", "independent_blocks") if k in persisted},
                })
                save({**result, "semantic_detector_probe": sem_out, "status": f"semantic detector running {mode_name} {label} {sid}"})
    sem_out["unstable_records"] = [
        {"mode": r["mode"], "set": r["set"], "id": r["id"]}
        for r in sem_out["records"]
        if r["repeats"].get("unique_z_values", 0) > 1
        or r["repeats"].get("unique_hit_hashes", 0) > 1
        or r["repeats"].get("unique_predictions", 0) > 1
    ]
    result["semantic_detector_probe"] = sem_out
    result["not_run"] = ["full dual-channel 10+10 repeated 5 stopped due CPU-bound lexical replay cost", "CPU encoder repeated 10 stopped due infeasible runtime"]
    result["status"] = "complete"
    save(result)
    print(OUT)


if __name__ == "__main__":
    main()
