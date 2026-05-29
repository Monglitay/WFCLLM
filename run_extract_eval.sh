#!/bin/bash
set -e
cd /root/autodl-tmp/WFCLLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate WFCLLM

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="logs/extract_eval_${TIMESTAMP}.log"
mkdir -p logs

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG"; }

WATERMARKED_FILE="data/watermarked/humaneval_20260517_200008_merged.jsonl"
EXTRACT_OUTDIR="data/results/run_${TIMESTAMP}"
NEG_DETAILS="data/results/negative/negative_corpus_details.jsonl"

log "水印文件: $WATERMARKED_FILE"
log ">>> Step 1: 提取检测"

HF_HUB_OFFLINE=1 python -u run.py --config configs/base_config.json --phase extract \
    --input-file "$WATERMARKED_FILE" \
    --output-dir "$EXTRACT_OUTDIR" 2>&1 | tee -a "$LOG"

POSITIVE_DETAILS=$(ls "$EXTRACT_OUTDIR"/humaneval_*_details.jsonl 2>/dev/null | head -1)
if [ -z "$POSITIVE_DETAILS" ]; then
    log "错误: 找不到 positive details 文件"
    exit 1
fi
log "positive details: $POSITIVE_DETAILS"

if [ ! -f "$NEG_DETAILS" ]; then
    log ">>> 生成负样本 extract details..."
    HF_HUB_OFFLINE=1 python -u run.py --config configs/base_config.json --phase extract \
        --input-file data/negative_corpus.jsonl \
        --output-dir data/results/negative 2>&1 | tee -a "$LOG"
fi

log ">>> Step 2: benchmark 评估"
EVAL_OUTDIR="data/eval/run_${TIMESTAMP}"

HF_HUB_OFFLINE=1 python -u scripts/evaluate.py bench \
    --dataset humaneval \
    --config configs/base_config.json \
    --watermarked-dirs data/watermarked \
    --negative-corpus data/negative_corpus.jsonl \
    --positive-details "$POSITIVE_DETAILS" \
    --negative-details "$NEG_DETAILS" \
    --output-dir "$EVAL_OUTDIR" 2>&1 | tee -a "$LOG"

log "完成！结果目录: $EVAL_OUTDIR"
