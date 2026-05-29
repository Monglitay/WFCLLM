#!/bin/bash
set -e
cd /root/autodl-tmp/WFCLLM
source activate WFCLLM
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

LOG_DIR="logs"
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="$LOG_DIR/pipeline_${TIMESTAMP}.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"
}

log "=========================================="
log "完整流水线开始"
log "日志文件: $LOG"
log "=========================================="

# ── Step 1: 生成负样本 ──────────────────────────
log ""
log ">>> Step 1/4: 生成负样本语料"
python -u run.py --config configs/base_config.json --phase generate-negative --force 2>&1 | tee -a "$LOG"
log "<<< Step 1 完成"

# ── Step 2: 训练 token-channel ──────────────────
log ""
log ">>> Step 2/4: 训练 token-channel 模型"
python -u run.py --config configs/base_config.json --phase token-channel-train --force 2>&1 | tee -a "$LOG"
log "<<< Step 2 完成"

# ── Step 3: 水印嵌入 ────────────────────────────
log ""
log ">>> Step 3/4: 水印嵌入生成"
# 先开启 token_channel
python -c "
import json
with open('configs/base_config.json') as f:
    cfg = json.load(f)
cfg['watermark']['token_channel']['enabled'] = True
cfg['watermark']['token_channel']['channel_mode'] = 'dual-channel'
with open('configs/base_config.json', 'w') as f:
    json.dump(cfg, f, indent=2, ensure_ascii=False)
print('token_channel enabled')
" 2>&1 | tee -a "$LOG"

python -u run.py --config configs/base_config.json --phase watermark --force 2>&1 | tee -a "$LOG"
log "<<< Step 3 完成"

# ── Step 4: 提取 + 评估 ─────────────────────────
log ""
log ">>> Step 4/4: 提取检测 + benchmark 评估"

# 找最新的水印文件
WATERMARKED_FILE=$(ls -t data/watermarked/humaneval_2026*.jsonl 2>/dev/null | head -1)
if [ -z "$WATERMARKED_FILE" ]; then
    log "错误: 找不到水印文件"
    exit 1
fi
log "水印文件: $WATERMARKED_FILE"

# 提取检测（生成 positive details）
EXTRACT_OUTDIR="data/results/instruct_run_${TIMESTAMP}"
python -u run.py --config configs/base_config.json --phase extract \
    --input-file "$WATERMARKED_FILE" \
    --output-dir "$EXTRACT_OUTDIR" 2>&1 | tee -a "$LOG"

POSITIVE_DETAILS=$(ls "$EXTRACT_OUTDIR"/humaneval_*_details.jsonl 2>/dev/null | head -1)
if [ -z "$POSITIVE_DETAILS" ]; then
    log "错误: 找不到 positive details 文件"
    exit 1
fi
log "positive details: $POSITIVE_DETAILS"

# 生成负样本的 extract details（如果还没有）
NEG_DETAILS="data/results/negative/negative_corpus_details.jsonl"
if [ ! -f "$NEG_DETAILS" ]; then
    log "生成负样本 extract details..."
    python -u run.py --config configs/base_config.json --phase extract \
        --input-file data/negative_corpus.jsonl \
        --output-dir data/results/negative 2>&1 | tee -a "$LOG"
fi

# benchmark 评估
EVAL_OUTDIR="data/eval/instruct_run_${TIMESTAMP}"
python -u scripts/evaluate.py bench \
    --dataset humaneval \
    --config configs/base_config.json \
    --watermarked-dirs data/watermarked \
    --negative-corpus data/negative_corpus.jsonl \
    --positive-details "$POSITIVE_DETAILS" \
    --negative-details "$NEG_DETAILS" \
    --output-dir "$EVAL_OUTDIR" 2>&1 | tee -a "$LOG"

log "<<< Step 4 完成"
log ""
log "=========================================="
log "流水线全部完成！"
log "结果目录: $EVAL_OUTDIR"
log "完整日志: $LOG"
log "=========================================="
