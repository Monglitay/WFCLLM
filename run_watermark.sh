#!/bin/bash
set -e
cd /root/autodl-tmp/WFCLLM
LOG="logs/watermark_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[$(date)] 开始嵌入水印" | tee "$LOG"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate WFCLLM
HF_HUB_OFFLINE=1 python -u run.py \
    --config configs/base_config.json \
    --phase watermark \
    --force 2>&1 | tee -a "$LOG"

echo "[$(date)] 嵌入完成" | tee -a "$LOG"
