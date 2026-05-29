#!/bin/bash
set -e
cd /root/autodl-tmp/WFCLLM
source /root/miniconda3/etc/profile.d/conda.sh
conda activate WFCLLM

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_A="logs/watermark_A_${TIMESTAMP}.log"
LOG_B="logs/watermark_B_${TIMESTAMP}.log"
mkdir -p logs data/watermarked_A data/watermarked_B

echo "[$(date)] 启动并行嵌入：A(0-81) B(82-163)" | tee -a "$LOG_A"

# 进程 A：前 82 条
HF_HUB_OFFLINE=1 python -u run.py \
    --config configs/base_config.json \
    --phase watermark \
    --force \
    --sample-offset 0 \
    --sample-limit 82 \
    --output-dir data/watermarked_A \
    2>&1 | tee -a "$LOG_A" &
PID_A=$!

# 进程 B：后 82 条
HF_HUB_OFFLINE=1 python -u run.py \
    --config configs/base_config.json \
    --phase watermark \
    --force \
    --sample-offset 82 \
    --output-dir data/watermarked_B \
    2>&1 | tee -a "$LOG_B" &
PID_B=$!

echo "[$(date)] A PID=$PID_A  B PID=$PID_B"
echo "[$(date)] 等待两个进程完成..."

wait $PID_A
STATUS_A=$?
wait $PID_B
STATUS_B=$?

echo "[$(date)] A 退出码=$STATUS_A  B 退出码=$STATUS_B"

if [ $STATUS_A -ne 0 ] || [ $STATUS_B -ne 0 ]; then
    echo "[错误] 有进程失败，跳过合并"
    exit 1
fi

# 合并结果
MERGED="data/watermarked/humaneval_${TIMESTAMP}_merged.jsonl"
mkdir -p data/watermarked
FILE_A=$(ls -t data/watermarked_A/humaneval_*.jsonl 2>/dev/null | head -1)
FILE_B=$(ls -t data/watermarked_B/humaneval_*.jsonl 2>/dev/null | head -1)

if [ -z "$FILE_A" ] || [ -z "$FILE_B" ]; then
    echo "[错误] 找不到输出文件"
    exit 1
fi

cat "$FILE_A" "$FILE_B" > "$MERGED"
TOTAL=$(wc -l < "$MERGED")
echo "[$(date)] 合并完成：$MERGED（共 $TOTAL 条）"
