#!/bin/bash
set -e
cd /root/autodl-tmp/WFCLLM
source activate WFCLLM
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

LOG="logs/tc_small_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

echo "[$(date)] 小量 token-channel 训练测试开始 (10样本, 3epoch)" | tee "$LOG"

python -u -c "
import sys
sys.path.insert(0, '.')
from pathlib import Path
from wfcllm.watermark.token_channel.training.workflow import run_token_channel_train_workflow, TokenChannelTrainWorkflowConfig

cfg = TokenChannelTrainWorkflowConfig(
    dataset='humaneval',
    dataset_path='data/datasets',
    lm_model_path='data/models/deepseek-coder-7b-instruct/deepseek-ai/deepseek-coder-7b-instruct-v1___5',
    model_path=Path('data/models/token-channel-small-test'),
    cache_path=Path('data/token_channel/small_test_corpus.json'),
    context_width=128,
    hidden_size=64,
    batch_size=32,
    epochs=3,
    lr=0.001,
    entropy_threshold=0.5,
    diversity_threshold=1,
    split_ratio=0.9,
    seed=0,
    max_variants=5,
)
summary = run_token_channel_train_workflow(cfg)
print(f'DONE: train_rows={summary.train_rows}, val_rows={summary.validation_rows}, epochs={len(summary.epochs)}')
for e in summary.epochs:
    print(f'  Epoch {e.epoch}: train={e.train_loss:.4f} val={e.validation_loss:.4f}')
" 2>&1 | tee -a "$LOG"

echo "[$(date)] 完成" | tee -a "$LOG"
