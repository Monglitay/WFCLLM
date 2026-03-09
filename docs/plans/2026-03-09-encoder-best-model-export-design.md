# Encoder 最优模型导出设计

**日期：** 2026-03-09
**状态：** 已确认

## 背景

当前 encoder 训练结束后，最优模型仅存在于 `data/checkpoints/encoder/encoder_epochN.pt` 中，watermark 和 extract 阶段需要通过 `run_state.json` 传递 checkpoint 路径才能加载微调权重。目标是让最优模型自动导出到固定路径，便于下游阶段直接读取。

## 目标

1. 训练时自动将最优模型保存到 `data/models/encoder/best_model.pt`
2. watermark、extract、eval-only 阶段统一从该路径加载微调权重
3. 将当前已训练的最优模型迁移到新路径

## 设计

### 保存格式

```python
{
    "model_state_dict": model.state_dict(),
    "config": dataclasses.asdict(encoder_config),
    "best_metric": float,   # val_loss（越小越好）
    "epoch": int,
}
```

### 保存路径

`{EncoderConfig.output_model_dir}/best_model.pt`，默认为 `data/models/encoder/best_model.pt`。

### 变更清单

#### 1. `wfcllm/encoder/config.py`
新增字段：
```python
output_model_dir: str = "data/models/encoder"
```

#### 2. `wfcllm/encoder/trainer.py`
- `ContrastiveTrainer.__init__` 接收 `output_model_dir`
- 每次发现新最优 val_loss 时，调用 `_export_best_model()` 保存到 `output_model_dir/best_model.pt`

#### 3. `configs/base_config.json`
encoder 节点新增：
```json
"output_model_dir": "data/models/encoder"
```

#### 4. `run.py`
- encoder 阶段：将 `output_model_dir` 传入 `ContrastiveTrainer`
- watermark/extract/eval-only 阶段：优先从 `data/models/encoder/best_model.pt` 加载；不存在则 fallback 到 `run_state.json` 中的 checkpoint 路径

### 迁移当前最优模型

`encoder_epoch10.pt`（val_loss=0.049316）按新格式写入 `data/models/encoder/best_model.pt`。

## 不变的内容

- `WatermarkConfig.encoder_model_path` 仍指向基础预训练模型 `data/models/codet5-base`（tokenizer 等需要从此处加载）
- checkpoint 机制（`data/checkpoints/encoder/`）保持不变，用于训练恢复
