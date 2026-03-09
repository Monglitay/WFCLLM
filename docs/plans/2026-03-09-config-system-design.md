# run.py 配置系统重构设计文档

日期：2026-03-09

## 背景

当前 `run.py` 所有参数通过硬编码默认值 + CLI 覆盖实现，缺乏可持久化的配置文件机制。需要引入 JSON 配置文件支持，方便管理不同实验配置。

## 目标

- 以 JSON 文件作为配置格式，建立 `configs/` 目录存放不同配置文件
- `configs/base_config.json` 作为默认配置，包含三个阶段所有参数
- 保留 CLI 参数方式，CLI 优先级高于 JSON 配置文件
- `--config` CLI 参数指定配置文件路径，默认为 `configs/base_config.json`

## 目录与文件结构

```
configs/
└── base_config.json       # 默认配置，包含三个阶段所有参数
run.py                     # 改动：新增 --config 参数 + load/merge 逻辑
```

## JSON 结构

按阶段分组，每个配置文件独立完整（不做继承合并）：

```json
{
  "encoder": {
    "model_name": "Salesforce/codet5-base",
    "embed_dim": 128,
    "use_lora": true,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q", "v"],
    "use_bf16": true,
    "data_sources": ["mbpp", "humaneval"],
    "max_seq_length": 256,
    "negative_ratio": 0.5,
    "lr": 2e-5,
    "batch_size": 64,
    "epochs": 10,
    "margin": 0.3,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "num_workers": 8,
    "pin_memory": true,
    "checkpoint_dir": "data/checkpoints/encoder",
    "results_dir": "data/results",
    "local_model_dir": "data/models",
    "local_dataset_dir": "data/datasets"
  },
  "watermark": {
    "secret_key": "",
    "lm_model_path": "",
    "prompt": "def solution():",
    "output_file": "data/results/watermarked.py",
    "encoder_model_path": "data/models/codet5-base",
    "encoder_embed_dim": 128,
    "encoder_device": "cuda",
    "margin_base": 0.1,
    "margin_alpha": 0.05,
    "max_retries": 5,
    "temperature": 0.8,
    "top_p": 0.95,
    "top_k": 50,
    "max_new_tokens": 512,
    "eos_token_id": null,
    "enable_fallback": true
  },
  "extract": {
    "secret_key": "",
    "code_file": "",
    "embed_dim": 128,
    "z_threshold": 3.0
  }
}
```

## CLI 变更

`build_parser()` 新增参数：

```
--config   配置文件路径（默认: configs/base_config.json）
```

其余所有现有 CLI 参数保持不变。运行控制参数（`--phase`、`--status`、`--reset`、`--force`、`--eval-only`、`--checkpoint`）只走 CLI，不放入 JSON。

## 配置加载与合并逻辑

`run.py` 内部新增两个函数：

```python
def load_config(config_path: Path) -> dict:
    """读取 JSON 配置文件，返回按阶段分组的 dict。"""

def apply_config(args: argparse.Namespace, config: dict) -> argparse.Namespace:
    """用 JSON 配置填充 args 中值为 None 的字段（CLI 优先）。"""
```

优先级链：**JSON 配置文件 → CLI 参数覆盖（CLI 最高优先级）**

`apply_config` 在 `main()` 中 `parse_args()` 后立即调用，之后所有 `run_*` 函数照旧使用 `args`。

### 特殊处理

- `--no-lora` / `--no-bf16` 是 `store_true` flag，默认 False：
  - CLI 传了 `--no-lora` → `use_lora = False`（CLI 优先）
  - CLI 未传 → 从 JSON 读取 `use_lora` 值应用
- `lora_target_modules`、`data_sources` 是列表类型，直接从 JSON 读取

## 错误处理

- 配置文件不存在 → 打印明确错误信息并退出
- JSON 解析失败 → 打印错误信息并退出
- 配置文件缺少某阶段的 key → 用空 dict 处理，不报错（该阶段参数全靠 CLI 或 config 类默认值）
