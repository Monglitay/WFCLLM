# Offline Resources Download & README Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将项目所需全部 HuggingFace 离线资源（模型、Tokenizer、数据集）下载到项目本地 `data/` 目录，修改代码使其优先从本地路径加载，并在 README 中补全所有资源的下载方式和默认存放路径，使项目在内网环境下可以一键部署运行。

**Architecture:** 资源分四类：
1. `Salesforce/codet5-base` —— 编码器底座模型 + Tokenizer（阶段一二三均需）
2. `google-research-datasets/mbpp` + `openai/openai_humaneval` —— 编码器训练数据集（阶段一）
3. 代码生成 LLM（如 CodeLlama） —— 阶段二水印嵌入（用户自备，README 说明推荐型号和下载命令）
4. tree-sitter 语言包 —— AST 解析依赖（pip 离线包）

下载后：① 修改 `EncoderConfig`/`WatermarkConfig` 支持本地路径；② 修改 `train.py`/`run.py` 自动检测本地资源；③ 更新 README。

**Tech Stack:** Python + `huggingface_hub.snapshot_download`、`datasets`、conda WFCLLM 环境

---

## 完整资源清单（经过全代码库扫描）

| # | 资源类型 | HuggingFace ID | 用途 | 引用位置 | 大小 | 目标本地路径 |
|---|---------|---------------|------|---------|------|-------------|
| 1 | **模型 + Tokenizer** | `Salesforce/codet5-base` | 编码器底座（阶段一预训练；阶段二/三做语义投影） | `wfcllm/encoder/config.py`, `wfcllm/encoder/model.py`, `wfcllm/encoder/train.py`, `wfcllm/watermark/config.py`, `run.py`, `tests/encoder/test_model.py`, `tests/encoder/test_dataset.py` | ~850 MB | `data/models/codet5-base/` |
| 2 | **数据集** | `google-research-datasets/mbpp` (config: `full`) | 编码器对比学习训练数据（阶段一） | `wfcllm/encoder/train.py:40` | ~1 MB | `data/datasets/mbpp/` |
| 3 | **数据集** | `openai/openai_humaneval` | 编码器对比学习训练数据（阶段一） | `wfcllm/encoder/train.py:45` | ~0.2 MB | `data/datasets/humaneval/` |
| 4 | **代码生成 LLM** | 用户自备（推荐 `codellama/CodeLlama-7b-hf`） | 阶段二：LLM 生成含水印代码 | `run.py:245-246`（`AutoModelForCausalLM.from_pretrained(args.lm_model_path)`） | ~13+ GB | 用户自定义，通过 `--lm-model-path` 指定 |

> **注：** codet5-base 在本地 `~/.cache` 中有两个 snapshot（`02cd2d31` 含完整 tokenizer + pytorch_model.bin，`fcedbb79` 只有 safetensors 权重），实际有效 snapshot 为 `02cd2d31`（refs/main 指向它）。HF hub 存储实际上是 blob 共享，尽管显示 3.4 GB，实际唯一数据约 850 MB。
>
> **注：** `wfcllm/watermark/config.py` 中有 `encoder_model_path: str = "Salesforce/codet5-base"` 字段，但 `run.py` 中实际并未使用此字段（而是用 `EncoderConfig.model_name`），计划中一并修复此不一致。

---

### Task 1: 下载 CodeT5-base 模型 + Tokenizer 到项目本地

**Files:**
- Create: `data/models/codet5-base/` （通过 snapshot_download 创建）

**Step 1: 运行下载**

```bash
conda run -n WFCLLM python - <<'EOF'
from huggingface_hub import snapshot_download
import os

local_dir = "data/models/codet5-base"
os.makedirs(local_dir, exist_ok=True)

print("Downloading Salesforce/codet5-base (model + tokenizer)...")
path = snapshot_download(
    repo_id="Salesforce/codet5-base",
    local_dir=local_dir,
    local_dir_use_symlinks=False,
)
print(f"Downloaded to: {path}")
for f in sorted(os.listdir(local_dir)):
    size = os.path.getsize(os.path.join(local_dir, f))
    print(f"  {f}: {size/1024/1024:.1f} MB")
EOF
```

**Step 2: 验证模型 + Tokenizer 均可从本地路径加载**

```bash
conda run -n WFCLLM python - <<'EOF'
from transformers import AutoTokenizer, T5EncoderModel

local_path = "data/models/codet5-base"
tok = AutoTokenizer.from_pretrained(local_path)
print(f"Tokenizer OK: vocab_size={tok.vocab_size}")
model = T5EncoderModel.from_pretrained(local_path)
params = sum(p.numel() for p in model.parameters()) / 1e6
print(f"Model OK: {params:.1f}M params")

# 确认 tokenizer 文件已包含在目录内（不是仅 model 权重）
import os
files = os.listdir(local_path)
assert "tokenizer_config.json" in files, "tokenizer_config.json missing!"
assert "vocab.json" in files or "tokenizer.json" in files, "vocab files missing!"
assert "config.json" in files, "config.json missing!"
print("PASS: model + tokenizer both present at data/models/codet5-base/")
EOF
```

期望：`PASS: model + tokenizer both present at data/models/codet5-base/`

**Step 3: Commit**

```bash
git commit --allow-empty -m "chore: download codet5-base to data/models/codet5-base (not tracked in git)"
```

---

### Task 2: 下载 MBPP 数据集到项目本地

**Files:**
- Create: `data/datasets/mbpp/`

**Step 1: 下载**

```bash
conda run -n WFCLLM python - <<'EOF'
from datasets import load_dataset
import os

cache_dir = "data/datasets/mbpp"
os.makedirs(cache_dir, exist_ok=True)
ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir=cache_dir)
print(f"MBPP splits: { {k: len(v) for k,v in ds.items()} }")
assert len(ds["train"]) == 374
print("PASS: MBPP downloaded to data/datasets/mbpp/")
EOF
```

**Step 2: 验证离线加载（模拟内网）**

```bash
conda run -n WFCLLM python - <<'EOF'
from datasets import load_dataset
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir="data/datasets/mbpp")
print(f"Offline load OK: { {k: len(v) for k,v in ds.items()} }")
EOF
```

---

### Task 3: 下载 HumanEval 数据集到项目本地

**Files:**
- Create: `data/datasets/humaneval/`

**Step 1: 下载**

```bash
conda run -n WFCLLM python - <<'EOF'
from datasets import load_dataset
import os

cache_dir = "data/datasets/humaneval"
os.makedirs(cache_dir, exist_ok=True)
ds = load_dataset("openai/openai_humaneval", cache_dir=cache_dir)
print(f"HumanEval splits: { {k: len(v) for k,v in ds.items()} }")
assert len(ds["test"]) == 164
print("PASS: HumanEval downloaded to data/datasets/humaneval/")
EOF
```

**Step 2: 验证离线加载**

```bash
conda run -n WFCLLM python - <<'EOF'
from datasets import load_dataset
import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
ds = load_dataset("openai/openai_humaneval", cache_dir="data/datasets/humaneval")
print(f"Offline load OK: { {k: len(v) for k,v in ds.items()} }")
EOF
```

---

### Task 4: 更新 EncoderConfig 添加本地资源路径字段

**Files:**
- Modify: `wfcllm/encoder/config.py`

**Step 1: 写失败测试**

在 `tests/encoder/test_config.py`（若不存在则新建）：

```python
def test_encoder_config_has_local_paths():
    from wfcllm.encoder.config import EncoderConfig
    cfg = EncoderConfig()
    assert cfg.local_model_dir == "data/models"
    assert cfg.local_dataset_dir == "data/datasets"
```

**Step 2: 运行确认失败**

```bash
conda run -n WFCLLM pytest tests/encoder/test_config.py::test_encoder_config_has_local_paths -v
```

期望：FAIL（字段不存在）

**Step 3: 在 EncoderConfig 中添加两个字段**

在 `wfcllm/encoder/config.py` 的 `# Paths` 区块末尾添加：

```python
    # Paths
    checkpoint_dir: str = "data/checkpoints/encoder"
    results_dir: str = "data/results"
    local_model_dir: str = "data/models"      # 本地模型根目录（离线部署）
    local_dataset_dir: str = "data/datasets"  # 本地数据集根目录（离线部署）
```

**Step 4: 运行测试**

```bash
conda run -n WFCLLM pytest tests/encoder/test_config.py -v
```

期望：PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/config.py tests/encoder/test_config.py
git commit -m "feat(encoder): add local_model_dir and local_dataset_dir to EncoderConfig"
```

---

### Task 5: 修改 train.py 支持从本地路径加载数据集和模型

**Files:**
- Modify: `wfcllm/encoder/train.py:34-53` (`load_code_samples`)
- Modify: `wfcllm/encoder/train.py:97-145` (`main`)

**Step 1: 修改 `load_code_samples` 支持 `local_dataset_dir`**

将函数签名和实现改为：

```python
def load_code_samples(
    data_sources: list[str],
    local_dataset_dir: str | None = None,
) -> list[dict]:
    """Load code samples from HuggingFace datasets.

    Args:
        data_sources: Dataset names ("mbpp", "humaneval").
        local_dataset_dir: Optional local cache root, e.g. "data/datasets".
            Each dataset lives at <local_dataset_dir>/<name>/.
            If None, uses HF default global cache.
    """
    samples: list[dict] = []

    for source in data_sources:
        if source == "mbpp":
            cache_dir = str(Path(local_dataset_dir) / "mbpp") if local_dataset_dir else None
            ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir=cache_dir)
            for split in ds:
                for item in ds[split]:
                    samples.append({"code": item["code"], "source": "mbpp"})
        elif source == "humaneval":
            cache_dir = str(Path(local_dataset_dir) / "humaneval") if local_dataset_dir else None
            ds = load_dataset("openai/openai_humaneval", cache_dir=cache_dir)
            for split in ds:
                for item in ds[split]:
                    canonical = item.get("canonical_solution", "")
                    prompt = item.get("prompt", "")
                    code = prompt + canonical
                    samples.append({"code": code, "source": "humaneval"})

    return samples
```

**Step 2: 修改 `main()` 使用本地路径**

在 `main()` 中：

① 解析 `model_name` 为本地路径（若存在）：

```python
# Resolve model path: prefer local if available
local_codet5 = Path(config.local_model_dir) / "codet5-base"
if local_codet5.exists() and (local_codet5 / "config.json").exists():
    effective_model = str(local_codet5)
    print(f"  Using local model: {effective_model}")
else:
    effective_model = config.model_name
    print(f"  Using HF Hub model: {effective_model}")
```

② 传入 `local_dataset_dir` 给 `load_code_samples`：

```python
code_samples = load_code_samples(
    config.data_sources,
    local_dataset_dir=config.local_dataset_dir,
)
```

③ 用 `effective_model` 加载 tokenizer 和 SemanticEncoder：

```python
tokenizer = AutoTokenizer.from_pretrained(effective_model)
# 创建临时 config 以便 SemanticEncoder 用本地路径
from dataclasses import replace
config_for_model = replace(config, model_name=effective_model)
model = SemanticEncoder(config=config_for_model)
trainer = ContrastiveTrainer(model, train_loader, val_loader, config_for_model)
```

**Step 3: 运行现有测试确认无回归**

```bash
conda run -n WFCLLM pytest tests/encoder/ -v
```

**Step 4: Commit**

```bash
git add wfcllm/encoder/train.py
git commit -m "feat(encoder): load datasets and model from local paths when available"
```

---

### Task 6: 修复 WatermarkConfig.encoder_model_path 不一致

**Files:**
- Modify: `wfcllm/watermark/config.py`
- Modify: `run.py:208-274` (`run_watermark`)、`run.py:277-341` (`run_extract`)

**问题：** `WatermarkConfig` 有 `encoder_model_path: str = "Salesforce/codet5-base"` 字段，但 `run.py` 实际从 `EncoderConfig.model_name` 读取 encoder 路径，两者不一致。

**Step 1: 修改 WatermarkConfig 默认值对齐本地路径约定**

在 `wfcllm/watermark/config.py` 中：

```python
    # Encoder
    encoder_model_path: str = "data/models/codet5-base"  # 本地路径优先；可传 HF Hub ID 作回退
    encoder_embed_dim: int = 128
    encoder_device: str = "cuda"
```

**Step 2: 修改 run.py 中 run_watermark() 和 run_extract() 自动检测本地模型**

在两处 `enc_config = EncoderConfig(embed_dim=embed_dim)` 之后加入：

```python
from pathlib import Path
local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
if local_codet5.exists() and (local_codet5 / "config.json").exists():
    enc_config.model_name = str(local_codet5)
    print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
else:
    print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
```

**Step 3: Commit**

```bash
git add wfcllm/watermark/config.py run.py
git commit -m "fix(watermark): align encoder_model_path default to local path convention"
```

---

### Task 7: 修改 run.py run_encoder() 自动使用本地路径

**Files:**
- Modify: `run.py:165-205` (`run_encoder`)

在 `config = EncoderConfig()` 的 arg 覆盖逻辑之后，当用户未显式指定 `--model-name` 时自动使用本地模型：

```python
# 若用户未指定 --model-name，自动检测本地
if args.model_name is None:
    local_codet5 = Path(config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        config.model_name = str(local_codet5)
        print(f"[自动] 使用本地模型: {config.model_name}")
    else:
        print(f"[回退] 使用 HF Hub 模型: {config.model_name}")
```

**Commit:**

```bash
git add run.py
git commit -m "feat(run): auto-detect local codet5-base in run_encoder"
```

---

### Task 8: 更新 .gitignore 保证数据目录不入库

**Files:**
- Modify: `.gitignore`

**Step 1: 检查当前 .gitignore**

```bash
cat .gitignore
```

**Step 2: 确认以下条目存在（如缺则添加）**

```gitignore
# 离线资源（模型权重、数据集）— 大二进制文件不入库
data/models/
data/datasets/
data/checkpoints/
data/results/
data/run_state.json

# 保留目录占位文件
!data/models/.gitkeep
!data/datasets/.gitkeep
!data/checkpoints/.gitkeep
!data/results/.gitkeep
```

**Step 3: 创建占位文件**

```bash
touch data/models/.gitkeep data/datasets/.gitkeep
```

**Step 4: 确认 data/mbpp（旧路径）也被忽略**

```bash
git check-ignore -v data/mbpp data/models data/datasets
```

若 `data/mbpp` 未被忽略，在 `.gitignore` 中加 `data/mbpp/`。

**Step 5: Commit**

```bash
git add .gitignore data/models/.gitkeep data/datasets/.gitkeep
git commit -m "chore: update .gitignore for offline resource dirs, add .gitkeep placeholders"
```

---

### Task 9: 更新 README.md 补全「离线资源准备」章节

**Files:**
- Modify: `README.md`

**目标：** 在「环境安装」和「快速开始」之间插入完整的「离线资源准备」章节，覆盖**所有 4 类资源**（模型 + tokenizer、两个数据集、代码生成 LLM），以及离线打包传输方法。

在 `README.md` 的 `---` 分隔线（「环境安装」章节末尾）之后，`## 快速开始` 之前，插入以下内容：

````markdown
---

## 离线资源准备（离线/内网环境必读）

> 目标服务器处于内网，无法访问 HuggingFace Hub。需在联网环境预先下载所有资源，再打包传至目标服务器。

### 资源清单

| 资源 | HuggingFace ID | 用途 | 默认本地路径 | 大小 |
|------|---------------|------|-------------|------|
| CodeT5-base 模型 + Tokenizer | `Salesforce/codet5-base` | 语义编码器底座（阶段一/二/三） | `data/models/codet5-base/` | ~850 MB |
| MBPP 数据集 | `google-research-datasets/mbpp` | 编码器训练数据（阶段一） | `data/datasets/mbpp/` | ~1 MB |
| HumanEval 数据集 | `openai/openai_humaneval` | 编码器训练数据（阶段一） | `data/datasets/humaneval/` | ~0.2 MB |
| 代码生成 LLM | 用户自备（推荐见下方） | 生成含水印代码（阶段二） | 任意路径，`--lm-model-path` 指定 | ≥13 GB |

---

### 1. 下载 CodeT5-base 模型 + Tokenizer

```bash
conda activate WFCLLM

# 方式一：脚本下载（推荐，确保 tokenizer 文件完整）
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="Salesforce/codet5-base",
    local_dir="data/models/codet5-base",
    local_dir_use_symlinks=False,
)
print("Done: data/models/codet5-base/")
EOF

# 方式二：huggingface-cli
huggingface-cli download Salesforce/codet5-base \
    --local-dir data/models/codet5-base \
    --local-dir-use-symlinks False
```

下载后目录包含：`config.json`、`pytorch_model.bin`（~850 MB）、`tokenizer_config.json`、`vocab.json`、`merges.txt`、`special_tokens_map.json`、`added_tokens.json`。

**验证：**
```bash
conda run -n WFCLLM python -c "
from transformers import AutoTokenizer, T5EncoderModel
tok = AutoTokenizer.from_pretrained('data/models/codet5-base')
m = T5EncoderModel.from_pretrained('data/models/codet5-base')
print('OK:', sum(p.numel() for p in m.parameters())//1_000_000, 'M params, vocab:', tok.vocab_size)
"
```

---

### 2. 下载训练数据集

```bash
conda activate WFCLLM

# MBPP（~1 MB）
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("google-research-datasets/mbpp", "full", cache_dir="data/datasets/mbpp")
print("MBPP:", {k: len(v) for k, v in ds.items()})
EOF

# HumanEval（~0.2 MB）
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("openai/openai_humaneval", cache_dir="data/datasets/humaneval")
print("HumanEval:", {k: len(v) for k, v in ds.items()})
EOF
```

---

### 3. 代码生成 LLM（阶段二）

阶段二需要一个代码生成 LLM，**不随本仓库分发**，需用户自行下载并通过 `--lm-model-path` 指定路径。

推荐模型：

| 模型 | HuggingFace ID | 显存需求 |
|------|---------------|---------|
| CodeLlama-7b-hf | `codellama/CodeLlama-7b-hf` | ~14 GB (FP16) |
| DeepSeek-Coder-6.7b-base | `deepseek-ai/deepseek-coder-6.7b-base` | ~13 GB (FP16) |
| StarCoder2-7b | `bigcode/starcoder2-7b` | ~14 GB (FP16) |
| CodeLlama-13b-hf | `codellama/CodeLlama-13b-hf` | ~26 GB (FP16) |

**下载（以 CodeLlama-7b 为例）：**
```bash
# 方式一
huggingface-cli download codellama/CodeLlama-7b-hf \
    --local-dir data/models/codellama-7b \
    --local-dir-use-symlinks False

# 方式二
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="codellama/CodeLlama-7b-hf",
    local_dir="data/models/codellama-7b",
    local_dir_use_symlinks=False,
)
EOF
```

**使用：**
```bash
python run.py --phase watermark \
    --lm-model-path data/models/codellama-7b \
    --secret-key mysecret \
    --prompt "def fibonacci(n):"
```

---

### 4. 打包传输到目标服务器

```bash
# 打包（约 850 MB+，LLM 模型单独处理）
tar -czf wfcllm_resources.tar.gz \
    data/models/codet5-base \
    data/datasets/mbpp \
    data/datasets/humaneval

# 传输
scp wfcllm_resources.tar.gz user@server:/path/to/WFCLLM/

# 目标服务器解压
ssh user@server "cd /path/to/WFCLLM && tar -xzf wfcllm_resources.tar.gz"
```

解压后运行 `python run.py` 会**自动检测** `data/models/codet5-base/` 和 `data/datasets/` 下的本地资源，无需额外配置。

````

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add complete offline resources section to README (model+tokenizer+datasets+LLM)"
```

---

### Task 10: 端到端验证

**Step 1: 验证所有本地资源可离线加载**

```bash
conda run -n WFCLLM python - <<'EOF'
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

from transformers import AutoTokenizer, T5EncoderModel
from datasets import load_dataset

# 1. CodeT5 model + tokenizer
tok = AutoTokenizer.from_pretrained("data/models/codet5-base")
print(f"[OK] CodeT5 tokenizer: vocab_size={tok.vocab_size}")
model = T5EncoderModel.from_pretrained("data/models/codet5-base")
print(f"[OK] CodeT5 model: {sum(p.numel() for p in model.parameters())//1_000_000}M params")

# 2. MBPP dataset
ds_mbpp = load_dataset("google-research-datasets/mbpp", "full", cache_dir="data/datasets/mbpp")
print(f"[OK] MBPP: { {k: len(v) for k,v in ds_mbpp.items()} }")

# 3. HumanEval dataset
ds_he = load_dataset("openai/openai_humaneval", cache_dir="data/datasets/humaneval")
print(f"[OK] HumanEval: { {k: len(v) for k,v in ds_he.items()} }")

print("\nALL RESOURCES OK - offline deployment ready")
EOF
```

**Step 2: 验证 run.py 自动检测本地路径**

```bash
conda run -n WFCLLM python run.py --status
```

期望：正常输出阶段状态，无 import 错误。

**Step 3: 运行完整测试套件**

```bash
conda run -n WFCLLM pytest tests/ -v
```

期望：所有测试通过（无新增失败）。

**Step 4: Final commit**

```bash
git add -A
git commit -m "chore: all offline resources ready, tests passing"
```

---

## 目录结构（最终状态）

```
data/
├── models/
│   ├── .gitkeep              # git 占位符
│   └── codet5-base/          # Salesforce/codet5-base（~850 MB，不入库）
│       ├── config.json
│       ├── pytorch_model.bin
│       ├── tokenizer_config.json
│       ├── vocab.json
│       ├── merges.txt
│       ├── special_tokens_map.json
│       └── added_tokens.json
├── datasets/
│   ├── .gitkeep              # git 占位符
│   ├── mbpp/                 # google-research-datasets/mbpp（~1 MB，不入库）
│   └── humaneval/            # openai/openai_humaneval（~0.2 MB，不入库）
├── checkpoints/
│   └── encoder/              # 训练产出 checkpoint（不入库）
└── results/                  # 评估报告（不入库）
```

## 执行顺序

- **Task 1-3**（下载资源）：可并行执行，依赖网速
- **Task 4-5**（修改 EncoderConfig + train.py）：顺序执行，4 先于 5
- **Task 6-7**（修改 WatermarkConfig + run.py）：可与 4-5 并行
- **Task 8**（.gitignore）：独立，任何时候可执行
- **Task 9**（README）：Task 1-7 完成后执行
- **Task 10**（验证）：最后执行
