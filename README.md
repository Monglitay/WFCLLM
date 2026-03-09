# WFCLLM：基于语句块语义特征的生成时代码水印

**WFCLLM (WaterMark for Code LLM)** 是一套针对代码大语言模型（Code LLM）的**生成时水印**系统。

核心思路：在 LLM 逐 Token 生成代码期间，实时拦截 AST 语句块，通过拒绝采样将水印特征注入代码的语义分布，
使生成代码在语义层面携带可统计验证的不可见标记，且对等价代码变换（混淆、重命名等）具备鲁棒性。

> 方案详见：[docs/基于语句块的语义特征的生成时代码水印方案.md](docs/基于语句块的语义特征的生成时代码水印方案.md)

---

## 系统架构

```
阶段一：编码器预训练
  代码数据集 → AST 切块 → 等价变换（正样本）/ 破坏性变换（负样本）
  → 对比学习微调 CodeT5 → 鲁棒语义编码器 E
         ↓
阶段二：生成时水印嵌入
  LLM + 编码器 E → 实时 AST 拦截 → 节点熵 → 方向向量派生
  → 余弦投影检验 → 拒绝采样回滚 → 含水印代码
         ↓
阶段三：提取与验证
  待测代码 → AST 解析 → 语义打分 → DP 去重 → Z 分数检验 → 有/无水印判决
```

---

## 环境安装

**要求：** conda 已安装，CUDA（可选）

```bash
# 1. 创建环境
conda create -n WFCLLM python=3.11 -y
conda activate WFCLLM

# 2. 安装依赖（PyTorch 请参照目标服务器 CUDA 版本单独安装）
pip install torch                   # 或按服务器 CUDA 版本安装
pip install -r requirements.txt
```

> **离线环境：** 提前下载 `Salesforce/codet5-base` 模型权重与 Tokenizer 至本地，
> 详见下方「离线资源准备」章节。

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
    --dataset humaneval
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

---

## 快速开始

### 查看各阶段状态

```bash
python run.py --status
```

### 运行全流程（自动跳过已完成阶段）

```bash
python run.py
```

### 单独运行某阶段

```bash
# 阶段一：训练语义编码器
python run.py --phase encoder

# 阶段一（自定义超参数）
python run.py --phase encoder --epochs 5 --lr 1e-4 --batch-size 16

# 阶段二：对 humaneval 数据集批量嵌入水印，输出 JSONL
python run.py --phase watermark \
    --lm-model-path data/models/deepseek-coder-7b \
    --secret-key mysecret \
    --dataset humaneval \
    --output-dir data/watermarked

# 阶段二（MBPP 数据集）
python run.py --phase watermark \
    --lm-model-path data/models/deepseek-coder-7b \
    --secret-key mysecret \
    --dataset mbpp

# 阶段三：检测水印 JSONL，输出统计报告（自动读取阶段二输出路径）
python run.py --phase extract \
    --secret-key mysecret

# 阶段三（指定 JSONL 文件）
python run.py --phase extract \
    --secret-key mysecret \
    --input-file data/watermarked/humaneval_20260309_120000.jsonl \
    --extract-output-dir data/results

# 强制重跑某阶段（忽略已完成标记）
python run.py --phase encoder --force

# 清除所有断点状态
python run.py --reset
```

---

## 各模块说明

### Phase 1 — `wfcllm.encoder`

```python
from wfcllm.encoder import EncoderConfig, SemanticEncoder

config = EncoderConfig(model_name="Salesforce/codet5-base", epochs=10)
# 完整训练流程见 wfcllm/encoder/train.py main()
```

关键 API：
- `EncoderConfig` — 超参数与路径配置（dataclass）
- `SemanticEncoder` — 基于 CodeT5 + LoRA 的语义编码器模型

### Phase 2 — `wfcllm.watermark`

```python
from wfcllm.watermark import WatermarkConfig, WatermarkGenerator
from wfcllm.watermark import WatermarkPipeline, WatermarkPipelineConfig

# 单条生成（底层 API）
config = WatermarkConfig(secret_key="mysecret")
generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, config)
result = generator.generate("def fibonacci(n):")
print(result.code)

# 批量 pipeline（推荐）
pipeline_config = WatermarkPipelineConfig(
    dataset="humaneval",          # "humaneval" 或 "mbpp"
    output_dir="data/watermarked",
    dataset_path="data/datasets",
)
pipeline = WatermarkPipeline(generator=generator, config=pipeline_config)
output_path = pipeline.run()     # 返回 JSONL 文件路径
```

关键 API：
- `WatermarkConfig` — 水印嵌入参数（secret_key, embed_dim, margin 等）
- `WatermarkGenerator.generate(prompt)` → `GenerateResult`
- `WatermarkPipelineConfig` — pipeline 配置（dataset, output_dir, dataset_path）
- `WatermarkPipeline.run()` → JSONL 路径（每行含 id/prompt/generated_code/embed_rate 等字段）

### Phase 3 — `wfcllm.extract`

```python
from wfcllm.extract import ExtractConfig, WatermarkDetector
from wfcllm.extract import ExtractPipeline, ExtractPipelineConfig

# 单条检测（底层 API）
config = ExtractConfig(secret_key="mysecret")
detector = WatermarkDetector(config, encoder, tokenizer)
result = detector.detect(code_str)
print(result.is_watermarked, result.z_score)

# 批量 pipeline（推荐）
pipeline_config = ExtractPipelineConfig(
    input_file="data/watermarked/humaneval_20260309_120000.jsonl",
    output_dir="data/results",
)
pipeline = ExtractPipeline(detector=detector, config=pipeline_config)
report_path = pipeline.run()     # 返回 JSON 报告路径
```

关键 API：
- `ExtractConfig` — 提取参数（secret_key, z_threshold, embed_dim）
- `WatermarkDetector.detect(code)` → `DetectionResult`
- `ExtractPipelineConfig` — pipeline 配置（input_file, output_dir）
- `ExtractPipeline.run()` → JSON 报告路径（含 summary/per_sample 统计字段）

---

## 目录结构

```
WFCLLM/
├── run.py                  # 统一运行入口
├── requirements.txt        # 依赖清单
├── CLAUDE.md               # 开发规范（Git、测试、环境）
├── wfcllm/                 # 主包
│   ├── common/             # 共享工具（AST 解析、代码变换）
│   ├── encoder/            # 阶段一：语义编码器预训练
│   ├── watermark/          # 阶段二：生成时水印嵌入
│   └── extract/            # 阶段三：提取与验证
├── experiment/             # 前期实验代码（仅供算法参考，禁止 import）
├── tests/                  # 测试代码（pytest）
├── docs/                   # 设计文档与实验记录
│   ├── plans/              # 实施计划文档
│   └── experiment/         # 实验记录
└── data/                   # 数据、模型检查点、运行结果
    ├── checkpoints/encoder/
    ├── watermarked/        # 水印 JSONL 数据集（阶段二输出）
    ├── results/            # 检测报告 JSON（阶段三输出）
    └── run_state.json      # 断点状态（自动生成，已 gitignore）
```

---

## 开发规范

- **分支模型：** main（稳定）/ develop（开发主线）/ feature/*（功能分支）
- **Commit 格式：** `<type>: <description>`，type 取值 feat/fix/refactor/test/docs/chore
- **测试：** `conda run -n WFCLLM pytest tests/ -v`
- **环境：** 严禁在 base 环境安装依赖或运行项目

详细规范见 [CLAUDE.md](CLAUDE.md)。
