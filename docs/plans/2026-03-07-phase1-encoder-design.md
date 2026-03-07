# 阶段一设计：鲁棒语义编码器 E 的预训练

## 核心目标

训练一个"重执行语义、轻表面结构"的特征提取器，使其对代码的等价变换（如混淆攻击）具备强抗漂移能力。

## 技术选型

| 项目 | 选择 | 理由 |
|------|------|------|
| 基座模型 | CodeT5-base (Salesforce/codet5-base) | 方案文档指定，encoder-decoder 架构取 encoder 部分 |
| 训练数据 | MBPP + HumanEval | 覆盖更多代码样本，前期实验已验证 MBPP pipeline |
| 损失函数 | Triplet Margin Loss (cosine) | 与方案文档一致的边界损失 |
| 负样本比例 | 简单:困难 = 1:1 | 平衡区分能力 |
| 微调策略 | LoRA (可选，默认开启) | 8-12GB 显存即可训练，冻结主体只训练 adapter + projection |
| 精度 | BF16 (可选，默认开启) | 进一步降低显存占用和加速训练 |

## 实现方案

采用端到端单体 Pipeline（方案 A）：一次性完成 `common/` + `encoder/` 所有模块，与方案文档对应，结构最清晰。

## 模块设计

### 1. `wfcllm/common/ast_parser.py` — Tree-sitter 封装

统一的 tree-sitter Python 解析接口，供阶段一~三复用。

```python
class PythonParser:
    """单例 tree-sitter Python 解析器"""
    def parse(self, source: str) -> tree_sitter.Tree

def extract_statement_blocks(source: str) -> list[StatementBlock]

@dataclass
class StatementBlock:
    block_id: str
    block_type: Literal["simple", "compound"]
    node_type: str
    source: str
    start_line: int
    end_line: int
    depth: int
    parent_id: str | None
    children_ids: list[str]
```

参照 `experiment/statement_block_split/split.py` 重写，去掉 MBPP 耦合，变为纯 source → blocks 函数。

### 2. `wfcllm/common/transform/` — 变换规则引擎

```
common/transform/
├── __init__.py
├── base.py          # Rule 基类、Match 数据类
├── engine.py        # TransformEngine（正/负模式）
├── positive/        # 39 条正变换规则（6 个文件）
└── negative/        # 36 条负变换规则（8 个文件）
```

参照 `experiment/statement_block_transform/` 重写，保持 Rule 的 detect/apply 接口，去掉 CLI 脚本逻辑。engine 支持 `mode="positive"|"negative"` 和排列组合生成。

### 3. `wfcllm/encoder/dataset.py` — Triplet 数据集构造

构造流程：
1. 加载 MBPP + HumanEval 数据集
2. 提取语句块
3. 对每个语句块构造三元组：
   - **anchor** = 原始代码块
   - **positive** = 随机选一个等价变换变体
   - **negative** = 50% 随机无关联代码块 + 50% 破坏性变异变体
4. 数据划分：训练 80% / 验证 10% / 测试 10%

```python
class TripletCodeDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx) -> dict:
        return {
            "anchor_input_ids": ...,
            "positive_input_ids": ...,
            "negative_input_ids": ...,
        }
```

### 4. `wfcllm/encoder/model.py` — CodeT5 编码器封装

架构：CodeT5 Encoder → (可选 LoRA adapter) → [CLS] pooling → Linear(hidden_size, 128) → L2 normalize → 语义向量 u

```python
class SemanticEncoder(nn.Module):
    def __init__(self, config: EncoderConfig):
        self.encoder = T5EncoderModel.from_pretrained(config.model_name, ...)
        # 可选 LoRA：冻结主体，注入 adapter 到 q, v 层
        if config.use_lora:
            self.encoder = get_peft_model(self.encoder, lora_config)
        # 可选 BF16：encoder 权重以 bfloat16 加载
        self.projection = nn.Linear(hidden_size, config.embed_dim)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        """返回 L2 归一化的 float32 语义向量 (batch_size, embed_dim)"""
```

设计决策：
- 使用 T5EncoderModel（只需 encoder）
- 投影到 128 维（加速余弦计算，保留表达能力）
- L2 归一化使余弦距离 = 点积
- LoRA（默认开启）：只训练 adapter + projection，显存降至 4-6GB
- BF16（默认开启）：encoder 权重以 bfloat16 加载，projection 保持 float32

### 5. `wfcllm/encoder/trainer.py` — 对比学习训练

损失函数：$L = \sum_{i} \max\{0, \delta - \cos(E(x_i), E(x_i^+)) + \cos(E(x_i), E(x_i^-))\}$

```python
class ContrastiveTrainer:
    def train_epoch(self) -> dict
    def validate(self) -> dict
    def train(self) -> None  # 多 epoch + early stopping + checkpoint
```

默认超参数：

| 参数 | 默认值 |
|------|--------|
| learning_rate | 2e-5 |
| batch_size | 32 |
| epochs | 10 |
| margin (δ) | 0.3 |
| embed_dim | 128 |
| max_seq_length | 256 |
| warmup_ratio | 0.1 |
| early_stopping_patience | 3 |
| use_lora | True |
| lora_r | 8 |
| lora_alpha | 16 |
| use_bf16 | True |

Checkpoint 保存到 `data/checkpoints/encoder/`。

### 6. `wfcllm/encoder/evaluate.py` — 编码器质量评估

评估指标：
1. **余弦分离度** — 正样本对 vs 负样本对的平均余弦相似度差值
2. **Recall@K** — 在 K 个候选中找到正样本的比率（K=1,5,10）
3. **投影符号准确率** — 模拟阶段二/三的符号判定

输出：stdout 打印 + JSON 保存到 `data/results/`。

### 7. `wfcllm/encoder/config.py` — 统一配置

```python
@dataclass
class EncoderConfig:
    model_name: str = "Salesforce/codet5-base"
    embed_dim: int = 128
    # LoRA (可选，默认开启)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["q", "v"])
    # 精度 (可选，默认 BF16)
    use_bf16: bool = True
    # 数据
    data_sources: list[str] = field(default_factory=lambda: ["mbpp", "humaneval"])
    max_seq_length: int = 256
    negative_ratio: float = 0.5
    # 训练
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    margin: float = 0.3
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3
    checkpoint_dir: str = "data/checkpoints/encoder"
    results_dir: str = "data/results"
```

### 8. 入口脚本 `wfcllm/encoder/train.py`

```
python -m wfcllm.encoder.train [--config config.yaml]
```

## 最终文件结构

```
wfcllm/
├── common/
│   ├── __init__.py
│   ├── ast_parser.py
│   └── transform/
│       ├── __init__.py
│       ├── base.py
│       ├── engine.py
│       ├── positive/
│       │   ├── __init__.py
│       │   ├── api_calls.py
│       │   ├── syntax_init.py
│       │   ├── control_flow.py
│       │   ├── expression_logic.py
│       │   ├── identifier.py
│       │   └── formatting.py
│       └── negative/
│           ├── __init__.py
│           ├── api_calls.py
│           ├── control_flow.py
│           ├── expression_logic.py
│           ├── identifier.py
│           ├── data_structure.py
│           ├── exception.py
│           ├── system.py
│           └── syntax_init.py
├── encoder/
│   ├── __init__.py
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── trainer.py
│   ├── evaluate.py
│   └── train.py
tests/
├── common/
│   ├── test_ast_parser.py
│   └── transform/
│       ├── test_engine.py
│       ├── test_positive_rules.py
│       └── test_negative_rules.py
├── encoder/
│   ├── test_dataset.py
│   ├── test_model.py
│   └── test_trainer.py
```

## 迁移规范

- experiment/ 代码仅作为算法逻辑参考
- 所有逻辑按 wfcllm/ 架构与编码规范重写
- 严禁 wfcllm/ 中 import experiment 模块
