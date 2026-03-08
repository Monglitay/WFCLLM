# WFCLLM: 基于语句块语义特征的生成时代码水印

> 方案文档：`docs/基于语句块的语义特征的生成时代码水印方案.md`

## 项目架构

```
wfcllm/              # 主包
├── encoder/          # 阶段一：鲁棒语义编码器预训练
├── watermark/        # 阶段二：生成时水印嵌入
├── extract/          # 阶段三：提取与验证
└── common/           # 共享工具（AST 解析、配置等）

experiment/           # 前期实验代码（仅供算法逻辑参考）
tests/                # 测试代码
docs/                 # 文档与设计
data/                 # 数据
```

## Git 规范

**分支模型（Git Flow 简化版）：**
- `main`：稳定版本，里程碑合并
- `develop`：开发主线，日常合入
- `feature/*`：功能分支，从 develop 创建，PR 合入 develop
- `experiment/*`：实验分支，从 develop 创建，PR 合入 develop

**Commit 格式：** `<type>: <description>`

type 取值：feat / fix / refactor / test / docs / chore

## 测试规范

- 框架：pytest
- 目录：`tests/`，子包测试放 `tests/<subpackage>/`
- 命名：`test_<module>.py`，测试函数 `test_<行为描述>()`
- 要求：每个公共函数/类必须有对应测试
- 运行：`HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v`
- **必须加 `HF_HUB_OFFLINE=1`**：测试使用本地离线模型/数据集，禁止访问 HF Hub
- 本地模型路径：`data/models/codet5-base/`；本地数据集路径：`data/datasets/`
- 涉及 `SemanticEncoder` 的测试须在 `EncoderConfig` 中传入 `model_name="data/models/codet5-base"`

## 代码迁移规范

- `experiment/` 下的代码仅作为算法逻辑参考
- **严禁** `wfcllm/` 中 import `experiment` 下的任何模块
- 所有逻辑必须按 `wfcllm/` 的架构与编码规范重写

## 环境管理

- Conda 环境名：`WFCLLM`
- 依赖清单：`requirements.txt`
- 核心业务依赖（tree-sitter 等）精确锁定版本号
- CUDA / PyTorch 等硬件相关依赖不锁版本，视目标服务器适配
- **严禁** 在 `base` 环境中安装业务依赖或运行项目代码
