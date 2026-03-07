# 阶段零设计：工程初始化与规范确立

## 概述

搭建标准化项目框架，确立代码与工程规范。不涉及代码迁移，仅建骨架和写规范。

## 方案选择

**方案 A：先规范后架构** — 先写 CLAUDE.md 确立规范，再按规范建目录结构，最后处理依赖和 Git。

## 交付物

### 1. CLAUDE.md 规范文档

放在项目根目录，包含以下板块：

- **项目概述：** 一句话说明 + 指向方案文档
- **项目架构：** `wfcllm/` 按阶段分包（encoder/watermark/extract/common），`experiment/` 仅供参考
- **Git 规范：** Git Flow 简化版（main → develop → feature/\* | experiment/\*），Commit 格式 `<type>: <description>`
- **测试规范：** pytest 框架，`tests/test_<module>.py` 命名，公共函数必须有测试
- **代码迁移规范：** experiment/ 仅参考，严禁跨目录 import，所有逻辑按新架构重写
- **环境管理：** Conda 环境 `WFCLLM`，`requirements.txt` 核心依赖锁版本，CUDA/PyTorch 不锁

### 2. 目录结构骨架

```
wfcllm/
├── __init__.py          # __version__ = "0.1.0"
├── encoder/
│   └── __init__.py
├── watermark/
│   └── __init__.py
├── extract/
│   └── __init__.py
└── common/
    └── __init__.py
```

子包内部不在阶段零创建，留到各阶段按需建立。

### 3. requirements.txt

从 Conda 环境 `WFCLLM` 导出实际版本，核心业务依赖精确锁定，CUDA/PyTorch 不锁。

### 4. Git 分支整理

1. master 合并到 main
2. 从 main 创建 develop 分支
3. 后续开发在 feature/* 分支进行，PR 合入 develop
4. master 确认合入后可删除或归档

## 不包含

- 代码迁移（留到后续阶段）
- 模型路径解耦（延后处理）
- CI/CD、pre-commit hooks 等自动化工具链
