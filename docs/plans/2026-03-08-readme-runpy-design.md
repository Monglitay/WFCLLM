# README.md 与 run.py 设计文档

日期：2026-03-08

## 背景

项目三个核心阶段（encoder / watermark / extract）已全部实现。需要：
1. 一份面向使用者与协作开发者的 README.md
2. 一个统一入口 run.py，支持全流程运行、单阶段运行和断点续跑

## README.md 设计

**语言：** 中文为主，技术标识（类名、命令）保留英文

**章节结构：**
1. 项目简介 + 核心思路（三阶段流程概述）
2. 系统架构（文字 ASCII 图，三阶段流向）
3. 环境安装（conda WFCLLM + requirements.txt）
4. 快速开始（run.py 典型用法示例）
5. 各模块 API 说明（encoder / watermark / extract 各自公共接口）
6. 目录结构
7. 开发规范（简要，指向 CLAUDE.md 和 docs/）

**定位：** 使用者关注"怎么跑"，不重复方案文档的数学推导

## run.py 设计

### CLI 接口

```bash
python run.py                           # 全流程，自动跳过已完成阶段
python run.py --phase encoder           # 只运行阶段一
python run.py --phase watermark         # 只运行阶段二
python run.py --phase extract           # 只运行阶段三
python run.py --status                  # 查看各阶段完成情况
python run.py --reset                   # 清除断点状态
python run.py --phase encoder --force   # 强制重跑（忽略已完成标记）
```

支持透传参数覆盖默认 config，例如：
```bash
python run.py --phase encoder --epochs 5 --lr 1e-4
```

### 断点状态文件

路径：`data/run_state.json`

```json
{
  "encoder": {
    "done": true,
    "checkpoint": "data/checkpoints/encoder/best.pt",
    "completed_at": "2026-03-08T10:00:00"
  },
  "watermark": {"done": false},
  "extract": {"done": false}
}
```

### 阶段依赖

- `watermark` 依赖 `encoder` 完成（读取 checkpoint 路径）
- `extract` 依赖 `encoder` 完成（共享编码器）
- 全流程顺序：encoder → watermark → extract

### 参数透传规则

- encoder 阶段：透传给 `EncoderConfig`（epochs, lr, batch-size, margin 等）
- watermark 阶段：透传给 `WatermarkConfig`（secret-key, margin-base 等）
- extract 阶段：透传给 `ExtractConfig`（secret-key, z-threshold 等）

## 文件位置

- `README.md`：项目根目录
- `run.py`：项目根目录
- `data/run_state.json`：运行时生成（git ignore）
