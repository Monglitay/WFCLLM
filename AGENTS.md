# AGENTS.md

适用于仓库根目录下的全部文件。先遵循 `CLAUDE.md`，再遵循本文件。

## 当前范围

- 只维护 Gated Lineage，方法名固定为 `gated_semantic_window_v1`。
- 只维护五个 Full Reproduction Profile：
  Python/HumanEval、Python/MBPP、C++/HumanEvalPack、
  Java/HumanEvalPack、JavaScript/HumanEvalPack。
- 完整链固定为
  `encoder → gate-data → gate-train → generate → calibrate → detect → report → audit`。
- 执行必须从新的 run root 和状态文件开始。不要加入旧状态迁移、外部历史
  Bundle、旧 candidate/report/schema reader 或恢复入口。
- 历史内容只从 Git 历史与既有 Archival Tag 恢复；不要在当前树创建
  `archive/`、`legacy/` 或历史副本。

## Public Execution Surface

- `run.py` / `wfcllm.cli.entry`
- `scripts/experiments/run_gated_experiment.sh`
- 五个 `scripts/experiments/run_*_full.sh`
- `scripts/wfcllm_prepare_gated_experiment.py`
- `scripts/evaluate.py`
- `scripts/wfcllm_metric_contract.py`

缺少本地模型、数据集、source catalog、negative input 或密钥必须失败。不得
增加 canonical-solution、简化 detector 或测试后端替代路径。测试依赖只能由
测试注入，产物必须带完整非正式标记。

## 当前模块

- `wfcllm.method`：唯一 preset、配置和 artifact 路径。
- `wfcllm.windowing` / `wfcllm.lang`：Python、C++、Java、JavaScript
  statement-window 合同。
- `wfcllm.encoder`：每数据集 semantic projection 训练。
- `wfcllm.gate`：source、key bank、Gate 数据、训练和当前 Bundle。
- `wfcllm.generation`：Gate 生成、改写、finalizer 和输出。
- `wfcllm.detection`：严格 code-only 校准与检测。
- `wfcllm.audit`：detector input、no-quality-gate 和 artifact 完整性。
- `wfcllm.evaluation`：生成后 Pass@1/Pass@k 与当前 Metric Contract；
  Metric Contract 只消费 Pass@1。
- `wfcllm.cli` / `wfcllm.orchestration`：当前八阶段入口和 fresh-run 状态。

## 数据与秘密

- 正式 detector positive input 只能是 `inputs/final_code.jsonl`。
- 每行必须且只能有 `id`、`dataset`、`prompt`、`final_code`。
- generation sidecar、audit、Gate 数据和 correctness 结果不得进入 detector。
- 不得把 Deployment Key、Training Key Bank、Holdout Key Bank 或原始秘密写入
  配置、状态、日志、公开 JSON/JSONL 或文档。
- 模型、数据集、keys、source catalog、Gate cache、checkpoint 和 run tree
  都是 Local Reproduction Resource；不得提交。

## 实现风格

- 在现有模块边界内做最小直接改动。
- 库模块使用 `from __future__ import annotations`。
- 路径用 `pathlib.Path`；文本读写显式指定 UTF-8。
- 使用现代类型标注和 dataclass。
- 无效配置、数据或 artifact 通常抛 `ValueError`。
- 模块 logger 使用 `logging.getLogger(__name__)`。
- 不从历史目录或退出范围的模块导入生产代码。

## 验证

使用 `WFCLLM` conda 环境并保持 Hugging Face 离线：

```bash
conda run -n WFCLLM python run.py --status --state-file /tmp/wfcllm-state.json
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest --collect-only -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest tests/ -q
bash -n scripts/experiments/run_gated_experiment.sh \
  scripts/experiments/run_*_full.sh
```

资源限制导致失败时，报告首个失败命令、首条错误和受影响文件；不要把 fixture
测试称为真实 GPU 复现。

## Git

- 编辑前检查 `git status --short`，保留无关和用户已有改动。
- 不提交 `data/` 内容、模型、数据集、keys、checkpoint、run、日志或论文。
- 未经明确要求，不暂存、提交、推送、建分支或创建 PR。
- 不改写分支、worktree、tag 或 Git 历史。
