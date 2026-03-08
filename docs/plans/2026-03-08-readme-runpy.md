# README & run.py Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Create `README.md` (中文项目文档) and `run.py` (统一 CLI 入口，支持单阶段/全流程运行与断点续跑).

**Architecture:** README 面向使用者，描述三阶段流程与 run.py 用法。run.py 通过 `data/run_state.json` 跟踪阶段完成状态，用 argparse 实现 `--phase / --status / --reset / --force`，各阶段 runner 直接调用现有模块的 `main()` 或公共 API。

**Tech Stack:** Python stdlib (argparse, json, pathlib, datetime), 现有 `wfcllm.*` 模块，pytest

---

### Task 1: 写 README.md

**Files:**
- Create: `README.md`

**Step 1: 写 README.md**

内容结构如下，完整写入：

```markdown
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
> 并通过 `--model-name /path/to/local/codet5` 指定本地路径。

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

# 阶段一（自定义参数）
python run.py --phase encoder --epochs 5 --lr 1e-4 --batch-size 16

# 阶段二：生成含水印代码
python run.py --phase watermark \
    --lm-model-path /path/to/codellama \
    --secret-key mysecret \
    --prompt "def fibonacci(n):" \
    --output-file data/results/watermarked.py

# 阶段三：检测代码是否含水印
python run.py --phase extract \
    --code-file data/results/watermarked.py \
    --secret-key mysecret

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

config = WatermarkConfig(secret_key="mysecret")
generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, config)
result = generator.generate("def fibonacci(n):")
print(result.code)
```

关键 API：
- `WatermarkConfig` — 水印嵌入参数
- `WatermarkGenerator.generate(prompt)` → `GenerateResult`

### Phase 3 — `wfcllm.extract`

```python
from wfcllm.extract import ExtractConfig, WatermarkDetector

config = ExtractConfig(secret_key="mysecret")
detector = WatermarkDetector(config, encoder, tokenizer)
result = detector.detect(code_str)
print(result.is_watermarked, result.z_score)
```

关键 API：
- `ExtractConfig` — 提取参数（secret_key, z_threshold）
- `WatermarkDetector.detect(code)` → `DetectionResult`

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
    ├── results/
    └── run_state.json      # 断点状态（自动生成，已 gitignore）
```

---

## 开发规范

- **分支模型：** main（稳定）/ develop（开发主线）/ feature/*（功能分支）
- **Commit 格式：** `<type>: <description>`，type 取值 feat/fix/refactor/test/docs/chore
- **测试：** `conda run -n WFCLLM pytest tests/ -v`
- **环境：** 严禁在 base 环境安装依赖或运行项目

详细规范见 [CLAUDE.md](CLAUDE.md)。
```

**Step 2: 确认文件存在**

```bash
head -5 README.md
```
预期输出：`# WFCLLM：基于语句块语义特征的生成时代码水印`

**Step 3: Commit**

```bash
git add README.md
git commit -m "docs: add project README in Chinese"
```

---

### Task 2: run.py 状态管理与 CLI 骨架（TDD）

**Files:**
- Create: `tests/test_run.py`
- Create: `run.py`

**Step 1: 写失败测试**

```python
# tests/test_run.py
import json
import sys
from pathlib import Path

import pytest


# ── 将项目根目录加入 sys.path（如果需要）
sys.path.insert(0, str(Path(__file__).parent.parent))

from run import RunState, PHASES


class TestRunState:
    def test_phases_order(self):
        assert PHASES == ["encoder", "watermark", "extract"]

    def test_initial_state_all_pending(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        for phase in PHASES:
            assert state.is_done(phase) is False

    def test_mark_done_persists(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder", checkpoint="data/checkpoints/encoder/encoder_epoch5.pt")

        # 重新加载
        state2 = RunState(state_file)
        assert state2.is_done("encoder") is True
        assert state2.get("encoder", "checkpoint") == "data/checkpoints/encoder/encoder_epoch5.pt"

    def test_reset_clears_all(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder")
        state.reset()
        assert state.is_done("encoder") is False

    def test_status_dict(self, tmp_path):
        state_file = tmp_path / "run_state.json"
        state = RunState(state_file)
        state.mark_done("encoder", checkpoint="x.pt")
        status = state.status()
        assert status["encoder"]["done"] is True
        assert status["watermark"]["done"] is False
        assert status["extract"]["done"] is False
```

**Step 2: 运行测试，验证失败**

```bash
conda run -n WFCLLM pytest tests/test_run.py -v
```
预期：`ImportError: cannot import name 'RunState' from 'run'`（run.py 不存在）

**Step 3: 实现 RunState 和 CLI 骨架**

```python
# run.py
"""统一运行入口：支持全流程、单阶段运行与断点续跑。

用法：
    python run.py                          # 全流程
    python run.py --phase encoder          # 只跑阶段一
    python run.py --status                 # 查看状态
    python run.py --reset                  # 清除断点
    python run.py --phase encoder --force  # 强制重跑
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PHASES = ["encoder", "watermark", "extract"]
DEFAULT_STATE_FILE = Path("data/run_state.json")


class RunState:
    """断点状态管理：读写 data/run_state.json。"""

    def __init__(self, path: Path = DEFAULT_STATE_FILE):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path, encoding="utf-8") as f:
                return json.load(f)
        return {phase: {"done": False} for phase in PHASES}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)

    def is_done(self, phase: str) -> bool:
        return self._data.get(phase, {}).get("done", False)

    def get(self, phase: str, key: str) -> str | None:
        return self._data.get(phase, {}).get(key)

    def mark_done(self, phase: str, **kwargs) -> None:
        self._data[phase] = {
            "done": True,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            **kwargs,
        }
        self._save()

    def reset(self) -> None:
        self._data = {phase: {"done": False} for phase in PHASES}
        self._save()

    def status(self) -> dict:
        return {
            phase: {
                "done": self._data.get(phase, {}).get("done", False),
                **{k: v for k, v in self._data.get(phase, {}).items() if k != "done"},
            }
            for phase in PHASES
        }
```

**Step 4: 运行测试，验证通过**

```bash
conda run -n WFCLLM pytest tests/test_run.py -v
```
预期：所有测试 PASS

**Step 5: Commit**

```bash
git add tests/test_run.py run.py
git commit -m "feat(run): add RunState state management with tests"
```

---

### Task 3: run.py — argparse CLI 与 --status / --reset

**Files:**
- Modify: `run.py`（追加 CLI 部分）
- Modify: `tests/test_run.py`（追加 CLI 测试）

**Step 1: 追加 CLI 测试**

```python
# 追加到 tests/test_run.py
import subprocess


class TestCLI:
    def test_status_exits_zero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", "run.py", "--status"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "encoder" in result.stdout

    def test_reset_exits_zero(self, tmp_path, monkeypatch):
        # 使用临时状态文件以免影响真实 data/
        monkeypatch.chdir(tmp_path)
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python",
             str(Path(__file__).parent.parent / "run.py"), "--reset"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "重置" in result.stdout or "reset" in result.stdout.lower()

    def test_unknown_phase_exits_nonzero(self):
        result = subprocess.run(
            ["conda", "run", "-n", "WFCLLM", "python", "run.py", "--phase", "invalid"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0
```

**Step 2: 运行测试，验证失败**

```bash
conda run -n WFCLLM pytest tests/test_run.py::TestCLI -v
```
预期：`FAIL`（run.py 没有 main() 和 argparse）

**Step 3: 在 run.py 末尾追加 CLI**

```python
# run.py 末尾追加

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="WFCLLM 统一运行入口",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase",
        choices=PHASES,
        help="运行指定阶段（不指定则运行全流程）",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="查看各阶段完成情况",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="清除断点状态，重头开始",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="强制重跑指定阶段（忽略已完成标记）",
    )
    # Encoder 参数
    parser.add_argument("--model-name", default=None, help="CodeT5 模型名称或本地路径")
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--margin", type=float, default=None)
    parser.add_argument("--no-lora", action="store_true", help="禁用 LoRA")
    parser.add_argument("--no-bf16", action="store_true", help="禁用 BF16")
    # Watermark 参数
    parser.add_argument("--secret-key", default=None, help="水印密钥")
    parser.add_argument("--lm-model-path", default=None, help="代码生成 LLM 路径")
    parser.add_argument("--prompt", default=None, help="生成代码的输入提示")
    parser.add_argument("--output-file", default=None, help="保存生成代码的路径")
    # Extract 参数
    parser.add_argument("--code-file", default=None, help="待检测代码文件路径")
    parser.add_argument("--z-threshold", type=float, default=None, help="Z 分数阈值")
    return parser


def cmd_status(state: RunState) -> None:
    print("=== WFCLLM 阶段状态 ===")
    for phase in PHASES:
        info = state.status()[phase]
        done_str = "✓ 完成" if info["done"] else "○ 未完成"
        extras = {k: v for k, v in info.items() if k not in ("done", "completed_at")}
        extra_str = "  " + str(extras) if extras else ""
        print(f"  {phase:10s} {done_str}{extra_str}")


def cmd_reset(state: RunState) -> None:
    state.reset()
    print("已重置所有阶段状态。")


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    state = RunState()

    if args.status:
        cmd_status(state)
        return 0

    if args.reset:
        cmd_reset(state)
        return 0

    phases_to_run = [args.phase] if args.phase else PHASES

    for phase in phases_to_run:
        if state.is_done(phase) and not args.force:
            print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
            continue
        rc = run_phase(phase, args, state)
        if rc != 0:
            print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
            return rc

    return 0


def run_phase(phase: str, args: argparse.Namespace, state: RunState) -> int:
    """分发到各阶段 runner，返回退出码。"""
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
    }
    return runners[phase](args, state)


def run_encoder(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段一实现在 Task 4。"""
    print("[encoder] 未实现")
    return 1


def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段二实现在 Task 5。"""
    print("[watermark] 未实现")
    return 1


def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """占位：阶段三实现在 Task 6。"""
    print("[extract] 未实现")
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

**Step 4: 运行 CLI 测试**

```bash
conda run -n WFCLLM pytest tests/test_run.py -v
```
预期：所有测试 PASS（run_phase 占位符不影响 status/reset 测试）

**Step 5: Commit**

```bash
git add run.py tests/test_run.py
git commit -m "feat(run): add CLI argparse with --status, --reset, --force"
```

---

### Task 4: run.py — 阶段一 Encoder Runner

**Files:**
- Modify: `run.py`（实现 `run_encoder`）

**Step 1: 理解 encoder train.py 的接口**

`wfcllm/encoder/train.py` 已有 `main(config: EncoderConfig)` 函数，训练完成后
checkpoint 保存在 `config.checkpoint_dir/encoder_epoch{N}.pt`。

**Step 2: 实现 run_encoder**

将 `run.py` 中的 `run_encoder` 占位替换为：

```python
def run_encoder(args: argparse.Namespace, state: RunState) -> int:
    """阶段一：训练语义编码器。"""
    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main
    from pathlib import Path
    import glob

    print("=== 阶段一：语义编码器预训练 ===")

    config = EncoderConfig()
    if args.model_name:
        config.model_name = args.model_name
    if args.embed_dim:
        config.embed_dim = args.embed_dim
    if args.lr:
        config.lr = args.lr
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.epochs = args.epochs
    if args.margin:
        config.margin = args.margin
    if args.no_lora:
        config.use_lora = False
    if args.no_bf16:
        config.use_bf16 = False

    try:
        encoder_main(config)
    except Exception as e:
        print(f"[错误] 编码器训练失败：{e}", file=sys.stderr)
        return 1

    # 找到最新的 checkpoint 文件
    ckpt_pattern = str(Path(config.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else config.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path)
    print(f"[完成] 编码器训练完毕，checkpoint: {checkpoint_path}")
    return 0
```

**Step 3: 手动验证（无完整 GPU 环境时可 mock 测试）**

```bash
# 检查 import 路径正确，干运行（会因无 GPU 或数据而很快出错，但不应 ImportError）
conda run -n WFCLLM python -c "
from run import run_encoder, RunState
import argparse
print('import OK')
"
```
预期：`import OK`

**Step 4: Commit**

```bash
git add run.py
git commit -m "feat(run): implement encoder phase runner"
```

---

### Task 5: run.py — 阶段二 Watermark Runner

**Files:**
- Modify: `run.py`（实现 `run_watermark`）

**Step 1: 理解 WatermarkGenerator 接口**

`WatermarkGenerator.__init__` 需要：
- `model`：代码生成 LLM（transformers AutoModelForCausalLM）
- `tokenizer`：LLM 的 tokenizer
- `encoder`：`SemanticEncoder` 实例
- `encoder_tokenizer`：encoder 的 tokenizer
- `config`：`WatermarkConfig`

`generator.generate(prompt)` → `GenerateResult.code`

**Step 2: 实现 run_watermark**

将 `run.py` 中的 `run_watermark` 占位替换为：

```python
def run_watermark(args: argparse.Namespace, state: RunState) -> int:
    """阶段二：生成含水印代码。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.watermark.config import WatermarkConfig
    from wfcllm.watermark.generator import WatermarkGenerator

    print("=== 阶段二：生成时水印嵌入 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not args.lm_model_path:
        print("[错误] --lm-model-path 为必填参数", file=sys.stderr)
        return 1
    prompt = args.prompt or "def solution():"

    encoder_checkpoint = state.get("encoder", "checkpoint")
    embed_dim = args.embed_dim or 128

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    # 加载代码生成 LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_tokenizer = AutoTokenizer.from_pretrained(args.lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(args.lm_model_path).to(device)

    config = WatermarkConfig(
        secret_key=args.secret_key,
        encoder_embed_dim=embed_dim,
        encoder_device=device,
    )
    generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, config)

    try:
        result = generator.generate(prompt)
    except Exception as e:
        print(f"[错误] 水印生成失败：{e}", file=sys.stderr)
        return 1

    # 保存结果
    output_file = args.output_file or "data/results/watermarked.py"
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    Path(output_file).write_text(result.code, encoding="utf-8")

    state.mark_done(
        "watermark",
        output_file=output_file,
        embedded_blocks=result.embedded_blocks,
        total_blocks=result.total_blocks,
    )
    print(f"[完成] 生成代码已保存至 {output_file}")
    print(f"       嵌入块: {result.embedded_blocks}/{result.total_blocks}")
    return 0
```

**Step 3: Import 验证**

```bash
conda run -n WFCLLM python -c "
from run import run_watermark
print('import OK')
"
```
预期：`import OK`

**Step 4: Commit**

```bash
git add run.py
git commit -m "feat(run): implement watermark phase runner"
```

---

### Task 6: run.py — 阶段三 Extract Runner

**Files:**
- Modify: `run.py`（实现 `run_extract`）

**Step 1: 理解 WatermarkDetector 接口**

`WatermarkDetector.__init__` 需要：
- `config`：`ExtractConfig`（secret_key, z_threshold）
- `encoder`：`SemanticEncoder` 实例
- `tokenizer`：encoder tokenizer
- `device`

`detector.detect(code)` → `DetectionResult`

**Step 2: 实现 run_extract**

将 `run.py` 中的 `run_extract` 占位替换为：

```python
def run_extract(args: argparse.Namespace, state: RunState) -> int:
    """阶段三：检测代码水印。"""
    import torch
    from transformers import AutoTokenizer
    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.extract.config import ExtractConfig
    from wfcllm.extract.detector import WatermarkDetector

    print("=== 阶段三：水印提取与验证 ===")

    # 前置检查
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1
    if not args.secret_key:
        print("[错误] --secret-key 为必填参数", file=sys.stderr)
        return 1
    if not args.code_file:
        print("[错误] --code-file 为必填参数", file=sys.stderr)
        return 1
    code_path = Path(args.code_file)
    if not code_path.exists():
        print(f"[错误] 文件不存在：{args.code_file}", file=sys.stderr)
        return 1

    encoder_checkpoint = state.get("encoder", "checkpoint")
    embed_dim = args.embed_dim or 128
    z_threshold = args.z_threshold or 3.0

    # 加载编码器
    enc_config = EncoderConfig(embed_dim=embed_dim)
    encoder = SemanticEncoder(config=enc_config)
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    config = ExtractConfig(secret_key=args.secret_key, embed_dim=embed_dim, z_threshold=z_threshold)
    detector = WatermarkDetector(config, encoder, tokenizer, device=device)

    code = code_path.read_text(encoding="utf-8")
    try:
        result = detector.detect(code)
    except Exception as e:
        print(f"[错误] 检测失败：{e}", file=sys.stderr)
        return 1

    verdict = "【含水印】" if result.is_watermarked else "【无水印】"
    print(f"\n{verdict}")
    print(f"  Z 分数:   {result.z_score:.4f}（阈值 {z_threshold}）")
    print(f"  p 值:     {result.p_value:.6f}")
    print(f"  独立块:   {result.independent_blocks}/{result.total_blocks}")
    print(f"  命中块:   {result.hit_blocks}")

    state.mark_done(
        "extract",
        code_file=args.code_file,
        is_watermarked=result.is_watermarked,
        z_score=result.z_score,
    )
    return 0
```

**Step 3: Import 验证**

```bash
conda run -n WFCLLM python -c "
from run import run_extract
print('import OK')
"
```
预期：`import OK`

**Step 4: Commit**

```bash
git add run.py
git commit -m "feat(run): implement extract phase runner"
```

---

### Task 7: 收尾 — .gitignore + 全量测试

**Files:**
- Modify: `.gitignore`（追加 data/run_state.json）
- 无新文件

**Step 1: 检查 .gitignore 是否已存在**

```bash
cat .gitignore 2>/dev/null || echo "不存在"
```

**Step 2: 追加 run_state.json**

如果 `.gitignore` 不存在：
```bash
echo "data/run_state.json" > .gitignore
```

如果已存在，追加一行：
```
data/run_state.json
```

**Step 3: 运行全量测试**

```bash
conda run -n WFCLLM pytest tests/test_run.py -v
```
预期：所有测试 PASS

**Step 4: 验证 run.py --status 正常**

```bash
conda run -n WFCLLM python run.py --status
```
预期输出类似：
```
=== WFCLLM 阶段状态 ===
  encoder    ○ 未完成
  watermark  ○ 未完成
  extract    ○ 未完成
```

**Step 5: 验证 --help 输出完整**

```bash
conda run -n WFCLLM python run.py --help
```
预期：显示所有参数说明，无错误

**Step 6: Commit**

```bash
git add .gitignore run.py
git commit -m "chore: add run_state.json to gitignore and finalize run.py"
```
