# Config System Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 `run.py` 引入 JSON 配置文件支持，建立 `configs/` 目录，`configs/base_config.json` 为默认配置，CLI 参数优先级高于 JSON。

**Architecture:** 新增 `load_config()` 函数读取 JSON，`run_phase()` 及各 `run_*()` 函数接收 config dict，先用 JSON 值初始化各阶段 dataclass，再用 CLI 参数覆盖。`main()` 负责加载配置并传递给下游。

**Tech Stack:** Python 标准库（argparse、json、pathlib），pytest，已有的 EncoderConfig / WatermarkConfig / ExtractConfig dataclass。

---

### Task 1: 创建 `configs/base_config.json`

**Files:**
- Create: `configs/base_config.json`

**Step 1: 创建目录和文件**

```bash
mkdir -p configs
```

在 `configs/base_config.json` 写入以下内容（JSON 不支持科学记数法，lr 用小数）：

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
    "lr": 0.00002,
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

**Step 2: 验证 JSON 合法**

```bash
python -c "import json; json.load(open('configs/base_config.json')); print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add configs/base_config.json
git commit -m "chore: add configs/ directory with base_config.json"
```

---

### Task 2: 新增 `load_config()` 函数并测试

**Files:**
- Modify: `run.py`（顶部新增函数，不改动其他逻辑）
- Test: `tests/test_run_config.py`

**Step 1: 写失败测试**

新建 `tests/test_run_config.py`：

```python
"""Tests for run.py config loading."""
import json
import sys
from pathlib import Path

import pytest

# run.py 在项目根目录，需加入 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))
from run import load_config


def test_load_config_returns_dict(tmp_path):
    cfg = {"encoder": {"lr": 0.001}, "watermark": {}, "extract": {}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result["encoder"]["lr"] == 0.001


def test_load_config_missing_phase_ok(tmp_path):
    """缺少某阶段的 key，返回空 dict 而不报错。"""
    cfg = {"encoder": {"lr": 0.001}}
    f = tmp_path / "cfg.json"
    f.write_text(json.dumps(cfg))
    result = load_config(f)
    assert result.get("watermark", {}) == {}


def test_load_config_file_not_found(tmp_path):
    with pytest.raises(SystemExit):
        load_config(tmp_path / "nonexistent.json")


def test_load_config_invalid_json(tmp_path):
    f = tmp_path / "bad.json"
    f.write_text("{ not valid json }")
    with pytest.raises(SystemExit):
        load_config(f)
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: FAIL（`load_config` 尚未定义）

**Step 3: 在 `run.py` 中实现 `load_config()`**

在 `run.py` 的 `PHASES = ...` 定义之后、`class RunState` 之前插入：

```python
DEFAULT_CONFIG_FILE = Path("configs/base_config.json")


def load_config(config_path: Path) -> dict:
    """读取 JSON 配置文件，返回按阶段分组的 dict。"""
    if not config_path.exists():
        print(f"[错误] 配置文件不存在：{config_path}", file=sys.stderr)
        sys.exit(1)
    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"[错误] 配置文件解析失败：{e}", file=sys.stderr)
        sys.exit(1)
```

**Step 4: 运行测试确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 4 PASSED

**Step 5: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: add load_config() with file-not-found and invalid-json handling"
```

---

### Task 3: `build_parser()` 新增 `--config` 参数

**Files:**
- Modify: `run.py`（`build_parser()` 函数）
- Test: `tests/test_run_config.py`（追加用例）

**Step 1: 写失败测试**

在 `tests/test_run_config.py` 追加：

```python
from run import build_parser


def test_parser_default_config():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.config == Path("configs/base_config.json")


def test_parser_custom_config():
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/my.json"])
    assert args.config == Path("configs/my.json")
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::test_parser_default_config tests/test_run_config.py::test_parser_custom_config -v
```

Expected: FAIL（`args` 没有 `config` 属性）

**Step 3: 在 `build_parser()` 中新增参数**

在 `build_parser()` 函数内，`parser.add_argument("--phase", ...)` 之前插入：

```python
parser.add_argument(
    "--config",
    type=Path,
    default=DEFAULT_CONFIG_FILE,
    help=f"配置文件路径（默认: {DEFAULT_CONFIG_FILE}）",
)
```

**Step 4: 运行测试确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 6 PASSED

**Step 5: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: add --config CLI argument with default base_config.json"
```

---

### Task 4: 重构 `run_encoder()` 使用 config dict

**Files:**
- Modify: `run.py`（`run_phase()`、`run_encoder()` 签名与实现）
- Test: `tests/test_run_config.py`（追加用例）

**Step 1: 写失败测试**

在 `tests/test_run_config.py` 追加：

```python
from unittest.mock import patch, MagicMock
from run import run_encoder


def test_run_encoder_uses_json_lr(tmp_path):
    """JSON 中 lr 覆盖 EncoderConfig 默认值，CLI 未传 lr。"""
    args = build_parser().parse_args([])
    args.eval_only = False
    args.model_name = None
    args.embed_dim = None
    args.lr = None
    args.batch_size = None
    args.epochs = None
    args.margin = None
    args.no_lora = False
    args.no_bf16 = False

    config = {"encoder": {"lr": 0.123, "model_name": "Salesforce/codet5-base",
                          "embed_dim": 128, "use_lora": True, "lora_r": 16,
                          "lora_alpha": 32, "lora_dropout": 0.1,
                          "lora_target_modules": ["q", "v"], "use_bf16": True,
                          "data_sources": ["mbpp", "humaneval"],
                          "max_seq_length": 256, "negative_ratio": 0.5,
                          "batch_size": 64, "epochs": 10, "margin": 0.3,
                          "warmup_ratio": 0.1, "early_stopping_patience": 3,
                          "num_workers": 8, "pin_memory": True,
                          "checkpoint_dir": str(tmp_path / "ckpt"),
                          "results_dir": str(tmp_path / "results"),
                          "local_model_dir": str(tmp_path / "models"),
                          "local_dataset_dir": str(tmp_path / "datasets")}}

    captured = {}

    def fake_encoder_main(cfg):
        captured["lr"] = cfg.lr
        # 模拟没有 checkpoint 文件
        pass

    state = MagicMock()
    state.get.return_value = None

    with patch("run.encoder_main", fake_encoder_main), \
         patch("glob.glob", return_value=[]):
        run_encoder(args, config, state)

    assert captured["lr"] == pytest.approx(0.123)


def test_run_encoder_cli_overrides_json_lr(tmp_path):
    """CLI --lr 优先于 JSON lr。"""
    args = build_parser().parse_args(["--lr", "0.999"])
    args.eval_only = False
    args.model_name = None
    args.embed_dim = None
    args.batch_size = None
    args.epochs = None
    args.margin = None
    args.no_lora = False
    args.no_bf16 = False

    config = {"encoder": {"lr": 0.123, "model_name": "Salesforce/codet5-base",
                          "embed_dim": 128, "use_lora": True, "lora_r": 16,
                          "lora_alpha": 32, "lora_dropout": 0.1,
                          "lora_target_modules": ["q", "v"], "use_bf16": True,
                          "data_sources": ["mbpp", "humaneval"],
                          "max_seq_length": 256, "negative_ratio": 0.5,
                          "batch_size": 64, "epochs": 10, "margin": 0.3,
                          "warmup_ratio": 0.1, "early_stopping_patience": 3,
                          "num_workers": 8, "pin_memory": True,
                          "checkpoint_dir": str(tmp_path / "ckpt"),
                          "results_dir": str(tmp_path / "results"),
                          "local_model_dir": str(tmp_path / "models"),
                          "local_dataset_dir": str(tmp_path / "datasets")}}

    captured = {}

    def fake_encoder_main(cfg):
        captured["lr"] = cfg.lr

    state = MagicMock()
    state.get.return_value = None

    with patch("run.encoder_main", fake_encoder_main), \
         patch("glob.glob", return_value=[]):
        run_encoder(args, config, state)

    assert captured["lr"] == pytest.approx(0.999)
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::test_run_encoder_uses_json_lr tests/test_run_config.py::test_run_encoder_cli_overrides_json_lr -v
```

Expected: FAIL（`run_encoder` 签名不接受 config 参数）

**Step 3: 重构 `run_encoder()`**

将 `run_encoder(args, state)` 改为 `run_encoder(args, config, state)`，并将内部 `EncoderConfig` 初始化改为先从 JSON dict 构建，再用 CLI 覆盖：

```python
def run_encoder(args: argparse.Namespace, config: dict, state: RunState) -> int:
    """阶段一：训练语义编码器。"""
    import glob

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.train import main as encoder_main

    print("=== 阶段一：语义编码器预训练 ===")

    enc_cfg = config.get("encoder", {})

    # eval-only 分支
    if args.eval_only:
        from wfcllm.encoder.train import evaluate_only

        checkpoint = args.checkpoint or state.get("encoder", "checkpoint")
        if not checkpoint:
            print("[错误] 未找到 checkpoint，请用 --checkpoint 指定路径", file=sys.stderr)
            return 1
        if not Path(checkpoint).exists():
            print(f"[错误] checkpoint 不存在：{checkpoint}", file=sys.stderr)
            return 1

        cfg = EncoderConfig(**enc_cfg) if enc_cfg else EncoderConfig()
        if args.model_name:
            cfg.model_name = args.model_name
        if args.embed_dim:
            cfg.embed_dim = args.embed_dim
        if args.no_lora:
            cfg.use_lora = False
        if args.no_bf16:
            cfg.use_bf16 = False

        try:
            evaluate_only(checkpoint, cfg)
        except Exception as e:
            print(f"[错误] 评测失败：{e}", file=sys.stderr)
            return 1
        return 0

    cfg = EncoderConfig(**enc_cfg) if enc_cfg else EncoderConfig()
    # CLI 覆盖
    if args.model_name:
        cfg.model_name = args.model_name
    if args.embed_dim:
        cfg.embed_dim = args.embed_dim
    if args.lr:
        cfg.lr = args.lr
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.epochs:
        cfg.epochs = args.epochs
    if args.margin:
        cfg.margin = args.margin
    if args.no_lora:
        cfg.use_lora = False
    if args.no_bf16:
        cfg.use_bf16 = False

    # 若 model_name 仍是 HF Hub ID，尝试自动检测本地模型
    local_codet5 = Path(cfg.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        cfg.model_name = str(local_codet5)
        print(f"[自动] 使用本地模型: {cfg.model_name}")
    else:
        print(f"[回退] 使用 HF Hub 模型: {cfg.model_name}")

    try:
        encoder_main(cfg)
    except Exception as e:
        print(f"[错误] 编码器训练失败：{e}", file=sys.stderr)
        return 1

    ckpt_pattern = str(Path(cfg.checkpoint_dir) / "encoder_epoch*.pt")
    checkpoints = sorted(glob.glob(ckpt_pattern))
    checkpoint_path = checkpoints[-1] if checkpoints else cfg.checkpoint_dir

    state.mark_done("encoder", checkpoint=checkpoint_path)
    print(f"[完成] 编码器训练完毕，checkpoint: {checkpoint_path}")
    return 0
```

注意：原来的 `from wfcllm.encoder.train import main as encoder_main` 需改为模块级别可 patch 的形式。在函数内保持 import，但 patch 路径用 `run.encoder_main`。实际测试中 patch `wfcllm.encoder.train.main` 或直接在函数内 import 后 patch `run.encoder_main`，以 patch 路径能匹配为准。

> **注意**：`EncoderConfig(**enc_cfg)` 要求 `enc_cfg` 中的 key 必须与 dataclass 字段完全匹配。`base_config.json` 的 encoder 部分已与 `EncoderConfig` 字段一一对应，因此直接解包即可。

**Step 4: 运行测试确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 所有测试 PASSED

**Step 5: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: run_encoder accepts config dict, JSON values as base with CLI override"
```

---

### Task 5: 重构 `run_watermark()` 使用 config dict

**Files:**
- Modify: `run.py`（`run_watermark()` 签名与实现）
- Test: `tests/test_run_config.py`（追加用例）

**Step 1: 写失败测试**

在 `tests/test_run_config.py` 追加：

```python
from run import run_watermark


def test_run_watermark_missing_secret_key_exits():
    """secret_key 为空时应返回错误码 1。"""
    args = build_parser().parse_args([])
    args.secret_key = None
    args.lm_model_path = None
    args.prompt = None
    args.output_file = None
    args.embed_dim = None

    config = {"watermark": {"secret_key": "", "lm_model_path": "/some/path"}}
    state = MagicMock()

    rc = run_watermark(args, config, state)
    assert rc == 1


def test_run_watermark_json_prompt_used(tmp_path):
    """CLI 未传 --prompt，使用 JSON 中的 prompt 值。"""
    args = build_parser().parse_args([])
    args.secret_key = "testkey"
    args.lm_model_path = "/fake/lm"
    args.prompt = None
    args.output_file = str(tmp_path / "out.py")
    args.embed_dim = None

    wm_cfg = {
        "secret_key": "",
        "lm_model_path": "/fake/lm",
        "prompt": "def hello():",
        "output_file": str(tmp_path / "out.py"),
        "encoder_model_path": "data/models/codet5-base",
        "encoder_embed_dim": 128,
        "encoder_device": "cpu",
        "margin_base": 0.1,
        "margin_alpha": 0.05,
        "max_retries": 5,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "max_new_tokens": 512,
        "eos_token_id": None,
        "enable_fallback": True,
    }
    config = {"watermark": wm_cfg}

    captured = {}

    fake_result = MagicMock()
    fake_result.code = "def hello(): pass"
    fake_result.embedded_blocks = 1
    fake_result.total_blocks = 1

    def fake_generate(prompt):
        captured["prompt"] = prompt
        return fake_result

    fake_generator = MagicMock()
    fake_generator.generate = fake_generate

    state = MagicMock()
    state.is_done.return_value = True
    state.get.return_value = None

    with patch("run.WatermarkGenerator", return_value=fake_generator), \
         patch("run.SemanticEncoder"), \
         patch("run.AutoTokenizer"), \
         patch("run.AutoModelForCausalLM"), \
         patch("run.torch"):
        run_watermark(args, config, state)

    assert captured["prompt"] == "def hello():"
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::test_run_watermark_missing_secret_key_exits tests/test_run_config.py::test_run_watermark_json_prompt_used -v
```

Expected: FAIL

**Step 3: 重构 `run_watermark()`**

```python
def run_watermark(args: argparse.Namespace, config: dict, state: RunState) -> int:
    """阶段二：生成含水印代码。"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.watermark.config import WatermarkConfig
    from wfcllm.watermark.generator import WatermarkGenerator

    print("=== 阶段二：生成时水印嵌入 ===")

    wm_cfg = config.get("watermark", {})

    # 前置检查：优先用 CLI，回退到 JSON
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1

    secret_key = args.secret_key or wm_cfg.get("secret_key", "")
    lm_model_path = args.lm_model_path or wm_cfg.get("lm_model_path", "")
    if not secret_key:
        print("[错误] secret_key 为必填参数（--secret-key 或配置文件）", file=sys.stderr)
        return 1
    if not lm_model_path:
        print("[错误] lm_model_path 为必填参数（--lm-model-path 或配置文件）", file=sys.stderr)
        return 1

    prompt = args.prompt or wm_cfg.get("prompt", "def solution():")
    output_file = args.output_file or wm_cfg.get("output_file", "data/results/watermarked.py")

    embed_dim = args.embed_dim or wm_cfg.get("encoder_embed_dim", 128)
    encoder_model_path = wm_cfg.get("encoder_model_path", "data/models/codet5-base")

    # 加载编码器
    enc_cfg = config.get("encoder", {})
    enc_config = EncoderConfig(**enc_cfg) if enc_cfg else EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    encoder_tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    # 加载代码生成 LLM
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lm_tokenizer = AutoTokenizer.from_pretrained(lm_model_path)
    lm_model = AutoModelForCausalLM.from_pretrained(lm_model_path).to(device)

    # 构建 WatermarkConfig（从 JSON，secret_key 用解析后的值）
    wm_config_fields = {k: v for k, v in wm_cfg.items()
                        if k not in ("lm_model_path", "prompt", "output_file")}
    wm_config_fields["secret_key"] = secret_key
    wm_config_fields["encoder_device"] = device
    wm_config_fields["encoder_embed_dim"] = embed_dim
    wm_config = WatermarkConfig(**wm_config_fields)

    generator = WatermarkGenerator(lm_model, lm_tokenizer, encoder, encoder_tokenizer, wm_config)

    try:
        result = generator.generate(prompt)
    except Exception as e:
        print(f"[错误] 水印生成失败：{e}", file=sys.stderr)
        return 1

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

**Step 4: 运行测试确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 所有测试 PASSED

**Step 5: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: run_watermark accepts config dict, JSON values as base with CLI override"
```

---

### Task 6: 重构 `run_extract()` 使用 config dict

**Files:**
- Modify: `run.py`（`run_extract()` 签名与实现）
- Test: `tests/test_run_config.py`（追加用例）

**Step 1: 写失败测试**

在 `tests/test_run_config.py` 追加：

```python
from run import run_extract


def test_run_extract_json_z_threshold_used(tmp_path):
    """CLI 未传 --z-threshold，使用 JSON 中的值。"""
    code_file = tmp_path / "test.py"
    code_file.write_text("def foo(): pass")

    args = build_parser().parse_args([])
    args.secret_key = "testkey"
    args.code_file = str(code_file)
    args.z_threshold = None
    args.embed_dim = None

    config = {
        "extract": {
            "secret_key": "",
            "code_file": "",
            "embed_dim": 128,
            "z_threshold": 2.5,
        },
        "encoder": {},
    }

    captured = {}

    fake_result = MagicMock()
    fake_result.is_watermarked = False
    fake_result.z_score = 1.0
    fake_result.p_value = 0.5
    fake_result.independent_blocks = 3
    fake_result.total_blocks = 5
    fake_result.hit_blocks = 2

    def fake_detect(code):
        return fake_result

    fake_detector = MagicMock()
    fake_detector.detect = fake_detect

    state = MagicMock()
    state.is_done.return_value = True
    state.get.return_value = None

    with patch("run.WatermarkDetector", return_value=fake_detector) as mock_detector, \
         patch("run.SemanticEncoder"), \
         patch("run.AutoTokenizer"), \
         patch("run.torch"):
        run_extract(args, config, state)

    # 验证 ExtractConfig 用了 JSON 的 z_threshold=2.5
    call_args = mock_detector.call_args
    ext_cfg = call_args[0][0]
    assert ext_cfg.z_threshold == pytest.approx(2.5)
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::test_run_extract_json_z_threshold_used -v
```

Expected: FAIL

**Step 3: 重构 `run_extract()`**

```python
def run_extract(args: argparse.Namespace, config: dict, state: RunState) -> int:
    """阶段三：检测代码水印。"""
    import torch
    from transformers import AutoTokenizer

    from wfcllm.encoder.config import EncoderConfig
    from wfcllm.encoder.model import SemanticEncoder
    from wfcllm.extract.config import ExtractConfig
    from wfcllm.extract.detector import WatermarkDetector

    print("=== 阶段三：水印提取与验证 ===")

    ext_cfg = config.get("extract", {})

    # 前置检查：CLI 优先，回退 JSON
    if not state.is_done("encoder"):
        print("[错误] 请先完成阶段一（encoder）", file=sys.stderr)
        return 1

    secret_key = args.secret_key or ext_cfg.get("secret_key", "")
    code_file_str = args.code_file or ext_cfg.get("code_file", "")
    if not secret_key:
        print("[错误] secret_key 为必填参数（--secret-key 或配置文件）", file=sys.stderr)
        return 1
    if not code_file_str:
        print("[错误] code_file 为必填参数（--code-file 或配置文件）", file=sys.stderr)
        return 1
    code_path = Path(code_file_str)
    if not code_path.exists():
        print(f"[错误] 文件不存在：{code_file_str}", file=sys.stderr)
        return 1

    embed_dim = args.embed_dim or ext_cfg.get("embed_dim", 128)
    z_threshold = args.z_threshold or ext_cfg.get("z_threshold", 3.0)

    # 加载编码器
    enc_cfg = config.get("encoder", {})
    enc_config = EncoderConfig(**enc_cfg) if enc_cfg else EncoderConfig(embed_dim=embed_dim)
    local_codet5 = Path(enc_config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        enc_config.model_name = str(local_codet5)
        print(f"[自动] 编码器使用本地模型: {enc_config.model_name}")
    else:
        print(f"[回退] 编码器使用 HF Hub: {enc_config.model_name}")
    encoder = SemanticEncoder(config=enc_config)
    encoder_checkpoint = state.get("encoder", "checkpoint")
    if encoder_checkpoint and Path(encoder_checkpoint).exists():
        ckpt = torch.load(encoder_checkpoint, map_location="cpu")
        encoder.load_state_dict(ckpt["model_state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = encoder.to(device)
    tokenizer = AutoTokenizer.from_pretrained(enc_config.model_name)

    ext_config = ExtractConfig(secret_key=secret_key, embed_dim=embed_dim, z_threshold=z_threshold)
    detector = WatermarkDetector(ext_config, encoder, tokenizer, device=device)

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
        code_file=code_file_str,
        is_watermarked=result.is_watermarked,
        z_score=result.z_score,
    )
    return 0
```

**Step 4: 运行测试确认通过**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 所有测试 PASSED

**Step 5: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: run_extract accepts config dict, JSON values as base with CLI override"
```

---

### Task 7: 更新 `main()` 和 `run_phase()` 串联配置加载

**Files:**
- Modify: `run.py`（`main()`、`run_phase()` 函数）
- Test: `tests/test_run_config.py`（追加用例）

**Step 1: 写失败测试**

在 `tests/test_run_config.py` 追加：

```python
import subprocess


def test_main_uses_custom_config(tmp_path):
    """--config 指向自定义文件，配置能正确加载。"""
    cfg = {
        "encoder": {
            "model_name": "Salesforce/codet5-base", "embed_dim": 128,
            "use_lora": True, "lora_r": 16, "lora_alpha": 32,
            "lora_dropout": 0.1, "lora_target_modules": ["q", "v"],
            "use_bf16": True, "data_sources": ["mbpp", "humaneval"],
            "max_seq_length": 256, "negative_ratio": 0.5,
            "lr": 0.00002, "batch_size": 64, "epochs": 10, "margin": 0.3,
            "warmup_ratio": 0.1, "early_stopping_patience": 3,
            "num_workers": 8, "pin_memory": True,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "results_dir": str(tmp_path / "results"),
            "local_model_dir": str(tmp_path / "models"),
            "local_dataset_dir": str(tmp_path / "datasets"),
        },
        "watermark": {"secret_key": "", "lm_model_path": "", "prompt": "x",
                      "output_file": "", "encoder_model_path": "",
                      "encoder_embed_dim": 128, "encoder_device": "cpu",
                      "margin_base": 0.1, "margin_alpha": 0.05,
                      "max_retries": 5, "temperature": 0.8, "top_p": 0.95,
                      "top_k": 50, "max_new_tokens": 512,
                      "eos_token_id": None, "enable_fallback": True},
        "extract": {"secret_key": "", "code_file": "", "embed_dim": 128, "z_threshold": 3.0},
    }
    cfg_file = tmp_path / "test_config.json"
    cfg_file.write_text(json.dumps(cfg))

    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", "run.py",
         "--config", str(cfg_file), "--status"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode == 0
    assert "encoder" in result.stdout


def test_main_config_not_found_exits(tmp_path):
    """--config 指向不存在的文件，程序应退出非零。"""
    result = subprocess.run(
        ["conda", "run", "-n", "WFCLLM", "python", "run.py",
         "--config", str(tmp_path / "no.json"), "--status"],
        capture_output=True, text=True,
        env={**__import__("os").environ, "HF_HUB_OFFLINE": "1"},
    )
    assert result.returncode != 0
```

**Step 2: 运行测试确认失败**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py::test_main_uses_custom_config tests/test_run_config.py::test_main_config_not_found_exits -v
```

Expected: FAIL

**Step 3: 更新 `main()` 和 `run_phase()`**

将 `main()` 改为：

```python
def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config(args.config)
    state = RunState()

    if args.status:
        cmd_status(state)
        return 0

    if args.reset:
        cmd_reset(state)
        return 0

    phases_to_run = [args.phase] if args.phase else PHASES

    for phase in phases_to_run:
        if state.is_done(phase) and not args.force and not args.eval_only:
            print(f"[跳过] {phase}（已完成，使用 --force 强制重跑）")
            continue
        rc = run_phase(phase, args, config, state)
        if rc != 0:
            print(f"[失败] {phase} 阶段退出码 {rc}", file=sys.stderr)
            return rc

    return 0
```

将 `run_phase()` 改为：

```python
def run_phase(phase: str, args: argparse.Namespace, config: dict, state: RunState) -> int:
    """分发到各阶段 runner，返回退出码。"""
    runners = {
        "encoder": run_encoder,
        "watermark": run_watermark,
        "extract": run_extract,
    }
    return runners[phase](args, config, state)
```

**Step 4: 运行全部测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/test_run_config.py -v
```

Expected: 所有测试 PASSED

**Step 5: 验证 --status 能正常运行**

```bash
conda run -n WFCLLM python run.py --status
```

Expected: 打印三个阶段状态，无报错

**Step 6: Commit**

```bash
git add run.py tests/test_run_config.py
git commit -m "feat: wire load_config into main(), pass config dict through run_phase to run_*"
```

---

### Task 8: 全量测试验证

**Step 1: 运行全部测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/ -v
```

Expected: 所有测试 PASSED，无回归

**Step 2: 手动验证 --help 输出**

```bash
conda run -n WFCLLM python run.py --help
```

Expected: 帮助信息中包含 `--config` 参数说明

**Step 3: 验证自定义配置文件**

```bash
cp configs/base_config.json configs/test_custom.json
conda run -n WFCLLM python run.py --config configs/test_custom.json --status
```

Expected: 正常打印状态，无报错

**Step 4: 清理临时配置文件**

```bash
rm configs/test_custom.json
```

**Step 5: 最终 Commit（若有残留改动）**

```bash
git add -A
git commit -m "test: verify full test suite passes after config system refactor"
```
