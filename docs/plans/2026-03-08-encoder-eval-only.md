# Encoder Evaluate-Only Mode Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 让已训练好的 encoder checkpoint 可以独立评测，不重新训练，通过 `run.py --phase encoder --eval-only --checkpoint <path>` 触发。

**Architecture:** 两处改动：(1) `train.py` 新增 `evaluate_only(checkpoint, config)` 函数，复用已有评测逻辑；同时修复 `main()` 中 `random_split` 缺少固定 seed 的 bug（否则测试集每次不同）。(2) `run.py` 加 `--eval-only` flag，`run_encoder` 分支到 `evaluate_only`。损失函数、模型结构、数据管道不动。

**Tech Stack:** Python 3.10+, PyTorch, pytest

---

### Task 1: 修复 `train.py` 的 `random_split` — 加固定 seed

**Files:**
- Modify: `wfcllm/encoder/train.py:162`

**背景：** 当前 `random_split(dataset, [n_train, n_val, n_test])` 没有传 `generator`，每次 split 随机不同，导致 `evaluate_only` 无法复现训练时的测试集。

**Step 1: 修改 `random_split` 调用，加 `generator` 参数**

找到 `train.py` 第 162 行：

```python
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])
```

替换为：

```python
    _split_gen = torch.Generator().manual_seed(42)
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=_split_gen)
```

**Step 2: 运行现有测试确认无回归**

```bash
conda run -n WFCLLM pytest tests/encoder/test_train.py tests/encoder/test_evaluate.py -v
```

期望：全部 PASS

**Step 3: Commit**

```bash
git add wfcllm/encoder/train.py
git commit -m "fix: use fixed seed in random_split to make test set reproducible"
```

---

### Task 2: 在 `train.py` 新增 `evaluate_only()` 函数

**Files:**
- Modify: `wfcllm/encoder/train.py`（在 `if __name__ == "__main__":` 之前插入）

**Step 1: 在 `if __name__ == "__main__":` 之前插入 `evaluate_only` 函数**

```python
def evaluate_only(checkpoint_path: str, config: EncoderConfig | None = None) -> None:
    """Load a saved checkpoint and run evaluation only (no training).

    Uses the same fixed seed as main() so the test split is identical.
    """
    from dataclasses import replace

    if config is None:
        config = EncoderConfig()

    print("=== Encoder Evaluation (eval-only) ===")

    local_codet5 = Path(config.local_model_dir) / "codet5-base"
    if local_codet5.exists() and (local_codet5 / "config.json").exists():
        effective_model = str(local_codet5)
        print(f"  Using local model: {effective_model}")
    else:
        effective_model = config.model_name
        print(f"  Using HF Hub model: {effective_model}")

    config_for_model = replace(config, model_name=effective_model)

    # Rebuild dataset with same pipeline + same seed as main()
    print("Loading data...")
    code_samples = load_code_samples(config.data_sources, local_dataset_dir=config.local_dataset_dir)
    blocks = prepare_blocks_with_variants(code_samples)
    triplets = build_triplets_from_blocks(blocks, negative_ratio=config.negative_ratio, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(effective_model)
    dataset = TripletCodeDataset(triplets, tokenizer, max_length=config.max_seq_length)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    _split_gen = torch.Generator().manual_seed(42)
    _, _, test_ds = random_split(dataset, [n_train, n_val, n_test], generator=_split_gen)

    test_loader = DataLoader(
        test_ds, batch_size=config.batch_size,
        num_workers=config.num_workers, pin_memory=config.pin_memory,
    )
    print(f"  Test set size: {len(test_ds)}")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SemanticEncoder(config=config_for_model)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {checkpoint_path}")

    # Collect embeddings
    all_anchor, all_positive, all_negative = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            a = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            neg = model(batch["negative_input_ids"].to(device), batch["negative_attention_mask"].to(device))
            all_anchor.append(a.cpu())
            all_positive.append(p.cpu())
            all_negative.append(neg.cpu())

    anchor_embs = torch.cat(all_anchor)
    pos_embs = torch.cat(all_positive)
    neg_embs = torch.cat(all_negative)

    sep_metrics = cosine_separation(anchor_embs, pos_embs, neg_embs)
    r1 = recall_at_k(anchor_embs, pos_embs, k=1)
    r5 = recall_at_k(anchor_embs, pos_embs, k=5)
    r10 = recall_at_k(anchor_embs, pos_embs, k=10)
    mrr = mean_reciprocal_rank(anchor_embs, pos_embs)
    map_score = mean_average_precision(anchor_embs, pos_embs)
    wsc = watermark_sign_consistency(anchor_embs, pos_embs, num_directions=64, seed=42)

    pos_cos = torch.nn.functional.cosine_similarity(anchor_embs, pos_embs, dim=1)
    neg_cos = torch.nn.functional.cosine_similarity(anchor_embs, neg_embs, dim=1)
    f1_metrics = pair_f1_metrics(pos_cos, neg_cos)

    eval_metrics = {
        **sep_metrics,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "mrr": mrr,
        "map": map_score,
        "watermark_sign_consistency": wsc,
        **f1_metrics,
    }

    print("\n=== Evaluation Results ===")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    report_path = save_evaluation_report(eval_metrics, config.results_dir)
    print(f"\nReport saved to {report_path}")
```

**Step 2: 确认文件语法正确**

```bash
conda run -n WFCLLM python -c "from wfcllm.encoder.train import evaluate_only; print('OK')"
```

期望：打印 `OK`，无 ImportError

**Step 3: Commit**

```bash
git add wfcllm/encoder/train.py
git commit -m "feat: add evaluate_only() to train.py for checkpoint-based eval without retraining"
```

---

### Task 3: 在 `run.py` 加 `--eval-only` 参数，接入 `evaluate_only()`

**Files:**
- Modify: `run.py`

**Step 1: 在 `build_parser()` 中加 `--eval-only` 和 `--checkpoint` 参数**

找到 `run.py` 中 `--force` 参数定义之后（约 L92），插入：

```python
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="只跑评测，不训练（需配合 --phase encoder）",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="评测用的 checkpoint 路径（不传则从 run_state.json 读取）",
    )
```

**Step 2: 在 `run_encoder()` 函数开头加 eval-only 分支**

找到 `run_encoder` 函数（约 L165），在 `print("=== 阶段一：语义编码器预训练 ===")` 之后插入：

```python
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

        config = EncoderConfig()
        if args.model_name:
            config.model_name = args.model_name
        if args.embed_dim:
            config.embed_dim = args.embed_dim
        if args.no_lora:
            config.use_lora = False
        if args.no_bf16:
            config.use_bf16 = False

        try:
            evaluate_only(checkpoint, config)
        except Exception as e:
            print(f"[错误] 评测失败：{e}", file=sys.stderr)
            return 1
        return 0
```

**Step 3: 确认语法正确**

```bash
conda run -n WFCLLM python -c "import run; print('OK')"
```

期望：`OK`

**Step 4: 确认 --help 显示新参数**

```bash
conda run -n WFCLLM python run.py --help | grep eval-only
```

期望：显示 `--eval-only` 说明行

**Step 5: Commit**

```bash
git add run.py
git commit -m "feat: add --eval-only flag to run.py for standalone encoder evaluation"
```

---

### Task 4: 验证端到端可用性（dry-run）

**Step 1: 确认 checkpoint 存在**

```bash
ls data/checkpoints/encoder/
```

期望：能看到 `encoder_epoch9.pt`（或类似文件）

**Step 2: 运行评测**

```bash
HF_DATASETS_OFFLINE=1 OMP_NUM_THREADS=8 python -u run.py --phase encoder --eval-only 2>&1 | tee logs/$(date +%m%d_%H%M%S)_eval.log
```

期望：
- 打印 `=== Encoder Evaluation (eval-only) ===`
- 打印各指标（recall@1、MRR、watermark_sign_consistency 等）
- 打印 `Report saved to data/results/evaluation_report.json`

**Step 3: 查看报告**

```bash
cat data/results/evaluation_report.json
```

期望：JSON 包含 `recall@1`、`mrr`、`watermark_sign_consistency`、`pair_f1` 等字段
