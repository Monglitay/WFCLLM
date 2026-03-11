# 水印检测改造设计文档

**日期：** 2026-03-11
**分支：** develop-LSH
**参考：** `docs/水印检测改造.md`

---

## 背景

当前检测层（`wfcllm/extract/`）使用 `{-1, 1}` 计分和硬编码 `z_threshold=3.0` 做水印判定。根据 SEMSTAMP 论文，需改为 `{0, 1}` 计分并引入基于假阳性率（FPR）的动态阈值 $M_r$，以适配不同编程语言的代码分布。

---

## 变更范围

### 1. 计分层（`{-1, 1}` → `{0, 1}`）

**文件：** `wfcllm/extract/scorer.py`

打分规则更新：
```python
# 旧
score = 1 if result.passed else -1
# 新
score = 1 if result.passed else 0
```

**文件：** `wfcllm/extract/config.py`

`BlockScore.score` 注释更新：`+1 (hit) or -1 (miss)` → `1 (hit) or 0 (miss)`

**文件：** `wfcllm/extract/dp_selector.py`

逻辑不变。平局偏向子节点（`self_score > children_sum` 才选自身）在 `{0, 1}` 计分下语义正确——等价于最大化独立命中数，且偏向更多独立样本。

---

### 2. 假设检验层（硬编码阈值 → FPR 阈值 $M_r$）

**文件：** `wfcllm/extract/config.py`

字段重命名：
```python
# 旧
z_threshold: float = 3.0
# 新
fpr_threshold: float = 3.0  # M_r，由校准脚本生成；默认值 3.0 仅作占位
```

**文件：** `wfcllm/extract/hypothesis.py`

构造函数参数及判定条件更新：
- 参数 `z_threshold` → `fpr_threshold`
- 判定 `z_score > self._z_threshold` → `z_score >= self._fpr_threshold`

**文件：** `wfcllm/extract/detector.py`

传参更新：
```python
# 旧
HypothesisTester(config.z_threshold, ...)
# 新
HypothesisTester(config.fpr_threshold, ...)
```

---

### 3. 校准层（新增）

**新增文件：** `wfcllm/extract/calibrator.py`

`ThresholdCalibrator` 类，职责：
- 接收无水印代码语料（JSONL，字段 `generated_code`，格式与 `ExtractPipeline` 输入一致）
- 对每个样本运行 `BlockScorer` + `DPSelector`，计算 SEMSTAMP Z 分
- 对 Z 分升序排列，取第 `(1 - fpr)` 百分位作为 $M_r$
  - 例：FPR=0.01 → 取第 99 百分位 Z 分
- 返回 `{"fpr": r, "fpr_threshold": M_r, "n_samples": N}`
- 不依赖 `HypothesisTester`，直接操作原始 Z 分列表（避免循环依赖）

**新增文件：** `scripts/calibrate.py`

CLI 入口：
```
python scripts/calibrate.py \
  --input data/negative_corpus.jsonl \
  --output threshold.json \
  --fpr 0.01 \
  --secret-key <key> \
  --model data/models/codet5-base
```

输出 `threshold.json`，用户将 `fpr_threshold` 值手动填入 `ExtractConfig` 后部署。

---

### 4. 测试同步更新

所有现有测试一次性同步，不留兼容 shim：

| 文件 | 变更内容 |
|------|----------|
| `tests/extract/test_config.py` | `z_threshold` 断言 → `fpr_threshold`（2处） |
| `tests/extract/test_hypothesis.py` | `HypothesisTester(z_threshold=...)` → `fpr_threshold`（4处） |
| `tests/extract/test_detector.py` | `ExtractConfig(z_threshold=3.0)` → `fpr_threshold`（1处） |
| `tests/extract/test_scorer.py` | 未命中断言 `score == -1` → `score == 0` |
| `tests/extract/test_dp_selector.py` | `_make_score` 中 `-1` 传值 → `0` |
| `tests/extract/test_calibrator.py` | 新增，覆盖 FPR 百分位逻辑 |

---

## 数据流

```
负样本 JSONL
    → calibrate.py
    → ThresholdCalibrator
        → BlockScorer + DPSelector（复用现有）
        → Z 分列表 → percentile(1-fpr)
    → threshold.json（M_r）

待测代码
    → WatermarkDetector.detect()
        → BlockScorer（{0,1} 计分）
        → DPSelector（最大化命中数）
        → HypothesisTester（z >= M_r）
    → DetectionResult
```

---

## 不在本次范围内

- DP 选择策略调整（保持现有平局偏向子节点逻辑）
- 校准结果自动写入 `ExtractConfig`（职责分离，由用户手动填入）
- 向后兼容 `z_threshold` 字段（直接重命名，同步更新测试）
