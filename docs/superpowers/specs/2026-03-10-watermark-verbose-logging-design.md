# 水印嵌入详细日志设计

**日期：** 2026-03-10

## 目标

在水印嵌入过程中，通过 `logging.DEBUG` 级别输出每个语句块的完整 LSH 签名信息，便于调试和分析失败原因（签名落错区域 vs margin 不足）。

## 变更范围

### `wfcllm/watermark/verifier.py`（已实现）

扩展 `VerifyResult` dataclass，新增 `lsh_signature` 字段（带默认值保持向后兼容）：

```python
@dataclass
class VerifyResult:
    passed: bool
    min_margin: float
    lsh_signature: tuple[int, ...] = ()
```

`ProjectionVerifier.verify()` 将已计算的 `sig` 放入返回值，无需额外计算。

### `wfcllm/watermark/retry_loop.py`（已实现）

`RetryLoop.run()` 每次尝试的日志新增 `entropy`/`sig`/`in_valid`：

```
[retry K/M] entropy=0.8342 margin_thresh=0.0014 sig=(1,0,1,...) in_valid=False min_margin=0.0227 passed=False
  text='...'
```

### `wfcllm/watermark/generator.py`（已实现）

**simple block 首次检测**（`_verify_block`）：
```
[simple block] node=X parent=Y entropy=0.1234 margin_thresh=0.0500
  sig=(1,0,1,...) in_valid=True valid_set_size=8 min_margin=0.0823 passed=True
  text='x = foo(a, b)'
```

**compound fallback**（`_try_passive_fallback`）：
```
[compound fallback] node=Z parent=W entropy=0.2345 margin_thresh=0.0500
  sig=(1,1,0,...) in_valid=True valid_set_size=8 min_margin=0.0910 passed=True
```

## 日志字段说明

| 字段 | 含义 |
|------|------|
| `entropy` | `block_entropy`，节点类型查表求和，影响 `margin_thresh` |
| `margin_thresh` | `margin_base + margin_alpha × entropy`，本块要求的最小置信度 |
| `sig` | d-bit LSH 签名 tuple，如 `(1,0,1,0,1,0,1,0)` |
| `in_valid` | 签名是否在当前块的 valid_set G 中（`False` 说明落错区域） |
| `valid_set_size` | `len(valid_set)`，即 `round(gamma × 2^d)` |
| `min_margin` | 到所有超平面的最小绝对余弦距离（越大越稳定） |
| `passed` | 最终嵌入决策：`in_valid AND min_margin > margin_thresh` |

## 失败原因诊断

- `in_valid=False` → 签名落在错误区域，调阈值无效，需更多 retry 或调整 `lsh_gamma`
- `in_valid=True` 且 `passed=False` → `min_margin ≤ margin_thresh`，可降低 `margin_base`/`margin_alpha`

## 不变更内容

- `LSHSpace`、`WatermarkKeying`、`StatementInterceptor` 不做改动
- 日志级别保持 `DEBUG`，不影响生产环境输出
