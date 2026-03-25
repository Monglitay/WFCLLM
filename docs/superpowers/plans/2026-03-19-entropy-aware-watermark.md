# Entropy-Aware Adaptive Watermarking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `WFCLLM` 落地基于 `entropy_profile + piecewise_quantile` 的块级自适应 `gamma_i` 水印协议，并在提取端以 canonical block contracts + adaptive hypothesis testing 完成闭环。

**Architecture:** 先引入共享 `block_contract` 构建器，把 simple statement block 的切分、`entropy_units`、`k_i` 和 `gamma_effective_i` 固化为 canonical truth；再将嵌入端改造为按块 `k_i` 派生有效区域，并将 metadata 写入 watermark JSONL；最后在提取端重算 contracts、执行对齐校验，并使用泊松二项正态近似替换固定二项检验。

**Tech Stack:** Python 3.11, dataclasses, tree-sitter-python, pytest, scipy, JSON, argparse

**Spec:** `docs/superpowers/specs/2026-03-19-entropy-aware-watermark-design.md`

---

## Preflight

- 在开始实现前，优先使用独立 worktree，避免污染当前工作区
- 所有测试命令统一加 `HF_HUB_OFFLINE=1`
- 优先跑最小范围测试，再逐步扩大到 pipeline 回归

**Recommended setup**

```bash
git worktree add ../WFCLLM-entropy-aware -b feature/entropy-aware-watermark develop
cd ../WFCLLM-entropy-aware
conda activate WFCLLM
git status
```

---

## File Structure

### New files

- `wfcllm/common/block_contract.py` — canonical `BlockContract`、`BlockContractBundle` 与 shared builder
- `wfcllm/watermark/entropy_profile.py` — `EntropyProfile` 的加载、校验与序列化
- `wfcllm/watermark/gamma_schedule.py` — `fixed | linear | bucket | piecewise_quantile` 调度器
- `wfcllm/extract/alignment.py` — contract 对齐校验与 `AlignmentReport`
- `tests/common/test_block_contract.py` — block contract 纯逻辑测试
- `tests/watermark/test_entropy_profile.py` — profile 加载与校验测试
- `tests/watermark/test_gamma_schedule.py` — 分位点插值与 `k_i` 量化测试
- `tests/extract/test_alignment.py` — 结构/数值 mismatch 测试
- `tests/test_calibrate_script.py` — `scripts/calibrate.py` 的 CLI smoke tests

### Modified files

- `wfcllm/watermark/entropy.py` — 从浮点表升级到 `entropy_units`
- `wfcllm/watermark/config.py` — 新增 adaptive gamma 配置
- `wfcllm/watermark/keying.py` — `derive(parent_node_type, k)` 替换固定 `gamma`
- `wfcllm/watermark/generator.py` — 运行时按块 `k_i` 验证，收尾阶段输出 canonical contracts
- `wfcllm/watermark/retry_loop.py` — 重试日志与按块 `k_i` 的验证参数传递
- `wfcllm/watermark/pipeline.py` — 写出 adaptive metadata 到 watermark JSONL
- `wfcllm/extract/config.py` — 扩展 `BlockScore`、`DetectionResult`、adaptive detection 配置
- `wfcllm/extract/scorer.py` — 将 `gamma_effective` 绑定到每个 block score
- `wfcllm/extract/hypothesis.py` — fixed / adaptive 双模统计检验
- `wfcllm/extract/detector.py` — contract 重算、alignment、adaptive 模式切换
- `wfcllm/extract/pipeline.py` — 写出 alignment 与 adaptive 检测结果
- `run.py` — 新增 CLI 与配置解析
- `scripts/calibrate.py` — 扩展成 profile 构建 + threshold 校准双功能
- `README.md` — 更新使用说明
- `configs/base_config.json` — 引入默认 adaptive 配置
- `configs/humaneval_10_config.json` — 引入子集实验 adaptive 配置
- `tests/watermark/test_entropy.py` — 适配 `entropy_units`
- `tests/watermark/test_keying.py` — 适配 `k_i`
- `tests/watermark/test_generator.py` — generator adaptive metadata tests
- `tests/watermark/test_pipeline.py` — watermark 输出字段测试
- `tests/extract/test_config.py` — adaptive detection config tests
- `tests/extract/test_hypothesis.py` — adaptive hypothesis tests
- `tests/extract/test_detector.py` — detector contract tests
- `tests/extract/test_pipeline.py` — extract output adaptive 字段测试
- `tests/test_run_config.py` — CLI / config parser tests

---

### Task 1: Canonical entropy units and block contracts

**Files:**
- Create: `wfcllm/common/block_contract.py`
- Modify: `wfcllm/watermark/entropy.py`
- Modify: `tests/watermark/test_entropy.py`
- Test: `tests/common/test_block_contract.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_estimate_block_entropy_units_returns_int():
    estimator = NodeEntropyEstimator()
    units = estimator.estimate_block_entropy_units("x = 1")
    assert isinstance(units, int)
    assert units > 0


def test_build_block_contracts_is_deterministic():
    code = "x = 1\nreturn x\n"
    first = build_block_contracts(code)
    second = build_block_contracts(code)
    assert [(c.ordinal, c.block_text_hash, c.entropy_units) for c in first] == [
        (c.ordinal, c.block_text_hash, c.entropy_units) for c in second
    ]
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_entropy.py \
  tests/common/test_block_contract.py -v
```

Expected: FAIL with missing `estimate_block_entropy_units` / missing `build_block_contracts`

- [ ] **Step 3: Write the minimal implementation**

```python
ENTROPY_SCALE = 10000


@dataclass(frozen=True)
class BlockContract:
    ordinal: int
    block_id: str
    node_type: str
    parent_node_type: str
    block_text_hash: str
    start_line: int
    end_line: int
    entropy_units: int
    gamma_target: float = 0.0
    k: int = 0
    gamma_effective: float = 0.0


class NodeEntropyEstimator:
    def estimate_block_entropy_units(self, block_source: str) -> int:
        if not block_source.strip():
            return 0
        tree = PythonParser().parse(block_source)
        return self._sum_entropy_units(tree.root_node)

    def estimate_block_entropy(self, block_source: str) -> float:
        return self.estimate_block_entropy_units(block_source) / ENTROPY_SCALE
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_entropy.py \
  tests/common/test_block_contract.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/common/block_contract.py wfcllm/watermark/entropy.py \
  tests/watermark/test_entropy.py tests/common/test_block_contract.py
git commit -m "feat: add canonical block contract builder"
```

---

### Task 2: Add entropy profiles and piecewise-quantile schedule

**Files:**
- Create: `wfcllm/watermark/entropy_profile.py`
- Create: `wfcllm/watermark/gamma_schedule.py`
- Test: `tests/watermark/test_entropy_profile.py`
- Test: `tests/watermark/test_gamma_schedule.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_entropy_profile_loads_quantile_units(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps({
        "profile_id": "python__demo__v1",
        "version": 1,
        "language": "python",
        "model_family": "demo",
        "entropy_scale": 10000,
        "sample_count": 10,
        "summary": {"min_units": 100, "max_units": 500, "mean_units": 300, "median_units": 280},
        "quantiles": {"p10_units": 120, "p50_units": 280, "p75_units": 360, "p90_units": 420, "p95_units": 480},
    }))
    profile = EntropyProfile.load(path)
    assert profile.quantile_units("p75") == 360


def test_piecewise_quantile_schedule_interpolates_and_quantizes():
    schedule = PiecewiseQuantileSchedule(
        profile=profile,
        anchor_quantiles=["p10", "p50", "p75", "p90", "p95"],
        anchor_gammas=[0.95, 0.75, 0.55, 0.35, 0.25],
        gamma_min=0.25,
        gamma_max=0.95,
    )
    result = schedule.resolve(entropy_units=320, lsh_d=4)
    assert 0.25 <= result.gamma_target <= 0.95
    assert 1 <= result.k <= 15
    assert result.gamma_effective == result.k / 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_entropy_profile.py \
  tests/watermark/test_gamma_schedule.py -v
```

Expected: FAIL with missing modules/classes

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass(frozen=True)
class EntropyProfile:
    profile_id: str
    version: int
    language: str
    model_family: str
    entropy_scale: int
    sample_count: int
    summary: dict[str, int]
    quantiles: dict[str, int]

    @classmethod
    def load(cls, path: str | Path) -> "EntropyProfile":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def quantile_units(self, name: str) -> int:
        return self.quantiles[f"{name}_units"]


@dataclass(frozen=True)
class GammaResolution:
    gamma_target: float
    k: int
    gamma_effective: float
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_entropy_profile.py \
  tests/watermark/test_gamma_schedule.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/entropy_profile.py wfcllm/watermark/gamma_schedule.py \
  tests/watermark/test_entropy_profile.py tests/watermark/test_gamma_schedule.py
git commit -m "feat: add entropy profile and adaptive gamma schedule"
```

---

### Task 3: Plumb adaptive config and block-sized key derivation

**Files:**
- Modify: `wfcllm/watermark/config.py`
- Modify: `wfcllm/extract/config.py`
- Modify: `wfcllm/watermark/keying.py`
- Modify: `tests/watermark/test_keying.py`
- Modify: `tests/watermark/test_config.py`
- Modify: `tests/extract/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_keying_derive_returns_exact_k_signatures():
    keying = WatermarkKeying("secret", d=4)
    valid = keying.derive("module", k=5)
    assert len(valid) == 5


def test_adaptive_gamma_config_defaults():
    cfg = WatermarkConfig(secret_key="k")
    assert cfg.adaptive_gamma.enabled is False
    assert cfg.adaptive_gamma.strategy == "piecewise_quantile"
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_keying.py \
  tests/watermark/test_config.py \
  tests/extract/test_config.py -v
```

Expected: FAIL because `WatermarkKeying(..., d=4)` and nested config do not exist

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass
class AdaptiveGammaConfig:
    enabled: bool = False
    strategy: str = "piecewise_quantile"
    profile_path: str | None = None
    profile_id: str | None = None
    gamma_min: float = 0.25
    gamma_max: float = 0.95
    anchor_quantiles: list[str] = field(default_factory=lambda: ["p10", "p50", "p75", "p90", "p95"])
    anchor_gammas: list[float] = field(default_factory=lambda: [0.95, 0.75, 0.55, 0.35, 0.25])


class WatermarkKeying:
    def __init__(self, secret_key: str, d: int):
        self._key = secret_key.encode("utf-8")
        self._d = d

    def derive(self, parent_node_type: str, k: int) -> frozenset[tuple[int, ...]]:
        if not 1 <= k < 2 ** self._d:
            raise ValueError("k must be between 1 and 2^d - 1")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_keying.py \
  tests/watermark/test_config.py \
  tests/extract/test_config.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/config.py wfcllm/extract/config.py wfcllm/watermark/keying.py \
  tests/watermark/test_keying.py tests/watermark/test_config.py tests/extract/test_config.py
git commit -m "feat: add adaptive gamma config and block-sized keying"
```

---

### Task 4: Emit canonical adaptive metadata from watermark generation

**Files:**
- Modify: `wfcllm/watermark/generator.py`
- Modify: `wfcllm/watermark/retry_loop.py`
- Modify: `wfcllm/watermark/pipeline.py`
- Modify: `tests/watermark/test_generator.py`
- Modify: `tests/watermark/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_generate_result_contains_block_contracts(monkeypatch, fake_generator):
    result = fake_generator.generate("def f():\n    return 1\n")
    assert result.block_contracts
    assert result.adaptive_mode in {"fixed", "piecewise_quantile"}


def test_pipeline_writes_adaptive_metadata(tmp_path, mock_generator):
    output_path = pipeline.run()
    record = json.loads(Path(output_path).read_text(encoding="utf-8").strip())
    assert "blocks" in record
    assert "adaptive_mode" in record
    assert "profile_id" in record
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_generator.py \
  tests/watermark/test_pipeline.py -v
```

Expected: FAIL because `GenerateResult` and pipeline JSONL do not include adaptive metadata

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass
class GenerateResult:
    code: str
    stats: EmbedStats
    block_contracts: list[BlockContract] = field(default_factory=list)
    adaptive_mode: str = "fixed"
    profile_id: str | None = None
    alignment_summary: dict[str, object] = field(default_factory=dict)


final_contracts = build_block_contracts(
    code=ctx.generated_text,
    profile=self._entropy_profile,
    schedule=self._gamma_schedule,
    lsh_d=self._config.lsh_d,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/watermark/test_generator.py \
  tests/watermark/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/watermark/generator.py wfcllm/watermark/retry_loop.py \
  wfcllm/watermark/pipeline.py tests/watermark/test_generator.py tests/watermark/test_pipeline.py
git commit -m "feat: wire adaptive gamma into watermark generator"
```

---

### Task 5: Add extract-side contract alignment and invalid-sample handling

**Files:**
- Create: `wfcllm/extract/alignment.py`
- Modify: `wfcllm/extract/detector.py`
- Modify: `wfcllm/extract/pipeline.py`
- Test: `tests/extract/test_alignment.py`
- Modify: `tests/extract/test_detector.py`
- Modify: `tests/extract/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_alignment_report_flags_structure_mismatch():
    report = compare_block_contracts(
        embedded=[embedded_contract],
        rebuilt=[],
    )
    assert report.structure_match is False
    assert report.first_mismatch_reason == "block_count_mismatch"


def test_alignment_report_flags_numeric_mismatch():
    report = compare_block_contracts(
        embedded=[embedded_contract],
        rebuilt=[replace(embedded_contract, entropy_units=embedded_contract.entropy_units + 1)],
    )
    assert report.structure_match is True
    assert report.numeric_match is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/extract/test_alignment.py \
  tests/extract/test_detector.py \
  tests/extract/test_pipeline.py -v
```

Expected: FAIL because `compare_block_contracts` and adaptive alignment fields do not exist

- [ ] **Step 3: Write the minimal implementation**

```python
@dataclass
class AlignmentReport:
    structure_match: bool
    numeric_match: bool
    mismatch_count: int
    first_mismatch_reason: str | None = None
    first_mismatch_block_ordinal: int | None = None


def compare_block_contracts(embedded, rebuilt) -> AlignmentReport:
    if len(embedded) != len(rebuilt):
        return AlignmentReport(False, False, 1, "block_count_mismatch", None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/extract/test_alignment.py \
  tests/extract/test_detector.py \
  tests/extract/test_pipeline.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/alignment.py wfcllm/extract/detector.py wfcllm/extract/pipeline.py \
  tests/extract/test_alignment.py tests/extract/test_detector.py tests/extract/test_pipeline.py
git commit -m "feat: add adaptive contract alignment checks"
```

---

### Task 6: Replace fixed-binomial testing with adaptive hypothesis testing

**Files:**
- Modify: `wfcllm/extract/config.py`
- Modify: `wfcllm/extract/scorer.py`
- Modify: `wfcllm/extract/hypothesis.py`
- Modify: `tests/extract/test_hypothesis.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_adaptive_gamma_sequence_uses_poisson_binomial_normal_approximation():
    scores = [
        BlockScore(block_id="0", score=1, min_margin=0.2, gamma_effective=0.25, selected=True),
        BlockScore(block_id="1", score=1, min_margin=0.2, gamma_effective=0.75, selected=True),
        BlockScore(block_id="2", score=0, min_margin=0.2, gamma_effective=0.50, selected=True),
    ]
    result = HypothesisTester(fpr_threshold=1.0).test(scores, total_blocks=3, mode="adaptive")
    expected_mu = 1.5
    expected_var = 0.25 * 0.75 + 0.75 * 0.25 + 0.50 * 0.50
    assert result.expected_hits == pytest.approx(expected_mu)
    assert result.variance == pytest.approx(expected_var)


def test_fixed_mode_matches_old_formula():
    scores = [BlockScore(block_id=str(i), score=1, min_margin=0.1, gamma_effective=0.5, selected=True) for i in range(20)]
    result = HypothesisTester(fpr_threshold=3.0).test(scores, total_blocks=20, mode="fixed")
    assert result.z_score == pytest.approx((20 - 10) / math.sqrt(5))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v
```

Expected: FAIL because `BlockScore.gamma_effective`, `mode`, `expected_hits`, and `variance` do not exist

- [ ] **Step 3: Write the minimal implementation**

```python
class HypothesisTester:
    def test(self, selected_scores, total_blocks, mode: str = "fixed") -> DetectionResult:
        if mode == "fixed":
            gamma = self._gamma
            mu = len(selected_scores) * gamma
            variance = len(selected_scores) * gamma * (1 - gamma)
        else:
            gammas = [score.gamma_effective for score in selected_scores]
            mu = sum(gammas)
            variance = sum(g * (1 - g) for g in gammas)
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/extract/test_hypothesis.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add wfcllm/extract/config.py wfcllm/extract/scorer.py wfcllm/extract/hypothesis.py \
  tests/extract/test_hypothesis.py
git commit -m "feat: support adaptive hypothesis testing"
```

---

### Task 7: Expose adaptive mode through pipelines, CLI, configs, and calibrate script

**Files:**
- Modify: `run.py`
- Modify: `configs/base_config.json`
- Modify: `configs/humaneval_10_config.json`
- Modify: `scripts/calibrate.py`
- Modify: `tests/test_run_config.py`
- Test: `tests/test_calibrate_script.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_parser_accepts_entropy_profile():
    parser = build_parser()
    args = parser.parse_args(["--gamma-strategy", "piecewise_quantile", "--entropy-profile", "configs/demo.json"])
    assert args.gamma_strategy == "piecewise_quantile"
    assert args.entropy_profile == "configs/demo.json"


def test_build_entropy_profile_subcommand_writes_json(tmp_path):
    log_path = tmp_path / "watermark.log"
    log_path.write_text("wfcllm.watermark.generator DEBUG [simple block] entropy=1.2345\n")
    output_path = tmp_path / "profile.json"
    completed = subprocess.run(
        [
            sys.executable, "scripts/calibrate.py", "build-entropy-profile",
            "--input-log", str(log_path),
            "--output", str(output_path),
            "--language", "python",
            "--model-family", "demo-model",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/test_run_config.py \
  tests/test_calibrate_script.py -v
```

Expected: FAIL because parser flags / subcommands are missing

- [ ] **Step 3: Write the minimal implementation**

```python
parser.add_argument("--gamma-strategy", default=None)
parser.add_argument("--entropy-profile", default=None)
parser.add_argument("--profile-id", default=None)
parser.add_argument("--adaptive-detection-mode", default=None)
parser.add_argument("--strict-contract", action="store_true")

subparsers = parser.add_subparsers(dest="command")
build_profile = subparsers.add_parser("build-entropy-profile")
threshold = subparsers.add_parser("calibrate-threshold")
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/test_run_config.py \
  tests/test_calibrate_script.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add run.py configs/base_config.json configs/humaneval_10_config.json \
  scripts/calibrate.py tests/test_run_config.py tests/test_calibrate_script.py
git commit -m "feat: expose adaptive gamma via cli and configs"
```

---

### Task 8: Add full round-trip regression coverage and finish docs

**Files:**
- Modify: `README.md`
- Create: `tests/extract/test_adaptive_roundtrip.py`
- Modify: `tests/watermark/test_pipeline.py`
- Modify: `tests/extract/test_pipeline.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_adaptive_roundtrip_preserves_contracts(tmp_path):
    record = make_watermark_record_with_blocks()
    result = detector.detect(
        record["generated_code"],
        embedded_contracts=record["blocks"],
        embedded_metadata=record,
    )
    assert result.mode == "adaptive"
    assert result.alignment_ok is True
    assert result.contract_valid is True


def test_tampered_metadata_is_marked_invalid(tmp_path):
    record = make_watermark_record_with_blocks()
    record["blocks"][0]["entropy_units"] += 1
    result = detector.detect(
        record["generated_code"],
        embedded_contracts=record["blocks"],
        embedded_metadata=record,
    )
    assert result.contract_valid is False
    assert result.alignment_ok is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/extract/test_adaptive_roundtrip.py \
  tests/watermark/test_pipeline.py \
  tests/extract/test_pipeline.py -v
```

Expected: FAIL because round-trip contract validity is not fully wired

- [ ] **Step 3: Write the minimal implementation and README updates**

```markdown
## Adaptive Gamma Watermarking

1. 构建 `entropy_profile`
2. 在 `watermark.adaptive_gamma.enabled=true` 下运行阶段二
3. 在 `extract.adaptive_detection.mode=prefer-adaptive` 下运行阶段三
4. 若出现 `alignment_failed` 或 `adaptive_contract_invalid`，优先检查 block contracts 和 profile 配置
```

- [ ] **Step 4: Run the targeted tests, then a broader regression sweep**

Run:

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest \
  tests/common/test_block_contract.py \
  tests/watermark/test_entropy.py \
  tests/watermark/test_entropy_profile.py \
  tests/watermark/test_gamma_schedule.py \
  tests/watermark/test_keying.py \
  tests/watermark/test_generator.py \
  tests/watermark/test_pipeline.py \
  tests/extract/test_alignment.py \
  tests/extract/test_hypothesis.py \
  tests/extract/test_detector.py \
  tests/extract/test_pipeline.py \
  tests/extract/test_adaptive_roundtrip.py \
  tests/test_run_config.py \
  tests/test_calibrate_script.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add README.md tests/extract/test_adaptive_roundtrip.py \
  tests/watermark/test_pipeline.py tests/extract/test_pipeline.py
git commit -m "test: add adaptive watermark end-to-end coverage"
```

---

## Final Verification

- [ ] Run the full targeted regression suite listed in Task 8
- [ ] Run `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/watermark tests/extract tests/common -v`
- [ ] Manually inspect one adaptive watermark JSONL sample and one extract details JSONL sample
- [ ] Confirm summary JSON distinguishes:
  - `fixed` vs `adaptive`
  - `alignment_failed`
  - `adaptive_contract_invalid`

## Notes for the Implementer

- 不要把 `gamma_target` 当作统计真值；统计与运行时 keying 一律基于 `k_i` 和 `gamma_effective_i`
- 不要让嵌入端与提取端各自复制 contract 构建逻辑；必须共享 `wfcllm/common/block_contract.py`
- 不要把结构 mismatch 和数值 mismatch 混为一谈；结构 mismatch 必须硬失败
- 优先保持 fixed 模式测试稳定，adaptive 能力以“新增路径”形式接入
