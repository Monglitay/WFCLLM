# HumanEval Pass@1 提升实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在单次本地生成且不使用隐藏测试选择的前提下，以语法边界收口和更完整的 greedy 解码将 full164 Pass@1 提高到至少 30%。

**架构：** 新增一个与模型和水印器解耦的 HumanEval completion finalizer，由生成管线在基础生成后、水印嵌入前调用。finalizer 只依据 prompt、Python AST 和源码边界工作；管线另存逐样本 provenance，最终 detector input 合同不变。

**技术栈：** Python 3.11、`ast`、dataclass、pytest、Transformers 本地生成、现有 WFCLLM generation/detection pipeline。

---

## 文件结构

- 创建 `wfcllm/generation/completion_finalizer.py`：纯函数收口器与结果 dataclass。
- 创建 `tests/generation/test_completion_finalizer.py`：finalizer 的截断、Markdown、stub 和 fallback 测试。
- 修改 `wfcllm/generation/gated_pipeline.py`：注入 finalizer、写 provenance 和 manifest 计数。
- 修改 `tests/generation/test_gated_pipeline.py`：验证调用顺序、fallback、严格 detector schema 和 sidecar。
- 修改 `wfcllm/cli/runners.py`：按 config 绑定 HumanEval finalizer。
- 修改 `configs/wfcllm/gated_semantic_window_v1.json`：512-token greedy 解码和 finalizer 版本。
- 修改 `tests/integration/test_gated_fast_mainline.py`：验证生产 wiring 使用批准的 finalizer。
- 创建固定实验目录中的 `run-full164-pass-attempt2.sh`：隔离启动、日志、状态、工件哈希与成功/失败标记。

### 任务 1：目标函数收口器（TDD）

**文件：**
- 创建：`tests/generation/test_completion_finalizer.py`
- 创建：`wfcllm/generation/completion_finalizer.py`

- [ ] **步骤 1：编写失败测试**

```python
from wfcllm.generation.completion_finalizer import finalize_humaneval_program


PROMPT = 'def add(a, b):\n    """Return the sum."""\n'


def test_trims_incomplete_tests_after_complete_target() -> None:
    source = PROMPT + "    return a + b\n\nprint(add(1, 2"
    result = finalize_humaneval_program(PROMPT, source)
    assert result.applied is True
    assert result.code == 'def add(a, b):\n    """Return the sum."""\n    return a + b\n'
    compile(result.code, "<finalized>", "exec")


def test_does_not_accept_prompt_only_stub() -> None:
    result = finalize_humaneval_program(PROMPT, PROMPT + "print(")
    assert result.applied is False
    assert result.code == PROMPT + "print("
    assert result.reason == "no_complete_target_implementation"
```

- [ ] **步骤 2：运行红灯测试**

运行：

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_completion_finalizer.py -q
```

预期：collection FAIL，`ModuleNotFoundError: wfcllm.generation.completion_finalizer`。

- [ ] **步骤 3：实现最小收口器**

```python
@dataclass(frozen=True)
class ProgramFinalizationResult:
    code: str
    applied: bool
    reason: str


def finalize_humaneval_program(prompt: str, source: str) -> ProgramFinalizationResult:
    target_name = _target_name_from_prompt(prompt)
    for prefix in _longest_line_prefixes(source, minimum_chars=len(prompt)):
        try:
            tree = ast.parse(prefix)
        except SyntaxError:
            continue
        target_index, target = _find_target(tree, target_name)
        if not _has_completion_statement(target, prompt):
            continue
        code = ast.unparse(ast.Module(body=tree.body[: target_index + 1], type_ignores=[]))
        return ProgramFinalizationResult(code.rstrip() + "\n", True, "target_function_complete")
    return ProgramFinalizationResult(source, False, "no_complete_target_implementation")
```

- [ ] **步骤 4：运行绿灯测试**

运行同一步骤 2。预期：全部 PASS。

- [ ] **步骤 5：补充并通过边界测试**

增加：完整源码带顶层测试、Markdown 中文解释、prompt/source 不匹配、装饰函数、嵌套 helper、目标函数后新函数。预期全部 PASS。

- [ ] **步骤 6：提交**

```bash
git add wfcllm/generation/completion_finalizer.py tests/generation/test_completion_finalizer.py
git commit -m "feat: finalize HumanEval target functions"
```

### 任务 2：生成管线集成与 provenance（TDD）

**文件：**
- 修改：`wfcllm/generation/gated_pipeline.py`
- 修改：`tests/generation/test_gated_pipeline.py`

- [ ] **步骤 1：编写失败测试**

```python
def test_finalizer_runs_before_watermark_generator(tmp_path: Path) -> None:
    observed = {}

    def finalizer(prompt: str, source: str):
        return ProgramFinalizationResult("def f():\n    return 1\n", True, "ok")

    class CapturingGenerator(_Generator):
        def generate(self, **kwargs):
            observed.update(kwargs)
            return super().generate(**kwargs)

    pipeline = GatedGenerationPipeline(
        config=_config(tmp_path),
        bundle_loader=lambda _p: _bundle(),
        bundle_hasher=lambda _p: _digest("bundle"),
        base_model=_BaseModel("def f():\n    return 1\nprint("),
        generator=CapturingGenerator(),
        data_adapter=[{"id": "HumanEval/0", "prompt": "def f():\n"}],
        deployment_key=b"key",
        program_finalizer=finalizer,
        program_finalizer_name="humaneval_target_function_v1",
    )
    pipeline.run()
    assert observed["original"] == "def f():\n    return 1\n"
```

- [ ] **步骤 2：运行指定测试确认红灯**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_gated_pipeline.py::test_finalizer_runs_before_watermark_generator -q
```

预期：FAIL，constructor 不接受 `program_finalizer`。

- [ ] **步骤 3：实现最少管线集成**

为 constructor 增加 `program_finalizer` 和 `program_finalizer_name`，在 `_generate_base_program` 后调用，并收集 finalizer rows。用 `write_generation_sidecar_rows` 写 `generation/finalizer.jsonl`；manifest 增加名称、applied/fallback 计数。

- [ ] **步骤 4：验证严格 schema 与 fallback**

新增断言：final input 仍只有四字段；fallback 样本仍进入输出；sidecar 不含部署密钥或隐藏测试；manifest 计数与 164 固定分母一致。

- [ ] **步骤 5：运行 generation 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation/test_completion_finalizer.py tests/generation/test_gated_pipeline.py -q
```

预期：全部 PASS。

- [ ] **步骤 6：提交**

```bash
git add wfcllm/generation/gated_pipeline.py tests/generation/test_gated_pipeline.py
git commit -m "feat: finalize programs before watermark embedding"
```

### 任务 3：生产 wiring 与配置（TDD）

**文件：**
- 修改：`wfcllm/cli/runners.py`
- 修改：`configs/wfcllm/gated_semantic_window_v1.json`
- 修改：`tests/integration/test_gated_fast_mainline.py`

- [ ] **步骤 1：编写失败测试**

测试 `_build_local_gated_generation_pipeline` 在 `generation.program_finalizer=humaneval_target_function_v1` 时向管线传入 callable，未知名称时报 `ValueError`。

- [ ] **步骤 2：验证红灯**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/integration/test_gated_fast_mainline.py -q
```

预期：新增测试 FAIL，当前 runner 未解析 finalizer。

- [ ] **步骤 3：实现配置 wiring**

```python
finalizer_name = str(generation.get("program_finalizer", "none"))
if finalizer_name == "humaneval_target_function_v1":
    program_finalizer = finalize_humaneval_program
elif finalizer_name == "none":
    program_finalizer = None
else:
    raise ValueError("unsupported generation program_finalizer")
```

将 callable 与名称传入 `GatedGenerationPipeline`。配置改为：

```json
{
  "max_new_tokens": 512,
  "temperature": 0.0,
  "program_finalizer": "humaneval_target_function_v1"
}
```

- [ ] **步骤 4：运行 targeted 测试**

```bash
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/generation tests/integration/test_gated_fast_mainline.py -q
```

预期：全部 PASS。

- [ ] **步骤 5：运行静态与广泛回归**

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts
HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/method tests/generation tests/detection tests/integration -q
```

预期：全部 PASS；若资源缺失，记录首个错误和受影响文件，不伪报全通过。

- [ ] **步骤 6：提交**

```bash
git add wfcllm/cli/runners.py configs/wfcllm/gated_semantic_window_v1.json tests/integration/test_gated_fast_mainline.py
git commit -m "fix: complete HumanEval programs before embedding"
```

### 任务 4：16 题预检与 full164 attempt2

**文件：**
- 创建：固定实验根目录 `run-full164-pass-attempt2.sh`
- 创建：`full164-attempt2/`、`full164_state.attempt2.json`、日志与 provenance。

- [ ] **步骤 1：服务器预检**

检查 SSH、screen、进程、GPU 显存低于 500 MiB、磁盘至少 20 GB、既有 attempt1 哈希不变。

- [ ] **步骤 2：运行 16 题隔离预检**

使用相同 seed/config 与独立目录，只检查：16/16 ID、语法有效率、finalizer sidecar、无 Traceback。不得根据 HumanEval 测试结果选单题候选。

- [ ] **步骤 3：验收预检**

语法有效率低于 90% 时，只允许在已批准边界内将 `max_new_tokens` 调至 768；不允许多候选、隐藏测试选择或 canonical solution。

- [ ] **步骤 4：启动 full164 attempt2**

复用既有 gate-data/gate-train bundle，强制生成 164 条；随后运行 calibration、detect、report。screen 名 `wfcllm-full164-pass-a2`，全程 tee 到固定日志。

- [ ] **步骤 5：执行真实 Pass@1**

对 attempt2 `final_code.jsonl` 运行官方 HumanEval 测试，每题 10 秒超时，分母固定 164，并保存逐题结果。

- [ ] **步骤 6：最终验收**

断言 164 个唯一 ID、严格四字段、语法有效率至少 90%、Pass@1 至少 30%、检出率至少 85%、无弃权、target FPR 与最少窗口未变。写 attempt2 报告、SHA-256 清单和成功/失败标记。

- [ ] **步骤 7：结果比较**

报告 attempt1 → attempt2：语法有效率、Pass@1、检测率、通过且检出率，以及所有未达标项。不得把 calibration-negative AUROC 表述为独立测试 AUROC。

