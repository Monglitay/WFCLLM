# WFCLLM Gate-only Full Reproduction Core

本仓库只维护 WFCLLM 的 Gated Lineage。当前唯一方法是
`gated_semantic_window_v1`，唯一公开实验形态是从头训练语义 encoder 与
Gate 的 Full Reproduction Profile。退出当前范围的实现、配置、运行产物和
文档只从 Git 历史与既有 Archival Tag 恢复。

## Reproduction Coverage

公开覆盖固定为五个组合：

| Wrapper | 语言 | 数据集 | Canonical dataset size | Current diagnostic limit | 窗口合同 | 改写策略 |
|---|---|---|---:|---:|---|---|
| `run_python_humaneval_full.sh` | Python | HumanEval | 164 | 100 | `python-statement-window/v1` | `python_ast_equivalent` |
| `run_python_mbpp_full.sh` | Python | MBPP test | 500 | 100 | `python-statement-window/v1` | `python_ast_equivalent` |
| `run_cpp_humanevalpack_full.sh` | C++ | HumanEvalPack | 164 | 100 | `cpp-statement-window/v1` | `cpp_structural_equivalent` |
| `run_java_humanevalpack_full.sh` | Java | HumanEvalPack | 164 | 100 | `java-statement-window/v1` | `java_structural_equivalent` |
| `run_js_humanevalpack_full.sh` | JavaScript | HumanEvalPack | 164 | 100 | `js-statement-window/v1` | `js_structural_equivalent` |

五个配置位于 `configs/wfcllm/experiments/`。生成模型和 rewrite 模型由运行时
路径注入，不按模型复制配置。

当前诊断与迁移实验固定使用 100 条样本；完整正样本、校准负样本和独立负样本
都保留在分母中。开发容量统计中的 `17 × 18 = 306` 是 authored candidate
trials，不是 benchmark 样本数。当前保留的最好完整 run、指标与配置 SHA-256
见 [`docs/BEST_RESULTS.md`](docs/BEST_RESULTS.md) 和
[`docs/BEST_RESULTS.json`](docs/BEST_RESULTS.json)。

本地和服务器的旧运行树已移到仓库外的 cleanup 目录，最佳 run 与必要资源仍在
服务器的 `fresh-runs/` 中；清单见
[`docs/CLEANUP_MANIFEST_20261025.json`](docs/CLEANUP_MANIFEST_20261025.json) 和
[`docs/SERVER_CLEANUP_MANIFEST_20261025.json`](docs/SERVER_CLEANUP_MANIFEST_20261025.json)。
cleanup 目录只用于人工恢复，不是受支持的执行输入。

MBPP profile 默认使用 `mbpp_chat`：生成器通过模型的聊天模板请求完整程序，
再保留公开接口及其所需辅助声明。校准与独立负样本须使用相同的生成设置和
finalizer；标准测试只在生成后执行。

C++ 结构改写维持固定 18 次预算，保留前 12 次表达式候选。尾部尝试先提供互不
重叠表达式的组合；当基本类型局部变量的声明及全部引用均在窗口内时，最多用
6 个局部改名候选替换尾部尝试。宏参数、嵌套绑定、限定名称及窗口外引用使改名
不可用。候选顺序只由公开源码决定，仍须通过结构、语义余弦、精度、批次稳定性
和 margin 守卫；改名不授予语义等价证书。

Java 返回值和初始化表达式的布尔候选按表达式 AST 的根运算符识别。字符串、
数值三元和移位表达式不会因包含比较字符而被识别为布尔结果。

Gate 数据的语义探针按解析语句窗口的精确 UTF-8 区间评分，与在线生成、检测一致。
候选原文保留；语句外的尾注释不会改变该窗口的签名。Gate cache 额外保存评分区间
和文本 hash，Gate 数据 manifest 的 selection summary 绑定该审计文件的 hash。

## Fresh Reproduction Run

完整阶段顺序固定为：

```text
encoder → gate-data → gate-train → generate → calibrate → detect → report → audit
```

运行目录和状态文件必须不存在。当前执行路径不恢复旧状态、不接受外部历史
Gate Bundle，也不读取旧 schema 或旧报告布局。pilot 与 full Gate source
catalog 必须是两个不同文件。

在已经配置好的 `WFCLLM` conda 环境中准备本地资源：

```bash
export GENERATION_MODEL_PATH=/absolute/path/to/generation-model
# C++、Java、JavaScript 必须设置；Python AST 改写不需要。
export REWRITE_MODEL_PATH=/absolute/path/to/rewrite-model
export SEMANTIC_ENCODER_MODEL_PATH=/absolute/path/to/semantic-encoder-base
export GATE_BASE_MODEL_PATH=/absolute/path/to/gate-base-model
export DATASET_PATH=/absolute/path/to/local-datasets
export PILOT_SOURCE_CATALOG=/absolute/path/to/pilot-sources.jsonl
export FULL_SOURCE_CATALOG=/absolute/path/to/full-sources.jsonl
export NEGATIVE_INPUT=/absolute/path/to/held-out-negative-final-code.jsonl
export EXPERIMENT_ROOT=/absolute/path/to/a-new-run-root
```

实际运行示例：

```bash
bash scripts/experiments/run_python_humaneval_full.sh
```

缺少模型、数据集、source catalog、held-out negative input 或私有密钥时，
preflight/runner 会非零退出。Public Execution Surface 不会用 canonical
solution、简化检测器或测试后端替代真实资源。

## 统一 CLI

`run.py` 是 `wfcllm.cli.entry:main` 的薄入口。可单阶段运行当前链中的一个
阶段，但前置阶段必须已在同一 Fresh Reproduction Run 的状态文件中完成：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
conda run -n WFCLLM python run.py \
  --phase encoder \
  --config configs/wfcllm/experiments/python_humaneval_full.json \
  --run-dir /absolute/path/to/new-root/run \
  --state-file /absolute/path/to/new-root/run_state.json \
  --gate-source-catalog /absolute/path/to/full-sources.jsonl \
  --semantic-encoder-model-path /absolute/path/to/semantic-encoder-base
```

查看当前八阶段状态：

```bash
conda run -n WFCLLM python run.py \
  --status \
  --state-file /tmp/wfcllm-status.json
```

## Detector Input

唯一正式正样本检测输入是 `run/inputs/final_code.jsonl`，每行必须且只能包含：

```json
{"id":"HumanEval/0","dataset":"humaneval","prompt":"...","final_code":"..."}
```

generation audit、candidate sidecar、Gate 数据、pass/test 结果都不是检测输入。

## Calibration 与生成后评估

`NEGATIVE_INPUT` 是严格四字段的 held-out 未加水印生成。若条数低于当前
`target_negative_count`，`calibrate` 使用同一 generation model 在与正样本
不重叠的提示上补足；生成即采用，不使用 pass/test/正确性信号筛选。

当前 posthoc correctness 工具只支持 Pass@1，且只在生成完成后计算；
报告与 Metric Contract 只消费 Pass@1：

C++/HumanEvalPack 的公共脚本会编译并执行数据集原测试，使用 C++11、
OpenSSL 的 crypto/ssl 库及 Boost 头文件，编译和执行合计超时为 60 秒。
离线 Boost 头文件可用 `WFCLLM_CPP_INCLUDE_DIR` 指定；预检会检查真实工具链。
单独从生成代码计算时，向 `scripts/evaluate.py` 传入 `--language cpp`
和 `--dataset humanevalpack`。正确性 harness 的公共头文件前缀及生成 `main`
清理只用于运行原测试，检测输入仍使用原始 `inputs/final_code.jsonl`。

Java、JavaScript/HumanEvalPack 的公共脚本同样在生成后执行数据集原测试，
分别要求本地 `javac`/`java` 和 `node`。单独评估时使用 `--language java`
或 `--language js`、`--dataset humanevalpack` 与 `--generated-input`。
Java 编译与运行共享 10 秒超时；JavaScript 运行超时为 10 秒，非空标准输出
或标准错误按原评测约定判失败。语法错误、错误答案和超时均保留在完整分母中。
Java 的空 `Main` 占位清理仅发生于后评测 harness，不改写 detector input。

Java、JavaScript 使用 `humanevalpack_chat` 生成协议，通过本地 tokenizer
chat template 请求完整实现。只拆出首个相应语言代码围栏；不修正语法、函数名、
计算逻辑或打印语句。原始响应、实际模型输入及其 token 保存在 generation
sidecar，不能进入 detector。校准和独立负样本必须使用相同生成协议。

JavaScript 使用固定 18 项的表达式结构改写，不调用模型改写窗口。语句分隔、正则、
模板、多行和存在歧义的表达式不作推断；候选仍须通过作用域、结构、语义余弦、
精度、批次稳定性和 margin 守卫，不携带语义等价证书。
其 encoder 的四视图训练正样本使用公开等价变体，按源程序拆分训练、验证与测试集。

Java 使用 `java_structural_equivalent`，按固定 18 项轨迹构造括号、常量条件及
多语句表达式组合，不添加尾部注释，也不颁发语义等价证书。候选须通过同样的
结构、语义余弦、精度、批次稳定性与 margin 守卫。

C++ 的 semantic projection 使用 `wfcllm-public-region-cosine-balanced-all-planes/v4`：
训练正样本余弦下界为 0.90，语义项权重为 8，原型项权重为 2，训练 plane margin
为 0.005。四视图损失约束全部 12 个公开签名平面，原型项仍约束前 4 个平面。
Python、Java、JavaScript 仍使用 4 个训练平面。运行时余弦阈值仍为 0.90；
训练参数和平面数写入本轮 encoder provenance 与 manifest。
C++ 和 Java 的 semantic projection 使用正样本余弦下界 0.90、语义项权重 8、
原型项权重 2、训练 plane margin 0.005；Java 的 objective version 为
`wfcllm-public-region-cosine-balanced/v3`。Python、JavaScript 仍使用原始
`wfcllm-public-region-prototype/v2` 训练目标。
C++ 的训练视图来自其结构改写器的实际 1–3 语句评分窗口，保留完整源程序上下文，
按规范化文本去重；encoder provenance 记录 `cpp-runtime-scored-structural/v1`。

```bash
conda run -n WFCLLM python scripts/evaluate.py \
  /absolute/path/to/posthoc-correctness.jsonl \
  --output /absolute/path/to/run/reports/pass_report_posthoc.json
```

`--k` 与 Pass@k 已退出当前公开 surface。任何正确性结果都不得反馈到
generation、retry、candidate selection、calibration 或 detection。

为一个或多个当前 run 生成 Metric Contract JSONL：

```bash
conda run -n WFCLLM python scripts/wfcllm_metric_contract.py \
  --run-dir /absolute/path/to/run \
  --output /absolute/path/to/metric-contract.jsonl
```

Metric Contract 只读取已通过 audit 且核心 artifact SHA-256 未变化的当前
run，并无条件输出一行；未得到的 Pass@1 等指标记为 `null` 并附 caveat，
不因指标未达标而抑制输出。

## 离线验证

```bash
conda run -n WFCLLM python -m compileall wfcllm run.py scripts tests
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest --collect-only -q
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  conda run -n WFCLLM python -m pytest tests/ -q
bash -n scripts/experiments/run_gated_experiment.sh \
  scripts/experiments/run_*_full.sh
```

这些命令验证代码、合同和受控测试，不等于完成真实 GPU 科学复现。

## 文档

- `docs/WFCLLM_METHOD.md`：当前 Gate 方法。
- `docs/REPRODUCTION_ENVIRONMENT.md`：本地资源与运行环境。
- `docs/ARTIFACT_SCHEMA.md`：当前运行目录和 schema。
- `docs/STRICT_CODE_ONLY_DETECTOR.md`：严格四字段检测合同。
- `docs/NO_QUALITY_GATE_PROTOCOL.md`：正确性信号的因果隔离。
- `CONTEXT.md`：领域词汇。
- `docs/adr/`：仍然有效的架构决定。
