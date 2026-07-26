# Gated Lineage 合并分析（阶段 1 书面结论）

日期：2026-07-26。依据 spec `2026-07-26-consolidate-gated-reproduction-design.md` 阶段 1 要求产出。
本文只记结论与理由，不改代码。逐文件证据由六个并行只读分析（generation / detection / gate / cli / windowing+lang+datasets / method+semantic+configs+scripts）交叉产出。

## 基线事实

- 落点 HEAD 代码面 = `6e94ab2`（main 领先的 2 个提交仅动 README）。`0d29949` = `6e94ab2` + JS 支持提交。
- 四个 Archival Tag 已建立（`archive/serverA-{nocarrier,opencoder,cpp-basic-run,gated}-20260726`），父提交均为 `0d29949`，内容与来源逐字节核验一致（git 无法表示的 5 个空目录除外）。
- **cpp-basic-run 态基于 pre-JS 树（6e94ab2）**：其对 JS 相关文件的全部 D/M 与 `6e94ab2` 逐字节一致，是血统假象，不是刻意移除。一律不采纳其删除。
- **gated 态整体陈旧**：其全部 M/D 文件的 blob 均命中主线祖先历史提交（54b7221 / 19d16f4 / d329479 / e5b5b33 等），零独有内容。四个簇全部丢弃，归档 tag 即其保存形式。
- **nocarrier 即 MBPP 目标实跑代码**：MBPP formal run 的 `metadata/source_snapshot` 12 个文件与 nocarrier 态逐字节一致。
- 启动脚本证实目标↔工作区映射：JS/MBPP → nocarrier；OpenCoder → opencoder worktree；Java/C++×2 → cpp-basic-run；gamma25 驱动脚本导入的接口四态中仅 gated 不完整（缺 `KeyBlindAstEquivalentWindowRewriter`），实际宿主为 nocarrier 系代码面。
- nocarrier/opencoder 有大段逐字节相同的共同改动（曾互相同步），之后在 `window_rewriter.py` 真分叉。
- 三个传言中的「cpp-basic-run 独有模块目录」（`wfcllm/ablation`、`wfcllm/extract`、`wfcllm/common/transform`）经服务器端核实为**空目录**（0 个文件），无可合并内容。
- `__sync_tmp__` 散落补丁全部为旧中间态或与树内文件逐字节相同（`java_fast/` 是已同步完成的 staging 目录），整体不合并。唯一遗留疑问：`__sync_tmp__/cpp_humanevalpack_fast.json` 与树内参数不同（lsh_d 12/γ0.85 vs 2/γ0.75），C++ basic 目标实跑用哪套，凭其 `reference_report.json` 的 `lsh_config_sha256` 在实施时核对。
- `*.before-*` 手工快照（66 个）一律不进主线，归档 tag 已保存。

## 关键取舍决策（按决策阶梯）

**D1 exact window text（detection 真分叉）**。nocarrier 于 07-24 引入 `window_text` 精确源文切片（UTF-8 逐字节），旧语义为 `"\n".join(unit.text)`。按 run 完成时间，JS(07-23)/OpenCoder/Java/C++×2 均在 join 语义下产出，仅 MBPP formal(07-24) 在精确切片下产出。→ 保留两种语义为配置项（`exact_window_text`），**默认 false（join，覆盖 5 目标）**，MBPP 五个实验配置显式 true。阶梯规则 2。

**D2 校准产物 `null_hit_probability` 兼容**。opencoder 给 `GatedCalibrationArtifact` 增必填字段且 `from_payload` 精确匹配，会拒载历史校准 JSON。→ `from_payload` 对缺失该字段宽容为 None，schema 版本不动。理由：不为合并强迫重校准历史目标。

**D3 descriptor 祖先去重（cpp 真分叉）**。cpp 态在 `gate/data.py`、`windowing/partitioner.py`、`gated_generator.py`、`window_rewriter.py` 四处引入 `_descriptor_ancestor_node_types`（剥离与 direct_parent 重复的祖先尾项），改变 canonical_parent → 允许 LSH 区域。cpp/java 三目标在去重语义下产出；python/js 四目标在无去重语义下产出。→ **按 window contract version 键控**：cpp/java 契约启用去重，python/js v1 契约不启用。零配置即可同时复现全部目标；语言与历史行为的对应恰好完备。阶梯规则 2 的「保留差异点」变体，选择键控而非新增配置键，因该差异与语言契约一一对应。

**D4 Python 重写档位（nocarrier/opencoder 真分叉）**。window_rewriter 上 N（预算 12 + comprehension-alpha + `_wrap_window_source`）与 O（预算 48、无 alpha、`_compound_header_scaffold` 支持 elif）互斥。→ 文件基座取 cpp 版（含全部 Python 核心 + cpp/java 重写器），Python 核心采 N 实现并把 O 的 elif 处理并入 wrap；预算做成配置（默认 12 = N 档，覆盖多数目标），alpha 可关。opencoder 三个配置显式 48 + 关 alpha。阶梯规则 2。

**D5 generator.py 注册表**。N/O 同回退了基点 bc419af 的 checkpoint 注册表并加 MBPP 支持；`tests/generation/test_gated_replay.py`（27 用例，四态未动）断言注册表内部结构。→ 以基点（含注册表）为基座，仅嫁接 MBPP 纯增量函数。风险：服务器实跑为无注册表版；若 MBPP 复现比对失败，备选方案为整取 N 版并重写 replay 测试。

**D6 fast_experimental 治理**。nocarrier 硬编码 `fast_experimental = True` 并跳过全部 manifest 校验，是服务器临时 hack；cpp 有配置驱动 `_uses_fast_experimental_gate_training(config)` + experimental/formal 双分支。→ 取 cpp 治理骨架，nocarrier 复现配置写入对应 fast_experimental 键。同时弃用 nocarrier 对 `_require_formal_manifest` 的提前放行（与 NO_QUALITY_GATE 治理相悖）；若 nocarrier 复现在 gate-train 校验处失败，回退方案为恢复提前放行并记录。

**D7 运行时窗口上限统一**。opencoder `max_units=`（硬编码 1..3 + int 强转）与 nocarrier `max_units_override=`（上限取 bundle、可选）为同一意图两个阶段。→ 取 nocarrier 语义 + 补 int 强转。

**D8 校准/检测 CLI 收紧**。N∩O 逐字节相同的「calibrate 强制 `--negative-input`、detect 强制 `--input`」保留（阶段 3 的 ADR 0008 将另行改造 calibrate，不在合并期动）。

**D9 gated_pipeline 三方合并**。以 opencoder 重构后的 calibrate 骨架（先收集 scores + count-statistic 分支）为底，并入 nocarrier 的 quantile 分组、`_inclusive_empirical_threshold` 与规则选择链；decision 三分支（quantile 只比阈值 / count-statistic 用 observed_statistic / 其余 p-value+hit_rate）。五种校准分组由 `calibration_group_by` 配置互斥选择，语义可叠加。

**D10 scoring.py 叠加**。opencoder 逐 channel 循环骨架 + nocarrier `allow_unsuitable_windows` 正交参数，默认值均为基线行为（1 / False）。

**D11 gate/production.py**。opencoder 是 nocarrier 的严格早期阶段（trust_remote_code 三处相同、MBPP 辅助函数逐字节相同），取 nocarrier + cpp 两组正交增量叠加。`validation.py` 取 N=O 一致的 `_WORKER_POLL_TIMEOUT_SECONDS=180`。

**D12 run_gated_experiment.sh**。以 cpp 版为底（basic 回退、conda 探测、fast 跳 audit），补回 0d29949 JS 提交的两处（prepare 调用 env 前缀、full gate-train `--pilot-feasibility`）。

**D13 根级文件**。README/CLAUDE.md/AGENTS.md/MANIFEST.md/requirements.txt/run.py/pytest.ini 全部保留 HEAD 版（cpp 态 README 与 HEAD 逐字节相同；其余各态无 HEAD 之外信息）。

## 合并落地顺序（阶段 2 提交序列）

1. **恢复 JS 支持提交**：把 `6e94ab2→0d29949` 的 delta 整体落回（lang/js、multilanguage JS extractor、contracts、humanevalpack js、cli 三文件 JS 词条、gate/sources `oss_js`、detection js 契约行、js 配置与脚本）。
2. **windowing/datasets**：cpp 增量（`skip_before`、partitioner descriptor 键控去重、humanevalpack `_local_jsonl_split` 保留 js 元组）→ nocarrier 增量（defer 开关、mbpp.py、mbpp_interface.py、loaders/local）+ 双方测试并集。cpp 增量一律以 `6e94ab2` 为 diff 基准取补丁。
3. **generation**：C 基座（多语言重写器 + 规则常量化 + fast-certified 通道 + descriptor 键控）→ N Python 核心（wrap+alpha）→ O 增量（evidence_channels + 重写档位配置化）→ N MBPP 链路（finalizer+pipeline+generator 嫁接，同批）。
4. **gate**：N 组（production generator 段 + validation 180 + 测试）→ C 组（pipeline/data/production adapter 段 + 测试）。
5. **method/semantic 原子批**：config.py + presets.py + window_lsh.py `score_channels` + `gated_semantic_window_v1.json` + 两个测试（evidence_channels 必填锁步）。
6. **detection**：code_only → gated_windows（N 版 + D1/D7）→ scoring（D10）→ gated_pipeline（D9/D2）→ `__init__` 导出 → 测试并集。
7. **cli**：runners.py 分层叠加（共同块 → cpp 治理骨架 → nocarrier 特性 → opencoder evidence_channels）+ preflight（N=O 版）+ 集成测试。
8. **orchestration/prereq.py**（cpp 版，依赖 runners 的 `_uses_fast_experimental_gate_training`）。
9. **configs + scripts**：opencoder 3 配置 → nocarrier 5 MBPP 配置 + js fast M 版（补 fast_experimental/exact_window_text/档位键）→ cpp/java fast 配置 → `run_basic_multilanguage_experiment.py` + 集成测试 → `run_gated_experiment.sh` 合成版。
10. 阶段 2 验证门槛：至少一个数据集的完整八阶段链跑通（用仓库既有 fake/受控后端集成测试路径验证接线，另跑 `tests/` 相关套件）。

## 遗留核对项（实施期完成）

- C++ basic 目标 lsh 参数版本：以其 `reference_report.json` 的 `lsh_config_sha256` 对两套候选参数分别算哈希核对。
- D6 的 R1：合并后 fast_experimental 模式 gate-train 是否写全四个 experimental 标记（`experimental_only/diagnostic_only/not_official_method/formal_eligible`）。
- D3 键控去重落地后，跑 nocarrier 目标 detect 路径回归，确认 python/js 契约下 canonical_parent 不变。
- 主工作区未提交的 generator.py 增量经比对与 gated 谱系同源（MBPP 支持移植），同一内容经 nocarrier 态进入合并，主工作区文件保持不动。
