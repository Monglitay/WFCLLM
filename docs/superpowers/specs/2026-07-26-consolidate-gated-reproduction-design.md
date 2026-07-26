# Consolidate Gated Lineage Reproduction

## Problem Statement

我的论文实现与实验散在三处：本地 `PycharmProjects/WFCLLM`、服务器 A（AutoDL，`/root/autodl-tmp`）、服务器 B。我原本以为它们是同一份仓库的版本漂移，只需要合并；实际盘点后发现不是。

服务器 A 上不是一份仓库，而是 8 份目录副本，对应 4 个互不包含的 Code State。我最想复现的 HumanEval C++ / Java / JavaScript、MBPP、以及换模型对比，全部出自 Gated Lineage —— 不是我以为的 `codex/server-sync-js`。后者的 C++ 与 Java 语言适配器只有一行占位 docstring，HumanEvalPack 数据集适配器直接抛 `NotImplementedError`，物理上跑不出多语言。

更要紧的是，产出这些结果的代码有相当一部分**不在任何仓库里**：JavaScript 那条链的四个评测脚本共 872 行只存在于 run 目录；Python 那条链的驱动脚本、配置、输入行全在 `/tmp` —— 而 `/tmp` 挂在容器 overlay 根盘上，重启即失。同时 `reference_report.json` 不记录数据集、语言与模型，一份报告无法自证身份；每种语言的 TPR / pass / AUROC 提取路径各自为政，我想要的那张跨数据集指标表拼不出来。

AutoDL 实例随时可能回收，端口每次重启都变。我需要在它消失之前，把「能复现这些数字」这件事从一台临时机器上解绑，收敛成本地一份、结构清楚、没有第二个真相之源。

## Solution

在本地仓库里建立唯一真相之源：

- 把四个 Code State 以不可变 Archival Tag 固定在共同基点之上，忠实保留当时磁盘上的样子（含手工快照文件），历史真相与开发主线职责分离。
- 把只存在于 run 目录与易失盘上的评测代码收进主线并去掉硬编码。
- 建立统一 Metric Contract：输入一个 run 目录，输出一行自证身份的标准化指标记录（数据集、语言、模型、口径参数、TPR、pass、AUROC、Key Identifier、结果资格），使那张跨数据集的表可以一次生成；并让未来的报告自己记录身份。
- 产出可移植的环境备齐清单，使复现不再绑定在服务器 A 上。
- 实验产物、Gate Bundle、模型权重、输入语料一律不取 —— 凡代码能再生成的都不留；密钥不可再生成，取回但放在仓库之外，仓库只记 Key Identifier。

验收口径不是字节级一致，而是：**拿着收敛后的仓库，能跑出与七个 Reproduction Target 差不多的每数据集 TPR / pass / AUROC。** 若某个数字漂得离谱，当作代码逻辑缺陷来查，而不是补档二进制。

整理工作在从 `main` 拉出的独立分支上进行，完工后正常 merge 回 `main`，不改写远端历史、不 squash 中间步骤。

## User Stories

1. 作为论文作者，我想在本地仓库里找到唯一一份权威代码，这样我不必再判断三台机器上哪一份才是真的。
2. 作为论文作者，我想对每个 Reproduction Target 都能指出一个可 checkout 的 git ref，这样我知道那批数字是哪份代码跑出来的。
3. 作为论文作者，我想让 Archival Tag 忠实保留当时工作区的原样（包括手工快照文件），这样归档不会因为「清理」而失真。
4. 作为论文作者，我想让开发主线不含那些手工快照文件，这样日常工作的仓库是干净的。
5. 作为论文作者，我想一条命令生成跨数据集的指标表，这样我不必为每种语言手工拼数字。
6. 作为论文作者，我想让指标表的每一行都记录自己的口径参数，这样我不会把不可横比的行误当作可横比。
7. 作为论文作者，我想让指标表的每一行都标出结果资格（Formal Eligible / Diagnostic Only / 是否带 Experimental Bypass），这样我不会把诊断性结果当作方法结论写进论文。
8. 作为论文作者，我想让 Conditional TPR 与全样本 TPR 在表里明确区分，这样我不会把筛过样本的 TPR 与未筛的并列比较。
9. 作为论文作者，我想让报告自己记录数据集、语言与模型，这样以后一份报告脱离目录名也能自证身份。
10. 作为论文作者，我想让那些只存在于 run 目录里的评测脚本进入主线，这样它们不会随实例回收消失。
11. 作为论文作者，我想让那些只存在于易失盘上的驱动脚本、配置与输入行被抢救进仓库，这样 Python 那条目标不会在下次重启后无从复现。
12. 作为论文作者，我想让评测脚本里写死的阈值、模型名与绕过声明改为从 run 产物读取或从配置注入，这样报告内容不再由脚本作者手写断言。
13. 作为论文作者，我想让报告生成器无条件输出、达标与否只作为字段记录，这样不会出现「达标才写文件」的选择性报告。
14. 作为论文作者，我想拿到一份可执行的环境备齐清单（解释器与关键依赖版本、模型标识与 revision、数据集 fingerprint、离线环境变量），这样我换一台机器也能重建运行环境。
15. 作为论文作者，我想让模型权重与数据集靠重新下载获得而不是入库，这样仓库不背几十 GB 的二进制。
16. 作为论文作者，我想让 Gate Bundle 靠重跑 gate-train 得到而不是归档，这样我不为可再生成的产物付存储代价。
17. 作为论文作者，我想让那些跳过自己 gate-train、直接复用别的 run 输出的目标改为各自跑完 gate-train，这样它们不再互相绑定。
18. 作为论文作者，我想让 Deployment Key 与两组 key bank 被取回但放在仓库之外，这样复现能力保住而密钥不进版本库。
19. 作为论文作者，我想在仓库里看到每个 Reproduction Target 对应的 Key Identifier 与读取路径约定，这样我知道该拿哪把钥匙、从哪注入。
20. 作为论文作者，我想在整理过程中完全不扰动本地那批未提交的 Semantic-V3 改动，这样另一条实验线的工作不被替我决定收口方式。
21. 作为论文作者，我想让整个抢救与收集过程对服务器零写入，这样不会破坏还没盘清的原始现场。
22. 作为论文作者，我想让整理成果最终落在 `main` 上，这样唯一真相之源就是默认分支。
23. 作为论文作者，我想让整理过程的中间步骤在 `main` 的历史里可见，这样日后出问题能二分定位。
24. 作为第三方复现者，我想拿到仓库后按备齐清单准备环境、按指标表逐行重跑，这样我能独立验证论文里的数字。
25. 作为第三方复现者，我想知道哪些结果是 Diagnostic Only，这样我不会误以为它们已经通过全部准入门槛。
26. 作为审稿人，我想看到每个数字的口径（`target_fpr`、`minimum_reliable_windows`、是否筛样本、是否事后重生成样本），这样我能判断跨行比较是否成立。
27. 作为未来接手的 agent，我想在仓库里读到领域词汇表与架构决策记录，这样我不必重新盘一遍服务器就能理解术语与既定选择。
28. 作为未来接手的 agent，我想在一处读到七个 Reproduction Target 的完整位置、四个 Code State 的归属、以及三处密钥位置，这样这些不可再生的事实不随对话消失。
29. 作为未来接手的 agent，我想知道哪些已知效力问题尚未处理（诊断资格、事后重生成样本比例、改写选中数为零、某个 Gate 的 Suitable False Positive Rate 远超上限），这样我不会把它们当作已解决。
30. 作为未来接手的 agent，我想知道抢救工作到哪一步中断了、已经落地了什么，这样我不会重复劳动也不会遗漏。

## Implementation Decisions

### 要实现的流水线阶段序列

目标序列为八阶段，对每个数据集各走一遍：

```
每个数据集 ∈ { HumanEval, MBPP, HumanEvalPack-C++ / Java / JavaScript }

  encoder ──► gate-data ──► gate-train ──► Gate Bundle
                                              │
                                              ▼
  generate ──► calibrate ──► detect ──► report ──► audit

  ＋ pass 率评测（独立路径，不是阶段）
  ＋ Metric Contract 提取（每个 run 输出一行）
```

相对历史实现的三处改动：

- **`encoder` 从可选阶段提升为必经阶段，且每个数据集各训一个**（见 ADR 0006）。历史实现里它根本不在实验链上。
- **`gate-validate` 整体去掉**（见 ADR 0007）。手稿中不存在该准入闸门；随之删除的还有 float 与量化一致性检查、[[Suitable False Positive Rate]] 准入上限、gate 数据 feasibility 准入，以及正式合格与仅供诊断的 bundle 发布二分。
- **`calibrate` 自主补充负样本**（见 ADR 0008）。历史实现要求外部预制负样本语料并通过参数传入。

其余阶段的性质：

- `gate-data` 在历史编排脚本里被调用多次（含全量规模），产出分组索引、切分与 manifest。
- `calibrate` 是独立阶段且在 `detect` 之前：阈值由负样本校准得出，不是检测器内部行为。负样本语料一变，`detect` 的数字必然跟着变。
- pass 率评测不是阶段，走独立的 HumanEval 执行摘要路径；事后 pass 报告虽属可选阶段，但只能作为事后效用报告并携带完整标记。

### calibrate 自主补充负样本

- `calibrate` 须能在负样本语料不足或缺失时，自行从同一生成模型产出未加水印的补全来补齐，而不是依赖外部预制文件。手稿的表述正是「在留出的未加水印生成上按目标假阳率校准」。
- 目标负样本数由配置给出。补齐后的语料须记录来源构成：外部提供多少条、自动生成多少条、生成所用模型与解码配置。该构成进入校准产物，使阈值可追溯。
- 负样本必须与正样本评测集不重叠 —— 「留出」是硬要求，不是建议。
- 补充过程不得使用 pass / 测试 / 正确性代理做筛选，这是仓库对校准阶段的既有禁令。生成即采用。
- 现有的负样本语料指纹字段继续保留并覆盖补齐后的完整语料，使校准产物与其负样本一一对应。

### semantic encoder 改为每数据集一个

- 历史上七个目标共用同一个 encoder checkpoint。它由一个独立脚本训练，不经 `encoder` 阶段（故各 run 的该阶段状态均为未完成），训练语料是一份通用 local-only 源码目录，不属于任何 benchmark 数据集 —— 该 encoder 从未见过 C++、Java 或 JavaScript。
- 我们改为每个数据集各训一个 encoder（见 ADR 0006）。训练器本身无需改动：它的入口参数就是源码 catalog，而 catalog 本来按语言分。
- `encoder` 须从可选阶段提升为 gated 实验链的必经阶段；编排脚本需要增加该步。
- 受影响行的重跑数字与历史值的差异是方法改变导致的系统性变化，不是随机漂移；「跑出差不多的数字」这一口径对它们不再适用。
- 实施时须确认按语言切分后的语料是否足够：某语言的 gate 源目录仅数百条，而 encoder 训练器默认的训练组上限为一千二百组。

### 范围与血脉

- 本轮只收敛 Gated Lineage。Semantic-V3 Lineage 的 v3 语义实验推到后续轮次（见 ADR 0005）。
- Gated Lineage 的基点与 `main` 共享祖先，`main` 仅领先该祖先两个提交 —— 它是 `main` 的兄弟而非异种分叉。让 `main` 成为收敛版本不需要历史手术（见 ADR 0004）。
- 服务器 B 的处置不在本轮。

### Code State 与 Archival Tag

- 四个 Code State 在共同基点上并列为四个不可变 Archival Tag，各自固定一个 Code State 的原样内容（见 ADR 0001）。
- Archival Tag 保留 Gated Lineage 工作区里 66 个手工快照文件；开发主线不含它们。
- Archival Tag 不进入 `main` 的历史，不作为开发起点。
- 其中一个 Code State 在服务器上完全没有 git 目录，只能作为文件内容归档；其语言适配器已逐字节比对确认与另一 Code State 相同，但整体树不同，因此仍需独立成 tag。

### Metric Contract（第一条缝）

- 新增评测侧模块，暴露一个提取入口：输入一个 run 目录，输出一条 Metric Contract 记录。
- 记录字段至少包含：run 标识、数据集、语言、生成模型、样本数、`target_fpr`、`minimum_reliable_windows`、TPR（并标明是否为 Conditional TPR 及其资格规则）、pass 率、AUROC、Key Identifier、结果资格三元组（Formal Eligible / Diagnostic Only / Experimental Only）、是否带 Experimental Bypass、以及一个自由文本的口径警示列表。
- 身份（数据集、语言、模型）必须能从 run 产物自行推导，不得依赖报告里的新字段 —— 七个历史目标的报告已固化、无法回填。可用证据包括 gate 数据 manifest 里的解析器契约版本、source manifest、以及启动脚本里的环境变量。
- 该提取器读取 gate 数据 manifest 以推导身份，这不违反「检测器绝不读 gate-data」的规则：它是事后报告工具，不在检测路径上。实施时须保证它不成为检测器输入的泄漏通道 —— 它只读、只产出报告，绝不回写任何检测器可见的产物。
- 记录还须包含 semantic encoder 标识，因为该标识参与检测器身份契约，且每数据集一个 encoder 之后它逐行不同（见 ADR 0006）。
- 四个 JavaScript 评测脚本折进这一条缝，不各自成缝。它们写死的绕过声明、目标阈值、重生成模型名与模式，改为从 run 产物读取或从配置注入。
- 报告生成无条件输出，达标与否只作为字段记录，不得以未达标为由拒绝写出（见 ADR 0003）。
- pass 率复用仓库里已有的 HumanEval 执行摘要能力，不把易失盘上的 pass@10 脚本当作新代码搬入。
- 三种语言历史上的 AUROC 算法不同，统一必须选定一个定义；因此部分历史 AUROC 数字不会完全吻合，这落在验收容差内。

### 报告身份回填（第二条缝）

- CLI 运行侧的报告输出增加数据集、语言、生成模型三项身份字段，使未来的 run 自证身份。
- 该改动只对未来的 run 生效，对七个历史目标零收益，因此不得成为 Metric Contract 提取器的前置依赖。

### 取什么、不取什么

- 实验产物一律不取。Gate Bundle 一个不取，靠重跑 gate-train 生成（见 ADR 0002）。模型权重不取，改记模型标识与 revision。输入语料不取，靠仓库里的准备脚本加数据集 fingerprint 再生成。
- 有几个目标的启动脚本跳过了自己的 gate-train、直接指向别的 run 的输出目录；复现这些目标时必须让它们各自跑完 gate-train。
- Deployment Key 与两组 key bank 不可由代码生成，必须取回；放在仓库之外的本地目录（限制权限），仓库只记每个目标对应的 Key Identifier 与读取路径约定。运行时通过既有的密钥文件/环境变量注入通道传入 —— 代码本就设计为从外部注入，这条路径零改造。
- 服务器上共三处密钥位置，均需取回。

### 传输与隔离

- 从服务器取物全程只读：git 抓取走 ssh 只读通道，工作区代码走 rsync，均不在服务器上产生写入。服务器持久卷剩余空间已不足，任何服务器端打包都不可接受。
- 优先抢救易失盘上的内容 —— 那里等待的代价不可逆。
- 本地在独立 worktree 中作业，完全不扰动当前工作区那批未提交的 Semantic-V3 改动。
- 顺手把只存在于服务器上的一个 Semantic-V3 提交抓回本地作为保险；它属于后续轮次，但零成本且不做就是永久丢失。

### 环境备齐

- 产出可执行的备齐清单，记录：解释器与关键依赖的确切版本、Gate 基座模型与生成模型的标识与 revision、语义编码器基座、HumanEvalPack 数据集的 fingerprint、以及把缓存与环境重定向到本地路径的离线环境变量集合。
- 假定服务器 A 会消失；备齐清单必须能在一台新机器上重建运行环境。

## Testing Decisions

好的测试只断言外部可观察行为 —— 输入一个 run 目录、断言输出记录的字段与数值；不断言内部函数调用顺序、不断言私有中间结构。

### 缝一：Metric Contract 提取器

- 在临时目录中构造最小合成 run 目录（一份最小报告、一段检测曲线、一份 gate 数据 manifest、一份评测摘要），调用提取入口，断言输出记录的字段完整性与数值。
- 覆盖场景：全样本 TPR 与 Conditional TPR 两种口径分别被正确标注；缺少 AUROC 来源时的行为；结果资格三元组与 Experimental Bypass 被如实透传；身份推导在只有产物、没有报告身份字段时仍成立；口径警示列表在样本被事后重生成时被填充。
- 明确断言无条件输出：即使指标低于任何目标阈值，记录也必须被写出。
- 先例：`tests/integration/` 下 25 个脚本 CLI 测试的写法 —— import 脚本模块、对纯函数喂小输入、加一个在临时目录写文件并断言带版本号输出 schema 的端到端测试。

### 缝二：报告身份回填

- 断言运行侧产出的报告包含数据集、语言、生成模型三项字段且取值正确。
- 断言该字段缺失时提取器仍能工作（向后兼容七个历史目标）。
- 先例：`tests/integration/` 下既有的 detect / generate CLI 测试。

### 缝三：calibrate 的负样本补齐

- 断言外部语料条数低于目标值时补齐被触发、达到目标值时不被触发。
- 断言补齐后的来源构成被如实记录（外部条数、生成条数、模型与解码配置），且负样本语料指纹覆盖补齐后的完整语料。
- 断言补齐产出的负样本与正样本评测集不重叠。
- 断言补齐过程不读取任何 pass / 测试 / 正确性信号 —— 以缺失这些信号的输入跑通即可证明。
- 生成模型在测试中以桩替代，遵循仓库既有的模型边界打桩惯例。
- 先例：`tests/detection/test_gated_calibration.py` 与 `tests/integration/` 下的 detect CLI 测试。

### 移除 gate-validate 的回归防护

- 断言八阶段序列里不再出现该阶段，且以该阶段名调用时被拒绝。
- 断言 `generate` / `calibrate` / `detect` 直接从 `gate-train` 的候选 bundle 取用，且三者取到的是同一个 bundle。
- 断言历史产物里残留的「门控验证已跳过」「未验证候选 bundle」两个标记不再由新 run 产生。
- 先例：`tests/integration/test_orchestration.py`、`tests/integration/test_cli_resolution.py`。原 `tests/gate/test_validation.py` 整体删除。

### 复用的既有缝

- pass 率能力已有测试覆盖，不重复造；只在提取器侧断言它被正确调用与结果被正确读入。

### 语言适配器

- 若归档过程暴露出语言适配器的缺陷，按 `tests/lang/` 既有先例补测（该目录已有 Python 与 JavaScript 适配器测试与注册表测试）。

### 不用测试验证的部分

- Archival Tag 是 git 操作，靠人工核对 tag 指向的树内容与来源逐字节一致来验证。
- 环境备齐清单是配置与 shell，靠在一台干净机器上实际执行来验证。

## Out of Scope

- 不重跑任何实验。本轮交付的是「能跑出差不多数字」的能力，不是那次重跑本身。
- 不处理 Semantic-V3 Lineage 的收口，包括本地那批未提交的 v3 语义改动。
- 不处理服务器 B 与本地另一条独立实验线的双向分叉。
- 不清理本地那 38 个指向同一提交的垃圾分支，也不清理服务器上 43 个指向本地路径的失效 worktree 元数据。
- 不修复已知的结果效力问题：多语言目标的诊断资格、JavaScript 目标中过半样本被事后用另一模型重生成、其改写选中数为零、以及某个被复用的 Gate 的 Suitable False Positive Rate 远超准入上限。这些如实记录进指标表的资格与警示字段，处置留待后续轮次。
- 不改动服务器上任何文件。
- 不把密钥以任何形式（含加密）放入版本库。
- 不重构 Gate 训练与检测的核心算法。

## Further Notes

### 七个 Reproduction Target

以服务器 A 上的位置标识（这些是数据而非仓库结构，是不可再生的事实，故在此记录）：

| 数据集 | 生成模型 | run 位置（`/root/autodl-tmp` 下） | 口径 | 资格 |
| --- | --- | --- | --- | --- |
| HumanEvalPack-JS full164 | deepseek，164 中 84 个事后用 7b 重生成 | `wfcllm-experiments/js-full-chain-20260723/fast-fullgate-js-full164/run` | `target_fpr` 0.31 / `minimum_reliable_windows` 1 | Diagnostic Only + Experimental Bypass |
| HumanEval-Python full164 | OpenCoder-8B | `wfcllm-runs/opencoder-8b-multichannel2-zscore-external-full164-20260724-013000` | 0.05 / 2 | 待复核 |
| MBPP full interface-aware | 待确认 | `wfcllm-experiments/mbpp-full-interface-aware-gated-semantic-20260724-goal` | 0.05 / 2 | 七者中唯一带完整溯源 metadata |
| HumanEvalPack-Java full164 | deepseek-1.3b | `wfcllm-experiments/java-full-chain-20260723/fast-gated-java-full164` | 0.05 / 2 | Diagnostic Only |
| HumanEvalPack-C++ full164 | deepseek-1.3b | `wfcllm-experiments/cpp-full-chain-20260722/fast-gated-cpp-equiv-d2g075-full164-20260723` | 0.05 / 2 | Diagnostic Only |
| C++ basic full | 待确认 | `wfcllm-experiments/cpp-basic-full-20260722/run` | 待查 | 无 metadata |
| HumanEval-Python gamma25 | deepseek-1.3b | `/tmp/gamma25_gate_pass113` | Conditional TPR（`reliable_window_count >= 3` 筛除 34/164，`hit_rate >= 0.6` 判检出） | 待复核 |

JavaScript 目标的目标摘要文件已核对 SHA-256 一致。Python gamma25 目标报的是 pass@1 与 pass@10 两个值，以及留出负样本 FPR。

### 四个 Code State 与目标归属

- Gated Lineage 主工作区（有 git，逾百处未提交改动，含 66 个手工快照文件）→ JavaScript 与 MBPP 两个目标。
- 其上的换模型 worktree（有 git，二十余处未提交改动）→ OpenCoder-8B 目标。
- 一个**完全没有 git 目录**的副本 → Java、C++ full164、C++ basic 三个目标。它独有三个模块目录，与主工作区相差七十余处。
- 一个更早期的副本（语言适配器仍全为占位）→ Python gamma25 目标所复用的 Gate 来源。此归属尚需最终确认：其启动脚本经由环境变量间接引用仓库路径，且该目录在服务器上触发 git 的 dubious ownership 保护，绕过需要写入 git 配置，故未处理。

### 易失盘暴露

Python gamma25 目标的驱动脚本、配置、输入行、pass@10 补全脚本全部位于容器 overlay 根盘上的 `/tmp`，重启即失。该目录另有约百个分析脚本与约百个 patch 文件，以及三个同族对照实验变体。排除一个约 300 MB 的无关二进制后，需抢救量约 110 MB。

驱动脚本导入仓库的私有接口（三个带前导下划线的运行时函数）并硬编码改写上限；它当时导入的是哪份包，脚本本身没有记录。

### 抢救进度

只读抢救已启动过一次并被中断，落地 79 个文件约 1.2 MB 在本地仓库之外的暂存目录，**尚未覆盖目标目录本身**。服务器全程零写入。续跑时应从头覆盖。

### 环境规格

解释器 Python 3.11.15；torch 2.11.0+cu130（CUDA 13.0）；transformers 4.46.3；numpy 2.4.2；scipy 1.17.1；datasets 4.6.1；accelerate 1.14.0；tokenizers 0.20.3；tree-sitter 0.25.2 及 cpp 0.23.4 / java 0.23.5 / python 0.25.0 —— **JavaScript 的 tree-sitter 语法包未安装**，而 JavaScript 解析器有两百行实现，其解析依赖待查。

conda 环境以 prefix 形式存放在数据卷上；环境变量已把 HuggingFace、pip、conda、torch 的缓存全部重定向到数据卷并开启全面离线。

生成模型与基座：deepseek-coder-1.3b-instruct、deepseek-coder-7b-instruct、OpenCoder-1.5B-Instruct、OpenCoder-8B-Instruct、codet5-base、codet5-small、qwen2.5-coder-0.5b，合计约 33 GB，一律不入库、改记标识与 revision。HumanEvalPack 数据集的 HuggingFace fingerprint 已记录。

GPU 型号未能确认 —— 非交互 shell 下 `nvidia-smi` 权限被拒。

### 访问方式

服务器 A 为 AutoDL 实例，SSH 端口每次重启都会变化；本地 SSH 配置里记录的端口已过期。持久卷使用率 88%（剩余约 11 GB），因此禁止任何服务器端打包或克隆。

### 工作分支的 CLAUDE.md 落后于 main

被 rsync 到本地的那条血脉分支落后 `main` 三十余个提交，其 `CLAUDE.md` 因此缺了五组正好管着本轮工作的规则：gated 预设属实验性且不得基于 fake/offline 接线测试声称有效；gated 八阶段序列与「fake backend 绝不可发布 formal validated bundle」；检测器只读官方四字段输入、已验证 bundle 与冻结校准产物，绝不读 gate-data、checkpoint、生成审计行或候选 sidecar；gated 真跑所需的私有 key bank 与 runtime deployment key 要求，以及不得把密钥材料写入 config、run state、日志、manifest 或报告；git 卫生上不得提交 gate-data、key bank、已验证 bundle、生成的 JSONL 与 run 目录内容。

实施须以 `main` 的完整规则为准，而不是工作副本里那份陈旧版本。本轮的取物策略已对照核过：密钥出库、产物不取、Gate Bundle 不取三项与上述规则一致。

### 已落地的文档

领域词汇表与八条架构决策记录已在本轮写入仓库：词汇表覆盖水印密钥、Gate、结果资格、三副本收敛四组共 21 个术语；决策记录覆盖 Archival Tag 表达方式、不归档 Gate Bundle、统一 Metric Contract、收敛落在 `main`、本轮范围与诊断资格、semantic encoder 改为每数据集一个、去掉 gate-validate、以及 calibrate 自主补充负样本。本 spec 不重复其内容。

### 已量出的 gate-validate 移除影响面

可整块删除约 2100 行：门控验证模块本体约 1039 行、门控流水线里的验证入口与其发布 manifest 版本常量、门控配置里的四个准入阈值数据类、CLI 侧的阶段入口。

需同步修改的注册与编排点六处：编排状态模块的 gate 阶段清单、编排前置检查、CLI 入口注册、配置解析器、门控包导出、CLI 分派表。

方法配置是硬约束：方法配置模块把门控验证段当作必需段并强制其存在且取固定值，预设里也带着该段，方法产物模块定了它的产物路径契约 —— 因此所有 gated 配置都要去掉这一段。

最麻烦的一处是 bundle 解析路径。现在 `generate` / `calibrate` / `detect` 经由「解析已验证 bundle」入口取 bundle，并按配置在「未验证候选」与「已验证」两个运行时 bundle 类之间二选一，同时往输出打两个标记。移除后须让该入口直接解析 `gate-train` 的候选 bundle，二选一分支收敛为一条，两个运行时类合一，两个标记连带删除；生成侧那条「实验性」与「未验证候选」互斥断言随之消失。

审计侧检查门控验证发布 schema 的那处需要移除。三个 shell 脚本与一个门控 CLI 脚本引用了该阶段。测试侧十八个文件涉及相关概念，其中门控验证专测整体删除，其余修改断言。
