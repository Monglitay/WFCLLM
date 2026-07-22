# WFCLLM 可复现门控语义窗口水印设计

- 日期：2026-07-15
- 状态：用户已批准书面规格；实现计划已生成；尚未授权进入代码实施
- 目标分支：`codex/gated-semantic-window-watermark`
- 交付方式：仅在本地分支实现；不 push、不创建 PR，不提交模型、数据集、
  checkpoint、运行日志或大体积实验产物

## 1. 摘要

本设计用一个小型代码编码模型替代当前无法从最终代码严格恢复的动态代理窗口。
同一个门模型同时在生成端和检测端运行，并只读取最终代码中能够重新获得的信息。
门模型从左到右形成互不重叠、最多包含三个 Python 语句单元的窗口，分别输出：

1. 当前窗口是否应当结束；
2. 已结束窗口是否值得尝试语义 LSH 嵌入。

一个窗口是一次完整水印载体。如果原始窗口没有可靠进入由密钥和父节点信息派生的
允许 LSH 区域，生成端回退到窗口起点并重新生成整个窗口，而不是只重写最后一个
语句。重写后的窗口可以改变内部语句数量和结构，但必须保持同一父级作用域、仍由
1～3 个语句单元组成，并被门模型稳定恢复为一个完整窗口。

密钥（例如字符串 `"0101"`）是生成允许 LSH 区域的秘密种子，不是逐窗口嵌入的
显式位串。本设计不训练“可靠 0 / 可靠 1”分类器。门模型不读取密钥、LSH 命中、
重试记录或候选日志；它学习的是与具体部署密钥无关的窗口边界和稳定可重写性。

本设计将门模型的数据生成、训练和稳定性验收加入主编排。完整的 gated 工作流是：

```text
gate-data
  -> gate-train
  -> gate-validate
  -> generate
  -> calibrate
  -> detect
  -> report
  -> audit
```

现有正式 preset 和五阶段流程继续保留作为基线。新 gated preset 才启用前三个门模型
阶段，或引用一个已经通过 `gate-validate` 的不可变门模型 bundle 后从 `generate`
开始。

## 2. 现状与核心冲突

当前 main 已经具备以下能力：

- `wfcllm.generation` 在 token 生成过程中识别结构边界、保存回退状态、执行重试，
  并把最终代码与候选 sidecar 分开保存；
- `wfcllm.semantic` 负责语义编码、密钥派生、LSH 签名、有效签名集合与 margin；
- `wfcllm.detection` 只读取严格的最终代码输入，并在最终代码上构造固定代理窗口；
- `wfcllm.audit` 强制执行 detector-input 隔离和 no-quality-gate 协议；
- `wfcllm.diagnostics` 保存不能作为正式方法证据的选择器和上界分析；
- `wfcllm.orchestration` 维护阶段注册、依赖、运行状态和 skip/force 行为。

当前最重要的冲突是：生成端窗口由动态生成状态、回退和重试决定；检测端看不到这些
状态，只能枚举固定的最终代码代理窗口。因此检测器不能声称恢复生成端实际使用的
窗口。

新方法必须满足更强的合同：

> 给定同一份最终代码、同一解析规则、同一门模型 bundle 和同一公开推理配置，
> 生成端接受的窗口必须能够由检测端从最终代码重新得到；生成 sidecar 不得参与。

## 3. 目标与非目标

### 3.1 目标

- 为 Python 定义生成端和检测端共享的语句单元与窗口划分规则；
- 使用同一个小门模型生成互不重叠的可复现窗口；
- 对未命中窗口执行整个窗口的语义重写；
- 保留当前“密钥 + 父节点信息 -> 允许 LSH 区域”的语义水印机制；
- 门模型和改写模型都不读取部署密钥；
- 将门模型训练数据生成、模型训练和稳定性验收纳入整体编排；
- 保持 `inputs/final_code.jsonl` 为唯一正式正样本 detector input；
- 保留现有方法作为独立可复现实验基线；
- 在进入门模型训练前先验证整体窗口重写和 LSH 本身是否可行。

### 3.2 非目标

- 不设计显式的逐窗口 0/1 载荷；
- 不让门模型预测密钥、LSH 签名或具体允许区域；
- 不让门模型决定动态重试次数；
- 不使用生成日志、候选、隐藏状态或 logits 进行检测；
- 不在正式生成或候选选择中使用测试通过、执行结果、正确性标签或质量代理；
- 不证明任意 Python 重写在数学上功能等价；
- 第一版不支持跨作用域窗口、超过三个语句的窗口或跨语言窗口；
- 第一版不把函数定义开头或类定义开头作为载体；
- 不删除或覆盖现有正式 preset、代理窗口检测器和诊断方法。

## 4. No-Quality-Gate 合同

`CLAUDE.md` 与 `docs/NO_QUALITY_GATE_PROTOCOL.md` 禁止把 pass、test、correctness
及其代理作为生成、重试、候选选择、校准或检测的控制信号。本设计选择保持这一正式
协议，而不是静默改变它。

因此，正式门模型训练标签只能由以下允许信号产生：

- 解析器确认的结构边界；
- 候选是否仍是同一父级作用域中的 1～3 个完整语句单元；
- 与具体部署密钥无关的多训练密钥 LSH 命中统计；
- LSH margin；
- batch、精度、量化和设备下的数值稳定性；
- 重写预算、候选顺序和回退成本；
- 公开、固定、与正确性无关的长度与结构约束。

正式门模型标签、阈值和候选选择不得读取或派生自：

- 单元测试、benchmark 测试或隐藏测试；
- 运行时执行结果；
- pass@k；
- 正确性判定；
- LLM 质量评分；
- 语义等价 oracle；
- 任何命名或结构上充当 quality/correctness/reward 的代理。

功能正确率仍然必须测量，但只能在正式生成、检测和选择完成后运行，并写入带完整
posthoc marker 的报告。功能正确率是本方法是否值得继续研究和是否可以晋升为正式
方法的停止条件，不是单个候选的在线选择信号。

如果将来需要“运行时正确性门控”版本，它必须作为违反当前 no-quality-gate 合同的
独立诊断方法，带 `diagnostic_only=true` 和 `not_official_method=true`，不能与本
规格的正式结果混报。

## 5. 术语

### 5.1 语句单元

由共享 Python 解析器产生的最小候选载体单位。括号内部表达式不是独立语句单元。

### 5.2 开放窗口

从当前未提交起点到最新完整语句单元的连续序列。开放窗口尚未成为正式水印载体。

### 5.3 已关闭窗口

门模型或硬结构规则已经确定结束位置的连续、非重叠语句序列。

### 5.4 候选 0

基础模型第一次正常生成的原始窗口。候选 0 始终先检查，不计入额外重写预算。

### 5.5 重写候选

从窗口起点回退后，由与密钥无关的固定采样过程重新生成的完整窗口版本，按生成顺序
编号为候选 1～N。

### 5.6 可靠命中

候选的 LSH 签名处于由密钥与父节点描述派生的允许区域，且 margin 和规定的数值
稳定性检查均通过。

### 5.7 门模型 bundle

不可变的门模型权重、tokenizer、输入规范、输出阈值、量化合同、解析合同版本和内容
哈希。生成和检测必须引用同一个 bundle。

## 6. Python 语句单元合同

### 6.1 基础划分

共享解析器必须正确处理：

- 普通简单语句；
- 使用分号连接的多个简单语句；
- 字符串、注释和其他非语句分隔位置中的分号；
- 单行 `if` 及其内联 body；
- 多行括号表达式；
- `if/elif/else`；
- `for/while`；
- `try/except/else/finally`；
- `with`；
- `match/case`；
- `async`、`await`、`yield`；
- 多层缩进与父级作用域；
- Tree-sitter 或解析器恢复节点，并把不可靠恢复片段标为不可承载。

例如：

```python
x = 1; y = 2
```

产生两个语句单元，而：

```python
result = process("a;b", value)
```

仍然只有一个语句单元。

对于：

```python
if condition: x = 1; y = 2
```

产生一个复合语句开头单元和两个内联 body 单元。复合开头与 body 属于不同父级上下文，
不能进入同一个窗口。

注释、空行和说明文字不产生语句单元，但保留在最终代码中。

### 6.2 第一版排除语句

以下语句仍然进入上下文，但不能成为水印窗口的一部分，并形成硬分隔点：

- `pass`；
- `break`；
- `continue`；
- `raise`；
- `import`；
- `from ... import ...`；
- `global`；
- `nonlocal`；
- `del`；
- `assert`；
- 函数定义开头；
- 类定义开头；
- 解析器恢复且边界不确定的片段。

`return` 不排除。

### 6.3 允许但保守处理的语句

普通赋值、属性/下标赋值、原地修改、函数调用、嵌套调用、`return`、`yield`、
`yield from`、`await` 和复合语句开头均可进入候选集合。允许进入候选集合不代表门模型
必须把它们判为适用。

复合语句开头第一版只能形成单语句窗口。它不能与子 body 共同组成窗口，也不能在重写
时改变外部控制角色，例如 `if` 不能改成 `for`。

## 7. 窗口合同

### 7.1 不变量

一个正式窗口必须：

- 由 1～3 个连续语句单元组成；
- 与其他正式窗口不重叠；
- 不跨函数、直接父节点、缩进层级或硬分隔点；
- 使用最终代码中可恢复的起点和终点；
- 在生成与检测的规定推理配置中得到相同跨度；
- 对复合语句开头遵守单语句窗口限制。

### 7.2 在线关闭

门模型在每个完整语句单元后运行。输入窗口长度小于 3 时，它可以输出继续或结束；
达到 3 时强制结束。父级变化、函数结束和硬分隔点也强制结束。

如果边界输出处于不确定区，固定回退规则是：立即关闭当前窗口并把该窗口标为
`boundary_uncertain`；它不允许承载正式水印。该规则保证流程前进，并避免用微小浮点
差异强行决定证据窗口。

### 7.3 整体重写

未命中时必须回退到窗口第一个语句之前，重新生成整个窗口。最终重写可以把三条语句
改成两条，或把一条改成两三条，但仍须满足全部窗口不变量。

接受重写候选之前，系统必须从同一前文状态重新运行解析器和门模型。候选只有在以下
条件全部成立时才可继续进行 LSH 检查：

- 产生恰好一个可恢复窗口；
- 内部没有被门模型提前关闭；
- 最后一个语句后稳定关闭；
- 最终窗口仍被适用性头接受；
- 不跨父级或硬分隔点；
- 候选生成没有读取密钥或之前的命中失败原因。

这些是结构和水印稳定性检查，不是功能正确性检查。

## 8. 父节点描述与密钥派生

部署密钥（例如 `"0101"`）是秘密种子，而不是水印位串。对于每个最终窗口，系统构造
公开、可重算的父节点描述：

```text
language_contract_version
+ ancestor_node_type_path
+ direct_parent_node_type
+ first_unit_direct_child_ordinal
+ compound_header_role_if_any
```

不得加入 prompt、task ID、生成 seed、候选编号、重试次数、生成日志窗口 ID 或运行
设备。

允许区域由现有密钥派生机制计算：

```text
allowed_region = HMAC_or_existing_key_derivation(
    secret_key,
    canonical_parent_descriptor,
)
```

候选通过现有语义编码与 LSH 规则得到签名。可靠命中要求有效集合成员关系和 margin
同时通过。

父节点描述必须按接受后的最终代码重新计算。当前窗口语句数量改变后，后续语句的直接
子节点 ordinal 也随最终代码更新；检测端从最终代码得到相同值。

## 9. 门模型合同

### 9.1 输入

每次推理输入包含：

- 当前开放窗口的 1～3 个语句单元；
- 当前窗口之前最近最多 3 个解析器语句单元；
- 每个单元的公开类型；
- 当前缩进层级；
- 直接父节点类型和祖先类型路径；
- 当前窗口语句数；
- 当前窗口 token 数；
- 第一语句的直接子节点 ordinal；
- 解析合同版本与规范化版本。

前文按解析器语句取得，而不是按前一个门模型窗口取得，减少边界误差的级联输入变化。

门模型不得读取：

- 部署密钥或训练密钥；
- LSH 签名、允许区域、margin 或命中结果；
- 目标位；
- 候选编号和重试次数；
- 生成日志、sidecar、hidden states 或生成模型 logits；
- 测试、执行、正确性、质量或 reward 字段。

### 9.2 输出

同一个 encoder 前向计算产生两个独立标量：

- `close_probability`：当前窗口是否结束；
- `suitable_probability`：若当前窗口结束，是否值得执行候选 0 检查和额外重写。

第一版不输出目标位、预计重试次数、正确性风险、候选质量或具体 LSH 命中概率。

### 9.3 规模与运行方式

第一版微调已有的 30M～80M 代码预训练 encoder，目标配置约 60M 参数，不从零训练。
门模型和语义 LSH encoder 使用独立权重与独立 bundle 标识，避免一方更新静默改变另一方。

正式推理必须：

- 关闭 dropout 和所有训练随机性；
- 固定 tokenizer、padding、truncation 和最大输入长度；
- 固定 pooling；
- 使用 bundle 声明的正式精度/量化格式；
- 验证输入规范与模型 bundle 的内容哈希；
- 对模型版本不匹配返回错误，不静默回退。

第一版门输入上限固定为 512 个门模型 tokenizer token。规范输入超过 512 token 时不
截断当前窗口来强行预测，而是执行 `close_and_skip`；该窗口不作为正式水印证据。

### 9.4 阈值与不确定区

`gate-train` 在训练验证集上拟合模型，不决定正式阈值。`gate-validate` 在独立验证 split
上冻结：

- close 的低阈值与高阈值；
- suitable 的接受阈值；
- 最大输入 token；
- 正式量化配置；
- 允许的 float/量化输出偏差；
- 稳定窗口一致性合同。

close 输出位于低阈值与高阈值之间时视为边界不确定。suitable 未达到接受阈值时一律
跳过。阈值冻结后不得在检测测试集上调整。

## 10. 门模型数据生成阶段（`gate-data`）

### 10.1 职责

`gate-data` 负责产生可分组、可追溯、与部署密钥隔离的门模型训练数据，不训练模型，
也不产生正式水印检测结论。

### 10.2 数据来源

数据按以下来源组合：

- 真实开源 Python 项目，作为主要来源；
- MBPP 的训练 split；
- 当前 main 已有生成结果，排除与最终评测重叠的任务；
- 至少三个不同代码生成模型产生的代码；
- 人工构造的解析边界与危险语法，仅用于合同和困难集。

HumanEval 全集默认保留为最终评测，不进入门模型训练、阈值选择或训练密钥分析。
如未来改变这一政策，则不得继续把 HumanEval 报告为未见测试集。

真实项目按 repository 分组。生成任务按 task 分组。函数、同一起点的长度 1～3 窗口、
前文 1～3 版本、候选 0～6、精度变体和预算变体必须保持在同一个 split。

### 10.3 候选窗口枚举

对同一作用域中的每个允许起点，枚举：

```text
W1 = [s1]
W2 = [s1, s2]
W3 = [s1, s2, s3]
```

遇到父级变化、复合开头限制或硬分隔点时停止扩展。W1、W2、W3 是不同候选载体，
各自产生自己的重写候选池。

对固定窗口 W，只生成一条有序轨迹：

```text
candidate_index = 0: 原始窗口
candidate_index = 1..6: 六个按实际生成顺序保存的整体重写
```

重写提示始终使用同一个规范上下文：当前窗口、最近最多三个前文语句、父级结构和
“生成结构兼容的替代表达”要求。提示、sampling 参数和随机 seed 计划不读取任何密钥。

候选生成后不得按命中、margin 或其他结果重新排序。

### 10.4 公平比较

固定窗口 W 的前文 1、2、3 输入版本共享完全相同的候选轨迹和标签。重写预算使用同一
轨迹的前缀：

```text
R=1: candidate 0 + candidate 1
R=3: candidate 0 + candidate 1..3
R=6: candidate 0 + candidate 1..6
```

候选 0 不计入 `rewrite_budget`。必须同时记录：

- `original_hit`；
- `rewrite_budget`；
- `actual_rewrites_generated`；
- `evaluated_candidate_count`；
- `first_reliable_hit_index`；
- `first_reliable_hit_source`。

正式第一版以 R=3 产生主要标签；R=1 和 R=6 只用于成本曲线与消融。

### 10.5 训练密钥银行

`gate-data` 使用多个训练专用随机密钥评估一个候选池在不同允许区域下的稳定可重写性。
部署密钥和最终评测密钥不得进入该银行。

公开 manifest 只保存：

- 训练密钥银行 ID；
- 密钥数量；
- 生成算法版本；
- 不可逆内容摘要。

秘密密钥值保存在本地 secret store 或进程输入中，不写入公开 JSON/JSONL、日志、spec
或 Git。

### 10.6 正式数据字段

`gate-data/window_groups.jsonl` 每行表示一个不可拆分的数据组，至少包含：

```text
group_id
source_kind
source_id
repository_or_task_group
function_group
language
parser_contract_version
parent_descriptor_without_secret
preceding_units_up_to_3
candidate_window_lengths_available
split
```

`gate-data/candidate_attempts.jsonl` 每行表示一个候选，至少包含：

```text
group_id
window_length
candidate_index
candidate_code
unit_count
parse_status
same_parent_scope
boundary_span
lsh_signature_by_training_key_id
margin_by_training_key_id
stable_across_precision_modes
stable_across_batch_modes
reliable_hit_by_training_key_id
generation_seed_id
rewrite_config_id
```

这些是训练 sidecar，不是 detector input。它们默认保存在本地 run 目录，不提交 Git。

### 10.7 标签生成

对每个固定窗口和训练密钥，按候选顺序计算在 R=1、3、6 内是否出现可靠命中。汇总多
训练密钥后得到：

- `reliable_success_rate_r1`；
- `reliable_success_rate_r3`；
- `reliable_success_rate_r6`；
- `unstable_candidate_rate_r3`；
- `expected_first_hit_index_r3`；
- `structurally_valid_rewrite_rate_r3`。

`suitable_target` 由冻结的数据配置阈值把 R=3 的多密钥成功率、结构有效率与不稳定率
转换得到。该阈值只使用 gate-data 的训练/开发部分，不查看最终检测结果。

第一版固定使用 32 个训练密钥，并采用以下标签合同：

```text
suitable_target = true，当且仅当：
  reliable_success_rate_r3 >= 0.60
  且 structurally_valid_rewrite_rate_r3 >= 2/3
  且 unstable_candidate_rate_r3 <= 0.10
```

这里 `reliable_success_rate_r3` 按“候选 0 加前三个重写是否至少有一个可靠命中”在
32 个训练密钥上求比例；结构有效率和不稳定率只统计三个额外重写，避免原始窗口天然
有效把重写能力虚高。阈值或训练密钥数发生变化时必须产生新的 label contract ID 和
gate-data dataset ID，不能原地覆盖旧标签。`gate-validate` 另用 8 个未参与标签生成的
holdout 训练域密钥检查 key 泛化；部署密钥仍与两组完全隔离。

`close_target` 使用因果规则生成：

1. 当前窗口达到硬结构边界或长度 3 时，目标为 close；
2. 当前窗口未到硬边界且已达到 suitable 训练阈值时，目标为 close；
3. 当前窗口未达到 suitable 阈值且长度小于 3 时，目标为 continue。

这避免用未来尚未生成的语句为当前输入制造不可学习的边界标签。close 与 suitable 使用
不同输出头；在长度 3 或硬边界上可能出现 `close=true, suitable=false`。

### 10.8 数据量与可行性批次

第一批只收集 2,000～5,000 个独立窗口起点，用于判断：

- 结构有效候选是否足够；
- R=1、3、6 的收益曲线；
- 多训练密钥下正负标签比例；
- 不同窗口长度和语句类型的覆盖；
- LSH 稳定性是否已经成为主瓶颈。

只有 `gate-data` 可行性报告通过后，才扩展到 20,000～50,000 个独立窗口起点。
派生的前文、预算、候选和精度行不计入独立样本数。

## 11. 门模型训练阶段（`gate-train`）

### 11.1 输入与前置条件

`gate-train` 只接受：

- 通过 schema 校验的 gate-data manifest；
- 固定 split manifest；
- 冻结的解析和输入规范；
- 预训练 encoder 的本地路径和内容哈希；
- 训练配置；
- 标签配置版本。

如果 group 跨 split、HumanEval 污染、密钥泄漏、候选顺序缺失或 schema 不兼容，阶段
必须失败，不能自动修复后继续。

### 11.2 模型与目标

一个共享 encoder 加两个输出头：

```text
encoder(input)
  -> close head
  -> suitable head
```

第一版训练目标包括：

1. close 二分类损失；
2. suitable 二分类损失；
3. 同一窗口前文 1～3 版本的决策一致性损失；
4. 浮点教师与正式量化学生的一致性损失；
5. 对 suitable 假阳性施加高于假阴性的代价。

不加入：

- 重试次数回归；
- 正确性或质量风险回归；
- 0/1 标签；
- LSH 签名预测；
- 候选排序；
- 强化学习；
- 测试结果监督。

### 11.3 batch 不变性

同一输入在不同 batch 中变化首先按工程缺陷处理：检查 eval 模式、padding、attention
mask、pooling、未初始化状态和非确定算子。只有在实现正确后仍存在可测数值差异时，
才由一致性目标和不确定区吸收。训练不得把明显的 batch 实现 bug 当作数据增强。

### 11.4 输出 bundle

`gate-train` 产生候选 bundle，但不把它标为可用于正式生成。候选 bundle 包含：

- encoder 权重；
- 两个输出头；
- tokenizer；
- 训练配置；
- 输入规范；
- parser contract ID；
- label contract ID；
- split manifest 摘要；
- gate-data manifest 摘要；
- float checkpoint hash；
- 候选量化 checkpoint；
- 训练与开发指标。

模型、checkpoint 和训练日志保存在本地 artifact 目录，不提交 Git。

## 12. 门模型验收阶段（`gate-validate`）

### 12.1 职责

`gate-validate` 使用与训练和阈值拟合隔离的验证分组，冻结正式 bundle 和阈值，并证明
它满足生成端/检测端复现合同。只有本阶段通过，gated `generate` 才可运行。

### 12.2 验证矩阵

至少覆盖：

- 单样本推理；
- batch size 2、4、8；
- 同一样本与不同邻居样本组合；
- 浮点参考模型；
- 正式量化模型；
- CPU；
- 可用时的 GPU；
- 两次独立进程运行；
- 不同窗口长度；
- 前文 1、2、3；
- 主要 Python 语句类型；
- 未见 repository、任务和代码生成模型。

### 12.3 硬验收条件

- 共享解析器在合同夹具和验证 final code 上必须产生 100% 相同语句跨度；
- 正式接受为水印证据的窗口必须在全部规定推理模式中保持 100% 相同跨度；
- 全体样本的 close/suitable 离散决策一致率至少为 99.9%；
- 所有不一致样本必须进入不确定或跳过路径，不得作为正式证据；
- float 与正式量化模型的 accepted-window 集合一致率至少为 99.9%；
- bundle 中的 checkpoint、tokenizer、阈值和 contract hash 必须完整；
- 未见分组上的 suitable 假阳性率不得超过冻结验证配置中的 5%；
- 验证数据不得与 gate-data 的 train/dev 分组重叠；
- 验证不得读取最终 watermark 检测测试集。

没有通过时不得使用 `--force` 把候选 bundle 标记为正式可用。`--force` 最多用于重新
执行阶段，不能绕过验收结论。

### 12.4 正式 bundle manifest

验收通过后输出 `gate_bundle_manifest.json`，包含：

- `gate_model_id`；
- 模型、tokenizer、阈值和规范的内容哈希；
- `validated=true`；
- 验证矩阵摘要；
- parser contract ID；
- 输入规范 ID；
- 正式精度/量化模式；
- 最大输入 token；
- close/suitable 阈值；
- 训练数据 manifest 摘要；
- 明确的 `contains_secret=false`。

## 13. 正式生成阶段集成

### 13.1 前置条件

gated preset 的 `generate` 必须引用：

- 一个 `validated=true` 的 gate bundle；
- 匹配的解析合同；
- 匹配的语义 encoder 和 LSH 配置；
- 部署密钥的安全输入；
- 固定默认重写预算 3；
- 与训练数据生成一致的 key-blind rewrite 配置。

bundle hash 不匹配或验证状态缺失时必须失败。

### 13.2 运行算法

1. 基础模型正常生成代码；
2. 共享解析器确认完整语句单元；
3. 保存最近三个语句起点所需的模型、KV、解析器和门状态 checkpoint；
4. 将允许语句加入开放窗口；
5. 门模型判断 continue/close；
6. 硬边界、长度 3 或稳定 close 关闭窗口；
7. 边界不确定则关闭并跳过该窗口；
8. suitable 未通过则保留原窗口并继续；
9. suitable 通过时，先检查候选 0；
10. 候选 0 可靠命中则以 0 次重写接受；
11. 未命中则从窗口起点回退并生成完整重写候选；
12. 每个候选先重新解析、重跑门模型并检查结构/稳定性；
13. 通过后才使用部署密钥检查 LSH；
14. 找到第一个可靠命中立即接受；
15. 三次重写都失败则精确恢复候选 0；
16. 接受改变语句数量的候选后，重建生成状态并按最终代码继续；
17. 最终只把严格 schema 的 `final_code` 行送入 detector input。

候选模型不接收密钥、LSH 失败原因或目标区域。sampling seed 计划不得从密钥派生。

### 13.3 复合语句开头

复合语句开头形成单语句窗口，并在 body 生成之前决定是否接受重写。接受最终 header 后，
基础模型从该 header 继续生成 body。不得等 body 完成后只改 header 而保留旧 body。

### 13.4 运行时信号限制

正式生成只允许解析边界、门模型输出、LSH 证据、margin、预算计数和回退状态参与控制。
任何测试、执行或质量字段出现在候选选择路径都属于审计失败。

## 14. 最终代码检测阶段集成

检测器只接收严格的最终代码行：

```json
{
  "id": "...",
  "dataset": "...",
  "prompt": "...",
  "final_code": "..."
}
```

窗口恢复和水印评分只读取 `final_code`；`prompt` 不参与窗口、父节点或 LSH 证据计算。

检测步骤：

1. 使用共享解析器产生基础语句和硬边界；
2. 从左到右运行同一个 gate bundle；
3. 形成互不重叠、最多三个语句的窗口；
4. 忽略边界不确定窗口；
5. 忽略 suitable 未通过或数值不稳定窗口；
6. 从最终代码构造父节点描述；
7. 使用部署密钥重新计算允许 LSH 区域；
8. 对当前窗口计算语义签名和 margin；
9. 可靠位于允许区域记为 hit；
10. 稳定但不位于允许区域记为 miss；
11. 不稳定或 margin 不足记为 abstain；
12. 使用独立无水印校准集冻结的统计规则聚合；
13. 可靠窗口不足时返回 `insufficient_evidence`，不能自动当作无水印。

检测器不得读取 gate-data、gate-train 日志、候选 0、重写候选、重试次数、生成 audit、
candidate sidecar、raw attempt summary 或 posthoc pass report。

当前固定代理窗口检测器继续保留，名称和报告必须明确其为 legacy/baseline，不能把它的
proxy windows 描述为新方法的真实门窗口。

## 15. 检测校准

无水印校准代码必须运行与正样本完全相同的：

```text
parser -> gate -> stability filtering -> keyed LSH -> aggregation
```

校准集与 gate 训练、gate 验证和最终测试分离。校准冻结：

- 单窗口背景命中分布；
- 可靠窗口数量条件下的样本统计量；
- `insufficient_evidence` 的最小窗口数；
- 目标 5% FPR 阈值；
- empirical p-value 规则。

名义 `gamma` 不能替代经验校准。门模型改变了被评分窗口的分布，必须在同一 gate bundle
下重新校准。

## 16. 编排与 CLI

### 16.1 阶段注册

新增三个非 legacy、可独立运行的一等阶段：

```text
gate-data
gate-train
gate-validate
```

现有阶段保持：

```text
generate
calibrate
detect
report
audit
```

`ALL_PHASES` 和 phase registry 注册全部八个阶段。现有默认 preset 继续使用五阶段序列。
新的 gated preset 使用八阶段序列。

为支持两种工作流，编排器增加受验证的 config-driven phase sequence：

- preset 显式给出 `runtime.default_phases` 时，按该顺序执行；
- 每个名称必须属于 `ALL_PHASES`；
- 未给出时回退现有 `PHASES` 五阶段；
- run state 记录 resolved phase sequence 和 preset hash；
- status 同时显示未适用、已跳过、完成和失败状态。

### 16.2 阶段依赖

```text
gate-train     requires gate-data
gate-validate  requires gate-train
generate       requires validated gate bundle for gated preset
calibrate      requires generate or a declared negative corpus input
detect         requires generate + calibrate
report         requires detect
audit          requires all resolved official phases
```

如果配置引用外部已经验收的本地 gate bundle，`gate-data`、`gate-train` 和
`gate-validate` 可以从 resolved sequence 中省略；`generate` 仍必须验证 bundle manifest
和 hash。

### 16.3 CLI 行为

CLI 增加三个 phase 名称及其阶段专属配置入口，但 `run.py` 继续保持 thin shim。业务逻辑
分别位于 package 模块中，不能堆入 `run.py` 或 argparse 文件。

CLI 不接受直接把 secret key 写入公开 resolved config。secret 通过现有安全机制或运行时
环境注入，resolved config 只保存 secret ID/摘要和 `redacted=true`。

## 17. 配置合同

新 gated preset 至少包含：

```text
method.name
method.windowing.enabled
method.windowing.max_units = 3
method.windowing.max_preceding_units = 3
method.windowing.excluded_statement_types
method.windowing.compound_header_singleton = true
method.gate.bundle_path
method.gate.bundle_sha256
method.gate.require_validated = true
method.gate.uncertain_boundary_policy = close_and_skip
method.rewrite.max_attempts = 3
method.rewrite.candidate_zero = original_window
method.rewrite.key_blind = true
method.semantic.parent_descriptor_version
method.semantic.encoder_id
method.semantic.lsh parameters
gate_data sources / split / rewrite trajectory / training key bank ID
gate_train base encoder / optimization / consistency weights
gate_validate device / batch / precision matrix / acceptance thresholds
runtime.default_phases
```

配置解析必须拒绝：

- `max_units > 3`；
- gated generate 引用未验收 bundle；
- gate bundle 与 parser/semantic contract 不匹配；
- secret 明文出现在 public config；
- 正式方法启用 correctness/quality/test selector；
- candidate 生成配置读取密钥；
- detector 配置读取 generation sidecar。

## 18. Artifact 布局

在现有 `data/runs/<run_id>/` 下增加：

```text
data/runs/<run_id>/
  gate-data/
    manifest.json
    window_groups.jsonl
    candidate_attempts.jsonl
    split_manifest.json
    training_key_bank_manifest.json
    feasibility_summary.json
  gate-train/
    resolved_training_config.json
    training_metrics.jsonl
    development_summary.json
    candidate_bundle_manifest.json
  gate-validate/
    validation_summary.json
    agreement_details.jsonl
    gate_bundle_manifest.json
  generation/
    audit.jsonl
    candidate_sidecar.jsonl
    raw_attempt_summary.jsonl
  inputs/
    final_code.jsonl
  calibration/
    reference_calibration.json
  detection/
    positive_details.jsonl
    reference_negative_details.jsonl
    model_negative_details.jsonl
  reports/
    reference_report.json
    model_negative_report.json
    pass_report_posthoc.json
  audit/
    detector_input_integrity.json
    no_quality_gate_integrity.json
    artifact_integrity.json
    gate_bundle_integrity.json
```

模型权重可位于本地 `data/models/gate/<gate_model_id>/`，由 manifest hash 引用。模型、
训练数据、checkpoint、候选代码、大型 JSONL 和运行日志不提交 Git。

`inputs/final_code.jsonl` 仍是唯一正式正样本 detector input，行字段仍严格限制为：

```text
id, dataset, prompt, final_code
```

## 19. 模块边界

### 19.1 新共享窗口模块

新增共享包 `wfcllm.windowing`，负责：

- Python 语句单元；
- 父节点描述；
- 排除语句；
- 窗口状态；
- 门模型输入序列化；
- 生成端与检测端共用的确定性推理适配；
- bundle/contract 校验。

生成与检测必须依赖该包，不得各复制一套划窗逻辑。

### 19.2 `wfcllm.method`

增加 gated 方法 config、preset、bundle 引用和 artifact 路径合同；保留现有 preset 不变。

### 19.3 `wfcllm.generation`

保留 token 生成、KV/context checkpoint、审计和最终代码 pipeline；增加最多三个语句起点
checkpoint、整个窗口回退、候选 0 和门模型重验证。旧 retry/state machine 保留为基线，
不与新 gated preset 混用。

### 19.4 `wfcllm.semantic`

保留 encoder、LSH、keying、margin 和 verifier；将输入改为共享窗口规范化文本，并增加
版本化父节点描述适配。不得把门模型输出混入 LSH 签名。

### 19.5 `wfcllm.detection`

新增 gate-window extraction 和 scoring path；保留 proxy-window path 作为明确命名的
baseline。code-only validator 继续拒绝候选和生成字段。

### 19.6 `wfcllm.audit`

扩展检查：

- gate bundle 完整性；
- detector 输入隔离；
- training/deployment key 隔离；
- no-quality-gate 字段和调用链；
- gated preset 是否误读 sidecar；
- posthoc marker；
- resolved phase sequence 与 artifacts 的一致性。

### 19.7 `wfcllm.diagnostics`

保存 oracle 窗口、正确性选择、全候选最优选择、外层多 seed 上界及其他不正式方法。
这些输出继续携带 `diagnostic_only=true` 和 `not_official_method=true`。

### 19.8 `wfcllm.cli` 与 `wfcllm.orchestration`

注册前三个门模型阶段、config-driven phase sequence、依赖和状态。低层训练/数据逻辑不
进入 CLI runner；runner 只验证输入、调用 package pipeline 并记录 artifact/status。

## 20. 测试设计

### 20.1 共享解析器与窗口

- 分号与字符串分号；
- 单行/多行复合语句；
- 多行括号；
- comments/blank lines；
- `try/match/async/yield/await`；
- 排除语句硬边界；
- 不跨父级；
- 最大三语句；
- 复合 header singleton；
- 生成端和检测端 exact-span 一致。

### 20.2 gate-data

- group 不跨 split；
- HumanEval 污染检测；
- 同一候选轨迹复用三个前文版本；
- R=1/3/6 是同一轨迹前缀；
- candidate 0 不计 rewrite budget；
- 候选顺序不可重排；
- training key 不写公开 artifact；
- no-quality 字段拒绝；
- deterministic resume 与 manifest hash。

### 20.3 gate-train

- 两个输出头 shape/范围；
- padding/mask；
- group sampler；
- context consistency；
- quantization consistency；
- checkpoint resume；
- bundle manifest；
- 禁止从零错误加载或 bundle contract mismatch。

模型边界使用 mock/fake encoder 完成默认离线测试，真实本地模型只用于显式 smoke。

### 20.4 gate-validate

- batch 邻居变化；
- repeated process；
- float/quantized；
- CPU/GPU 可选矩阵；
- uncertain close-and-skip；
- failed bundle 不能进入 generate；
- accepted window exact-span consensus。

### 20.5 generation

- candidate 0 hit 不重写；
- R=3 计数；
- 整个窗口回退；
- 三语句变两语句；
- 一语句变三语句；
- 四语句候选拒绝；
- 父级变化拒绝；
- 内部提前 close 拒绝；
- 全失败精确恢复 candidate 0；
- 改写后生成状态重建；
- rewriter 不收到 key；
- 不调用 test/correctness/quality 信号。

### 20.6 detection 与 audit

- strict final-code-only schema；
- prompt 不参与窗口和评分；
- generation sidecar 拒绝；
- same gate bundle 恢复窗口；
- unstable window abstain；
- insufficient evidence；
- negative calibration；
- bundle/hash mismatch；
- no-quality-gate integrity；
- posthoc marker。

### 20.7 集成测试

用 fake parser/encoder/gate/generator 运行八阶段最小流程，并验证：

- phase prerequisites；
- existing five-phase preset 不回归；
- external validated bundle 可以跳过前三阶段；
- run-state 正确恢复；
- detector 只打开 final-code artifact；
- gated 和 legacy 报告明确分开。

## 21. 分阶段本地交付

### 阶段 A：共享语句与窗口合同

修改范围：共享窗口模块、generation/detection 适配、相关测试。

完成条件：两端在全部合同夹具上得到 100% 相同基础语句跨度。

失败回退：只修解析合同，不开始收数据。

### 阶段 B：gate-data

修改范围：数据 schema、pipeline、CLI phase、artifact/audit、测试。

完成条件：2,000～5,000 独立起点的可行性批次可恢复、无泄漏且候选轨迹公平。

失败回退：回到阶段 A 或重写候选生成合同。

### 阶段 C：无门模型可行性检查

修改范围：只读分析/正式允许的 LSH 稳定性统计。

完成条件：R=3/R=6 存在足够的结构有效、数值稳定命中，且增加预算带来可测收益。

失败回退：修改语义 LSH、允许区域或 key-blind 重写，不训练 gate。

### 阶段 D：gate-train 与 gate-validate

修改范围：gate dataset/model/trainer/validation、bundle、测试。

完成条件：通过第 12.3 节全部硬验收条件。

失败回退：减少输入、扩大真实独立数据、提高不确定区或缩小语句类型；不得伪造通过。

### 阶段 E：生成接入

修改范围：generation/state/checkpoint/retry/pipeline、preset、artifact。

完成条件：整个窗口回退和可变语句数候选在 fake 与本地 smoke 中可恢复。

失败回退：限制为保持语句数的诊断模式，正式 gated preset 不启用。

### 阶段 F：最终代码检测

修改范围：detection gate path、calibration、metrics、audit。

完成条件：不读取 sidecar 时能够重建相同 accepted windows，并完成经验校准。

失败回退：退回固定代理窗口 baseline，不声称恢复门窗口。

### 阶段 G：稳定性和正式比较

比较当前 main、无 gate、生成端 gate、两端 gate、固定 1/2 窗口和最大 3 动态窗口。

完成条件：固定 5% FPR 下检测能力不低于基线，重写成本下降，且 posthoc pass@1 相对
基线的下降不超过预先冻结的 2 个百分点。

失败回退：不晋升为正式方法，保留 gated 结果为 diagnostic/experimental。

## 22. 主要实验指标

- 原始 candidate 0 天然命中率；
- R=1、3、6 的新增可靠命中率；
- 平均实际重写次数；
- 首次可靠命中候选位置；
- 边界不确定率；
- suitable 接受率与假阳性率；
- accepted-window 跨 batch/精度/设备 exact-span 一致率；
- 每个样本可靠 hit/miss/abstain 数；
- insufficient-evidence 比例；
- TPR@5%FPR 与 empirical p-value；
- 生成 token、延迟和显存；
- 各语句类型、窗口长度和代码来源的分层结果；
- posthoc pass@1；
- 格式化和轻微修改后的水印保留率。

posthoc pass 指标不得回流到 gate-data 标签、阈值、候选选择或检测。

## 23. 风险与强制回退

### 风险 1：整体窗口本身没有足够 LSH 可控性

处理：阶段 C 停止，不训练 gate，先修改语义机制或重写策略。

### 风险 2：门模型边界在量化或 batch 中漂移

处理：扩大不确定区、close-and-skip、固定正式量化 bundle；仍不达标则退回“生成端
gate + 检测端固定规则”，并停止声称窗口复现。

### 风险 3：可变语句数导致生成状态无法正确恢复

处理：先限制候选保持语句数作为 diagnostic；正式方法在状态重建验证前不启用。

### 风险 4：门模型记住训练密钥

处理：多训练密钥、部署密钥隔离、holdout key 评估；发现 key-specific 行为则 bundle
验收失败。

### 风险 5：代码功能下降

处理：只在正式流程完成后测 posthoc pass；超过 2 个百分点非劣界则方法不晋升。不得
用测试结果在候选间择优。

### 风险 6：数据行数很大但独立窗口不足

处理：所有统计按 group 计数，派生版本不算独立数据；学习曲线按 repository/task/window
group 增长绘制。

### 风险 7：现有正式方法被新阶段破坏

处理：保留五阶段默认回退、旧 preset 和 proxy detector；加入兼容集成测试。新 gated
preset 未通过前不替换 `evidence_retry_seed7x3`。

## 24. 本地分支与提交边界

本规格和后续实现位于：

```text
codex/gated-semantic-window-watermark
```

本地实现允许修改：

- `wfcllm/` 活跃包；
- `tests/` 活跃测试；
- `configs/wfcllm/` 新 preset；
- `scripts/` 中必要的低层入口；
- `docs/` 中正式合同；
- `run.py` 的 thin-shim 合同保持不变。

不得修改或提交：

- 本地模型、数据集和 checkpoint；
- `data/runs/` 大型产物；
- 密钥；
- 本地日志；
- papers/review-stage；
- 与本方法无关的未跟踪用户文件；
- archive/legacy 历史材料，除非只增加必要的迁移说明。

在用户审查本 spec 前不开始实现。在用户批准后，下一步是编写逐文件实现计划，再按阶段
A～G 本地完成。是否 commit、push 或创建 PR 必须遵守用户后续明确授权；本次仅创建
分支和 spec，不提交 Git。

## 25. 最终决策摘要

本规格冻结以下第一版决策：

- 选择生成端和检测端共用同一个门模型；
- 最大窗口长度 3；
- 最近最多 3 个解析器前文语句；
- 两个独立输出头：close 与 suitable；
- 原始窗口为 candidate 0，不计重写预算；
- 默认额外重写预算 3，R=1/6 为消融；
- 未命中时重写整个窗口；
- 最终候选可改变内部语句数，但仍为 1～3 条并保持父级；
- key + 可恢复父节点信息派生允许 LSH 区域；
- 不使用显式 0/1 标签；
- gate-data、gate-train、gate-validate 进入 gated 端到端流程；
- 门模型不读取密钥、LSH 结果、候选日志或正确性代理；
- 检测只读取 final code；
- 功能正确率只做 posthoc；
- 不通过无 gate 可行性或稳定性验收时必须停止或回退，不能强行晋升。
