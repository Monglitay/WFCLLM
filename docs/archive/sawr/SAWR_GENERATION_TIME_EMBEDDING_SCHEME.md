# SAWR 生成时嵌入方案

日期：2026-06-11

## 一句话版本

这个分支只做一件事：在模型生成代码的过程中，遇到函数体等可控位置时，短暂停下来，在当前位置尝试生成一条或几条语句；如果这些语句满足一个预先定义的本地嵌入规则，就把它们写进最终代码，否则继续尝试或回退。

提取和检测方案还没有确定，当前状态是待定。

`scientific_claims_enabled=false`。

## 为什么要重做

之前的 SAWR 方向积累了太多诊断框架、review gate、PR-LSU 术语和验证路线。它们让实现变得很重，而且还没有把最关键的问题解决掉：嵌入到底是在生成时发生，还是在代码生成完成后再修改。

这个分支把旧路线全部降级为历史参考。新的实现目标更窄：

- 不先做理论化容量分析；
- 不先做复杂 gate；
- 不先做提取器；
- 先把“生成时嵌入”这个动作跑起来；
- 先能跑 1 到 10 个样本；
- 稳定后再扩展到小批量或全量生成。

## 当前不做什么

- 不保留旧 SAWR 诊断框架。
- 不保留旧 review-stage gate 序列。
- 不继续把生成完成后的 rewrite 当作正式嵌入。
- 不实现当前提取器。
- 不声明水印可检测。
- 不声明 correctness/pass@1 结果。
- 不声明 TPR、FPR 或 controlled FPR。
- 不声明 semantic capacity。
- 不声明 correctness-preserving generation。
- 不声明 pass-cost improvement。
- 不声明 PR-LSU B2/B3 成功。
- 不做 matched-null/control 实验。
- 不做 training/fine-tuning。

## 名词解释

### 1. 样本

一个样本就是一次代码生成任务。它通常包含：

- `id`：样本编号，例如 `HumanEval/0`；
- `prompt`：给模型的代码前缀或函数签名；
- `final_code`：最终生成出来的完整代码。

例子：

```json
{
  "id": "HumanEval/demo_clamp",
  "prompt": "def clamp(x, lo, hi):\n    \"\"\"Return x limited to [lo, hi].\"\"\"\n"
}
```

### 2. 生成时嵌入

生成时嵌入指的是：模型还在生成代码时，系统就在当前位置控制候选语句的接受或重试。

它不是下面这种流程：

```text
先生成完整 final_code -> 再回头改 final_code -> 声称改写后的代码带嵌入
```

它必须是下面这种流程：

```text
生成到函数体中间 -> 生成候选语句 -> 判断候选语句是否满足嵌入规则 -> 满足才写入最终输出
```

### 3. 可控代码区域

可控代码区域就是我们愿意尝试嵌入的位置。第一版只考虑 Python 函数体。

例子：

```python
def clamp(x, lo, hi):
    """Return x limited to [lo, hi]."""
    # 这里开始是函数体，可以进入受控生成
```

暂时不把整文件、注释、字符串、变量名、格式化空格当作正式嵌入位置。

### 4. 位置路径

位置路径是给“当前在哪里生成代码”起的稳定名字。旧文档里叫 `ScopePath`。为了减少术语，可以理解成“代码位置 ID”。

例子：

```text
sample_id: HumanEval/demo_clamp
位置路径: module.clamp.body
含义: 当前在 clamp 函数的函数体里
```

第一版不需要设计复杂路径系统，只要能稳定地区分“这是哪个函数体”即可。

### 5. 语句候选

语句候选是模型刚生成出来、还没有最终确认写入代码的一条完整语句。

例子：

```python
if x < lo:
    return lo
```

或者：

```python
return x
```

这里的“完整”很重要。不能每生成一个 token 就判断一次，第一版应该等到语句或小 block 完整后再判断。

### 6. 小块候选

小块候选是比单条语句更大的候选，例如一个 `if` block：

```python
if x > hi:
    return hi
```

第一版可以把“单条 return 语句”和“简单 if-return block”都当作候选。先不要支持太复杂的嵌套结构。

### 7. 当前组合

当前组合就是正在等待判断的一组候选语句。旧文档里叫 `cluster` 或 `cluster buffer`。

为什么需要组合？因为有些嵌入规则可能不是看一条语句，而是看连续两条语句。

例子：

```python
if x < lo:
    return lo
if x > hi:
    return hi
```

这两段可以作为一个当前组合来判断。

### 8. 最大组合长度

最大组合长度限制当前组合最多包含几条候选语句。旧文档里叫 `max_cluster_statements`。

例子：

```text
max_cluster_statements = 3
```

意思是最多看连续 3 条候选。如果 3 条都没有满足嵌入规则，就不要无限等下去，应该 retry 或 fallback。

### 9. 嵌入规则

嵌入规则是一个本地函数，用来判断当前组合是否可以被接受。旧文档里叫 `predicate`。

第一版的嵌入规则只做生成控制，不是检测器，不是打分器，也不是统计检验。

它的输入可以是：

- 样本编号；
- 位置路径；
- 当前组合里的代码文本；
- 一个固定 seed；
- 简单配置。

它的输出只有两类：

```text
hit: 当前组合满足规则，可以接受
miss: 当前组合不满足规则，继续扩展、重试或回退
```

### 10. hit 和 miss

`hit` 表示当前组合满足嵌入规则。

`miss` 表示当前组合不满足嵌入规则。

这两个词只描述生成时控制结果，不能解释成“检测成功”或“检测失败”。

### 11. 接受并写入

接受并写入就是把候选语句正式放进最终代码。旧文档里叫 `commit`。

关键边界：只有在候选还没有进入最终代码之前，系统才能决定接受或重试。一旦完整 `final_code` 已经生成完，再回头自由改写，就不是正式生成时嵌入。

### 12. 清空当前组合

如果某个当前组合已经 hit，就必须清空当前组合。下一条语句要从新的组合重新开始。

例子：

```text
第 1 条语句 hit -> 清空当前组合
第 2 条语句 miss -> 当前组合只包含第 2 条，不能把第 1 条又拿回来拼
```

这条规则避免同一段已经接受的代码被重复算作后续机会。

### 13. 函数体结束 flush

flush 指的是：函数体结束或生成结束时，如果当前组合还没空，必须做最后一次判断。

例子：

```text
max_cluster_statements = 3
函数体结束时当前组合只有 2 条语句
```

这时不能因为没达到 3 条就直接失败。应该对这 2 条语句做最后一次判断：

- 如果 hit：接受这 2 条，不 retry；
- 如果 miss：回退或记录失败，不 claim。

### 14. 回退

回退就是嵌入尝试没有成功时，仍然让模型产出正常代码。旧文档里叫 `fallback`。

回退不是失败到没有输出，而是不声明这段代码完成了嵌入。

### 15. 审计记录

审计记录是给开发者看的生成过程记录。它可以说明：

- 哪个样本；
- 哪个函数体；
- 哪些候选被尝试；
- 哪个组合 hit；
- 最后为什么接受、重试或回退。

审计记录不能进入未来检测器输入。它只能用于复现和调试。

### 16. 最终代码输入

如果未来设计提取器，检测器输入仍然只能是最终代码本身。

格式应该保持类似：

```json
{"id": "HumanEval/demo_clamp", "final_code": "def clamp(...):\n    ..."}
```

不能把审计记录、seed、尝试次数、hit 位置等隐藏信息交给检测器。

## 完整流程

下面是第一版生成时嵌入 runner 应该做的事情。

### 第 0 步：读取样本

输入：

```json
{
  "id": "HumanEval/demo_clamp",
  "prompt": "def clamp(x, lo, hi):\n    \"\"\"Return x limited to [lo, hi].\"\"\"\n"
}
```

runner 记录样本 ID，并准备把模型输出接到 prompt 后面。

### 第 1 步：正常生成到函数体

模型看到 prompt：

```python
def clamp(x, lo, hi):
    """Return x limited to [lo, hi]."""
```

系统识别到现在位于 `clamp` 函数体内，于是进入受控生成模式。

内部状态可以写成：

```json
{
  "sample_id": "HumanEval/demo_clamp",
  "position_id": "module.clamp.body",
  "current_group": [],
  "accepted_hit_count": 0,
  "max_group_statements": 3,
  "retry_budget": 2
}
```

### 第 2 步：生成第一条候选

模型生成候选：

```python
if x < lo:
    return lo
```

系统还没有把它永久写入最终代码，而是先加入当前组合：

```json
{
  "current_group": [
    "if x < lo:\n    return lo"
  ]
}
```

调用嵌入规则：

```text
embedding_rule(sample_id, position_id, current_group) -> miss
```

因为 miss 且当前组合长度是 1，小于最大长度 3，所以继续等下一条语句。

### 第 3 步：生成第二条候选

模型生成候选：

```python
if x > hi:
    return hi
```

当前组合变成：

```json
{
  "current_group": [
    "if x < lo:\n    return lo",
    "if x > hi:\n    return hi"
  ]
}
```

再次调用嵌入规则：

```text
embedding_rule(sample_id, position_id, current_group) -> hit
```

这时系统执行：

1. 接受这两条候选；
2. 把它们写入最终代码；
3. 写只用于审计的过程记录（audit-only provenance）；
4. `accepted_hit_count += 1`；
5. 清空当前组合。

状态变成：

```json
{
  "current_group": [],
  "accepted_hit_count": 1
}
```

最终代码目前是：

```python
def clamp(x, lo, hi):
    """Return x limited to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
```

### 第 4 步：hit 后重新开始

模型继续生成：

```python
return x
```

因为上一个组合已经 hit 并清空，所以当前组合只能包含新语句：

```json
{
  "current_group": [
    "return x"
  ]
}
```

它不能变成：

```json
{
  "current_group": [
    "if x < lo:\n    return lo",
    "if x > hi:\n    return hi",
    "return x"
  ]
}
```

这就是“hit 后清零”的含义。

### 第 5 步：函数体结束时 flush

假设 `return x` 之后函数体结束。

当前组合非空：

```json
{
  "current_group": [
    "return x"
  ]
}
```

系统必须做最后一次判断：

```text
embedding_rule(sample_id, position_id, current_group, final_flush=true) -> miss
```

因为 miss，所以不声明这条语句形成嵌入。清空当前组合，正常结束生成。

最终代码：

```python
def clamp(x, lo, hi):
    """Return x limited to [lo, hi]."""
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x
```

### 第 6 步：写 artifact

runner 至少写出两类结果。

最终代码行：

```json
{
  "id": "HumanEval/demo_clamp",
  "final_code": "def clamp(x, lo, hi):\n    \"\"\"Return x limited to [lo, hi].\"\"\"\n    if x < lo:\n        return lo\n    if x > hi:\n        return hi\n    return x\n"
}
```

审计记录行：

```json
{
  "id": "HumanEval/demo_clamp",
  "audit_only": true,
  "detector_input_allowed": false,
  "position_id": "module.clamp.body",
  "event": "accepted_generation_time_group",
  "group_statement_count": 2,
  "final_flush": false,
  "scientific_claims_enabled": false
}
```

审计记录不能包含或诱导这些字段：

- `p_value`
- `tpr`
- `fpr`
- `correctness_result`
- `pass_cost`
- `detector_score`
- `watermark_params`
- `generation_ledger`
- `retry_ledger`

## partial group flush 例子

这个例子说明：最大组合长度不是必须达到。

配置：

```text
max_group_statements = 3
```

函数体只生成了两条语句：

```python
total = a + b
return total
```

函数体结束时，当前组合长度是 2，没有达到最大长度 3。系统仍然必须 flush：

```text
embedding_rule([total = a + b, return total], final_flush=true) -> hit
```

如果 hit，则接受这两条语句，并记录：

```json
{
  "event": "accepted_generation_time_group",
  "group_statement_count": 2,
  "final_flush": true,
  "retry_required": false,
  "audit_only": true,
  "detector_input_allowed": false,
  "scientific_claims_enabled": false
}
```

这不是检测成功，只是说明生成时控制规则接受了这个 partial group。

如果 flush 时是 miss，则记录：

```json
{
  "event": "closed_without_hit",
  "group_statement_count": 2,
  "final_flush": true,
  "audit_only": true,
  "detector_input_allowed": false,
  "scientific_claims_enabled": false
}
```

这时不能 claim 嵌入成功。

## 第一版模块边界

第一版实现只需要四个核心组件。

### 1. 边界检测器

职责：

- 判断当前生成是否进入 Python 函数体；
- 判断一条 statement 或小 block 是否完整；
- 判断函数体是否结束。

不做：

- 不跑 correctness；
- 不跑 extraction；
- 不跑 detector scoring；
- 不分析全量语义容量。

### 2. 在线状态机

职责：

- 保存当前样本和当前位置；
- 保存当前组合；
- 处理 hit、miss、flush、retry、fallback；
- 保证 hit 后清空当前组合。

### 3. 嵌入规则接口

职责：

- 接收当前组合；
- 返回 hit 或 miss；
- 保持本地、确定、可测试。

不做：

- 不调用 `wfcllm.extract`；
- 不调用 `wfcllm.watermark`；
- 不调用 legacy detector；
- 不调用 token-channel runtime；
- 不加载 `torch`、`transformers`、`datasets`。

模型加载只能发生在 CLI 的本地 Hugging Face sampler 路径中。

### 4. 生成 runner

职责：

- 读取样本；
- 调用模型；
- 在可控位置接入在线状态机；
- 写 final code artifact；
- 写 audit-only provenance。

## 小样本和全量运行计划

### 小样本 smoke

先跑 1 到 10 个样本，目标是验证工程流程：

- 模型可以正常加载；
- 生成可以开始和停止；
- 函数体边界能识别；
- statement candidate 能被截出来；
- 在线状态机能收到 candidate；
- 至少一个 candidate 能在进入最终代码前被接受并写入；
- artifact 分离正确；
- 失败样本有明确原因。

### 小批量运行

小样本稳定后，再跑几十到几百个样本，目标是验证：

- 程序不会卡死；
- resume 可用；
- 每个样本都有状态；
- 输出行数可审计；
- 失败类别可统计。

### 全量嵌入

全量嵌入只能说明“生成时控制 runner 产出了 final-code artifacts 和 audit-only provenance”。

在提取方案确定前，不能把全量嵌入结果说成检测实验，也不能报告 TPR/FPR。

## 模型和服务器边界

当前小样本 smoke 使用的模型路径：

```text
data/models/deepseek-coder-7b-instruct/deepseek-ai/deepseek-coder-7b-instruct-v1___5
```

推荐参数：

```text
--torch-dtype bf16
--device cuda
--temperature 0.0
--max-new-tokens 64
```

服务器上允许做：

- sync code；
- pytest/compile smoke；
- 1 到 10 个样本的生成时控制 smoke；
- 后续明确授权的小批量或全量嵌入。

服务器上当前不做：

- correctness/pass@1；
- detector scoring；
- full extraction；
- matched-null/control；
- training/fine-tuning。

## 提取状态

提取方案待定。

这个分支目前不选择：

下面这些都是未来检测方案里的术语，当前分支不定义也不实现它们。

- detector；
- scorer；
- thresholding rule；
- calibration set；
- FPR protocol。

如果未来要设计提取方案，必须重新定义：

- 检测器只能看到什么输入；
- 是否只看 final code；
- 是否需要 calibration；
- 如何避免使用审计记录；
- 可以报告哪些指标；
- 哪些指标仍然禁止报告。

当前默认约束是：未来 detector input 仍保持 final-code-only。

## 安全表述

可以说：

- “生成时嵌入 runner”
- “受控 candidate commit”
- “在线状态机接受了某个候选组合”
- “只用于审计的过程记录（audit-only provenance）”
- “最终代码 artifact 生成（final-code artifact generation）”
- “提取方案待定”
- “extraction pending”

不能说：

- “watermark detected”
- “semantic capacity”
- “TPR”
- “controlled FPR”
- “correctness-preserving generation”
- “pass-cost improvement”
- “PR-LSU B2/B3 success”

## 下一步实现顺序

建议按这个顺序落地：

1. 写一个最小 Python 函数体和 statement 边界检测器。
2. 写在线状态机，覆盖 hit、miss、hit 后清零、body close flush。
3. 写一个确定性的本地嵌入规则接口。
4. 写无模型的 replay smoke，验证状态机流程。
5. 接入本地 DeepSeek 小样本生成。
6. 写 final code artifact 和 audit-only provenance。
7. 跑 1 到 10 个样本。
8. 修稳定性和 resume。
9. 再决定是否跑小批量或全量嵌入。
10. 之后单独讨论提取方案。
