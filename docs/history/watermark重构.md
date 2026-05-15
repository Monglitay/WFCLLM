这是一个非常典型且具有挑战性的工程需求。在大型语言模型（LLM）推理过程中处理“语句块回滚”（通常涉及KV Cache的管理和随机数状态的恢复）是非常容易出bug且极易导致显存泄漏（OOM）的地方。

你原本的prompt已经包含了核心诉求，但我帮你进行了**结构化重组**，并加入了**专家角色设定**、**更明确的技术上下文（如显存管理的强调）**，以及**强制的交互式输出规范**。这样能确保大模型不会“一口气吐出一大堆无法运行的代码”，而是像一个资深同事一样跟你一步步排查和重构。

以下是为你优化后的 Prompt。你可以直接复制这段内容发给大模型（或者直接发给我，我们现在就开始）：

---

### 优化后的 Prompt

**【角色设定】**
你现在是一位资深的 LLM 推理系统工程师和算法专家，精通 Python、大模型生成机制（如 KV Cache 管理、采样算法）以及水印（Watermark）嵌入技术。

**【任务背景】**
我需要重构我的 `wfcllm` 项目中的 `watermark` 模块，特别是**“语句块回滚（Block Rollback）”**机制。目前的方案不够稳定，经常出现逻辑 Bug（附后有部分 DEBUG 日志示例）。我们需要设计并实现一套更健壮的回滚与拒绝采样方案。

**【核心目标与需求】**
请严格按照以下目标进行重构设计：

**1. 精准的简单语句块回滚（核心）**

* 能够正确识别新生成的简单语句块。
* 在拒绝采样触发后，能够完美回滚到该简单语句块生成**前**的状态（包括但不限于 token 序列、生成器状态、以及**必须妥善处理的 KV Cache 等，避免显存泄漏**）。

**2. 级联回退机制（复杂语句块支持）**

* 当简单语句块的重试次数超过预设阈值时，触发级联回退：将整个语句块回滚到上一层“复杂语句块”生成之前的状态。
* 在复杂语句块这一层重新进行水印嵌入和拒绝采样。
* 如果重试依然过多，支持继续向上层退回（需提供“最大退回层数”的阈值配置）。
* **注意：** 退回复杂语句块的功能必须是**可选的**，且默认配置为**关闭（False）**。

**3. 严格的显存与性能控制**

* 回滚过程中必须显式释放不需要的资源，显存占用必须保持正常，**绝对不能爆显存（OOM）**。

**4. 完善的 DEBUG 监控**

* 实现细粒度的 DEBUG 日志模式，需要能详细记录“每一个语句块、每一次嵌入尝试”的状态（如当前尝试次数、entropy、margin、通过/拒绝状态等）。

**5. 工程化收尾**

* 核心模块更新后，同步提供 `run.py` 中 API 调用的更新方案。
* 更新 `config` 中的相关配置项（如各种阈值、开关）。
* 更新 `README.md` 中关于该模块的使用文档。
* 提供充分的测试用例和测试建议。

**【参考资料与日志】**

* **文档依赖：** 具体细节请参考原程序以及 `docs/基于LSH的改进方案.md`。
* **错误日志示例：**

```text
wfcllm.watermark.generator DEBUG [simple block #2] node=return_statement parent=if_statement entropy=0.5679 margin_thresh=0.0011
  sig=(1, 0, 0) in_valid=False valid_set_size=4 min_margin=0.0085 passed=False
  text='return []'
wfcllm.watermark.generator DEBUG   [retry 1/50] sig=(1, 0, 0) in_valid=False min_margin=0.0216 margin_thresh=0.0011 passed=False
  text='result'
wfcllm.watermark.generator DEBUG   [retry 2/50] ...
# [中间省略部分重试日志]
wfcllm.watermark.generator DEBUG [simple block #9] node=expression_statement parent=module entropy=1.5790 margin_thresh=0.0013
  sig=(1, 1, 0) in_valid=False valid_set_size=4 min_margin=0.0059 passed=False
  text='open_brackets = 1'
  text='open_count -= 1'
wfcllm.watermark.generator DEBUG   [retry 1/50] ...

```
