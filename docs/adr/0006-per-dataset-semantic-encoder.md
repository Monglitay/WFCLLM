---
status: accepted
---

# Semantic encoder 改为每个数据集各训一个

历史上七个 Reproduction Target 共用同一个 semantic encoder checkpoint。它由一个独立脚本训练而成 —— 不经 `encoder` 阶段，因此各 run 的 `encoder` 阶段状态一律为未完成 —— 训练语料是一份通用的 local-only 源码目录，不属于任何 benchmark 数据集。也就是说这个共用 encoder 从未见过 C++、Java 或 JavaScript，却被用于这三种语言的实验。我们改为每个数据集各训一个 encoder。

## 与当前手稿的刻意分歧

当前手稿把语义编码器描述为冻结的：正文两处写明它是 frozen semantic encoder、frozen CodeT5-base，投影矩阵是 fixed public projection matrix；手稿中被训练的是门控自身的结构编码器与 closure / suitability 两个预测头，不是语义编码器。手稿因此没有语义编码器训练这一步，更没有按数据集分别训练。

我们仍然选择保留并按数据集拆分语义编码器训练，并由手稿侧补写 method 章节来对齐。这是刻意分歧，不是实现缺陷 —— 对照手稿审读代码的人不应把它当作 bug。补写手稿前，代码与手稿在这一点上不一致的状态是已知且被接受的。

## Considered Options

- **沿用共用 encoder**：与历史数字同口径，重跑差异只来自随机性。但它把跨语言迁移这个缺陷原样保留 —— 用只见过一种语言的语义投影去判断另外三种语言的窗口能否承载水印。
- **按手稿去掉编码器训练、改用冻结基座加固定投影**：代码与手稿立刻一致，流水线少一个阶段。已否决 —— 但这意味着手稿必须补写，见上。

## Consequences

- 重跑出的数字与历史值的差异不再只是随机漂移，而是方法改变导致的系统性变化。「跑出差不多的数字」这一验收口径对受影响的行不再适用，偏差是预期的而非缺陷。
- semantic encoder 标识参与检测器身份契约，因此 [[Metric Contract]] 必须逐行记录 encoder 标识，否则两行数字无法判断是否同一 encoder 产出。
- `encoder` 须从可选阶段提升为 Gated Lineage 实验链的必经阶段，实验编排脚本需要相应增加该步 —— 历史脚本里没有这一步。
- 有一个待验证假设值得在实施时确认：JavaScript 目标的 gate 数据里 [[Suitable Window]] 正样本数为 0（准入门槛要求至少 30），可能正源于 encoder 从未见过 JavaScript。若每数据集 encoder 让该计数变为非零，这条假设得到支持，且该目标有望脱离 [[Experimental Bypass]]。
- 风险：按语言切分后单个语料可能不足。某语言的 gate 源目录仅数百条，而 encoder 训练器默认的训练组上限为一千二百组，历史那份通用语料为数 MB 量级。实施时须确认语料是否足以训出可用 encoder；若不足，应扩充语料而不是下调组数上限。
