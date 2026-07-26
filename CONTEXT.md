# WFCLLM

生成时语义水印方法的实现与实验仓库。本文件只定义领域词汇 —— 不放实现细节、不放规格、不放待办。

## Language

### 水印密钥

**Deployment Key**:
嵌入侧与检测侧共用的对称水印密钥。换掉它，检测必然失效。
_Avoid_: 水印密钥, secret key, embed key

**Training Key Bank**:
训练 [[Gate]] 时轮换使用的密钥组。
_Avoid_: 训练密钥, training keys

**Holdout Key Bank**:
留出不参与训练的密钥组，用于检验 Gate 学到的是通用性质而非记住了 Training Key Bank。
_Avoid_: 验证密钥, holdout keys

**Key Identifier**:
Deployment Key 的 SHA-256 指纹。可公开出现在报告中，密钥本身不可。
_Avoid_: key hash, 密钥 id

### Gate

**Gate**:
判定单个代码窗口能否承载水印比特的判别器。
_Avoid_: 门控, classifier, filter

**Suitable Window**:
在改写预算 3 下命中率、结构合法率、稳定率三项同时达标的窗口 —— 即真能稳定承载水印比特的窗口。
_Avoid_: 可用窗口, valid window, 合格窗口

**Reliable Window**:
检测期被判定为证据可采信的窗口。与 Suitable Window 是不同判定，不可混用。
_Avoid_: 有效窗口, 可信窗口

**Gate Bundle**:
gate-train 阶段产出的可部署 Gate 权重包。由配置、gate 数据、基座模型与随机种子确定性生成。
_Avoid_: gate 模型, checkpoint, 权重

**Suitable False Positive Rate**:
Gate 把非 Suitable Window 判成 Suitable 的比率，取多运行配置下的最坏值。
_Avoid_: FPR, 假阳率

### 结果资格

**Formal Eligible**:
满足全部准入门槛、可作为方法结论被引用的运行。
_Avoid_: 正式结果, official, 可发表

**Diagnostic Only**:
仅供诊断的运行，不得作为方法结论被引用。
_Avoid_: 实验性, experimental, 探索性

**Experimental Bypass**:
为取得数字而主动放宽准入约束的运行。带 Experimental Bypass 的结果一律是 Diagnostic Only。
_Avoid_: 放宽, relaxed, workaround

**Conditional TPR**:
先按资格规则筛除部分样本、再在剩余样本上计算的 TPR。与全样本 TPR 不可横比。
_Avoid_: TPR

### 三副本收敛

**Lineage**:
一条独立演进的实验代码谱系。本仓库有两条：Gated Lineage 与 Semantic-V3 Lineage。
_Avoid_: 分支, fork, 实验线

**Code State**:
某次实验实际运行时磁盘上的完整代码内容，包含未提交改动。同一个 commit 可对应多个互不相同的 Code State。
_Avoid_: 版本, commit, 快照

**Archival Tag**:
固定某个 Code State 的不可变 git tag。只读归档用，不作为开发起点。
_Avoid_: 备份分支, release tag, 快照分支

**Reproduction Target**:
被明确指定、必须能被重跑出近似指标的一次历史运行。
_Avoid_: 实验, run, 基线

**Metric Contract**:
每个运行产出一行自证身份的标准化指标记录，含数据集、语言、模型与各项指标。
_Avoid_: 指标表, report schema, summary
