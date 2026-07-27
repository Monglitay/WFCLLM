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

**Reproduction Evaluation**:
为验证 Reproduction Coverage 所必需的生成后评估，包括 posthoc Pass@1、Metric Contract、检测与校准报告及审计；任何评估结果都不得反向参与生成或候选选择。
_Avoid_: 诊断分析, 质量门控, 候选选择

**Formal Eligible**:
满足全部准入门槛、可作为方法结论被引用的运行。
_Avoid_: 正式结果, official, 可发表

**Diagnostic Only**:
仅供诊断的运行，不得作为方法结论被引用。
_Avoid_: 实验性, experimental, 探索性

### Reproduction Core

**Reproduction Core**:
当前必须持续可复现的最小方法集合，只包含 Gated Lineage。非 Gate 方法即使曾被称为官方基线，也不属于 Reproduction Core。
_Avoid_: 当前实验, 官方基线, 全部 mainline

**Reproduction Coverage**:
Reproduction Core 必须维持的能力范围：可替换生成模型，覆盖 Python、C++、Java、JavaScript 四种 HumanEval 系列语言，并覆盖 Python MBPP；不要求保留每次历史调参运行。
_Avoid_: 七个历史运行, 所有旧配置, 单一基线

**Full Reproduction Profile**:
Reproduction Coverage 唯一公开的实验 profile，使用完整目标样本、相互独立的 pilot/full Gate 数据、完整校准和最终审计。缩短或放宽参数的 smoke/fast 运行只属于测试，不是实验 profile。
_Avoid_: fast profile, smoke result, quick experiment

**Fresh Reproduction Run**:
从当前 Full Reproduction Profile 开始、重新训练所需 encoder 与 Gate，并只产生和读取当前 artifact schema 的运行；旧状态、旧 bundle 和旧指标布局不属于受支持输入。
_Avoid_: resume old run, historical reader, artifact migration

**Public Execution Surface**:
运行 Reproduction Core 的受支持入口，只包括统一 CLI、五个 Full Reproduction Profile wrapper，以及它们直接调用或 Authoritative Documentation 明确要求的准备与生成后报告工具。
_Avoid_: low-level script, 调试脚本, 兼容 wrapper

**Lineage**:
一条独立演进的实验代码谱系。当前工作树只维护 Gated Lineage；退出 Reproduction Core 的 Lineage 只能通过 Historical Recovery Source 找回。
_Avoid_: 分支, fork, 实验线

**Code State**:
某次实验实际运行时磁盘上的完整代码内容，包含未提交改动。同一个 commit 可对应多个互不相同的 Code State。
_Avoid_: 版本, commit, 快照

**Archival Tag**:
固定某个 Code State 的不可变 git tag。只读归档用，不作为开发起点。
_Avoid_: 备份分支, release tag, 快照分支

**Historical Recovery Source**:
退出 Reproduction Core 的代码、配置和文档的唯一恢复来源，即 Git 历史与 Archival Tag；当前工作树不充当历史副本。
_Avoid_: archive 目录, legacy 入口, 工作树备份

**Authoritative Documentation**:
经当前实现核对、直接描述 Reproduction Core、Reproduction Coverage 或约束合同的文档。历史计划、设计草稿、进度汇报和研究过程记录只能通过 Historical Recovery Source 找回。
_Avoid_: 实现计划, 历史 spec, 设计修订副本

**Local Reproduction Resource**:
Full Reproduction Profile 所需但不进入 Git 的模型、数据集、密钥、Gate 数据、checkpoint、cache 和运行产物；仓库只记录其身份、合同与准备方式。
_Avoid_: 仓库数据, 内置模型, 已提交结果

**Metric Contract**:
每个运行产出一行自证身份的标准化指标记录，含数据集、语言、模型与各项指标。
_Avoid_: 指标表, report schema, summary
