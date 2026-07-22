# 面向 Python 代码语义水印的原子代码单元门控方案（方案 C）

**协议名：** SAGE-C（Semantic Atomic-unit Gated Evidence, Scheme C）  
**文档版本：** 0.3-rc2  
**语法目标：** Python 3.12  
**研究状态：** RC freeze；同一 reviewer 已关闭 C01/M02/M03，rc2 仅统一 M01 的 ECC-disabled 规范枚举，将由原 reviewer 定向复查；没有运行本方案训练、Oracle pilot 或端到端实验  
**水印类型：** keyed statistical zero-bit watermark  
**权威输入：** final Python code、固定 public artifacts/parameters、private key  

> 证据纪律：本文中的仓库数字只描述历史 WFCLLM/SAWR；所有 SAGE-C 阈值都是预注册用的 initial engineering thresholds 或 required experiments。没有把未运行的 Oracle、Gate 或 system 实验写成结果。

## 1. Executive conclusion

SAGE-C 是一条**协议可证伪、工程可实现、容量与检测力尚未实证**的 keyed statistical zero-bit 路线。一个载体始终只对应当前 `AtomicWatermarkUnit`；最多三个前序单元只进入 Gate，绝不进入当前单元的 `RegionV1`。正式 v1 先无密钥生成并冻结 baseline document，完成全 document 的 Pre-Gate 预算分配、所有 fixed candidate pools、双侧完整性判定与 commitment，之后才允许 target-key 接口工作。提取端不读取 prompt、logits、hidden states、candidate pool、retry trace 或 generation logs。

前两轮审查揭示了不能靠“更多训练”修复的协议缺陷，0.3 已作结构性收缩：

1. 删除调用者可选的 `document_nonce`，同一 `(code, protocol, key)` 的检测是唯一确定测试，不能靠枚举 nonce 重采样 target；
2. 删除需要不可见原序列的 ordinal/edit-DP 和 keyed alignment maximum，改用可由生成前缀和最终代码同样计算、对正式 candidate rewrite 不变的 causal structural 4-gram anchor；
3. 正式bit region不再依赖浮点神经embedding，而是current-only、alpha-normalized `RegionCurrentV2`上的公开离散realization partition；语义保持由generation-side `RewriteCertificateV1`、final-only `BlindCertificateV1`与pair verifier共同约束。Gate/Post仍是可训练的小模型，但只决定预算和`accept/erasure`，不能改observed bit；formal Post-X本身只读current unit。
4. target不再只寻址可跨文档重复的local anchor；它额外绑定`DocumentIdentityV1`。该identity由完整final AST/ownership tree重建，registered carrier realization先由BlindCertificate归一回唯一identity form，其他identifier、literal、decorator和结构内容不mask。它只在pool/plan commitment完成后进入HMAC，不进入Pre、候选枚举、pool variant或allocator。不同document identities即使local anchor相同也得到独立PRF地址；完整identity-equivalent artifact复制仍是明确的provenance non-goal。
5. 删除formal v1的Hamming/content-landmark target stream。每code-object最多4个selected slots与Hamming最少5个observations在结构上矛盾；v1唯一target是独立`K_unit` stream，擦除由三态evidence、dependence clusters和完整empirical null处理。任何ECC重新引入都是major protocol change，而不是可暗中开启的参数。
6. allocator所有Q12 utility算术改为checked signed INT64；`OperationalDigestV2`对整个constant/code-object graph编码确定性`DEF/REF`别名拓扑；lone-surrogate文档明确fail closed；Post-X增加exact-tensor collision与信息论coverage ceiling gate。

路线成立需要四个不可绕过的条件：

1. 受限 Python 单元类在 fixed public region 下有足够的 region-0/region-1 双侧可达率；
2. blind extraction Gate 在 final-stream deployment strata 上有极低 false accept、可独立风险认证；
3. integer-only reference evaluator 内离散 Gate state 与 `RegionV1` 可 100% 重放；
4. 完整detector-side Gate/erasure/anchor/document-identity/query-family selection在独立、分层且含same-key watermarked-sample-aware mosaic攻击的corpus上同时控制per-query、worst-stratum与atomic-family FPR，并保有detection power；embedding-side Pre plan另做causal audit但不成为blind evidence。

Gate 不能创造底层不存在的容量，也不能证明一般 Python 程序等价。若有限 rewrite language 的静态枚举或扩大 pool 的 Oracle 显示大量位置仍单侧可达，第一层直接 NO-GO，停止训练 Gate。当前建议是 **CONDITIONAL GO，仅批准 parser conformance、rewrite-catalog enumeration 与最小 Oracle feasibility study**；不批准大规模训练，不声称水印系统、经验 FPR、鲁棒性或容量已成功。

相对现有仓库，这不是 `evidence_retry_seed7x3` 的参数升级。现行生成边界器只覆盖 `if/for/while` compound layer，现行 detector row 仍含 `prompt`，现行 no-quality-gate 合同排斥 syntax/quality control fields，旧 SAWR 又以多语句窗口为载体。SAGE-C 需要新 parser contract、新 artifact registry、新 gate policy 和新 empirical-null detector；本任务只设计文档，不修改生产代码。

## 2. Problem definition

给定合法 Python 最终代码 \(D\)、public protocol \(P\)、public model artifacts \(M\) 和 private key \(K\)，定义确定性的原子单元流：

\[
\mathcal U_P(D)=(U_1,\ldots,U_T).
\]

嵌入端在生成当前单元 \(U_t\) 时，可读取 \(U_{t-3:t-1}\) 作为 feasibility context，但正式载荷只由当前单元候选 \(x_t\) 的 public region \(R_P(x_t)\) 与`(DocumentIdentityV1(D), candidate-independent local anchor a_t)`的keyed target决定。这里必须区分三件事：第8节causal structural 4-gram是**局部同步地址**；`DocumentIdentityV1`是阻断cross-document target复用的全模块binding；两者都不是semantic encoder输入。observed region/bit、candidate属于哪一side及RegionV1 rejection完全只读当前unit。若把前序canonical text/embedding或document identity送入RegionV1/Gate tensor，或让target-selected前序改写改变当前side/identity，均是协议失败。提取端从 \(D,P,M,K\) 重新得到每个单元的 `match / mismatch / erasure`，再做相关性修正和经验校准检测。

需要同时优化：

- functional/semantic preservation；
- effective evidence density；
- fixed-budget search efficiency；
- blind replay determinism；
- low Post-Gate false accept；
- controlled empirical FPR；
- formatting survival，以及对rename/local-edit会触发document-wide target rebinding的诚实availability分析；
- bounded latency 与 retry cost。

问题不是“给每行打 bit”，也不是“从 generation trace 证明曾成功嵌入”。formal evidence 的唯一权威来源是 final-code replay。

## 3. Threat model

### 3.1 参与者、秘密与信任边界

- Provider 持有 \(K\)，运行 generator、Gate 和 embedding protocol。
- Detector 持有同一 \(K\) 与 public artifacts，可不访问 generator。
- 用户/攻击者看到 final code 和全部 public protocol/model artifacts，但看不到 \(K\)。
- public region partition 不保密；安全性来自 anchor-specific PRF target，而非隐藏模型。
- Provider 被假设诚实执行 key-neutral planning，但可能有实现 bug；final code 本身不能证明 provider 没有让 key 回流到 pool。静态 taint、进程隔离、可选 HSM/TEE 与 signed plan commitment 是审计控制，不是水印 evidence。恶意 provider 不在可由 blind detector 解决的威胁内。

### 3.2 允许攻击

- formatting、comment/blank-line change；
- identifier rename；
- expression-level equivalent rewrite；
- unit insertion/deletion；
- 同 suite 内有限 reordering；
- dead-code insertion；
- 读取同一key下大量合法watermarked outputs后进行local-anchor字典、mosaic或cross-document transplant；
- public registered-marker normalization（如把`v+0`折回identity）；该攻击可无key擦除全部证据，是availability/removal attack而非forge；
- header/body edit；
- 在注册deployment population内以一次atomic request预提交至多\(Q_{max}=32\)个不同artifacts、选择公开strata、公开模型probing；服务在整个family封存前不返回中间结果，并做Holm或统一max-null校正。根据先前detector输出自适应构造后续code不属于v1 FPR保证，只能拒绝或标stress-only。

### 3.3 不保证的攻击

- 任意控制流重构、整函数重写、语义重新实现；
- 攻击者拥有 oracle detector key；
- 无限制、自适应、跨身份 detector oracle 查询；离线库只给单次校准 p-value，不能声称对无限查询仍有 1% family-wise FPR；
- 攻击者替换/伪造 public model artifact 却要求 detector 静默接受；
- 超出bounded k-gram structural-resynchronization scope的无界 insertion/deletion/reorder。
- 对公开registered marker做全局canonical folding后仍保持可检测；v1明确不作此保证。
- 执行环境通过`sys.settrace`、coverage/position events、traceback对象/格式化的行列信息、读取source bytes、`inspect`或code-object/line-table introspection来区分两个catalog realizations；这些观察被`OperationalDigestV2`明确排除。固定fresh CPython process中的普通`is`/alias关系属于合同并由alias graph覆盖；依赖外部模块预热状态、C-extension地址或跨进程绝对`id()`数值不属于合同。

对注册edit channel必须区分**结构地址**和**keyed target**：local anchor在一次局部编辑后可于第4个未改same-suite unit恢复，但`DocumentIdentityV1`对任何AST/identifier/literal/ownership变化都改变，因而整个文档target stream重新绑定。v1只保证format/comment/registered-carrier-alternative等保持identity的变换；rename、insert/delete/reorder是document-wide availability attack，不再宣称局部match survival。超出admission/query contract时返回`UNAVAILABLE`或不提供family-wise保证；若attack后仍positive，必须由held-out attack evaluation证明。

### 3.4 安全目标

- key secrecy 与 domain separation；
- candidate pool 及 Gate selection key-independent；
- final artifact不写显式`expected_target`、selected-position map、retry或key-dependent failure trace；但carrier的observed public region必然可见，这正是水印信号。安全要求是这些PRF output bits不泄露master key，也没有region之外的target-conditioned控制轨迹；
- detector API 不接受 caller-selected nonce、anchor、slot 或 subset；同一 canonical artifact 重复提交得到逐位相同结果；
- target HMAC必须绑定由final artifact自身唯一导出的`DocumentIdentityV1`；不同identity下重复local anchor的target在PRF假设下独立。identity不能由caller metadata覆盖，identity歧义或生成前后不一致即fail closed；
- artifact/version mismatch fail closed；
- null corpus上完整重放所有detector-side data-dependent selection；embedding-side Pre plan不属于证据，只做独立causal audit；
- 不把 parser recovery、神经边界输出或 generation audit 当 formal evidence。

## 4. Design goals and non-goals

### 4.1 Goals

1. AST/CST/token-aware 严格定义原子单元，嵌入/提取确定性对偶。
2. 一个 shared small Gate 同时支持 CTX-1/2/3 和 simple/header type。
3. 同 anchor、同 pool 的双侧 counterfactual labels。
4. fixed, key-independent attempt budget 与 deterministic candidate selection。
5. current-unit-only public realization region，且每个realization由semantic-preservation certificate约束；context不进入bit编码。
6. match/mismatch/erasure、dependence-aware aggregation 和 empirical FPR 全链路；formal v1明确禁用ECC。
7. reference backend、QAT、margin、conformance 的离散重放协议。
8. 明确 Oracle/Gate/System 三层 stop/go。

### 4.2 Non-goals

- 不承载任意用户 payload；formal v1没有ECC或codeword target。
- 不证明一般 Python 程序等价。
- 不保证在source/trace/code-object/position introspection下realizations不可区分；语义合同只覆盖固定CPython 3.12正常执行观察。
- 不让 Gate 使用 private key、target bit、candidate count、attempt index 或 retry outcome。
- 不用原 prompt、generation logits/hidden state、candidate sidecar 或 logs 检测。
- 不自动将所有 syntactic units 视为 carriers。
- 不以 QAT 或固定 seed 代替部署契约。
- 不继承旧多窗口混合载荷或 compound-body fallback。
- 不保证单函数/短片段可检测；v1 正式 detection object 是满足第 31、40 节 admission 的长 Python module，短代码必须计入 coverage denominator 并返回 `INSUFFICIENT`。
- 不保证对知道公开rewrite catalog的静态分析器隐蔽；显式marker可识别、可canonical-fold擦除，v1只研究keyed detection与受限自然编辑生存，不声称steganographic undetectability。
- 不保证任何会改变`DocumentIdentityV1`的rename、statement insertion/deletion/reorder或等价AST重写后仍可检测；这是为阻断same-key mosaic/transplant而选择的安全优先scope。完整identity-equivalent artifact复制也不能证明作者归属。

## 5. Terminology

| 术语 | 严格含义 |
|---|---|
| syntactic unit | strict parser在线性化结构中识别的`SimpleUnit`或`ClauseHeaderUnit`，无论是否可水印；“eligible”只属于policy层 |
| candidate unit / `FORMAL_ELIGIBLE(D0)` | 在Gate运行前，位于provider已知generated span、baseline RHS为canonical identity-form `ClosedValueV1`且通过公开grammar/certificate-domain/anchor-uniqueness policy的syntactic unit；它是所有容量指标不变的generation candidate denominator |
| `FORMAL_REPLAY_ELIGIBLE(D)` | final unit形状匹配**non-identity** registered rewrite form并通过`BlindCertificateV1`；identity form虽可certify但按规则erasure，从而blind extractor不需要猜它是否曾被选中 |
| pre-selected unit | 被 key-neutral document allocator 预留固定搜索预算的 candidate unit |
| attempted unit | 已生成完整 A-record pool 的 pre-selected unit；unused reservations不重分配 |
| dual-complete unit | 同一个实际 committed pool 在 target 解锁前已有至少一个 region-0 和一个 region-1 admissible candidate |
| embedded decision | 生成端找到 target-region 合格候选的内部结果；不是 detector 权威事实 |
| formal evidence | 最终代码经 strict replay 后命中registered non-identity form，并通过 parser、BlindCertificate/hard verifier、Post-Gate-X、RegionV1 non-boundary 与 artifact checks 的单元 |
| erased unit | 结构存在但因边界、低置信、版本、对齐或验证失败归为 erasure 的单元 |
| excluded unit | 规则层永不进入正式候选池的 syntactic unit；仍占 ast ordinal |
| oracle-only class | 只做隔离 feasibility/风险数据，不属于v1 formal carrier；任何开放都需新protocol/review |
| ast_ordinal | suite 内按所有 atomic syntactic units 计数的诊断顺序；excluded 也计数；永不进入 target anchor |
| candidate_ordinal | 仅 eligible candidate 的顺序；只用于统计，不用于 key anchor |
| provisional evidence | 生成中增量 parser 认为边界可能闭合的内部状态；最终 strict replay 前无效 |
| region | 当前单元 canonical realization 的公开 `RegionV1` 离散分区：`0 / 1 / boundary-erasure`；不是 Python denotation 的等价类 |
| document identity | `DocumentIdentityV1`：对registered carrier alternatives归一、但对其余完整AST内容不mask的candidate-independent全模块digest；只用于commitment和target HMAC，不进入候选池或Gate tensor |
| target bit | HMAC 从 private key、`DocumentIdentityV1`和candidate-independent local anchor 派生的期望 region；formal v1无ECC |
| Pre-Gate | 搜索前判断是否值得投入固定 attempt budget；不产生证据 |
| Post-Gate-G | generation-only、可读取 baseline/candidate pair 的候选 verifier/ranker；不产生 blind evidence |
| Post-Gate-X | extraction-authoritative、v1只读取 final current unit（其余三个slot固定BOS）的 intrinsic accept/erasure Gate；这样前序target-selected改写不会改变当前证据状态 |
| pairwise semantic verifier | 生成/标注时比较 baseline 与 candidate 的静态、测试、differential证据；key-independent，但提取端不可用 |
| blind validity Gate | Post-Gate-X 仅预测 final-observable replay/quality/robustness risks；不再预测“relative-to-hidden-baseline semantics” |
| zero-bit watermark | detector 判断“是否存在 keyed signal”，不恢复外部消息 |

## 6. Formal definition of AtomicWatermarkUnit

### 6.1 Grammar-level definition

对 Python 3.12 strict parse tree \(T(D)\)，定义：

\[
AtomicWatermarkUnit := SimpleUnit \mid ClauseHeaderUnit.
\]

为消除“eligible”一词混淆，本文把prompt中的暂名`EligibleSimpleStatement`拆成两层：grammar层的`SimpleUnit`包含全部可定位small statements；policy层的`FORMAL_ELIGIBLE SimpleUnit`才是candidate。这样被排除的return/yield/import等仍在结构流中，但不会被误称为可嵌入。

`SimpleUnit` 是 grammar `simple_stmt` 中一个 individual small statement，不是整个 `simple_stmts` logical line。Python 官方规范允许同一 logical line 中由 semicolon 分隔多个 simple statements，因此：

```python
x = 1; y = 2
```

必须产生两个 units，边界由 CST small-statement nodes 和 exact `SEMI` tokens 确定，禁止字符串 split。[Python simple statements](https://docs.python.org/3.12/reference/simple_stmts.html)

`ClauseHeaderUnit` 是 compound statement 每个 clause 从 clause 起始 token 到控制 suite 的最终 depth-zero colon（含 colon），不含 suite。集合为：

- `if`, `elif`, `else`；
- `while`, loop-`else`；
- `for`, loop-`else`；
- `try`, `except`, `except*`, try-`else`, `finally`；
- `with`；
- `match`, `case`；
- `def`, `class`；
- `async def`, `async for`, `async with`。

`else` 必须带 `owner_clause_type ∈ {if, for, while, try}`，不能只记录字符串 `else`。`case` 是 `match` 的 child clause header。`except*` 与 `except` 不混同。type parameter list 属于 def/class header 文本的一部分。

### 6.2 Header span

- 多行 header 仍为一个 unit。
- token depth 按 `()[]{}` 维护；header colon 是 CST 指定 suite 之前最后一个 bracket-depth-zero `COLON`，可跳过 lambda 内部 colon。
- decorators 不属于 header span，也不是 AtomicWatermarkUnit；它们作为 ordered `StructuralAuxiliary`附着到随后 def/class header。每个decorator expression按注册node/role/operator topology编码，identifier做current-expression equality/role masking、literal只保留type/shape bucket；ordered sequence用`SHA256(CBOR_D(["decorators/v1",...]))`得到`decorator_fingerprint`，并进入该header的`UnitSkeletonV1`和child scope identity。Gate只看decorator count/flags，不看hash。格式/registered alpha-rename不改变fingerprint；增加、删除、重排或改变operator会改变header及child subtree anchors。
- inline suite 在 header 后拆为独立 simple units：

```python
if condition: x = 1; y = 2
```

线性化为 `if condition:`、`x = 1`、`y = 2`。

### 6.3 Simple-unit completeness

Python 3.12 simple-unit taxonomy 包括 expression、assignment、annotated assignment、augmented assignment、assert、pass、del、return、yield、raise、break、continue、import、future、global、nonlocal 和 `type` statement。`await`/`yield` 可嵌在 expression/assignment AST 内，必须由 descendant flags 记录，不能只看顶层 node type。

### 6.4 Non-carriers 与 auxiliary records

- comments、blank lines、encoding declaration和standalone whitespace不是syntactic unit；
- module/function/class docstring在grammar上仍是`SimpleUnit`并占ast ordinal，但`is_docstring=true`、`RULE_EXCLUDED`，永不成为carrier；
- decorators不是AtomicWatermarkUnit，而是`StructuralAuxiliary`；
- suite/body整体、多个相邻statements或compound body不是AtomicWatermarkUnit，也不是正式载体。

### 6.5 四层集合

\[
\mathcal U_{syntactic}\supseteq\mathcal U_{candidate}\supseteq
\mathcal U_{selected};\qquad
\mathcal U_{formal}\text{ 由 final replay 独立重算。}
\]

`selected` 与 `formal` 不保证相等：stop truncation、最终 parse、artifact mismatch 或 Post-Gate-X 重放都可使 selected 变 erasure；反之，原代码中本来就存在的registered **non-identity** rewrite form即使从未搜索，也可能经盲重放成为null-like match/mismatch。canonical identity form永远erasure。这个规则避免 detector 依赖隐藏 generation state。

## 7. Deterministic parser and linearization contract

### 7.1 Normative bytes、encoding 与坐标

`parser_contract_id = py312.12-utf8-ast-libcst-token-cbor-v2`。正式输入是**原始 bytes**，不是实现相关的 Unicode string。

1. 只接受UTF-8或UTF-8-SIG。允许的源编码声明只有首/次行PEP-263位置中ASCII case-insensitive、去两端空白后value恰为`utf-8`的`coding[:=]`声明；`utf8`,`utf_8`等别名即使Python可解也在v1拒绝。BOM可有可无；有BOM且cookie不是上述exact value、检测到其他codec、多个冲突cookie或decode error时返回`ENCODING_UNAVAILABLE`。physical newlines只允许LF或CRLF；lone CR拒绝。这是有意的scope contraction，不声称支持任意PEP-263 codec。
2. 解码使用 strict UTF-8。保留 raw bytes、每个 physical line 的 raw start、newline kind 和 `cp_to_raw_byte[k]` prefix table；CRLF 占两个 raw bytes，tab 占一个 code point/byte，UTF-8-SIG 的首行 base offset 为 3。
3. CPython AST `col_offset/end_col_offset` 按 decoded line 的 UTF-8 byte offset解释；`tokenize`/LibCST 的 code-point column 经 `cp_to_raw_byte` 映射。所有内部 span 统一为 raw-byte 半开区间 `[start,end)`；不存在直接比较两套 column numbers 的实现捷径。
4. 固定 CPython 3.12 patch/container 执行 `ast.parse(..., feature_version=(3,12), type_comments=True)`；pinned LibCST 与 CPython `tokenize` 解析同一 decoded text。任一 strict parser失败，final artifact `PARSE_UNAVAILABLE`；Tree-sitter recovery 永不进入 formal path。
5. AST/CST/token 按 raw-byte span、node category、owner suite 做双向对齐；一对多或跨 span 歧义令相关 subtree erasure，歧义影响超过全部 units 的 10% 则 document unavailable。

必须发布两份独立 reference implementations 和 gold vectors，至少覆盖：ASCII/Unicode identifier、multibyte string、tab、LF/CRLF、UTF-8 BOM/cookie、semicolon、inline/multiline header、lambda/slice colon、async、decorator、`match/case`、`except*`。两实现的 spans、unit order、CBOR 与 anchors 必须 byte-exact。

### 7.2 Exhaustive Python 3.12 ownership table

| grammar object | atomic kind | owner/span rule | v1 carrier status |
|---|---|---|---|
| `Expr, Assign, AugAssign, AnnAssign, Return, Delete, Pass, Raise, Break, Continue, Import, ImportFrom, Global, Nonlocal, Assert, TypeAlias` | `SimpleUnit` | individual CST small-statement span; terminal semicolon不含在unit CBOR | 仅第10节certified Assign/AnnAssign可formal |
| expression中的`Yield/YieldFrom/Await/NamedExpr` | descendant flags | 不另建unit；附着到owner SimpleUnit | non-formal |
| `if/elif/else` | `ClauseHeaderUnit` | clause start至owning suite depth-zero colon | non-formal v1 |
| `for/async for/while`及loop-`else` | `ClauseHeaderUnit` | `else`带owner enum | non-formal v1 |
| `try/except/except*/else/finally` | `ClauseHeaderUnit` | continuation分别建unit；`except`与`except*`不同tag | non-formal v1 |
| `with/async with` | `ClauseHeaderUnit` | header-only，不含suite | non-formal v1 |
| `match/case` | `ClauseHeaderUnit` | case是match child clause | non-formal v1 |
| `def/async def/class` | `ClauseHeaderUnit` | 包含type params/signature/bases至colon，不含decorators/suite | non-formal v1 |
| decorators | `StructuralAuxiliary` | ordered attachment to next def/class | never atomic/carrier |
| comments/blank lines/encoding cookie | auxiliary bytes | 不占unit ordinal | never carrier |
| module/function/class first string Expr | `SimpleUnit,is_docstring` | 正常占all-unit order | rule-excluded |

任何合法 Python 3.12 AST statement/compound owner未命中注册表都是 `UNKNOWN_GRAMMAR_UNAVAILABLE`，不能走“默认允许”。

### 7.3 Header、semicolon 与 source-order 算法

header colon 由 owning CST suite 确定，再用 token bracket-depth 校验；不能取文本第一个/最后一个冒号。多行 header 是一个 unit。`if c: x=1; y=2` 顺序为 header、`x=1`、`y=2`。深度优先 source-order固定为：current clause header → its suite → continuation header → continuation suite。decorator不占 `U_t`。

为同时定义DFS stream与第8节same-suite anchor，`suite`是CST中一个statement-list ownership object，而不是缩进字符串。一个compound statement的initial header以及`elif/else/except/except*/finally/case` continuation headers都作为其**parent statement-list suite**的immediate atomic items，按source order参与该parent suite的causal 4-gram；每个clause body（包括inline suite）另建一个由相应header拥有的child suite。child body units不进入parent suite的4-gram。`match` header是parent suite item，每个`case` header也是该parent statement-list中match compound的continuation-role item，而每个case body有独立child suite。DFS只决定全局\(U_1…U_T\)与Gate context：发出header，递归发出其child suite，再发出下一continuation header。owner/parent/transition enums由这棵ownership tree唯一导出。两实现必须对该suite ID、immediate-item序列和DFS order同时给gold，不允许把“前一个global unit”误用为anchor的“前一个same-suite unit”。

每个 suite 记录包含 excluded units 的 `ast_ordinal`，全文件另有 debug source ordinal；两者仅用于 diagnostics/schema joins。`candidate_ordinal` 只用于统计。三个 ordinal 均禁止进入 target anchor、RegionV1 或 exported model tensor。

### 7.4 Context stream 与 suite transition

`U_1,...,U_T` 是上述全局 source-order stream，而不是同 suite eligible stream。CTX predecessor可跨 suite，并附公开 enum：`same_suite / parent_to_child / child_to_continuation / child_to_parent / sibling_suite`、clipped depth delta和LCA role。excluded units仍在流中。具体结构路径只用role enum，不输入绝对ordinal或内容hash。

### 7.5 Prompt/generated boundary与incremental parse

generation audit 可记录 `prompt/generated/mixed` byte provenance，prompt-owned unit固定不搜索；该字段不进入shared Gate、anchor、Post-Gate-X、RegionV1或formal eligibility，因为blind extractor不可恢复它。v1全部headers非载体，因此prompt既有def/class与generated nested def/class不会形成不一致header evidence。其他prompt unit若被final extractor接受，只作为与key独立的null-like evidence，必须包含在final-replay/null calibration。

incremental Tree-sitter/CST只可发provisional close事件；`ERROR`、未闭合括号/string、ambiguous colon、mixed span不分配预算。最终 path只能由7.1 strict parser确认。

### 7.6 CanonicalCurrentV2 / deterministic CBOR

`canonical_current` 是 RFC 8949 deterministic CBOR。顶层固定数组而非自由map：

```text
[schema=2, unit_kind_u8, node_kind_u16, clause_kind_u8,
 owner_kind_u16, flags_u32, payload]
```

`payload`按公开Python-3.12 role table做preorder数组；每个node为`[node_tag_u16,[role_tag_u16,value]...]`，roles按注册表序而非反射/字典序。整数在signed/unsigned 64-bit内用最短CBOR，之外用RFC8949 tag 2/3最短big-endian magnitude，source integer bit-length>4096则document unavailable；bytes/text使用definite length；identifier保留strict UTF-8 bytes；string decoded value不做Unicode normalization。literal编码为`[literal_type,value]`：`None/bool/Ellipsis`用固定tag，float用big-endian IEEE-754 binary64 raw bits（保留-0/inf/NaN payload），complex为两个raw-bit fields，bytes/str/int/tuple递归。float/complex在formal carrier中禁止，但parser仍确定编码。list/dict/set等由AST role sequence而非运行时hash iteration编码。locations、whitespace、冗余parentheses、quote/numeric spelling、普通comments、line layout、terminal semicolon不编码。header payload只含current header-owned fields；simple payload只含该small statement。unknown node/role、indefinite-length或其他noncanonical CBOR fail closed；decoder limits（depth64、nodes4096/unit、bytes1MiB/unit）写入manifest，formal current另受更小bounds。

Python source可用ASCII escape产生lone surrogate（例如`'\ud800'`），而RFC 8949 text要求Unicode scalar。v1不引入replacement character或实现相关surrogatepass：在任何AST `str` value（包括docstring、annotation、f-string constant part、nested container和noncarrier unit）中发现`U+D800..U+DFFF`，整个document唯一返回`CANONICAL_UNAVAILABLE_SURROGATE`，不构造unit CBOR、anchor、DocumentIdentity、Gate tensor或evidence。两套实现必须对escaped lone/high/low surrogate、mixed scalar和nested tuple给完全相同状态；原始UTF-8 bytes仍按7.1 strict解码。

在该CBOR上再定义两个不同、不可混用的public projection：

- `RegionCurrentV2`只处理当前unit：按preorder维护identifier equality table，把每个identifier spelling的首次出现编号为0、1…，编码`[alpha_id, node_role(load/store/del/attribute/keyword/binding)]`；后续相同spelling复用同一编号。它不查scope、前后unit或symbol table；literal type与**完整payload**、node/role/operator结构均保留。因而一致alpha-renaming保持projection，而registered rewrite的operator/literal realization仍可改变projection。
- `GateProjectionV2`按第17.12节进一步mask identifier和literal payload，只作为模型输入。

RegionV1严格hash deterministic-CBOR `RegionCurrentV2`，不直接hash原始identifier；规则验证仍使用完整`CanonicalCurrentV2`。三者的serializer/projection hashes、role table、decoder limits和至少1,000 hand/fuzz gold vectors都发布。格式化不改变它们；AST-level equivalent realization rewrite可改变`RegionCurrentV2`，这正是carrier freedom。

## 8. Stable anchor and resynchronization design

### 8.1 Candidate-invariant `UnitSkeletonV1`

0.1 的 ordinal/virtual-slot DP 被删除：extractor没有原序列，不能比较“expected role at deleted slot”，更不能用keyed match score决定结构身份。v1改用final-code自包含的local anchor。

对suite内每个syntactic unit计算公开skeleton：

- formal `Assign/AnnAssign`：保留unit/node kind、target binding topology、annotation topology、type-comment presence和effect flags，**整个RHS替换为`[CARRIER_RHS]`**；
- header：保留clause/owner kind、parameter/binding/operator topology、ordered `decorator_fingerprint`与decorator count，mask用户identifier与literal value；
- excluded/noncarrier simple：保留完整node/role/operator topology；identifier只用**当前单元内**preorder首次出现表编号并附load/store/del/binding role，绝不读取scope符号表、前后单元或future declaration；literal只保留type/shape bucket；
- 永不含exact source offset、ordinal、candidate text hash、key、target、pool、retry或origin。

`d_i=SHA256(CBOR_D(["SAGE-C/skeleton/v1",skeleton_i]))`。注册rewrite certificate必须证明候选不会改变`skeleton_i`；否则候选在pool过滤阶段淘汰。

### 8.2 Causal structural 4-gram anchor

固定历史半径 `r_anchor=3`。对当前suite序列，只使用当前unit及其三个同suite前驱；缺失位置依次用三个有位置标签的suite-BOS sentinels补齐：

\[
w_i=(d_{i-3},d_{i-2},d_{i-1},d_i).
\]

因此在原始生成过程中，当前unit闭合后即可只从生成前缀构造 \(w_i\)、结构路径和anchor；two-phase v1在冻结 \(D_0\) 后重算同一值，final extractor也只从observed code重算同一值。anchor不读取future unit。`unit_role_i`是固定ownership-edge enum（`module_item/function_body/class_body/if_body/elif_body/else_body/loop_body/loop_else/try_body/except_body/try_else/finally_body/with_body/match_case_body`）与当前`unit_kind/clause_kind`的fixed tuple；它不含suite index、source ordinal、span或用户文本。module scope token为固定常量。child suite token递归定义为owner header在parent suite的**candidate-invariant local signature**加clause-role：

\[
\begin{aligned}
\ell_i &= SHA256(CBOR_D([\texttt{"local/v2"},\text{scope\_token},\text{unit-role}_i,w_i])),\\
\text{scope\_token}_{child} &= SHA256(CBOR_D([\texttt{"scope/v2"},\text{scope\_token}_{parent},\ell_{owner},\text{clause-role}])),\\
a_i &= SHA256(CBOR_D([\texttt{"anchor/v2"},\text{protocol-id},\text{scope\_token},\ell_i,\text{unit-kind}])).
\end{aligned}
\]

这里`CBOR_D`就是第7.6节deterministic CBOR，domain strings是UTF-8 bytes，hash/token fields是定长bytes，enums是注册unsigned integers；不存在含糊字符串拼接。三个suite-BOS skeleton digests分别绑定`BOS_MINUS_3/BOS_MINUS_2/BOS_MINUS_1`，不能复用一个无位置sentinel。

在每个scope内，若两个units的`(unit-role, w_i)`相同，则这些units均`ANCHOR_AMBIGUOUS_ERASURE`；若owner header ambiguous，其child suite全部erasure。不能用source order、keyed signs或lexicographic任选一个来破除歧义。`text_identity=SHA256(CanonicalCurrentV2)`只用于dedup，不进入anchor。

local anchor本身允许不同documents的同型结构重复；它只负责局部同步，**不再单独充当PRF address**。第8.5节的`DocumentIdentityV1`与它共同进入target HMAC，阻断不同identity之间的target复用。删除`document_nonce`后，detector API不得接受nonce或identity override字段；同一canonical code重复检测得到相同identity/anchors/targets/statistic，nonce/override sweep gold test必须全部拒绝未知字段。

### 8.3 可证明的局部影响，而非原序列恢复

若在一个suite插入/删除一个unit，且不改变owner header，则编辑点之前的anchors不变；编辑位置和其后至多`r_anchor=3`个原units的windows变化；此后的`w_i,\ell_i,a_i`恢复byte-exact。若编辑产生/消除重复4-gram，受影响重复组额外erasure。若owner header本身或其三个同suite前驱变化，header local signature可能变化，因此对应child scope token及child subtree不声称resync；它贡献new-anchor null-like evidence或因Post/ambiguity而erasure。

extractor没有“旧anchor changed”标签，因此局部变化后的unique unit仍会形成新local anchor；正文不把它虚称为恢复旧slot。更重要的是，任何改变完整AST identity的编辑都会根据第8.5节重绑定**全部**expected targets，所以local structural resynchronization不等于watermark evidence survival。primary detector没有alignment hypotheses、edit penalty或DP cap。

离线edit evaluation同时拥有原/改代码，必须分别报告：

1. untouched-unit exact-anchor survival；
2. affected-window erasure率与new-anchor random evidence率；
3. wrong recovery率——v1算法不声称恢复旧slot，故任何changed anchor被报告为not recovered；
4. `DocumentIdentityV1`是否保持；不保持时按预注册availability attack报告detection survival与完整edited-null FPR，不把expected-target reset算局部恢复。

### 8.4 Formatting、rename、reorder与duplicate

- formatting、普通comment与冗余括号不改变canonical AST/identity，anchor和target均应exact不变，仍须gold验证；
- alpha-rename因identifier role masking通常不改变local skeleton，但`DocumentIdentityV1`有意保留identifier spelling，故target stream全局重绑定；v1不承诺rename evidence survival；
- reorder改变被移动unit、插入/删除边界及各边界之后至多3个同suiteunits，不承诺这些anchors；
- insertion/deletion影响同suite局部窗口，header附近可影响child subtree；
- 大量或精心构造duplicate k-grams可擦除signal，是允许的removal attack；
- 同`text_identity`复制品按公开cluster规则只保留最早source span，其余erasure；该tie-break只用于duplicate suppression，不用于anchor identity。

### 8.5 Document-bound `DocumentIdentityV1`

Round-2发现local anchor复用会让same-key合法watermarked samples充当`anchor→target bit`字典。v1因此定义一个**由代码自身承载、不可由请求覆盖**的全模块身份。它不是随机nonce，也不写sidecar/tag；它只把HMAC的单元target地址绑定到identity与local anchor。该一位PRF输出**不是**独立的MAC tag、artifact签名或provenance认证，不能据此声称文档完整性或作者归属。

`IdentityProjectionV1(D)`先复制strict AST/CST ownership model，再对每个RHS匹配registered template的Assign/AnnAssign做唯一归一：direct identity form按typed AST解析；non-identity form独立运行`BlindCertificateV1`。若能唯一反演typed `ClosedValueV1`（non-identity还须通过全部hard steps），则把RHS换成唯一canonical identity AST；无唯一反演或certificate失败就保留current AST，不猜测。不存在selected-position map。

投影后生成两个definite-length CBOR bytestring。`CanonicalNormalizedModuleV1`用7.6同一Python-3.12 node/role注册表，从`ast.Module`根递归编码全部node fields；identifier spelling、literal payload、annotation、decorator、docstring、excluded/noncarrier statement与statement-owned`type_comment`均保留，locations与module `type_ignores`排除。`OwnershipShapeV1`用7.3的CST ownership tree递归编码`[suite_role_u16,code_object_kind_u8,ordered_items]`；每个ordered item恰为`[UNIT,unit_kind_u8,clause_kind_u8,unit_role_u16]`、`[DECORATOR_AUX,attached_unit_position_u32,canonical_decorator_expression_bytes]`或`[CHILD_SUITE,owner_unit_position_u32,clause_role_u16,child_suite]`。positions是当前array内0-based link，仅用于该完整identity序列化，不进入local anchor/Gate；decorator link必须指向其后紧邻decorator-run后的def/class header，child link必须指向此前owner header，array order本身保留statement/clause顺序。ordinary comments、shebang、encoding/protocol comments、whitespace、quote/numeric spelling和其他已由CanonicalCurrent排除的non-runtime surface信息不进入。unknown node/role、out-of-range/wrong-type link、非法link direction或任一noncanonical CBOR使identity unavailable。


```text
identity_preimage = CBOR_D([
  "SAGE-C/document-identity/v1", parser_contract_id, u16(1),
  CanonicalNormalizedModuleV1_bytes,
  OwnershipShapeV1_bytes
])
document_identity = SHA256(identity_preimage)
```

固定`u16(1)`是SAGE-C formal protocol-major，不是待填变量；它避免把包含identity测试向量的最终`protocol_id`递归放回自身，实际target另绑定完整`protocol_id`。生成端在\(D_0\)冻结、pool生成前计算identity，但**不得把它输入Pre、pool variant、candidate generation/order/count、allocator utility或Post-G/X tensor**；它只进入document-plan commitment，且target API仍在全部pool/composition commitment之后才解锁。direct identity forms无需额外compile；其余replay-shaped forms复用每unit一次BlindCertificate cache，最多512次并受第27.1节同一CPU/RSS上限，所有baselines同付这项mandatory identity cost且actual ms另报。对\(D_1\)重新计算必须与commit中的identity byte-exact，否则artifact被withhold并暂停发布。提取端完全从final code重复上述流程；任何unit normalization歧义、certificate unavailable、surrogate、unknown grammar或identity mismatch都令document `DOCUMENT_IDENTITY_UNAVAILABLE`，不得降级为local-only target。

在SHA-256 collision resistance与HMAC-SHA256 PRF假设下，两个不同identity即使`a_i`相同也对应不同PRF inputs；从identity A观察到的region bit不预测identity B。identity-equivalent完整模块（允许format/comment变化及registered carrier alternatives）仍共享target stream；复制这种完整normalized artifact或把它重新surface-realize不能证明独立作者归属，这是明确的provenance non-goal。安全代价同样明确：identifier rename、任意literal/statement/ownership插入删除或重排会改变identity并全局重绑定targets。v1选择anti-mosaic而非AST-edit availability；任何未来的scope/Merkle局部绑定必须作为major protocol重新证明scope-transplant、authentication、query family与null，而不能在本协议中切换。

## 9. Context window definition

对当前 \(U_t\)：

- CTX-1：`[U_{t-1}, U_t]`；
- CTX-2：`[U_{t-2}, U_{t-1}, U_t]`；
- CTX-3：`[U_{t-3}, U_{t-2}, U_{t-1}, U_t]`。

v1明确选择第7.3节的**全局depth-first source-order**，不是“同suite内上一个eligible unit”：它保留header→body、child suite→continuation等真实生成邻接，同时excluded units不会被悄悄跳过；transition/LCA features让模型区分跨suite。same-suite predecessor另作预注册ablation：越界位置补`[SUITE_BOS]`，不得与global结果混训后冒充context增益。若same-suite在unseen groups更稳，只能作为新artifact/protocol在validation选择一次，不能逐sample切换。

缺失 predecessor 使用 `[BOS_UNIT]`，不是复制当前单元。每个unit输入：第17.12节`GateProjectionV2` tokens、unit/node/clause/parent/code-object types、suite depth、role path enums、transition type、is_header、is_multiline_header、inline-suite flag、decorator role count。`origin`、key、target bit、candidate count、attempt index、retry/failure、generator/repository id全不作为model input。

使用一个shared model；同anchor三views做paired batches/context dropout，不把width与budget混合。Pre只读frozen \(D_0\)；Post-G可读冻结D0 context；Post-X训练含key-independent final assemblies但其tensor只含current unit，formal extraction只读 \(D_1\)。predecessor编辑可改变随后3个Pre/Post-G诊断视图，却不会直接改变这些units的正式Post-X state或RegionV1；local anchor按第8节同样只影响编辑边界及其后三个同suiteunits。RegionV1输入严格为current-only `RegionCurrentV2`。

## 10. Candidate eligibility rules

### 10.1 Policy states与固定分母

每个 syntactic unit 被公开 rule engine 分到：

- `FORMAL_ELIGIBLE(D0)`：baseline identity-form可进入部署candidate pool；
- `PROMPT_OWNED_NOT_CANDIDATE`：语法/identity形状可能相同，但generation provenance表明来自prompt；不进pool或candidate denominator，且origin永不进Gate/anchor；
- `FORMAL_REPLAY_ELIGIBLE(D)`：final **non-identity** registered form通过BlindCertificate后可进入evidence hard path；identity只得到`BLIND_CERTIFIED_NON_EVIDENCE`并erasure；
- `ORACLE_ONLY`：只收集 feasibility/negative labels，不能形成正式载体；
- `RULE_EXCLUDED`：永不生成 candidates；
- `SYNTACTIC_ONLY`：例如零自由度 header，只用于结构/ordinal。
- `ZERO_ENTROPY_SYNTACTIC_ONLY`：grammar形态可解析但注册rewrite enumerator产生少于两个canonical-distinct **non-identity** forms；在Gate前确定。

`candidate denominator`永远是provider在冻结\(D_0\)上标出的全部generated-span `FORMAL_ELIGIBLE(D0)`，在Gate、allocator、pool和target之前冻结；prompt-owned形状相同单元单列，不进入该分母。另报`pre_selected/attempted/dual_complete/embedded/formal`分母。Gate不能通过缩小candidate denominator美化成功率。final blind evidence denominator另报全部public replay-shaped units，不能偷换成embedded positions。excluded不作为普通q0/q1负样本；它们是规则测试与OOD/hard-negative audit。

### 10.2 v1 formal simple-statement allowlist

初始只允许 `Assign` 与有 value 的 `AnnAssign`。`Assign` target必须是单一`Name`，或仅由`Name`构成且形状固定的tuple/list unpacking target；`AnnAssign` target必须是单一`Name`。两者均禁止Attribute/Subscript/Starred，且一个statement只能有一个assignment target（禁止`x=y=1`）。generation baseline RHS必须是**canonical identity literal形式的closed immutable value**，不能读取Name；final replay RHS则可为该identity或第10.3节唯一可反演的registered form。0.1的`SafeAtom := Name`被删除，因为在强不变量下它几乎没有canonical carrier entropy。`ClosedValueV1`为：

```text
ClosedAtom := Constant(None | bool | int[-2^31,2^31-1] |
                       str_utf8_len<=128 | bytes_len<=128)
ClosedValueV1 := ClosedAtom | Tuple(ClosedValueV1*, total_nodes<=16)
```

`CanonicalClosedLiteralV1`不是实现相关`repr`。它是generation baseline/pool的唯一source printer：`None/True/False`使用exact ASCII token；int使用无`+`、无underscore、无前导零的base-10 magnitude（零为`0`，负数为`-`加magnitude）；str使用单引号，`'`与`\\`转义，ASCII 0x20–0x7e中其余字符原样，控制字符与非ASCII Unicode scalar分别用lowercase `\\xhh`、`\\uhhhh`或`\\Uhhhhhhhh`，surrogate拒绝；bytes同样单引号，0x20–0x7e中除`'`/`\\`外原样，其余用lowercase`\\xhh`；tuple为`()`或`(`加逗号分隔的递归children加必需trailing comma再加`)`。printer不插入空白。generation-side `FORMAL_ELIGIBLE(D0)`要求baseline RHS bytes等于该printer输出；例如双引号、不同escape spelling或hex int虽可解析但不是generation candidate。

所有rewrite template先构造注册AST，而不是对source做字符串替换，再由`CanonicalExprPrinterV1`按固定precedence/parenthesis表打印。`v`位置总是嵌入上述canonical literal AST；一元/二元/条件/布尔模板的括号、空白与operator token有唯一generation输出。提取端按**AST template topology和typed Constant values**反演，不要求formatter保留quote、escape、parenthesis或空白拼写；只要反演typed value唯一、bounds满足、重建canonical identity后BlindCertificate/OperationalDigestV2全过即可。root匹配`ClosedLiteralAST`（Constant、递归Tuple，以及仅用于负int literal的`UnaryOp(USub,positive-int-Constant)`）的任何合法lexical spelling都归identity/non-evidence；只有注册non-identity operator topology可成为evidence。完整precedence表、每个template的source/format gold vectors和printer hash属于`rewrite_catalog_hash`。

### 10.3 Normative `RewriteCertificateV1`

候选不是任意LLM paraphrase，而是下表有限template的实例。`v`与identity operands均为closed literals；所有中间AST nodes≤24，folded constant bytes≤256。

| family | denotation type | ordered templates（不适用者写sentinel） | static precondition |
|---|---|---|---|
| `INT_ID_V1` | exact `int`，不含bool | `v`, `+v`, `v+0`, `v-0`, <code>v&#124;0</code>, `v^0`, `v*1`, `v//1`, `~~v` | exact evaluator无overflow/exception；结果type/value与v相同 |
| `BOOL_ID_V1` | `bool` | `v`, `not(not v)`, `v and True`, `v or False`, `True if v else False` | closed bool；compiled operational digest相同 |
| `STR_ID_V1` | `str` | `v`, `v+""`, `""+v`, `v*1` | resulting UTF-8≤128；digest相同 |
| `BYTES_ID_V1` | `bytes` | `v`, `v+b""`, `b""+v`, `v*1` | resulting bytes≤128；digest相同 |
| `TUPLE_ID_V1` | recursive immutable tuple | `v`, `v+()`, `()+v`, `v*1` | total nodes/bytes bounds；digest相同 |
| `NONE_ID_V1` | `NoneType` | identity only | always `ZERO_ENTROPY`; nonformal |

模板先由独立small-step exact evaluator验证Python builtin type/value：只实现上表grammar，int按Python无界整数后检查bounds，bool与int严格区分，str按Unicode code points、bytes按octets、tuple递归；每一步先检查node/byte/gas上限，任何未注册operator、exception或resource overflow为FAIL而非unknown-positive。第二个独立evaluator必须给全部bounded catalog forms相同typed result。

再把baseline与单一替换后的**完整module**用manifest固定的CPython 3.12 patch执行`compile(source,"<sage-c>","exec",flags=0,dont_inherit=True,optimize=0)`，比较alias-preserving `OperationalDigestV2`。每次compile都在独立fresh process中产生root module code object；serializer维护一张仅在本次遍历中使用的object-identity map。首次访问支持对象时按确定性遍历次序分配连续`u32 object_id`并编码`DEF(object_id,type_tag,payload)`，再次访问同一运行时对象编码`REF(object_id)`；raw address/`id()`数值绝不序列化。因此两个process只要value与sharing graph同构，就会产生相同first-visit IDs；equal-but-distinct与same-object会产生不同bytes。

遍历root为module code object，identity map跨所有nested code objects共享。`DEF(code)` payload按固定次序编码`co_argcount,co_posonlyargcount,co_kwonlyargcount,co_nlocals,co_stacksize,co_flags`，六者均须在`0..2^32-1`并编码为deterministic-CBOR unsigned integer；再遍历对象字段`co_code,co_consts,co_names,co_varnames,co_cellvars,co_freevars,co_exceptiontable,co_name,co_qualname`；`co_consts`及其他tuple自身也作为对象节点而非扁平values。支持节点为`None,bool,int,float(raw IEEE u64 bits),complex(two raw u64 bits),str strict UTF-8,bytes,tuple,frozenset,Ellipsis,code object`。int编码为`[sign_u8,minimal_unsigned_big_endian_magnitude]`且bit length≤4096；object ID在`0..2^32-1`。每个str节点还由protocol-bound、read-only CPython C helper原样记录`PyUnicode_CHECK_INTERNED`的四个3.12状态：`SSTATE_NOT_INTERNED`、`SSTATE_INTERNED_MORTAL`、`SSTATE_INTERNED_IMMORTAL`、`SSTATE_INTERNED_IMMORTAL_STATIC`；其他值或helper/runtime hash不符fail closed，helper不得调用`PyUnicode_Intern*`改变状态。四态合同与固定3.12 patch一并绑定；Python 3.12官方变更说明明确区分`IMMORTAL`与`IMMORTAL_STATIC`。[Python 3.12 C API changes](https://docs.python.org/3.12/whatsnew/3.12.html#c-api-changes) tuple按index；frozenset先以不含identity的recursive `ValueKeyV2(type,value)` bytes升序，equal key但typed value不等即fail closed，再按该序访问children；支持类型不能形成source constant cycle，检测到cycle-to-incomplete-DEF、unknown type、object数>1,000,000或depth>128均fail closed。singleton与small int/string等也走同一`DEF/REF`规则，不假设CPython必然intern。整个graph用deterministic CBOR fixed arrays编码并逐byte比较，另存SHA-256。

唯一有意忽略的是`co_filename,co_firstlineno,co_linetable`及其派生position tables，因为threat model排除source/position观察；CPython 3.12若出现任何其他code-object field，release checker必须显式列为included或documented-nonsemantic并触发新protocol，不能默认忽略。fresh process固定hash seed、locale、allocator/runtime image且禁止quickening profile复用。`OperationalDigestV2`相等覆盖固定fresh-process正常执行中由constants/code-object graph产生的普通`is/is not` equality pattern及同一fresh process中相同程序顺序触发的`sys.intern`结果；它不承诺跨外部预热interpreter、C-extension地址或绝对`id()`数值。release suite必须在两个独立serializer/compile processes上覆盖same-object与equal-distinct strings（含四种intern state和`sys.intern` probes）、bytes、small/large ints、nested tuples、frozensets、singletons及跨nested code objects sharing；任何included field、intern state或`DEF/REF`拓扑差异拒绝。

因此语义承诺仅覆盖固定CPython正常执行，不覆盖`sys.settrace`、source/code-object introspection或位置敏感coverage。该限制必须写入threat model，不能把certificate称为一般Python contextual-equivalence proof。

`AnnAssign` annotation、targets、type comment与statement surroundings byte-identical。每个candidate保存`template_id,is_identity,input_value_digest,exact_eval_digest,operational_digest_before/after,certificate_verdict`。独立verifier不运行LLM即可重放。每个baseline equivalence class完整枚举后按`CanonicalCurrentV2` dedup；identity form固定保留为pool record 0但`evidence_admissible=false`，少于2个non-identity canonical forms在Gate前标`ZERO_ENTROPY`。因此\(A=2\)（baseline+1 rewrite）的formal dual-complete率按定义为0，只作cost/label sanity baseline。

**`BlindCertificateV1(final_module,current_unit)`是提取端硬合同。** 它不需要隐藏\(D_0\)：(1) final RHS匹配全部可能的registered templates；所有匹配必须反演出相同`ClosedValueV1` typed value，若无解或typed values不一致则失败；多个template仅因对称性重合（如空string/tuple）时取最小public template ID；(2) 把当前RHS替换为该value的唯一canonical identity literal，得到\(D_{id}^{(i)}\)，其余final module bytes不变；(3) strict parse、target/annotation/skeleton不变量全过；(4)比较`OperationalDigestV2(final_module)==OperationalDigestV2(D_id^(i))`，包括完整`DEF/REF` alias graph。四步全过且selected canonical template为non-identity才是`BLIND_CERTIFIED_EVIDENCE`；identity-form返回`BLIND_CERTIFIED_NON_EVIDENCE`并强制erasure。这样extractor能证明“当前final realization属于注册operational-equivalence class”且从最终语法知道它携带显式registered marker，不需要隐藏selected-position sidecar；但仍只在固定CPython与第3节排除的introspection范围内成立。

字符串/整数域巨大，规范**不声称穷举全部values**。catalog release必须对有限template AST patterns做两两unification/overlap table，并由独立symbolic checker证明“任一overlap的反演typed value相同”；再对每个observed training/evaluation value完整枚举/dedup，加上整数边界、空/最大长度string/bytes/tuple和≥1,000,000 deterministic property-fuzz cases。symbolic proof失败或任一counterexample会删除相关template family/发布失败。该hard bit不能被Post-X覆盖。

### 10.4 禁止构造与Oracle-only classes

禁止float/complex/NaN、f-string、list/set/dict、Name读取、Attribute、Subscript、Slice、运行时BinOp/BoolOp/Compare/IfExp、Call、Await、Yield/YieldFrom、NamedExpr、Lambda和comprehension。它们可能触发overload、descriptor、iteration、allocation/aliasing、exception timing或scheduling。

- AugAssign；
- 所有一般 `Expr` statement；
- assignment/Expr 含 Attribute/Subscript、BinOp/BoolOp/Compare/IfExp on Names、Call、container literal/comprehension、lambda、walrus；
- `await`、`yield/yield from`；
- 可能改变 evaluation order、mutation、I/O、exception timing 或 coroutine scheduling 的表达式。

这些只做隔离Oracle。任何promotion都改变rewrite language与protocol ID，必须重新做parser/region/null/review；不能由一次Gate分数隐式开放。**理论警告：如果一种region representation对Python denotation完全不变，则等价候选必落同一region，容量为零。SAGE-C的RegionV1因此只能划分“经certificate证明等价的不同realizations”，不能被宣传为denotation-invariant semantic embedding。**

## 11. Excluded statements and rationale

| 类别 | 状态 | 原因 |
|---|---|---|
| pass/break/continue | RULE_EXCLUDED | 极低自由度；控制流敏感 |
| raise | RULE_EXCLUDED | exception type/timing/control transfer |
| import/import_from/future | RULE_EXCLUDED | 模块加载、副作用、compile semantics |
| global/nonlocal | RULE_EXCLUDED | scope binding directive |
| del | RULE_EXCLUDED | mutation/lifetime semantics |
| assert | RULE_EXCLUDED | optimization flag 与 exception semantics |
| return | RULE_EXCLUDED | API/control transfer；遵从第一版约束 |
| type statement | RULE_EXCLUDED | 3.12 scope binding 与 lazy type-alias semantics |
| docstring | RULE_EXCLUDED | runtime `__doc__`/documentation contract |
| yield/yield from | ORACLE_ONLY, v1 disabled | 改变 generator function 与 suspension behavior |
| await | ORACLE_ONLY, v1 disabled | scheduling/cancellation/exception timing |
| side-effecting calls | ORACLE_ONLY, v1 disabled | 无 blind proof of equivalence |

这些 units 仍进入结构流、role path、ast ordinal 与 contexts。训练中不把它们混为 `q0=q1=0` 的普通样本；规则层直接排除，另建 OOD audit set，以便验证 Gate 遇到 policy-violating input 时 fail closed。

## 12. Compound header handling

### 12.1 Syntactic identity 与 carrier eligibility 分开

所有第 6 节 header 都是 AtomicWatermarkUnit。carrier policy：

- `else:`, `try:`, `finally:`：`SYNTACTIC_ONLY`，canonical current-unit semantic freedom 为零，理论容量为零。
- `def/class/async def`：v1 `RULE_EXCLUDED`；改变 name、signature、defaults、annotations、bases/metaclass/decorators 可改变 public API 或 import-time behavior。prompt/generated origin 又不可 blind 重建。
- `if/elif/while/for/async for/with/async with/match/case/except/except*`：`ORACLE_ONLY` diagnostics；v1无promotion路径。

**v1永久决定：所有ClauseHeaderUnit均不形成formal carrier。** 原因不只语义风险：全document key-neutral planning先冻结完整baseline body，而header若在body后按target改写会破坏chronology；若重生成body，后续pool又间接依赖target-selected header。这个矛盾不是阈值可修复。任何header carrier必须是独立major protocol、独立文档和fresh review；不能从v1“promotion”。

### 12.2 v1中header的唯一作用

header只进入syntactic stream、suite ownership、candidate-invariant skeleton、anchor scope token和Gate context。incremental header永远只是parser state，不是provisional watermark evidence。`else/try/finally`等零自由度header作为zero-entropy controls；其余header可做**非promotion**诊断，测量短单元表示退化与控制流风险，但任何结果都不能改变v1 eligibility。

### 12.3 独立header研究的最低前提

若未来研究header，必须先提交不读取target-selected header的两侧body branch commitment、完整成本上界、body/header pair verifier以及“retry only current header”的证明，并创建新protocol与fresh review。上下文仍不得进入header region。0.3不提供这个算法，因此对v1的结论是明确`NO HEADER CARRIER`，也不存在promotion threshold。

## 13. Data sources

数据集不是把任意公开 GitHub 代码直接抓来训练，而是建立可审计的 provenance ledger。每个 source artifact 在任何派生前必须有 `source_id`、repository URL/commit、SPDX expression、文件路径、license evidence hash、采集日期、删除/opt-out 状态和 legal-use decision。只有经项目法务/数据治理允许的 permissive allowlist（初始为 MIT、BSD-2/3-Clause、Apache-2.0、ISC；MPL/GPL 等另行评估其再分发义务）进入训练；无法确定 license 的代码拒绝。CodeSearchNet 只作为候选 repository 索引，不能把其聚合数据自动等同于“许可清晰”。

数据来源分为七个互斥 provenance strata：

1. **真实仓库源码。** 按 repository 固定 commit 采样可解析 Python 3.12 文件，保留 project tests、type hints 和依赖 manifest；覆盖库、CLI、数据处理、异步服务、科学计算等领域。仓库必须通过 license allowlist、secret scan 和 clone quarantine。
2. **许可明确的教学/合成任务。** 自建、委托创作或明确允许再使用的函数级任务，专门覆盖 semicolon、inline suite、多行 header、nested flow、`match/case`、`except*`、decorator 和 async。合成任务保留生成规范，不能与真实测试集模板近重复。
3. **真实 prompt 上的当前 generator 轨迹。** prompt 来自训练 partition 的 docstring/signature/issue-style specification；收集正确与错误 completion、完整代码和 incremental prefix。prompt 只用于数据生成，不成为 Gate 输入或 detector 依赖。
4. **多 generator 候选。** 至少包含当前目标 generator、一个不同家族的开源 code model 和一个非同架构/非同 tokenizer generator；记录 generator family/version 仅用于分层、nuisance audit 和 leave-one-generator-out 评估，严禁作为 Gate feature。
5. **语义破坏与 hard negatives。** 从正确候选构造 evaluation-order swap、operator/constant mutation、binding change、exception swallowing、await/yield relocation、API-signature change、header predicate inversion 等；每个 mutation 都记录 operator 和是否由 tests/human/static oracle 检出。未被 tests 检出的 mutation不能被标为语义保持。
6. **编辑与解析 stress corpus。** 对合法样本产生 formatter、identifier alpha-renaming、comment change、局部 equivalent rewrite、单 unit insertion/deletion/reorder 和 header/body edit；另保留 truncated/unbalanced/incomplete prefixes，只训练 provisional parser/robustness auxiliary tasks，绝不作为 final formal evidence。
7. **same-key donor/mosaic security corpus。** 只在null/attack partitions用独立audit keys生成合法donor outputs；attacker view仅含donor final code与public artifacts。target来自clone-disjoint components，要求local skeleton/anchor可重复但`DocumentIdentityV1`不同；完整identity-equivalent copy单列，不混入forge FPR。donor、target、attack constructor tuning和held-out test按repository component与audit key双重隔离，绝不进入Gate训练。

HumanEval 与 MBPP 以及其已知衍生/镜像只进入封存的 external evaluation；不训练Gate、不设计RegionV1/rewrite catalog，不做threshold/erasure/parser policy调整。[HumanEval](https://arxiv.org/abs/2107.03374) 与 [MBPP](https://arxiv.org/abs/2108.07732) 的任务文本和 canonical solutions 均进入 contamination signatures。若训练 source 与其 MinHash/AST clone 命中，整个 connected component 从训练删除。

每个 stratum 都同时覆盖：正确/错误代码、可解析/不可解析前缀、simple/header、低/中/高候选熵、候选为空、单侧可达和双侧可达。数据报告按 source、unit class、generator、correctness、parse state、entropy bin 和 edit type 分层；任何一个大 stratum 不得贡献超过训练 anchors 的 35%，防止模型把 repository 或 generator style 当 feasibility。

## 14. Data generation pipeline

### 14.1 Split-before-expand 流程

数据生成必须按以下不可交换顺序执行：

1. ingest provenance 与 license decision；
2. secret/PII scan，固定 source bytes 和 source hash；
3. 建立 repository/task/function/template/clone 关系图，先分配 split；
4. 仅在各 split 内 strict parse、抽取 scope 和 AtomicWatermarkUnit；
5. 生成DocumentIdentity、anchor、current-unit canonical form 和 structural metadata；DocumentIdentity/hash字段只作binding/audit，禁tensor；
6. 在各partition内完整生成第14.2节四个key-independent public pool variants；calibration/risk/test另标由anchor实际路由的deployment variant，不接受外部seed；
7. 从同一 pool 计算 region-0 与 region-1 counterfactual labels；
8. 做 syntax/static invariant/unit tests/property tests/sandbox execution 和人工审计抽样；
9. 生成 CTX-1/2/3 paired views、edit variants、hard negatives、Post-Gate labels、exact Post-X tensor cells与独立null/mosaic records；
10. 写 immutable records、parent hashes 和 data-card counts；任何过滤都产生日志但不写 secret。

禁止先生成 anchors/candidates 再随机拆行。一个 function 的原始版本、所有 anchors、candidate attempts、generator variants、mutations、retry seed blocks 和 CTX views 由同一 `group_component_id` 锁在一个 split。

### 14.2 Anchor、rewrite enumeration 与attempt accounting

对合法、无key baseline \(D_0\)，strict parser先抽取全部syntactic units，计算`UnitSkeletonV1`、k-gram anchors、eligibility和zero-entropy状态。只有`FORMAL_ELIGIBLE`进入pool数据。

`A`永远表示**pool中总记录数**，不是rewrite调用数：

\[
C_{r,0}=D_0[U_i],\qquad C_{r,1},\ldots,C_{r,A-1}=\text{publicly ranked RewriteCertificateV1 alternatives/sentinels}.
\]

每个anchor只执行**一次**完整catalog枚举、certificate construction与canonical dedup，得到immutable `BaseCandidateCatalogRecord`；CTX views绝不重新生成candidate。固定\(R_{pool}=4\)个public pool variants只是对这同一物理catalog的四个deterministic rank/subset projections，不调用generator、不新建candidate bytes、不改变任何candidate label。非baseline form \(c\)在variant \(r\in\{0,1,2,3\}\)中的rank为

\[
h_{i,r}(c)=SHA256(CBOR_D([\texttt{"SAGE-C/pool-order/v2"},protocol\_id,anchor_i,u8(r),template\_id(c),CanonicalCurrentV2(c)])).
\]

按unsigned digest、template ID、canonical bytes升序取前\(A-1\)项；不足写按catalog ID固定排列的`NO_CANDIDATE(template_id,reason)` sentinels。初始`A=6`即baseline+5 additional attempts。formal deployment的唯一variant是

\[
r_i^{dep}=OS2IP_{BE}(SHA256(CBOR_D([\texttt{"SAGE-C/pool-variant/v2"},protocol\_id,anchor_i]))[0:2])\bmod4.
\]

它只由candidate-invariant public anchor确定，不是caller nonce，也不读取future/current candidate realization；相同\(D_0\)/protocol的pool跨keys byte-identical。训练为soft feasibility保存四个projection index lists，但同anchor的CTX-1/2/3共享同一个base catalog、同一四-projection family和同一labels；deployment只materialize其确定的一个projection并在target前检查actual dual-completeness。任何LLM proposal只作generator-shift、Post-G hard-negative或oracle-only诊断，除非它精确归一到上述已注册form；归一后仍按同一公开rank，不增加count或模板语言。

候选只能替换exact current span，不得增删sibling、重写prefix、改变skeleton或header/body。v1没有header pool。每个candidate的region、certificate、hard filters与blind scores均在key不可访问阶段计算。

### 14.3 分层语义与功能标签

单元级“语义保持”采用证据层级，绝不把神经相似度当真值：

- `E0_PARSE`：strict parse 且 span/structure 合法；
- `E1_STATIC`：binding、free names、effects、calls、control flags、type/comment constraints 不变量通过；
- `E2_TESTED`：原 project tests、task tests、metamorphic/property tests 在隔离 sandbox 中与 baseline 同结果；
- `E3_DIFFERENTIAL`：多输入 differential execution 的返回值、exception class、stdout/stderr、observable mutations 一致；
- `E4_AUDITED`：高风险/抽样 pair 经双人 blinded review 一致。

`pair_valid=1`要求RewriteCertificateV1、E0、E1、`OperationalDigestV2`全过；有可执行tests时还要求E2/E3，缺失tests不会把certificate-positive改成“数学等价”，而是单列`dynamic_evidence=unavailable`。Oracle分别报告certificate-only与certificate+dynamic两层。失败candidate按首失败层标hard negative。sandbox timeout/nondeterminism/environment dependency为unknown。该证据只支持固定CPython operational contract，不证明一般程序等价。

### 14.4 Region、Post-Gate-G、Post-Gate-X 与 final replay labels

`RegionV1`按第20.1节从candidate的current-only `RegionCurrentV2`产生`0/1/boundary`与64-bit margin；它不训练、不读context/key。每个**实际replicate pool**分别记录`Y_{0r},Y_{1r},Y_{both,r}=Y_{0r}Y_{1r}`、各侧admissible比例与最小rank，不能把不同pools的单侧成功拼成both。

标签和authority分开：

- `Post-Gate-G`：baseline/candidate pair输入；labels为certificate、E0–E4、quality与pair rank。只在generation/filter/ranking使用，extractor没有该输入且不运行它。
- `Post-Gate-X`：只输入final current unit（前三slots BOS）；soft labels仅为current-observable/intrinsic的quality rubric与format/projection-alpha-rename robustness；reference replay由artifact conformance给hard bit。syntax/structure、duplicate/correlation由规则层计算，**不含pairwise semantic head或`p_formal(relative baseline)`**。`rename_target`只测current projection/Gate稳定，不声称`DocumentIdentityV1`或expected target稳定。
- `FinalReplayCalibrationRecord`：用public RNG在已封存pools上抽取完整counterfactual assemblies，包含natural D0、自身baseline、dual-incomplete fallback、多个replacement D1、prompt-like prefix、format/rename/edit，以及`random-side marker null`：后者完整运行key-free Pre/allocator/pools/composition/guard，但每个dual slot用`SHA256("SAGE-C/null-side/v1"||anchor||null_release_id)&1`选预提交winner，绝不调用detector key。每次重新parse、anchor、Post-X、RegionV1；actual keyed D1只进held-out test。

X labels有唯一recipe。`CanonicalRaterDisplayV1`由pinned AST pretty-printer把**当前一个unit**渲染成标准空白的UTF-8 monospace plaintext，并在首行显示公开`unit_kind`；不显示前后unit、module、baseline、region、target、template ID、anchor、pool、repository、generator或origin。三名rater只看该display与冻结的1–5 readability/obvious-suspicion rubric。`quality_target=1`要求public lint/complexity hard rules全过、median≥4且无人勾选suspicion；不足3名即unknown。每个blind rater batch分别计算readability的ordinal Krippendorff \(\alpha_{ord}\)（距离`(r-r')²/16`）和suspicion的nominal \(\alpha_{susp}\)（不等为1、相等为0），任一<0.67时该batch全部human labels为unknown；不把两类回答拼成一个未定义alpha。display bytes、rubric/distance version、随机化与rater-release hash写入label manifest，独立团队对同一row可byte-exact重建stimulus。

`format_target=1`要求固定formatter artifact运行后current `RegionCurrentV2`与`GateProjectionV2` byte-exact；`rename_target=1`要求预注册的8个capture-avoiding local alpha-renames全部保持这些**current-unit fields**，但文档target因identity改变仍重绑定。formatter/rename版本上的strict parse与BlindCertificate也必须另行PASS，但它们只进入hard conformance记录，绝不混入这两个soft targets；否则current-only X会被要求预测自己看不到的module-level alias/certificate状态。`intrinsic_accept_target=quality∧format∧rename`，仅三项全known时定义。parser、BlindCertificate、backend replay和duplicate/cluster不是soft labels，分别是hard bits。人工/编辑真值只作监督，绝不进入tensor。

在训练任何Post-X前执行`PostXIdentifiabilityGateV1`。以导出的完整fixed-shape tensor **bytes**（不仅是SHA；hash相同仍逐byte确认）分cell \(h\)，只用label-known的model-validation rows计\(n_h^+,n_h^-\)。必须有\(N^+\ge1000,N^-\ge5000\)、至少20个repository components且单component占比≤10%；否则直接`IDENTIFIABILITY_SAMPLE_INSUFFICIENT`并判当前learned-X路线NO-GO。报告`collision_mass=Σ_{h:n_h^+n_h^->0}(n_h^++n_h^-)/N`与不可约binary error lower bound`E_Bayes=Σ_h min(n_h^+,n_h^-)/N`。再把每个cell视为不可拆0/1 item，令negative budget \(B=\lfloor0.002N^-\rfloor\)，用exact integer 0/1 knapsack求

\[
C_{oracle@0.2\%}=\frac{1}{N^+}\max_{A\subseteq H}\left\{\sum_{h\in A}n_h^+:\sum_{h\in A}n_h^-\le B\right\}.
\]

DP按tensor bytes lexicographic order处理，equal-positive ties取较少negative、再取lexicographically smaller accepted-cell bitset；row/group bootstrap以repository component为重采样单位。它是deployable-input information ceiling，不是模型实验结果。若point estimate<0.50或95% component-bootstrap lower<0.40，learned Post-X在Layer-2唯一裁决为`DELETE_LEARNED_POST_X_AND_REVIEW_RULE_ONLY_MAJOR_PROTOCOL`；当前learned-Gate路线NO-GO，不能把几乎全erasure归咎于模型容量。通过ceiling仍须第24节独立risk certification；unknown-label coverage另要求≥70%，否则同样NO-GO。

靠近RegionV1 interval、Post-X accept/reject thresholds、backend不一致、anchor ambiguity或label unknown的样本进入erasure/rejection训练。自然或fallback unit不是“relative semantic positive/negative”；它们只用于blind acceptance/null calibration，避免不可识别标签冲突。

### 14.5 不可解析前缀

incremental CST/token recovery 可产生 `provisional_parse_label`，训练生成时预算预测；它不能生成 final label。每条 prefix 保存 truncation position、open-bracket/indent state、recovery ambiguity count 和最终 continuation 的 strict parse outcome。最终代码 strict parse 失败时整个 document 是 `INVALID_DOCUMENT`，没有单位证据。

## 15. Paired CTX-1/2/3 construction

对每个 anchor \(a_t\) 只产生一次 candidate-pool family：

\[
\mathcal C_t=\{C_{t,r,j}:r=1,\ldots,R;\ j=0,\ldots,A-1\},
\]

初始 \(R=4,A=6\)。从完全相同的 `AtomicUnitRecord`、candidate IDs、labels、attempt budget 和 verifier results 派生三个 `ContextViewRecord`：

- CTX-1：\([U_{t-1},U_t]\)；
- CTX-2：\([U_{t-2},U_{t-1},U_t]\)；
- CTX-3：\([U_{t-3},U_{t-2},U_{t-1},U_t]\)。

缺失前序位置用带 position 的 `[BOS_UNIT]`，不能复制当前 unit。三 view 的唯一允许差异是可见 predecessor 数量和由此确定的 transition masks；pool、attempt order、candidate labels、current unit bytes、split 和 sample weight 完全相同。paired views 在同一 minibatch 内出现，或用 group sampler 保证一个 optimization window 内齐全。

Pre-Gate paired views固定来自同一 \(D_0\)。generation-only Post-G可读取同一冻结\(D_0\)的注册context来判断candidate pair质量；正式Post-X则固定使用`[BOS_UNIT,BOS_UNIT,BOS_UNIT,current]`，不读取前序单元。因此某个前序target-selected rewrite不会改变当前单元的blind accept状态，也不需要成本为\(A^3\)的context envelope。`FinalReplayCalibrationRecord`仍从自然、fallback、多个replacement、prompt-like、edit和null完整assemblies重新解析，以校准anchor、admission、聚类与整体分布；actual keyed \(D_1\)只进held-out system test，不回灌训练或threshold。contextual Post-X仅保留为nonformal ablation；若未来开放，必须定义并计费完整reachable-context envelope、注册新major protocol并重做FPR校准。

context dropout 独立采样 \(w\in\{1,2,3\}\)，但不删除中间 unit；curriculum 先 current-only/CTX-1，再均匀加入 CTX-2/3。评估同时报告 paired delta、bootstrap CI 和 permutation test，避免把 pool 随机性解释成 context gain。formal `attempt_budget` 是独立轴 \(A\in\{2,4,6,9\}\)，不与 context width 绑定；\(A\in\{16,32,64\}\)只用于nonformal generator-stress，超出catalog的positions是sentinels且不能制造容量。“retry 5”在本文唯一解释为 \(A=6\)：1 次 initial attempt + 5 次 additional attempts。

## 16. Candidate-pool protocol

### 16.1 Key-independent pool

正式v1采用document-wide two-phase planning。Phase P0用完全无key的generator产生baseline final code \(D_0\)，strict parse后冻结unit stream、anchors和baseline contexts。没有document nonce。Phase P1仍禁止key模块，并先执行固定global allocator：

1. `B_doc`以**reserved generation-equivalent units (RGE)**计，在看pool/target前由protocol固定。令\(L_i^{max}\)为catalog对该current unit给出的最大canonical token数（超过96即非formal），并固定`k_compile=32,k_post=8,k_sandbox=512,k_combo=32` RGE；即使没有tests也保守预留sandbox charge。每个CPython lexical code-object scope（module/function/async-function/class body）最多分配\(m_{scope}=4\)个units。若该scope已有\(m\in\{0,1,2,3\}\)个allocated units，加入\(i\)的marginal reservation为\(c_{i|m}=A(L_i^{max}+552)+32·2^m\)：末项精确预留第16.2节组合枚举从\(2^m\)增到\(2^{m+1}\)的新增compile checks。初始document rule为\(N_{reserve}=\min(128,\max(16,\lfloor L_{doc}/256\rfloor))\)，\(B_{doc}=N_{reserve}[A(96+552)+256]\)，用最大\(m=3\)边际保守定标。这些整数常量、tokenizer和变更规则写入protocol；RGE只用于公平reservation，actual candidate tokens、CPU/GPU ms和sandbox time另报，不能把RGE冒充wall time。另有不可超支的`PoolResourceContractV1`：每record至多96 Gate tokens、24 candidate AST nodes、1,024 exact-evaluator gas、一次fresh compile、一次Post-G和一次Post-X；每个有tests的pair最多5 CPU-seconds、512 MiB RSS、1 MiB combined output，超限统一变`dynamic_evidence=UNKNOWN`而非追加尝试；每scope组合compile次数精确为\(2^m\)。document的record、compile、Gate和sandbox hard caps分别由allocated units求和，OS quota达到即给剩余预留positions写固定`RESOURCE_SENTINEL`，不重分配、不延长deadline。protocol同时记录actual counts/CPU-ms/wall-ms/RSS，Layer-3按actual cost裁决；
2. Pre输出经独立校准的`LCB(q0),LCB(q1),LCB(qboth),cost`。部署LUT先给`p_q15∈0..32767`，再以checked INT64唯一转换`to_q12(p_q15)=floor(p_q15·4095/32767)`；因此`Q12_ONE=4095`，所有`*_q12`在`0..4095`并表示`x/4095`，端点精确映射0→0、32767→4095。任何LUT/转换越界即protocol failure。`cost_q12=max(cost0_q12,cost1_q12)`是第23.1节first-success fraction，`safe_q12=min(LCB0_q12,LCB1_q12,LCBboth_q12)`。未满足第21节\(G_i^{pre}\)的unit令signed INT64 \(u_i=-1\)并不分配；其余用checked signed INT64先乘后exact floor-div：`numerator_i = INT64(2^20) * INT64(safe_q12)`，`denominator_i=max(INT64(1),INT64(cost_q12))`，`u_i=floor_div_nonnegative(numerator_i,denominator_i)`。静态最大值是`2^20*4095=4,293,918,720<2^63-1`，禁止INT32、wrap或saturation；任何实现若在conversion/multiply/divide越界必须fail closed，不能改变排序；
3. 只对`u_i≥0`的`PRE_ACCEPT`集合按checked signed-INT64 tuple`(-u_i,anchor_hash)`得到唯一静态顺序；`u_i=-1`永不进入排序/分配。再逐项根据该unit code-object scope的当前\(m\)计算marginal \(c_{i|m}\)；若\(m=4\)或加入会超过`B_doc`则跳过继续，否则扣减budget并增加scope count。不会因marginal变化重排已冻结顺序。这是public partition-cap greedy，不按pool outcome回溯。no-Gate baseline按`(source_order,anchor_hash)`、rule-Gate按`(-rule_score,anchor_hash)`冻结顺序，二者都使用相同scope cap与marginal reservation；rule-Gate若为所有eligible units预枚举raw template/region sides，必须先按相同token/hash RGE逐项扣除该cheap census成本，预算不足的units取最低rule score，不能把预计算当免费oracle；
4. pool outcome产生的unused reservation永不再分配，避免候选/target feedback。报告absolute evidence、evidence/RGE、每document与processed documents，不能换分母宣称capacity。

每个pre-selected unit固定写恰好A条records：`C0=D0 current`加`A-1`templates/sentinels；无early stop。C0用于initial-attempt/accounting与identity reconstruction，但永远不进入side sets。对所有units完成Pre、allocator、pools、certificate、Post-G以及current-only Post-X scoring后，只在non-identity admissible records上计算同pool`has0`与`has1`。仅`DUAL_COMPLETE=(has0 && has1)`可进入target phase；其他slot在key前commit为`DUAL_INCOMPLETE_SKIP`，最终保持D0，且target接口从不接收该anchor。

`document_pool_commitment=SHA256(protocol || SHA256(D0 bytes) || DocumentIdentityV1(D0) || B_doc || allocator_arithmetic_id || plan_signal_guard status || ordered(slot status || pool_variant || A ordered candidate/sentinel hashes || winner_0 hash || winner_1 hash || scope_composition_certificate hash || all key-neutral scores || config hashes))`。只有全document commitment封存后，隔离key模块才可解锁。commitment是private audit，不是detector input。固定`(prompt, baseline RNG)`改变key时，document identity、anchor-derived pool variant、Pre/allocator/pools/order/count/scores/dual-status/two winners/scope certificates/guard/commitment必须byte-identical；API没有可选pool RNG、seed或identity。

上式不是字符串拼接实现：preimage是`CommitmentPreimageV2` deterministic-CBOR fixed array，所有hash为bytes32、整数最短编码、slots按anchor bytes升序、records按attempt index；schema hash绑定protocol。任一缺field、map/indefinite CBOR或不同ordering都fail closed。

target解锁前运行`PLAN_SIGNAL_GUARD`：composition-certified dual slots≥16、按第29节只看这些slots形成的public clusters≥8、exact \(n_{eff}\ge12\)。同一guard还做与target side无关的最坏界：`natural_nonidentity_replay_shapes(D0)+dual_slots≤512`，以及
`L_min=L_fixed+Σ_i min(L(w_i^0),L(w_i^1))≥4096`、`L_max=L_fixed+Σ_i max(L(w_i^0),L(w_i^1))≤65536`，其中`L`严格使用`CanonicalDocumentTokenCountV1`，`L_fixed`是去掉全部dual-slot baseline贡献后的其余document计数。这样任一target mux都满足static detector resource/length scope，不会在解锁后才发现target-conditioned admission failure。其结果随后写入commitment。失败时不调用任何key API，只能返回明确标记为**非水印**且不含protocol metadata的\(D_0\)，或`NO_FORMAL_ARTIFACT`；该document计入candidate/system coverage denominator。guard通过仍不保证final detector一定positive，因为duplicate suppression、Post replay和最终cluster sufficiency可能更保守；它只排除明显无信号/越界plan。guard、阈值与结果进private audit/commitment，不进提取输入。

在线逐unit选择被排除为formal v1：先前key-selected code会进入后续context，使后续Pre decision产生间接key依赖。v1先生成baseline再局部改写，candidate generator虽可访问已冻结baseline prefix，但被访问控制禁止读取future units；pairwise tests可在 \(D_0\) 的单unit替换副本上运行。该改变增加一次baseline pass，但真正满足全计划key-independence。

### 16.2 Dedup、过滤与排序

先按 exact normalized token bytes 去重，再按 current-unit canonical AST hash 去重；保留最小 attempt index 作为代表。重复仍计生成成本，但不增加覆盖率。非水印 hard filters（span、parse、binding/effect invariants、resource limits）对所有 candidates 一视同仁。不能因某侧 region 缺候选而调整 temperature、追加 attempt 或换 prompt。

只有**non-identity** candidate且RewriteCertificate、对单candidate assembly重放的`BLIND_CERTIFIED_EVIDENCE`、其他hard checks、Post-G、current-only Post-X和RegionV1 margin全过时才进入side sets。actual pool若不是dual-complete，**在target解锁前**固定skip；不存在“看完target才fallback”。dual-complete pool在key封存状态下就分别计算并commit `winner_0=argmin_{region=0}k(c)`与`winner_1=argmin_{region=1}k(c)`；target模块只能在这两个已提交bytestring间做constant-time mux，不能重新排序或扫描pool。公开ascending key为：

\[
k(c)=(-cert\_tier(c),-r^{G}_{rank,int},-p^{G}_{pair,int},-p^{X}_{accept,int},-p^{X}_{quality,int},
       canonical\_bytes,attempt\_index).
\]

其中`cert_tier=2`表示certificate且所有available E2/E3均PASS，`cert_tier=1`表示certificate PASS但dynamic evidence全为UNAVAILABLE；任何available dynamic FAIL都非admissible。其余模型量均为artifact的有符号integer logits/Q15 scores；所有整数范围、负号与byte lexicographic order均规范化。tie gold vectors覆盖每层。

**`ScopeCompositionCertificateV1`和`CompositionalityLemmaV1`是target前release proof obligation。** replacement spans两两不相交；每个registered RHS是closed、total、无Name/Call/Await/Yield/mutation的pure expression，并由exact evaluator证明产生与identity RHS type/value相同的immutable value；targets、annotations、statement order与control structure不变。对每个lexical CPython code-object scope，令其中dual-complete slots数\(m\le4\)，按anchor升序编号；在key仍sealed时枚举全部\(z\in\{0,1\}^m\)，把slot\(j\)替换为其已提交`winner_zj`，编译完整module，并要求该scope的递归`OperationalDigestV2`（含跨nested code objects共享的`DEF/REF` alias graph）及所有ancestor code-object digests与\(D_0\) byte-exact相同；同时每个完整组合的`DocumentIdentityV1`必须等于commit中的D0 identity。certificate按bitvector升序保存16-bit pass mask（实际只用\(2^m\)位）、各digest/identity hash与baseline hash。任一组合失败则该scope所有slots在target前变为`COMPOSITION_INCOMPLETE_SKIP`，budget不重分配，target API不接收其anchors。

数学上，按Python assignment evaluation顺序归纳，任意组合产生相同状态转移和exception behavior（在第3节排除的source/trace introspection外）；工程上，上述穷举又覆盖同一code object内所有target choices。不同code-object scopes通过“每个changed child及其ancestors的recursive digest仍等于baseline”做树归纳组合。parser substitution、UnitSkeleton、anchor、current-only Post-X与Region input逐slot独立。catalog实现还必须由第二个small-step interpreter和property tests验证该引理；不能只用“单替换tests都过”代替。

两侧都非空且两个winners均预提交，所以不存在target-conditioned count/retry/fallback。最终`BLIND_REPLAY(D1)`与`PAIR_VERIFY_DETERMINISTIC(D0,D1)`是defense-in-depth assertion，不能把blind replay写成功能验证；后者只重放exact evaluator、span/skeleton、OperationalDigestV2、DocumentIdentity与ScopeCompositionCertificate，不在target后重新运行可能flaky的project tests。E2/E3动态结果只在target前运行并commit为候选tier/诊断；held-out functionality只用于system evaluation，不控制某个target后的发布。在符合规范的实现里final assertions应由上述引理保证。若仍出现UNKNOWN/FAIL，不发布artifact、不回退D0、不以相同或新plan重试，并立即停用该protocol/key作为`POST_TARGET_SECURITY_INCIDENT`；任何一次该事件都使Layer-3 Hard criterion失败，因为release availability已可能成为一位侧信道。

### 16.3 泄漏审计

在同一frozen \(D_0\)下对1,000个keys重放planning，要求DocumentIdentity、document commitment、Pre/allocator、candidate/sentinel bytes/order/count/config、scores、dual/composition status、two winners、scope certificates与timeouts byte-identical。plan controller先验证完整commitment再一次性解锁target process；target interface有deny-by-default schema，只接收`(K_unit,protocol_id,document_identity,anchor_hash)`，不能调用generator、Gate、pool builder或compiler。selector使用的winner pair必须来自解锁时已冻结的只读plan mapping，运行时hash再次对document commitment做全量校验。

`AllocatorArithmeticV2` gold至少含：Q15 conversion`0→0,32767→4095,32766→4094`；`safe=0,cost=0→u=0`；`safe=4095,cost=0或1→u=4,293,918,720`；`safe=4095,cost=4095→u=1,048,576`；`safe=2048,cost=3→u=715,827,882`；以及utility相等时按anchor bytes升序。两个独立实现必须对所有`4096×{0,1,2,3,4094,4095}`边界组合、完整plan order和commitment byte-exact；schema和CBOR使用signed INT64。该conformance与Gate integer evaluator同为release hard gate。

统计probe检查**target解锁前**的pool size、duplicates、planning latency、seed/order、candidate tokens、timeouts、dual skip和commitment length；硬标准是byte/dataflow invariance 100%。probe按预注册MDE=AUROC 0.52做≥90% power设计，并要求one-sided 95% AUROC upper≤0.52；`p≥0.01`不再被当无泄漏证明。最终D1 source bytes/length与observed region是被授权的carrier output，本来可以随target变化，不得混入“planning independence”probe；另做side-conditioned质量/长度对称性审计以防隐蔽附加载荷，但不要求AUROC 0.5。candidate count、attempt index、generator/repository/source IDs、seed、retry、current/anchor hashes全禁tensor。任何key-selected code回流P0/P1直接protocol failure。

## 17. Dataset schema

### 17.1 Common contract

所有records用deterministic-CBOR logical schema并可映射Parquet。每张表物理包含`schema_version:semver, protocol_id:sha256, data_release_id:sha256, split:enum/null, group_component_id:sha256/null, parent_record_hashes:list[sha256], record_hash:sha256`。这些common fields公开到受控dataset；训练只可join/split/audit，嵌入/提取只校验protocol/integrity；全部禁tensor，可能泄漏source/split而不泄key；分别由schema/protocol/data/split manifest版本化。

表中“公开”指可在受控data/public registry出现，不等于写进final code；`监督/审计`永不作为tensor。所有hash/ID字段均禁model input。为节省重复行，形如`a,b:type`的单元格表示两个**独立物理字段**，且该行的公开性、阶段可用性、风险与version owner逐字段相同；没有把它们序列化为一个含糊字段。

### 17.2 `AnchorRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | key/bit或shortcut风险 | version owner |
|---|---:|---|---|---|---|---|
| `anchor_record_id:sha256` | 是 | join | audit | rebuild | content ID shortcut；禁tensor | schema |
| `source_artifact_id:sha256` | 条件 | grouping only | 否 | 否 | repo identity | data release |
| `document_identity:sha256,identity_projection_hash:sha256` | 是 | grouping/audit，禁tensor | compute pre-pool/commit | rebuild before PRF | full-code public binding；无key，不可request override | document-identity-v1 |
| `scope_token:sha256` | 是 | audit，禁tensor | rebuild | rebuild | public structure；非key | anchor-v2 |
| `unit_skeleton_cbor:bytes` | 是 | parser/feature source | rebuild | rebuild | masks carrier RHS；非target | skeleton-v1 |
| `unit_skeleton_hash:sha256` | 是 | audit | rebuild | rebuild | hash shortcut；禁tensor | skeleton-v1 |
| `kgram_hashes:fixed[sha256;4]` | 是 | audit | rebuild | rebuild | current/three-predecessor structure；禁tensor | anchor-v2 |
| `local_signature:sha256` | 是 | join only | rebuild | rebuild | 禁tensor | anchor-v2 |
| `unit_role:enum` | 是 | allowlisted input | input | input | 无 | grammar |
| `anchor_preimage:bytes` | 是 | conformance | PRF address component | PRF address component | 不含current text/key；不能脱离document identity使用 | anchor-v2 |
| `anchor_hash:sha256` | 是 | join only | PRF address component | PRF address component | 与document identity/per-unit response联合可成oracle；不发布response | hash suite |
| `scope_unique:bool` | 是 | rule label | hard gate | hard gate | false→erasure | anchor-v2 |
| `current_text_identity:sha256` | 条件 | dedup/audit only | dedup | dedup | content memorization；禁tensor/anchor | canonical-v2 |
| `ast_ordinal:uint32` | 条件 | diagnostics | audit | diagnostics | insertion-sensitive；禁tensor/anchor | linearizer |

**不存在**`document_nonce`、virtual slot、sync ordinal、caller-supplied identity或caller-supplied anchor字段；unknown input field fail closed。

### 17.3 `AtomicUnitRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `unit_record_id,anchor_record_id:sha256` | 是 | join | audit | rebuild | IDs禁tensor | schema |
| `raw_span:[u64,u64)` | 条件 | extraction | current replacement | rebuild | source location | parser |
| `canonical_current:bytes` | 是 | allowlisted tokens | input | input | public code only | canonical-v2 |
| `unit_kind,node_type,clause_type,owner_type,parent_type:enum` | 是 | input | input | input | 无 | grammar |
| `structural_role_path:list[enum]` | 是 | last-four role tokens/input；full path audit | rebuild | rebuild | 无ordinal/text；depth>4以`PATH_PREFIX_MANY`汇总 | linearizer/feature-v2 |
| `code_object_scope_token:sha256,code_object_kind:enum` | 是 | grouping/audit；kind可input，token禁tensor | allocator/cert | rebuild/audit | scope token shortcut；无key | linearizer/composition-v1 |
| `suite_depth:uint8` | 是 | clipped input | input | input | style shortcut受限 | feature-v2 |
| `multiline,inline_suite,docstring,descendant_effect_flags:bitset` | 是 | input/rule | rule/input | rule/input | 无key | parser/feature |
| `ast_ordinal,candidate_ordinal:u32/null` | 条件 | denominator only | audit | diagnostics | 禁tensor/anchor | policy |
| `eligibility_state:enum` | 是 | mask/label | rule gate | rule gate | public selection | policy-v2 |
| `exclusion_reason:enum/null` | 是 | OOD strata | rule gate | rule gate | trivial shortcut；excluded不进Gate | policy-v2 |
| `origin_label:enum` | 否 | stratification only | prompt skip audit | unavailable | causal shortcut；禁tensor/anchor/Post | data release |
| `parse_status:enum` | 是 | provisional label | provisional | strict rebuild | recovery不得formal | parser |

### 17.4 `CandidatePoolRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `pool_record_id,anchor_record_id:sha256` | 条件 | join | audit | 否 | generation trace | schema |
| `base_catalog_hash:sha256,catalog_candidate_ids:list[sha256]` | 条件 | shared candidate source | enumerate once/assert | 否 | catalog content shortcut；禁tensor | pool-v2 |
| `generator_artifact_id:sha256/null,rewrite_catalog_hash:sha256` | 是 | strata/audit，禁tensor | formal enumerator has null generator; catalog fixed | 否 | generator shortcut | protocol/generator registry |
| `pool_variant:u8(0..3),variant_derivation_hash:sha256` | 条件 | four-pool label grouping | derive exactly from protocol/anchor | 否 | variant/seed shortcut；禁tensor | pool-v2 |
| `attempt_budget_A:u16` | 是 | experiment factor，禁tensor | fixed | 否 | candidate-count shortcut | pool-v2 |
| `ordered_records:fixed[list[CandidatePlacement];A]` | 条件 | labels；placement=`[attempt_index,candidate_id/null,sentinel_reason/null]` | target前sealed | 否 | retry trace；不进final | pool-v2 |
| `has_admissible_0,has_admissible_1:bool` | 条件 | label | target前hard check | 否 | 无actual target | pool-v2 |
| `dual_complete:bool` | 条件 | label | target前commit | 否 | key-neutral；不进final | pool-v2 |
| `unique_count,duplicate_count:u16` | 条件 | label/audit | audit | 否 | shortcut；禁tensor | pool-v2 |
| `reservation_RGE:u32,allocator_state:enum` | 条件 | cost label | target前 | 否 | budget trace；禁tensor | allocator-v2 |
| `pool_commitment,document_plan_commitment:sha256` | 条件 | audit | seal/assert | 否 | compliance audit only | commitment-v2 |
| `winner_0_id,winner_1_id:sha256/null,scope_composition_certificate:sha256/null` | 条件 | causal/composition audit | target-before commit | unavailable | two-side plan trace；禁tensor/final | composition-v1 |
| `target_relation:enum(after_document_commit)` | 条件 | causal audit | assert | 否 | violation=critical leak | protocol-v2 |

### 17.5 `CandidateRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `candidate_record_id,base_catalog_hash:sha256` | 条件 | join；candidate可被多个projections引用 | audit | 否 | ID shortcut | schema |
| `catalog_index:u16,is_baseline:bool,is_identity:bool` | 条件 | strata only | fixed catalog/rule | identity recomputable | index shortcut；禁tensor；identity强制non-evidence | pool/policy-v2 |
| `template_id:enum,catalog_params_cbor:bytes` | 是 | certificate input/audit | enumerator | 可从final class重建但非必需 | 无key；模板shortcut不进Gate | rewrite-v1 |
| `candidate_source:bytes,candidate_canonical:bytes` | 条件 | Post-G/X input | replacement/input | final current rebuild | source content | data/canonical-v2 |
| `skeleton_unchanged:bool,span_valid,parse_valid:bool` | 是 | hard labels | hard filters | hard checks | 无 | verifier-v2 |
| `exact_eval_digest,operational_before,operational_after:sha256` | 条件 | pair labels | hard filter | unavailable baseline | privileged pair info；禁Post-X | certificate-v1/OperationalDigestV2 |
| `alias_graph_before,alias_graph_after:sha256` | 条件 | hard pair label/property audit | hard filter | final/current graph only | object-sharing truth；禁Post-X tensor | OperationalDigestV2 |
| `certificate_verdict:enum` | 是 | Post-G label | RewriteCertificate hard filter | reconstruct as BlindCertificate authority | no key；pair/original unavailable but inverse is final-only | certificate-v1 |
| `region_digest:bytes32,region_u64:u64,region:enum,margin_u64:u64` | 是 | labels | side selection | exact rebuild | public; no target | RegionV1 |
| `post_g_labels:struct,post_g_scores:struct` | 条件 | Post-G supervision | rank/filter | unavailable | privileged baseline relation | model release |
| `post_x_labels:struct,post_x_scores:struct` | 条件 | current-only Post-X supervision | current-only score | exact current-only inference | only final observables | model/calibration |
| `current_intrinsic_admissible:bool` | 条件 | pool label | dual check | recompute from current | key-neutral; no target/context | pool-v2 |
| `mutation_labels:struct,execution_labels:struct,human_labels:struct` | 否 | supervision/audit only | no tensor truth | unavailable | private tests/source | label release |

### 17.6 `ContextViewRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `context_view_id,paired_group_id:sha256` | 条件 | sampler | audit | rebuild | IDs禁tensor | schema/data |
| `width:enum(0,1,2,3)` | 是 | PRE/G use 1..3; X uses 0 | `pre_context_width` or X=0 | X fixed 0 | 无 | model contract |
| `context_source:enum(baseline,counterfactual_final,natural,actual_test)` | 条件 | sampler strata only | PRE/G baseline; X current-only | final current only | actual target trajectory不得train | data-v2 |
| `predecessor_canonical:fixed[bytes;3],current_canonical:bytes` | 条件 | predecessor only PRE/G; current all heads | PRE/G input; X predecessor=BOS | X current+BOS only | final code observables；scope mask强制 | canonical/model-v2 |
| `BOS_mask,transition_enum,depth_delta,unit_feature_tensor:fixed int` | 是 | sole structural input | input | input | 见allowlist；无ID/hash/origin | feature-v2 |
| `augmentation_seed:u64/null` | 否 | sampler only | 否 | 否 | key-independent但禁tensor；formal pool不用它 | data release |
| `shared_pool_label_hash:sha256` | 是 | parity assert | audit | 否 | 保证CTX配对 | label release |

### 17.7 `PreGateLabelRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `label_id,context_view_id:sha256` | 条件 | join | 否 | 否 | ID禁tensor | schema |
| `trials_R:u16,Y0_by_pool,Y1_by_pool,Yboth_by_pool:bitset[R]` | 是 | supervision | 否 | 否 | pool outcomes；禁input | label-v2 |
| `successes_0,successes_1,successes_both:u16` | 是 | supervision | 否 | 否 | 同上 | label-v2 |
| `y0_soft,y1_soft,yboth_soft:float32` | 是 | targets | 否 | 否 | 无actual bit | prior/label-v2 |
| `rho0,rho1:float32,side_corr:float32/null` | 是 | coverage/aux targets | 否 | 否 | 无 | label-v2 |
| `min_rank_0_by_pool,min_rank_1_by_pool:fixed[u16/null;R],cost0_soft,cost1_soft:float32` | 条件 | 第23.1节first-success cost targets；RGE另在PoolRecord | 否 | 否 | attempt trace；禁input | cost-v2 |
| `class_weight,confidence_weight:float32` | 是 | loss only | 否 | 否 | sampling only | recipe |

Gold反例：`R=2,Y0=(1,0),Y1=(0,1)`必须得到`Yboth=(0,0),successes_both=0`，不能得到both成功。

### 17.8 `PostGateLabelRecord`与`FinalReplayCalibrationRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `label_id,candidate_or_unit_id:sha256,head_scope:enum(G,X)` | 条件 | join/router | router | X only | ID/head shortcut禁tensor | schema |
| `syntax_valid,structure_valid:bool` | 是 | hard/aux labels | recompute | recompute | 无 | verifier |
| `pair_certificate,E2,E3,E4:bool/unknown` | 否 | **G-only** supervision | G-only hard/rank | unavailable | hidden baseline/private tests；禁X | label release |
| `quality_target,format_target,rename_target:bool/unknown` | 条件 | X supervision with masks；三者均按14.4节得到binary truth | X predict | X predict | no key; label missingness mask | rubric/edit release |
| `rater_display_hash,rubric_version_hash:sha256,rater_count:u8,agreement_ord_q14:i16,agreement_susp_q14:i16` | 条件 | label reproducibility/audit，禁tensor；每个`agreement_*_q14=round_half_away(clamp(alpha_*,-1,1)·16384)`，范围`[-16384,16384]` | 否 | 否 | human/source release metadata；无key；两种alpha均可表示负agreement | rater-display-v1 |
| `blind_certificate,replay_conformance:bool/unknown` | 是 | hard/audit only | hard verifier | hard verifier/startup | **不建learned head**；artifact mismatch→unavailable | certificate/deployment-v1 |
| `redundancy_target,correlation_target:float` | 条件 | aggregation audit only | deterministic cluster rules | deterministic aggregate | no target sign；**不进Post-X tensor/head** | clustering |
| `intrinsic_accept_target:bool/unknown` | 条件 | X asymmetric target；三项任一unknown则unknown并mask | threshold select | threshold apply | only conjunction of final-observable labels | policy-v2 |
| `false_accept_weight:float32,label_mask:bitset` | 否 | loss only | 否 | 否 | no model input | recipe |
| `erasure_reason_target:enum/null` | 条件 | X supervision | infer | infer | no key | policy-v2 |

`FinalReplayCalibrationRecord`是独立物理表，不把assembly信息塞进`PostGateLabelRecord`：

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `final_replay_record_id:sha256` | 条件 | join only | audit | 否 | ID禁tensor | schema |
| `assembly_kind:enum(natural,baseline_fallback,multi_replacement,prompt_like,format,rename,edit,random_side_null,wrong_key_null,sample_aware_mosaic)` | aggregate | sampler stratum only | key-free replay | unavailable | trajectory shortcut；禁tensor | final-replay-v2 |
| `assembly_seed:u64` | 否 | sampler only | key-free assembly | 否 | seed shortcut；禁tensor | data release |
| `whole_document_hash:sha256` | 条件 | clone/join audit | verify | recompute only | memorization；禁tensor | canonical-v2 |
| `document_identity:sha256` | 条件 | binding/attack strata，禁tensor | compute/assert | recompute | public full-code binding；无key | document-identity-v1 |
| `unit_final_tensor_hash:sha256` | 条件 | parity assertion | verify | recompute | hash shortcut；禁tensor | feature/model-v2 |
| `unit_final_tensor_bytes:fixed[1216,bytes],collision_cell_id:sha256` | 条件 | exact-cell grouping；bytes是Post-X实际输入 | verify | exact rebuild | cell ID禁input；tensor本身是deployable input | DeployableTensorBytesV2 |
| `actual_keyed:bool` | 否 | always false except sealed system-test mask | false during training/calibration | unavailable | direct trajectory leak；禁tensor | split/release |
| `post_x_logits_i16:fixed[4,i16],post_x_scores_q15:fixed[4,u16],post_x_state:enum` | aggregate | labels/calibration | key-free replay | exact infer | final-observable; no sign | model/calibration |
| `region_digest:bytes32,region:enum,margin:u64` | 是 | labels/calibration | exact | exact | public current property | RegionV1 |
| `blind_certificate,hard_status:enum` | 是 | hard labels | exact | exact | no key | verifier-v2 |
| `final_intrinsic_label:bool/unknown,label_mask:bitset` | 条件 | X supervision/risk cert | no model input | unavailable truth | missingness shortcut；mask only | label release |
| `public_stratum_id:enum/null` | 是 | final calibration | recompute | recompute | key-independent routing | null-release |

训练用它做Post-X与final-system calibration；嵌入可用于key-free整文档replay；提取只重建current-only tensor和public deterministic fields。所有seed/hash/assembly/label fields禁tensor。

### 17.9 `EmbeddingDecisionRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `decision_id,anchor_hash:sha256` | 否 | audit | join | anchor rebuild only | per-anchor trace | schema/anchor |
| `q0_q12,q1_q12,qboth_q12,cost_q12:u16` | 否 | diagnostics | checked range 0..4095 | not used | selection shortcut；不进final | gate/calibration-q12 |
| `utility_i64:i64` | 否 | diagnostics | checked allocator/order | not used | 允许-1 sentinel；禁止INT32/wrap；不进final | allocator-v2 |
| `document_identity:sha256` | 否 | causal audit | committed PRF binding | rebuild, not sidecar | public-code-derived；禁tensor | document-identity-v1 |
| `candidate_state,pre_selected_state,attempted_state,dual_complete_state:enum` | 否 | audit | state machine | unavailable | plan trace | protocol-v2 |
| `pool_commitment,document_commitment:sha256` | 否 | audit | target-before assert | unavailable | generation trace | commitment-v2 |
| `target_bit:secret bit` | **否** | **否** | isolated memory after commit | recompute transiently | direct key leakage | PRF-v2 |
| `selected_candidate_id:sha256/null` | 否 | audit | selection | unavailable | withtarget leaks | selection-v2 |
| `outcome:enum(success,keyneutral_skip,withheld_incident)` | 否 | audit | state | not authority | no target-side fallback enum | protocol-v2 |
| `blind_replay_status,pair_verify_status:enum(PASS,FAIL,UNKNOWN)` | 否 | test/audit | release gate | blind only recomputed | pair status unavailable extraction | verifier-v2 |
| `final_code_hash:sha256` | 条件 | audit | release | recompute | code identity | canonical-v2 |

### 17.10 `ExtractionEvidenceRecord`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `evidence_id,detector_run_id:sha256` | aggregate only | 否 | 否 | create | query linkage | detector schema |
| `anchor_hash:sha256` | 默认不逐项公开 | 否 | 否 | rebuild | keyed oracle when joined response | anchor-v2 |
| `document_identity:sha256` | aggregate only | 否 | 否 | rebuild before any PRF | full-code public binding；不可request override | document-identity-v1 |
| `parser_status,anchor_uniqueness:enum` | aggregate | 否 | 否 | rebuild | key-free | parser/anchor |
| `region_u64:u64,observed_region:enum,margin_u64:u64` | 可公开 | 否 | 否 | exact rebuild | public code property | RegionV1 |
| `expected_bit:secret bit` | **否** | 否 | 否 | transient PRF | direct key leakage | PRF-v2 |
| `post_x_state:enum(accept,reject,erasure)` | aggregate | 否 | 否 | integer evaluator | public-model result | model/calibration |
| `evidence_state:enum(match,mismatch,erasure,invalid,unavailable)` | aggregate | 否 | 否 | derive | per-unit release aids oracle | evidence-v2 |
| `erasure_reason:enum/null` | aggregate counts | 否 | 否 | derive | no sign | policy-v2 |
| `cluster_id:sha256,cluster_weight_q30:i64` | aggregate | 否 | 否 | derive | identifier default private；无bit sign | cluster-v2 |
| `ecc_state:enum(ECC_DISABLED_FORMAL_V1)` | 是 | 否 | 否 | constant/assert | 防止旧detector暗开codeword target；其他拼写fail closed | protocol-v3 |
| `contribution_q30:i64` | aggregate only | 否 | 否 | derive | keyed sign | statistic-v2 |
| `artifact_conformance:bitset,stratum_id:enum` | 是 | 否 | 否 | admission | no key | deployment/null release |

没有`alignment hypothesis/status`：anchor由observed final code唯一计算或erasure。

### 17.11 `ModelArtifactMetadata`

| field:type | 公开 | 训练 | 嵌入 | 提取 | risk | version owner |
|---|---:|---|---|---|---|---|
| `artifact_id:sha256` | 是 | bind | verify | verify | 无 | registry |
| `protocol_id:sha256` | 是 | bind | verify | verify | 无 | registry |
| `schema_version:semver` | 是 | bind | verify | verify | 无 | schema |
| `role:enum(tokenizer,pre_gate,post_g,post_x,integer_evaluator,generator)` | 是 | route | route | tokenizer/post_x/evaluator only | generator identity禁tensor | registry |
| `weights_hash,tokenizer_hash,generator_hash:sha256/null` | 是 | bind/audit | verify | relevant subset verify | no key | content registry |
| `architecture:enum(IntGateV1-S)` | 是 | QAT/export | reference | reference | no float fallback | model-v1 |
| `integer_opset:list[enum]` | 是 | QAT/export | reference | reference | unlisted op fail closed | evaluator-v1 |
| `operator_allowlist:list[enum]` | 是 | QAT/export | reference | reference | no dynamic dispatch | evaluator-v1 |
| `weight_dtype:enum(INT8),activation_dtype:enum(INT16),accumulator_dtype:enum(INT64)` | 是 | fake-quant | exact | exact | no key | quant-v2 |
| `saturation:enum(SAT16),round_shift_rule:enum(SRS_HALF_AWAY)` | 是 | fake-quant | exact | exact | no key | quant-v2 |
| `input_shapes:map[role,fixed_shape]` | 是 | simulate | exact | exact | shape mismatch→unavailable | input-v2 |
| `padding:struct{pad_id:u16,mask_rule:enum},BOS:struct{token_id:u16,slot_id:u8}` | 是 | simulate | exact | exact | no implicit padding | input-v2 |
| `truncation_rules:map[scope,enum(skip,long_sentinel)]` | 是 | simulate | exact | exact | never silent truncation | input-v2 |
| `dependency_hashes:map[enum(canonicalizer,region_projection,gate_projection,parser,skeleton,anchor,document_identity,operational_digest,intern_state_helper,feature_schema,rewrite_catalog,blind_certificate,composition,region_spec,resource_contract,allocator_arithmetic),sha256]` | 是 | bind | verify | verify | dependency integrity；CPython intern helper不可隐式替换 | protocol-v3 |
| `allocator_contract:struct{q12_one:u16=4095,utility_dtype:i64,scale:u32=1048576,floor_rule:enum,overflow:FAIL}` | 是 | simulate/audit | exact allocator | registry check | no secret；mismatch fail closed | allocator-v2 |
| `ecc_state:enum(ECC_DISABLED_FORMAL_V1)` | 是 | bind | assert | assert | old ECC artifact或其他disabled-state拼写均incompatible | protocol-v3 |
| `calibration_LUT:map[probability_head,fixed[u16;65536]],pre_lcb_tables:struct` | 是 | map/risk-cert output | exact score→Q15/LCB | Post-X LUT only | public decision；table mismatch fail closed | calibration release |
| `pre_thresholds_q15:struct{tau_side:u16,tau_both:u16,cmax:u16},post_g_thresholds_q15:struct{certificate:u16,dynamic:u16,pair_quality:u16}` | 是 | threshold-selection output | exact Pre/G decision | unused | public decisions | calibration release |
| `post_x_thresholds_i16:map[enum(quality,format,rename,intrinsic),struct{reject:i16,accept:i16}]` | 是 | threshold-selection output | Post-X current-only | exact Post-X | reject<accept且gap≥2048；band→erasure | calibration release |
| `reference_evaluator_hash:sha256` | 是 | export gate | startup gate | startup gate | binary/source identity | deployment-v2 |
| `reference_runtime_fingerprint:struct{container:sha256,os:string,kernel_abi:string,libc:string,compiler:string,flags:list[string],isa:list[enum],endianness:enum,threads:u8}` | 是 | export gate | startup gate | startup gate | mismatch→unavailable | deployment-v2 |
| `conformance_inputs_hash,conformance_outputs_hash:sha256` | 是 | export gate | startup gate | startup gate | exact vectors | deployment-v2 |
| `training_card_hash,model_card_hash,data_card_hash,calibration_manifest_hash,null_manifest_hash:sha256` | 是 | provenance | audit | audit | source aggregation only | releases |
| `compatibility_rules,fail_closed_rules:list[enum]` | 是 | test | enforce | enforce | no approximate fallback | protocol-v2 |

### 17.12 Sole exported feature allowlist

允许tensor字段只有：由`CanonicalCurrentV2`确定性投影的`GateProjectionV2` token IDs、BOS/pad mask，以及20.2节唯一的16-field structural vector：`unit_kind,node_type,clause_type,owner_type,parent_type,code_object_kind` enums、clipped `suite_depth∈[0,15]`、clipped signed depth delta（`≤-4,-3,…,3,≥4`）、suite-transition enum、一个注册8-bit `header/multiline/inline/docstring/effect` combined flag、Python赋值左值的`assignment_target_arity/assignment_target_shape`、closed-value type/size bucket、canonical token length bucket和`slot_kind∈{BOS,PAD,LONG,NORMAL}`。`GateProjectionV2`保留注册node/role/operator/literal-type/size tokens，并在BOS后发出最多四个从root到leaf的role-path tail tokens；更深prefix合成一个`PATH_PREFIX_MANY` token，路径不含任何sibling index/ordinal。identifier只保留current-unit equality/role index 0…15，更多映射`[IDENT_MANY]`；int只保留sign与bit-length bins`0,1,2–7,8–15,16–31,32+`，str/bytes/tuple只保留length/node bins`0,1,2–3,4–7,8–15,16–31,32–63,64–128,129+`；canonical token length使用同一组长度bins。每个field的value-ID table至多256项且随artifact发布。完整`CanonicalCurrentV2`供规则验证，`RegionCurrentV2`只供RegionV1。禁止：key/watermark target、nonce（协议中不存在）、document identity、anchor/text/content/**scope** hashes、absolute ordinal/span、raw identifier/literal payload、origin、repo/task/generator/source IDs、pool size/order/variant/seed/attempt/retry/timeout/latency、dual/composition outcome、commitment、mutation type、private tests/human labels。exporter按schema bitset做deny-by-default taint test；多一个字段即artifact invalid。该mask会限制Gate对value-specific reachability的预测，故actual-pool dual hard check不可移除；这是一项安全/泛化trade-off，不是免费改进。

`private_key`永不属于schema。final artifact可含可选固定`protocol_id` comment，或由API唯一指定；`DocumentIdentityV1`只能从code重建，不能由metadata提供。API出现nonce、identity override、subset、anchor或unknown field立即`UNAVAILABLE_REQUEST`。没有sidecar、plan或generation logs。

## 18. Train/validation/test split

### 18.1 分组与 clone graph

先构建无向关系图：同 repository/fork、同 task/spec、同 function ancestry、同 synthetic template、token MinHash 命中、AST subtree similarity 命中、identifier-normalized clone 命中、同 prompt 的多 generator outputs 均连边。connected component 是最小 split unit；repository 连接优先于函数级随机化。所有 descendants、anchors、pools、attempts、mutations、edits 和 paired views 继承 component split。

主数据按connected components/repositories分为：train 50%、model-validation 10%、Post calibration-map 5%、Pre calibration-map 5%、Post threshold-selection 2.5%、Pre threshold-selection 2.5%、joint risk-certification 10%、held-out test 15%。比例按repository/component而非expanded rows报告。全部post-training partitions互不重叠：Post map/select先冻结Post-G/X；只在train上重算deployment \(V\)并训练Pre head；随后Pre map/select；risk-cert只给最终全pipeline one-sided bounds，test只运行一次。另构造互不重叠的`null-admission-fit`、`null-tail-calibration`与`null-test` releases；前者只看public key-independent covariates拟合第31.1节strata/OOD，后两者才计算keyed statistic/tails。不能把Post map/select rows再拿来fit Pre。

- leave-one-generator-family-out test；
- leave-one-domain/repository-family-out test；
- sealed HumanEval/MBPP external test；
- 每个正式public stratum的null-tail calibration与null-test样本量按第31节目标alpha反算；`Qmax=32`且零错误时每stratum至少9,585，实际预注册下限10,000，稀疏stratum不合并冒充而是unsupported；
- attack test，与 clean threshold selection 分离。

每个public stratum的null releases必须分别含五个construction groups且各自满足第31节样本量：(1)自然未加水印代码；(2)第14.4节key-neutral random-side marker assemblies，用于匹配marker/evidence count；(3)在独立组件上生成的watermarked assembly用independent wrong key检测；(4)format/edit/error-channel null；(5)same-key donor-visible、different-DocumentIdentity sample-aware mosaic。group 2/3/5不能与positive test共享prompt/task/component；group 5的donor/constructor-tuning/target-test还按audit key与repository component分开。所有null与positive来自同支持长度/结构/evidence strata；同一prompt/task的variants属于同一connected component，不能跨任何calibration/test partition。repository/task/generator/domain虽然不作为query路由输入，仍作为hidden audit groups，在risk-cert/test上做simultaneous worst-group bounds。

### 18.2 泄漏检查

在 expansion 前和最终 records 后各做一次 exact hash、normalized token MinHash、AST path fingerprint、docstring/spec embedding nearest-neighbor 和 known benchmark signature scan。任何跨 split clone 把整个 connected component移动到 quarantine，而不是只删一个 candidate。报告每种 detector 的阈值、命中数量、人工抽样 precision 和被删除 component 数；generator/version 只能用于 stratification，不作为 features。

### 18.3 Balance、weights 与 hard negatives

训练 sampler 先按 unit class × source stratum × generator family × min-side bin 分层，再在 anchor 内保留完整 paired views/pools。`y0/y1` 单侧、双侧、皆不可达均显式采样；对 rare false-accept hard negatives 采用最高 5× weight，但 validation/test 保持自然 prevalence。hard-negative mining 每轮只从 train predictions 选择，固定 mining model hash 和 round；不得从 test 错例回灌。class balance 通过 loss weights/sampler 实现，不复制同一 anchor 到不同 split。

excluded statements规则层直接排除，不作普通Gate negatives；另建policy-OOD suite要求rule engine 100% fail closed。oracle-only classes只进隔离feasibility/diagnostic heads，永不进入v1 formal denominator。

## 19. Oracle feasibility study

### 19.1 目的与阶段

Oracle study在训练正式Gate前执行，回答“有限rewrite language + current-only RegionV1是否有双侧容量”。先做**零模型静态census**：对有限template grammar做完整overlap/compositional symbolic check；对所有observed anchors与预注册boundary/property-fuzz values枚举全部canonical-distinct certificates、RegionV1 sides与equivalence-class size。由于string/int value域巨大，不把样本census冒充全域穷举。随后在held-out repositories/tasks/generators运行\(A\in\{2,4,6,9\}\)。`A=16/32/64`只用于oracle-only generator stress，formal catalog超过9后全是sentinel，不能假装增加容量。

同时报告两个上界/下界：

1. `certificate capacity`：只计strict parse、skeleton invariant、exact evaluator与OperationalDigestV2（含alias graph）全过；
2. `certificate+dynamic lower bound`：另要求可用tests/differential checks全过，unknown不计成功；
3. `Post-X oracle ceiling`：假设perfect intrinsic accept selector；不把privileged pair labels部署化。

因此“Post-Gate 后有效容量”先以 label-oracle ceiling 和 public rule-verifier baseline 夹逼；若二者都低，无需训练 Gate。只有 Oracle GO 后才训练 learned Post-Gate，并在第二层检验其 FAR/容量。

### 19.2 测量协议

对每个anchor、public pool variant \(r\) 和region \(b\)只在non-identity records上计算`Y0r,Y1r,Yboth,r=Y0rY1r`、有限variant精确比例、最小rank、side correlation、canonical class size、certificate/dynamic survival、identity-preserving format replay，以及rename/edit后的projection replay与DocumentIdentity reset；identity C0另报但永不算side success。跨pool“ever reached side0 and side1”只作诊断，绝不称deployment dual-complete。formal四variants不报告伪随机Jeffreys posterior；不确定性来自按rewrite family/value type/generator/repository聚类的anchor/repository bootstrap与actual-deployment-variant binomial CI。只有另行抽样的oracle-only generator pools才可同时报Jeffreys posterior。所有结果报micro/macro；\(A=2\) dual rate应精确为0，是accounting gold check。

主要指标：Oracle 双侧可达率、region-0/1 success、

\[
S_{\min}=\min(S_0,S_1),\qquad
D=\Pr(Y_{0r}Y_{1r}=1),
\]

formal evidence/1000 generated tokens、attempt-success curve、marginal gain、zero-entropy share、format的DocumentIdentity+Region+anchor replay，以及rename/local-edit的projection/anchor replay与identity reset。headers只报noncarrier diagnostic，不进入容量分母或promotion。

### 19.3 Oracle go/no-go

预注册 initial engineering thresholds（test 前冻结）：

- formal catalog在`A=9`：macro min-side pool success ≥0.35、same-pool dual-complete anchors ≥50%，且95% component-CI lower分别≥0.30/0.45；
- 任一 side 与另一 side 的绝对 success gap ≤0.10；
- label-oracle formal evidence density ≥4/1000 generated tokens；
- RegionV1 gold/reference replay 100%，zero-edit anchor replay 100%，format replay ≥99.9%；
- 功能正确率相对 no-watermark attempt-0 下降不超过 2 percentage points；
- 95% CI 下界也必须超过相应阈值，不能只看 point estimate。

若static census显示绝大多数formal-shaped units少于两个canonical forms，或上述任一hard threshold失败，停止Gate training并把当前carrier/region路线判Layer-1 NO-GO；不能降阈值、扩大模型或重新引入multi-unit payload。headers不能补分母。Oracle GO只授权小规模Gate训练，不等于System GO。

## 20. Model architecture

### 20.1 Normative `RegionV1`：current-unit-only discrete realization partition

正式region不使用神经模型：

\[
d(x)=SHA256(CBOR_D([\texttt{"SAGE-C/RegionV1"},RegionCurrentV2(x)])),\qquad
u(x)=OS2IP_{BE}(d(x)[0:8]).
\]

固定\(\Delta_R=2^{57}\)，并令\(L_R=2^{63}-\Delta_R,H_R=2^{63}+\Delta_R\)，使半开中央rejection interval\([L_R,H_R)\)恰有\(2^{58}\)个值、占64-bit空间的\(1/64\)，且两侧各有\(2^{63}-2^{57}\)个值：

\[
R(x)=\begin{cases}
0,&u<L_R,\\
1,&u\ge H_R,\\
\bot,&\text{otherwise},
\end{cases}
\quad
m_R(x)=\begin{cases}
L_R-u,&R=0,\\
u-H_R+1,&R=1,\\
0,&R=\bot.
\end{cases}
\]

比较使用unsigned64-bit；\(u=L_R\)归erasure，\(u=H_R\)归region1并有最小离散margin 1。`Region hard pass`唯一含义是\(R\in\{0,1\}\)（等价于\(m_R\ge1\)），没有另一个未定义margin threshold。SHA-256、CBOR、domain bytes、endianness和gold vectors绑定protocol。random-oracle balance只是design assumption；class balance与dual reach必须由Oracle验证。RegionV1区分certificate-equivalent realizations，不是denotation-invariant semantic quotient。由于SHA avalanche，较大\(m_R\)不表示更耐语义编辑；它只记录到public rejection interval的整数距离，selection key不按它偏置。format/registered alpha-rename的**projection**鲁棒性来自byte-exact；只有format保持DocumentIdentity/target，rename有意触发target rebinding。

### 20.2 Shared integer-only Gate encoder

Gate不用Transformer/LayerNorm/Softmax。最终选择`IntGateV1-S`，而不是留下未决架构。`GateTokenizerV2`按注册AST role-table preorder发出`BOS, role-path-tail, node, role, operator, literal-type/size, identifier-equality/role, EOS` tokens；它不做BPE或Unicode normalization，完整4,096-entry ID table/hash属于artifact，unknown role/token即unavailable。每个unit slot固定96 positions。current unit若含BOS/EOS后超过96 tokens直接rule skip/erasure；超长predecessor整slot替换为`[BOS,LONG_UNIT,length_bucket,EOS,PAD×92]`，绝不截取prefix/tail。

全文反复使用的“canonical token length”只有一个定义：`CanonicalDocumentTokenCountV1(D)=Σ_{u∈U_syntactic(D)} len(GateTokenizerV2_unpadded(u))`，每个unit的计数含恰好一个BOS和一个EOS，不含PAD、LONG替代或跨unit separator，且在96-token slot裁决**之前**计原始确定性token IDs。第16节的`L_doc`、4,096–65,536 document admission、evidence/1000-token指标都指这个整数；generator自身BPE tokens只用于候选生成成本，必须另列，不能混为容量分母。tokenizer/hash或unknown unit变化即新protocol或unavailable。

token与position embeddings均为64维（`4096×64`与`96×64` INT8）。位置\(p\)先查`E_t[token_p]`与`E_p[p]`，做saturating INT16 add，再运行**一个**共享`Affine64→64 + ReLU`得到\(z_p\)。只对valid \((token,position)\) pairs按left-to-right顺序累加\(z_p\)到已静态证明不溢出的INT64，再按manifest shift做SRS得到64维token vector。非线性在求和前作用于token+position，因此不同token排列一般得到不同vector；gold vectors必须包含token permutation反例，禁止退化成丢序的plain embedding sum。

结构向量恰有16个字段：`unit_kind,node_type,clause_type,owner_type,parent_type,code_object_kind,suite_depth,depth_delta,transition,combined_flags,assignment_target_arity,assignment_target_shape,closed_value_type,closed_value_size,canonical_token_length,slot_kind`。这里两个`assignment_target_*`只描述Python AST赋值左值，和私有watermark target bit完全无关。每个字段按各自公开value table映射到0…255（含BOS/PAD/UNKNOWN，其中formal UNKNOWN fail closed），查共享形状为`[16,256,8]`但field-index隔离的INT8 table，依字段顺序拼成128维。`combined_flags`是注册8-bit bitset的枚举值；所有bin边界由17.12节定义。token vector与结构向量concat后经一个`Affine192→128 + ReLU`形成slot vector。四slot vectors按`[prev3,prev2,prev1,current]`拼成512维，经共享`Affine512→256 + ReLU → Affine256→128 + ReLU` pass trunk。PRE head为单层`128→7`；POST-X为单层`128→4`；POST-G把baseline/candidate两次128维pass输出concat后运行`Affine256→128 + ReLU → Affine128→4`。

上述**export graph**含`4096×64 + 96×64 + 64×64 + 64 + 16×256×8 + 192×128 + 128 + 512×256 + 256 + 256×128 + 128 + 128×7 + 7 + 128×4 + 4 + 256×128 + 128 + 128×4 + 4 = 528,975`个权重/偏置标量，低于0.8M和2M hard cap；artifact generator必须独立复算并逐tensor列出shape，任何差异使artifact invalid。第23.4节generator/repository nuisance heads是training-only的两个`128→256` heads；family label用public training-release salt的SHA-256 low byte映射到256 buckets，bucket collisions只削弱audit、不进入部署。training card计数它们，exporter必须证明零可达路径并剥离；含其bias的training graph为595,023 scalars，仍<1.0M。每个affine后的shift、saturation location与bias scale写入manifest。无BatchNorm、attention、normalization、dropout或float op。

shared backbone有三条scope明确的heads：`PRE`取一个`[1,4,96]`tensor（baseline current加最多三个前序D0 units）；`POST-G`取两个共享权重的`[1,4,96]`passes，前序slots相同，current slot分别是baseline和candidate，head只concat两次backbone vectors，不使用未注册difference/float op；`POST-X`取一个`[1,4,96]`tensor且前三slot强制BOS、current为final unit。各自输入在对应部署阶段可用；privileged labels永不进入input。model metadata分别绑定`PRE:[1,4,96]`,`POST_G:[1,2,4,96]`（pair axis依次baseline,candidate）与`POST_X:[1,4,96]`，shape或axis order不同即fail closed。

`DeployableTensorBytesV2`固定序列化每个pass的4个slots；每slot依次是96个big-endian UINT16 token IDs（192 bytes）、96个UINT8 valid-mask bytes和16个UINT8 structural field IDs，共304 bytes；单pass恰1,216 bytes，POST-G按baseline pass后candidate pass恰2,432 bytes。不存在padding alignment或native-endian字段。第14.4/17.8节的Post-X collision cell使用完整1,216 bytes逐byte分组，SHA-256只作索引；长度或hash-equal/bytes-different均fail closed。

formal deployment 的 `pre_context_width` 从 CTX-1/2/3 中仅在 validation 预注册选择一次；默认候选是 CTX-3，随后写入artifact。Pre不能逐样本挑最有利width。Post-G使用同一个冻结D0 width以便候选质量判断；Post-X的`post_x_context_width=0`是v1常量，嵌入finalization与提取均使用同一current-only tensor。研究仍同时报告三个paired Pre views，不能把Post-X的scope contraction误报成CTX比较结果。

### 20.3 统一、type-specific 与分模型比较

最终推荐`shared integer backbone + task-specific PRE/POST-G/POST-X heads + simple-only formal calibrator`。header heads仅作research diagnostics，不可formal；这样复用表示但不混淆pairwise truth与blind intrinsic acceptance。

必须在同一split/pool/compute比较完全统一heads、shared+task heads、simple/header分模型；formal winner只能是simple path。若复杂结构在unseen groups的risk-cert/latency没有明确收益，采用更简单的统一simple heads。header结果不能promotion v1。

### 20.4 成本预算

初始工程上限：batch-1 reference integer Gate evaluator≤5 ms/unit，全部Gate artifacts≤8 MiB，Gate自身wall-time相对generation baseline增幅≤15%；end-to-end generated candidate tokens≤unwatermarked generation 2×，end-to-end wall time≤unwatermarked 2×。RGE只在watermark policies间做same-budget accounting，不能除以“无search的no-watermark RGE”。这些必须实测；当前没有latency结果。

## 21. Pre-Gate outputs

对anchor \(i\) 和注册context view \(v\)，Pre-Gate输出经integer LUT映射的固定点值：

\[
(q_{i0}^{(v)},q_{i1}^{(v)},q_{i,both}^{(v)},\rho_{i0}^{(v)},\rho_{i1}^{(v)},\hat c_{i0}^{(v)},\hat c_{i1}^{(v)}).
\]

- \(q_0,q_1\)：fixed \(A\) 下，对第14.2节四个等权public pool variants，各 region 至少一个 formal-valid candidate 的校准成功率；deployment单位上的概率解释来自protocol hash把不同\((D_0,anchor)\)确定地路由到一个variant，必须再在actual deployment variants上校准；
- \(q^{both}\)：同一个A-sized deployment pool同时双侧可达的校准成功率；不是把两个不同variants拼接；
- \(\rho_0,\rho_1\)：各侧admissible records比例；
- \(\hat c_0,\hat c_1\)：各侧 first-success cost，归一化到 \([0,1]\)；

部署分数不是 `max(q0,q1)`，而是 calibration lower bounds 的

\[
q_{safe,i}=\min(q^-_{i0},q^-_{i1}),\qquad c_i^+=\max(\hat c_{i0},\hat c_{i1}),
\]

\[
G_i^{pre}=\mathbf1[q_{safe,i}\ge\tau_{pre}\land q^-_{i,both}\ge\tau_{both}\land c_i^+\le C_{max}].
\]

没有`p_parse/p_budget`未监督heads；parse是rule，budget worth由上述outputs和allocator导出。初始\(\tau_{pre}=0.65,\tau_{both}=0.55,C_{max}=0.75\)只在threshold-selection split选择并在risk-cert认证。Pre不读actual pool/bit、不产生evidence。reject仍属于固定candidate denominator，只是不属于pre-selected/attempted denominator。

## 22. Post-Gate outputs

`Post-Gate-G`输出`p_certificate,p_dynamic,p_pair_quality`与未校准integer ranking logit`r_rank`；输入是baseline+candidate，所以只用于generation。它的假阳性是pair-invalid candidate被G接受，仍必须被deterministic certificate/hard verifier拦截；G从不成为blind evidence。

`Post-Gate-X`只输出每项都有current-unit final-replay label的`p_quality,p_format,p_rename,p_intrinsic_accept`。曾考虑的`p_replay`被删除：规定backend下的重放是artifact conformance hard bit，不是应由样本模型猜测的概率。syntax/structure、`BlindCertificateV1`、anchor uniqueness、RegionV1 margin、duplicate/redundancy与correlation cluster也都是deterministic public values，不建learned heads；current-only model无法可靠判断document-level duplicate，强加该head反而会引入不可识别标签。没有`p_semantic_proxy`、learned`margin_region`或relative-baseline`p_formal`。

formal state有唯一三态规则。每个component \(m\in\{quality,format,rename\}\)有整数\(\theta_m^R<\theta_m^A\)，intrinsic head有\(T_R<T_A\)：hard parser/replay-shape/BlindCertificate/anchor/Region/backend先过；若全部component score\(\ge\theta_m^A\)且`l_accept≥T_A`则ACCEPT；若任一component score\(\le\theta_m^R\)或`l_accept≤T_R`则REJECT；其余为ERASURE。hard fail/backend mismatch在evidence层也归带原因erasure，不允许模型覆盖。REJECT在evidence层归erasure，只有ACCEPT+region 0/1才比较target产生match/mismatch。Post-X false accept定义为“接受但final-observable intrinsic conjunction=0”，不是无法blind识别的pairwise semantic failure。pairwise safety由generation release gate与system functionality单独评估。

## 23. Loss functions with formulas

### 23.1 双侧 soft labels

一个minibatch含\(N\)个anchor，\(i\in\{1,\ldots,N\}\)、view \(v\in\{1,2,3\}\)、region \(b\in\{0,1\}\)、public pool variant \(r\in\{0,1,2,3\}\)、record \(j\in\{0,\ldots,A-1\}\)。最终Pre监督必须对应**冻结deployment pipeline**：令\(H_{ijr}=1\)当且仅当record真实、**non-identity**且span/parse/skeleton/RewriteCertificate/`BLIND_CERTIFIED_EVIDENCE`/resource/Region-margin hard checks全过；\(F^{pair}_{ijr}=1\)当certificate PASS且没有任何available E2/E3 FAIL（全unavailable允许但tier较低）；\(A^G_{ijr},A^X_{ijr}\)分别是已冻结integer Post-G与Post-X thresholds的accept bits。于是record对region \(b\) 的admissibility严格为：

\[
V_{ijr,b}=H_{ijr}F^{pair}_{ijr}A^G_{ijr}A^X_{ijr}\mathbf1[R(x_{ijr})=b],\quad
Y_{ibr}=\mathbf1[\sum_{j=0}^{A-1}V_{ijr,b}>0].
\]

联合representation预训练时尚无冻结Post decisions，使用oracle auxiliary \(V^{oracle}=HF^{pair}y^X\mathbf1[R=b]\)；它不能作为最终q calibration truth。预训练后冻结shared backbone、Post-G/X weights与integer thresholds，重算上式全部\(V\)，随后**只更新Pre head**，最后在actual deployment variants独立校准。这样标签不会随同一优化step内的Post输出漂移。

对第14.2节完整枚举的\(R=R_{pool}=4\)个等权public pool variants，\(S_{ib}=\sum_rY_{ibr}\)。formal soft target是有限总体的精确比例，而不是把四个确定variants伪装成四次随机实验：

\[
\tilde y_{ib}=\frac{S_{ib}}{R}.
\]

同pool双侧变量与soft target为：

\[
Y_{i,both,r}=Y_{i0r}Y_{i1r},\qquad
\tilde y_{i,both}=\frac1R\sum_rY_{i,both,r}.
\]

另存side correlation和first-success cost：

\[
\operatorname{corr}_i=
\frac{\sum_r(Y_{i0r}-\bar Y_{i0})(Y_{i1r}-\bar Y_{i1})}
{\sqrt{\sum_r(Y_{i0r}-\bar Y_{i0})^2\sum_r(Y_{i1r}-\bar Y_{i1})^2}},
\quad \bar Y_{ib}=R^{-1}\sum_rY_{ibr},
\]

任一分母为0时`side_corr=null`，不填0。

\[
c_{ibr}=\frac{(\min\{j:V_{ijr,b}=1\}+1)\wedge(A+1)}{A+1},\qquad
\tilde c_{ib}=R^{-1}\sum_rc_{ibr}.
\]

若集合为空，约定\(\min\varnothing=A\)，故\(c_{ibr}=1\)；这使“无成功candidate”具有唯一、有限且可实现的cost label。candidate index从0开始，真实record的首次成功成本为\((j+1)/(A+1)\)，sentinel永远不能成为成功项。

deployment实际variant的hard labels在calibration/risk-cert上按repository component给frequentist bounds；四-variant平均不能替代actual-variant calibration。oracle-only随机generator pools若只抽取而未穷举，可另报Jeffreys Beta-Binomial posterior\((S+1/2)/(R+1)\)，但它不进入formal Pre target。若某diagnostic只能\(R=1\)，必须标`single_pool` low-precision，不把一次Bernoulli当真概率。

后续统一记\([z]_+=\max(z,0)\)，训练概率裁剪到\(p\in[\epsilon,1-\epsilon]\)、\(\epsilon=10^{-7}\)，并定义

\[
BCE(p,y)=-y\log p-(1-y)\log(1-p),
\]

\[
Huber(z;\delta)=\begin{cases}z^2/2,&|z|\le\delta,\\
\delta(|z|-\delta/2),&|z|>\delta.\end{cases}
\]

所有平均分母若有效样本数为0，该minibatch task返回0并记录`zero_label_batch`；sampler保证epoch级每task至少一个known label，否则训练run invalid。

### 23.2 Pre-Gate losses

\[
\mathcal L_{side}=\frac1{6N}\sum_{i,v,b}\operatorname{BCE}(q_{ib}^{(v)},\tilde y_{ib}),
\]

\[
\mathcal L_{both}=\frac1{3N}\sum_{i,v}\operatorname{BCE}(q_i^{both,(v)},\tilde y_i^{both}),
\quad
\mathcal L_{cost}=\frac1{6N}\sum_{i,v,b}\operatorname{Huber}(\hat c_{ib}^{(v)}-\tilde c_{ib};0.1).
\]

覆盖率target \(\tilde\rho_{ib}=R^{-1}\sum_r A^{-1}\sum_jV_{ijr,b}\)，其loss：

\[
\mathcal L_{cover}=\frac1{6N}\sum_{i,v,b}\operatorname{Huber}(\rho_{ib}^{(v)}-\tilde\rho_{ib};0.05).
\]

不强制 \(q_{safe}(CTX3)\ge q_{safe}(CTX1)\)：额外 context 可能揭露副作用，使可行性下降。使用容差一致性：

\[
\mathcal L_{ctx}=\frac{\sum_{i,b,v=1}^{2}\omega_{ibv}
[|q_{ib}^{(v+1)}-q_{ib}^{(v)}|-0.15]_+^2}
{\max(1,\sum_{i,b,v=1}^{2}\omega_{ibv})}.
\]

其中\(\omega_{ibv}=\mathbf1[|\tilde y_{ib}-0.5|\ge0.2]\)，完全由共享pool label确定，不读取尚未选择的deployment threshold或另一view prediction。边界/高不确定样本不被错误拉平；三个views仍各自对相同truth训练，因此额外context可降低误差但没有强制单调性。

### 23.3 Post-G/Post-X asymmetric、ranking 与 decision losses

G任务\(\mathcal M_G=\{certificate,dynamic,pair\_quality\}\)，X任务\(\mathcal M_X=\{quality,format,rename\}\)：

对任一task \(m\)，令label-known mask\(a_k^m\in\{0,1\}\)，class weight\(w_k^m\in[1/5,5]\)，则

\[
WBCE_{mask}(p^m,y^m)=
\frac{\sum_k a_k^mw_k^m BCE(p_k^m,y_k^m)}{\max(1,\sum_k a_k^mw_k^m)}.
\]

\[
L_G=|\mathcal M_G|^{-1}\sum_{m\in\mathcal M_G}WBCE_{mask}(p^m,y^m),\quad
L_X=|\mathcal M_X|^{-1}\sum_{m\in\mathcal M_X}WBCE_{mask}(p^m,y^m).
\]

`unknown`由label mask删除并报告coverage；prevalence weights截断\([1/5,5]\)。X intrinsic target \(y^X=\prod_{m\in\mathcal M_X}y^m\)只在三项labels全已知时定义。令\(M_X^{known}\)为该minibatch中此conjunction已知的样本数，\(p_k^X\)为独立intrinsic-accept head输出；false-accept代价20倍：

\[
L_X^{FA}=-(\max(1,M_X^{known}))^{-1}\sum_{k:y_k^X\ known}
[y_k^X\log p_k^X+20(1-y_k^X)\log(1-p_k^X)].
\]

同anchor内先从labels构造公开的排序grade：certificate fail为0；certificate pass且dynamic unavailable为1；available E2/E3全pass为2；在相同grade内以blinded quality target破tie。只对严格有序且labels已知的\((k^+,k^-)\)训练`r_rank`：

\[
L_{rank}=\mathbb E\log(1+\exp[-(r^{G}_{k^+}-r^{G}_{k^-})]).
\]

训练阶段对Post-X四个heads均固定fake-integer thresholds\(T_R^{h,tr}=-4096,T_A^{h,tr}=4096\)，并固定\(m_D=1024\)；这些不是部署threshold，也不从test选择。下式对intrinsic-accept logit\(\ell_X\)记\(T_R^{tr}:=T_R^{intrinsic,tr}\)、\(T_A^{tr}:=T_A^{intrinsic,tr}\)：

\[
L_{decision}=\mathbb E\begin{cases}
[m_D-(\ell_X-T_A^{tr})]_+^2,&y^X=1,\\
[m_D+(\ell_X-T_R^{tr})]_+^2,&y^X=0.
\end{cases}
\]

部署的\(T_R,T_A\)只在Post threshold-selection冻结。每个component/intrinsic head的selected integer gap必须满足\(T_A-T_R\ge2048\) logit units；gap内全部为erasure，不存在把连续概率“勉强四舍五入成bit”的路径。达不到该gap或risk-cert FAR的threshold tuple没有发布资格。

### 23.4 Calibration、batch、quantization 与 nuisance losses

只保留完全定义的Brier。令\(M_{cal}\)为当前head集合中label-known的scalar prediction/target pairs总数：

\[
L_{cal}=(\max(1,M_{cal}))^{-1}\sum_{k=1}^{M_{cal}}(p_k-y_k)^2.
\]

ECE只作固定15个equal-mass bins的评估指标；loss中不存在SoftECE项。posthoc monotone integer isotonic LUT只在calibration-map split拟合。

所有连续/fake-integer logits先按export scale换算到最终INT16 logit units，才进入下列loss；不能把不同layer的raw float量直接相减。对Post-X任一有decision band的head \(h\in\mathcal H_X=\{quality,format,rename,intrinsic\}\)及其固定training thresholds\(T_R^h<T_A^h\)，定义“保持reference state \(s\)”的margin：

\[
\mu_h(\ell;s)=\begin{cases}
\ell-T_A^h,&s=ACCEPT,\\
T_R^h-\ell,&s=REJECT,\\
\min(\ell-T_R^h,T_A^h-\ell),&s=ERASURE.
\end{cases}
\]

PRE不是一个可代入上述公式的虚构single logit。令所有量均为LUT后的Q15整数，训练期composite signed margin明确定义为

\[
g^{pre}=\min\{q_{0,Q15}-\tau_{pre,Q15},q_{1,Q15}-\tau_{pre,Q15},
q_{both,Q15}-\tau_{both,Q15},C_{max,Q15}-c_{0,Q15},C_{max,Q15}-c_{1,Q15}\}.
\]

`PRE_ACCEPT`当且仅当\(g^{pre}\ge0\)，否则`PRE_REJECT`；对应离散margin为\(\mu_{pre}(g;ACCEPT)=g\)、\(\mu_{pre}(g;REJECT)=-g-1\)。这里training thresholds是第21节initial values的nearest-Q15整数；部署仍只能用独立selection/risk-cert后冻结的thresholds。

记\(\boldsymbol\ell^{PRE}\)为PRE七个logits，\(\boldsymbol\ell^{GX}\)为Post-G四个与Post-X四个logits按固定顺序的拼接；Post-G ranking/heads没有blind accept authority，只参加normalized logit一致性项。对同一样本取batch-1 reference、其PRE/Post-X states与随机batch-size/companions/padding变体：

\[
\begin{aligned}
\mathcal L_{batch}^{PRE}=\mathbb E\bigg[&\frac1{7}\|\boldsymbol\ell_*^{PRE}-\boldsymbol\ell_{B,L}^{PRE}\|_2^2\\
&+5[m_D-\mu_{pre}(g^{pre}_{B,L};s_*^{pre})]_+^2
\mathbf1[\mu_{pre}(g^{pre}_*;s_*^{pre})\ge m_D]\bigg],\\
\mathcal L_{batch}^{G,X}=\mathbb E\bigg[&\frac1{8}\|\boldsymbol\ell_*^{GX}-\boldsymbol\ell_{B,L}^{GX}\|_2^2\\
&+5\sum_{h\in\mathcal H_X}[m_D-\mu_h(\ell_{B,L}^h;s_*^h)]_+^2
\mathbf1[\mu_h(\ell_*^h;s_*^h)\ge m_D]\bigg].
\end{aligned}
\]

reference自身距decision boundary少于\(m_D=1024\)时不参加对应state-margin一致性项（仍参加logit/classification loss），并在validation报告为boundary-hard；不伪造新label。RegionV1不在该loss内，因为它已经整数确定。QAT模拟最终IntGateV1 graph；\(s_{fp}^{pre},s_{fp}^h\)由FP32 fake-integer outputs与同一training thresholds产生，\(g_{fp}^{pre}\)和\(g_{int}^{pre}\)按上式分别重算：

\[
\begin{aligned}
\mathcal L_{quant}^{PRE}=\mathbb E\bigg[&\frac1{7}\|\boldsymbol\ell_{fp}^{PRE}-\boldsymbol\ell_{int}^{PRE}\|_2^2\\
&+5[m_D-\mu_{pre}(g_{int}^{pre};s_{fp}^{pre})]_+^2
\mathbf1[\mu_{pre}(g_{fp}^{pre};s_{fp}^{pre})\ge m_D]\bigg],\\
\mathcal L_{quant}^{G,X}=\mathbb E\bigg[&\frac1{8}\|\boldsymbol\ell_{fp}^{GX}-\boldsymbol\ell_{int}^{GX}\|_2^2\\
&+5\sum_{h\in\mathcal H_X}[m_D-\mu_h(\ell_{int}^h;s_{fp}^h)]_+^2
\mathbf1[\mu_h(\ell_{fp}^h;s_{fp}^h)\ge m_D]\bigg].
\end{aligned}
\]

shared representation \(h\) 接gradient-reversal heads预测generator/repository family；\(g,r\)是仅训练可见的family labels，`GRL`前向为identity、反向对backbone梯度乘\(-1\)，而nuisance heads自身正常最小化CE：

\[
\mathcal L_{nuis}=CE(d_{gen}(GRL(h)),g)+CE(d_{repo}(GRL(h)),r).
\]

它只降低 shortcut pressure，不替代 leave-one-group-out 测试；attempt、pool size、target/key/retry 根本不进入 input graph。总损失中的`PRE`与`G,X`上标分别表示只在相应branch representations/head labels上计算上述batch、quant、Brier与nuisance项；这也定义了\(L_{cal}^{PRE},L_{cal}^{G,X},L_{nuis}^{PRE},L_{nuis}^{G,X}\)，不存在跨branch缺失logit填零。

### 23.5 RegionV1没有训练loss

RegionV1由SHA-256与固定threshold定义；不存在triplet、invariance、projection、margin或QAT loss。balance、margin和双侧可达是Oracle measurements。candidate generator/Gate不能联训移动partition制造容量。

### 23.6 最终总损失、权重和 cost-aware selection

\[
\begin{aligned}
\mathcal L_{pre}={}&1.0L_{side}+0.75L_{both}+0.25L_{cover}+0.25L_{cost}+0.10L_{ctx}\\
&+0.10L_{batch}^{PRE}+0.10L_{quant}^{PRE}+0.25L_{cal}^{PRE}+0.20L_{nuis}^{PRE},\\
\mathcal L_{post}={}&2.0L_X^{FA}+0.5L_G+0.5L_X+0.5L_{rank}+0.25L_{decision}\\
&+0.10L_{batch}^{G,X}+0.10L_{quant}^{G,X}+0.25L_{cal}^{G,X}+0.20L_{nuis}^{G,X},\\
&\boxed{\mathcal L_{total}=\mathcal L_{pre}+\mathcal L_{post}}.
\end{aligned}
\]

这是shared Gate的监督multi-task objective，不是generator-to-detector的不可微“纯端到端”。训练严格分两步：Stage J在train用\(V^{oracle}\)辅助targets最小化上式，获得shared representation与Post heads；Post map/select后冻结backbone/Post；Stage P在train重算\(V^{dep}\)，只更新Pre head并最小化上式中的\(\mathcal L_{pre}\)。最终artifact从未用risk-cert/test梯度。显示的系数是initial recipe；model-validation只允许把每个loss group整体乘\(\{0,0.5,1,2\}\)，不是声称表内每个系数来自该集合。Pareto selection首要hard constraint是Post-Gate FAR；选定后冻结，不在test动态调权，也不默认用GradNorm。

validation model selection的cost-aware utility中，\(\pi\)是一个完整Gate/allocator policy；\(N_{formal,match}\)是final replay matches，\(N_{mismatch}\)是mismatches；\(C_{search}\)为reserved RGE除以no-Gate validation mean，\(C_{gate}\)为Gate integer-MAC count除以no-Gate候选rule-evaluation MAC-equivalent mean，二者均dimensionless并等权：

\[
U(\pi)=\frac{\mathbb E[N_{formal,match}-20N_{mismatch}]}
{\mathbb E[C_{search}+C_{gate}]},
\]

且必须同时满足 functionality、FAR、calibration、replay、FPR 和 evidence-density hard constraints；高 \(U\) 不能补偿违反任何 hard constraint。RL 的 action 会是 candidate-or-skip，reward 是 \(U\) 的噪声代理，而 sampling、parser、execution 与 region threshold 不可微，且 generator-style reward hacking 风险高。因此 v1 不使用 RL；只有 supervised baseline、Oracle 与 system tests 都通过后，contextual bandit 才可作为隔离 ablation，不能替代 validators。

简单 baseline：

\[
\mathcal L_{base}=\mathcal L_{side}+WBCE_{mask}(p^X,y^X),
\]

只有两个side heads与intrinsic head；没有`qboth` joint head、calibration、ranking、context/batch consistency、QAT、nuisance或auxiliary heads，用于辨认复杂目标的真实增益。

## 24. Calibration strategy

### 24.1 Pre-Gate conservative lower bounds

model-validation只选架构/损失。在Post已冻结并只用train重算\(V\)、训练Pre head之后，**Pre calibration-map 5%**拟合唯一monotone integer isotonic LUT，**Pre threshold-selection 2.5%**冻结bins/threshold candidates；joint risk-cert对最终chosen mapping/threshold给bounds。Pre bins只用可部署public covariates与score，不按generator/repository路由。每个risk-cert anchor使用第14.2节由其public anchor实际导出的deployment variant，不用四-variant平均label，也不参与训练；bin有\(n,s\)时CP lower bound：

\[
LCB_{CP}(s,n;\delta)=\begin{cases}
0,&s=0,\\
\operatorname{Beta}^{-1}(\delta;s,n-s+1),&s>0.
\end{cases}
\]

confidence family包含two sides+both×全部bins/heads；预注册\(\delta_{cell}=0.05/M_{pre}\) Bonferroni（若采用更紧simultaneous bound必须在test前固定）。同时做component-cluster bootstrap，最终取更保守bound。n<200或repositories<20按预注册public层次backoff；仍不足LCB=0。报告ECE/Brier/worst hidden-group calibration；unseen repo/task/generator ECE≤0.05是initial threshold，不是当前结果。

ECE固定为15个equal-count bins（相同score不拆bin，故实际bin数可<15）：\(ECE_{15}=\sum_m(n_m/n)|\bar p_m-\bar y_m|\)；binary actual-deployment labels用于formal ECE，四-variantsoft labels只报training ECE。Brier按23.4节。float temperature scaling仅作nonformal baseline；正式使用exported full integer isotonic LUT。若研究conformal lower bounds，必须用同一Pre map/select/risk split并作为预注册alternative，不能在test后择优。

### 24.2 Post-Gate risk control

在任何Post训练前，先通过14.4节`PostXIdentifiabilityGateV1`；失败的唯一裁决是learned Post-X路线NO-GO，不允许靠threshold把全部collision cells擦除后称成功。Post calibration-map 5%先拟合Post LUT；Post threshold-selection 2.5%枚举Post-X integer\((T_R,T_A,\{\theta_m^R,\theta_m^A\}_{m\in\mathcal M_X})\)，只用FinalReplayCalibrationRecord而非candidate rows，覆盖natural/key-neutral fallback/multi-D1/edit/null strata中的**registered non-identity replay forms**；identity rows只验证rule erasure，不进入Post FAR/coverage分母。在满足每对reject<accept、经验FAR≤0.2%的组合中选known-positive acceptance coverage最高者，lexicographic integer tuple破tie，然后永久冻结Post backbone/heads/LUT/thresholds。joint risk-cert按document/anchor family计算one-sided CP与component-bootstrap UCB并取max；正式要求overall及每个registered final-stream stratum UCB≤0.5%，同时known-positive acceptance coverage point≥50%、one-sided95% component-bootstrap lower≥40%，known-label coverage≥70%。confidence alpha按heads×strata×hidden audit groups同时分配，记cell allocation为\(\delta_{cell}\)。每cell negatives至少为\(\max[1000,\lceil\log(\delta_{cell})/\log(0.995)\rceil]\)，这是零false accept时让CP upper≤0.5%的最低量；出现failure时继续按精确CP反算更大n，不能固定1000后宣称达标。另要求≥20 repositories、单repo≤10%；不足即unsupported。unknown计入known-label coverage的分母，不能隐藏。

同一Post map/select partitions还按预注册alpha ledger选择Post-G的Q15 accept thresholds\(\gamma_{certificate},\gamma_{dynamic},\gamma_{pairquality}\)；`A^G=1`当三个scores均不低于threshold且deterministic hard/pair verifier PASS。dynamic truth unavailable时仍可由score筛选，但`cert_tier=1`；available FAIL无论score均拒绝。选择顺序先满足hard-verifier后pair-invalid acceptance=0，再最大化valid coverage，integer tuple lexicographic破tie。Post-G永远不能覆盖hard fail。

integer LUT、thresholds、RegionV1、quality/format/rename/intrinsic thresholds、strata/backoff与alpha ledger全写protocol。risk-cert/test失败不可回调。conformal是预注册alternative baseline，不得test择优。

### 24.3 Selection-aware calibration

同时报告all-eligible、pre-selected、attempted、dual-complete与final-blind populations。这里必须区分两种“selection”：Pre/allocator只是embedding-side预算决策，按方案C不构成evidence且blind detector故意不使用它；因此把隐藏Pre plan塞进null statistic反而违反final-code-only约束。formal null完整重放的是全部**detector-side** selection：rule、anchor uniqueness、Post-X、RegionV1、erasure、cluster、evidence guard、public stratum routing和query-family correction，且绝不只在embedded rows校准。作为独立causal audit，可在null D0上运行完整key-free Pre/allocator/pools并验证“条件于pre-selected/dual-complete”不会改变报告的blind-null tail；该audit不能筛掉null rows或成为detector输入。actual D1只做test。

## 25. Quantization-aware and batch-consistency training

### 25.1 Normative deterministic contract

RegionV1天然整数；Gate连续训练值无跨硬件承诺。formal只运行`IntGateV1` reference evaluator：

- CPython 3.12 patch version、LibCST/tokenize versions 与 canonicalizer hashes 固定；
- tokenizer vocab/normalization/hash 固定；未知 token、UTF-8、newline、tab 和 truncation rule 固定；
- tensor shapes按head固定：`PRE=[1,4,96]`、`POST_G=[1,2,4,96]`且pair axis为baseline/candidate、`POST_X=[1,4,96]`且前三slots为BOS；masked positions/slots使用固定PAD/BOS及mask；overlength按20.2处理；
- op allowlist仅`EmbeddingGather,IntAdd,IntMatMul,Int64Bias,SRS,SatInt16,ReLU,Concat,IntegerLUT,IntegerCompare`；无float、LayerNorm、Softmax、BatchNorm、dropout；
- weights INT8、activations INT16、accumulator/bias INT64。`Sat16(z)=min(32767,max(-32768,z))`；对shift\(s>0\)，`SRS(z,s)=sign(z)·floor((|z|+2^{s-1})/2^s)`（exact half away from zero）；所有layer bounds静态证明不溢出INT64；
- 每个概率head的最终INT16 logit以unsigned index`logit+32768`查完整65,536-entry UINT16 LUT；LUT输出的`Q15_PROB`范围严格是0…32767（值表示`integer/32767`），不能输出32768…65535。无插值、exp或float。`r_rank`保留INT16。accept/reject及Pre thresholds全是manifest中的INT16/Q15整数；
- allocator不属于neural graph但属于同一formal reference contract：`Q12_ONE=4095`、inputs 0…4095、utility/intermediates signed INT64、非负exact floor-div、最大4,293,918,720、`-1`唯一reject sentinel；任何INT32 conversion、wrap或saturation都使artifact invalid；
- exact reference evaluator source/binary/container hashes与single-thread batch1固定；`reference_runtime_fingerprint`还固定OS/kernel ABI、libc、compiler/version/flags、CPU ISA allowlist、endianness、thread count和禁用动态dispatch/JIT的flags。纯整数alternate仍须逐vector资格化，不能只因同为x86/AVX2即视为兼容；
- artifact、runtime、operator 或 conformance hash 不匹配即 `UNAVAILABLE_ARTIFACT`，不降级到近似 FP 模型。

PyTorch 官方文档明确不保证不同 release/platform 或 CPU/GPU 完全可复现，batch 与逐样本计算也可能不同；因此 formal correctness 不能靠“固定 seed”。[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### 25.2 QAT 的边界

QAT用上述integer graph fake quant、threshold margin与float/integer consistency，只改善训练收敛；**确定性来自finite integer semantics与reference evaluator，不来自QAT**。不使用generic QDQ Transformer或“AVX2即可”等宽泛合同。

每次发布至少10,000无secret conformance vectors和boundary fuzz；包含canonical bytes、RegionV1 digest/u64、每层integer tensors、all logits/LUT/states。reference及任何formal alternate必须**所有integer tensors byte-exact、0 flips**；`≤10^-4`不够formal，只能标`DIAGNOSTIC_NONFORMAL`。任何mismatch阻止启动。

### 25.3 必测矩阵

| 轴 | 水平 | 比较 | formal acceptance |
|---|---|---|---|
| batch size | 1,2,4,8 | 同样本独立 vs 混合 companions | normative B1 100%；batched backend 需上述 conformance |
| batch composition | all-short/all-long/mixed/adversarial boundary | integer logits 与 state | state flip 超标即不合格 |
| padding/shape | effective 16/32/64/96 per slot；固定/动态 diagnostic；G pair-axis swap反例 | masks、integer tensors | formal只接受head-specific fixed shapes；G axis order固定 |
| precision | FP32, FP16, BF16, exported integer | continuous delta 与 discrete flip 分开 | integer reference exact；其余训练/诊断 |
| device | reference CPU、第二 CPU、GPU | score/state | 未资格化设备不得 formal detect |
| process | 两次 clean process、两 container replicas | bytes/logits/state | reference outputs byte-identical |
| mode | single sample 与 mixed batch | state | 见 batch criterion |

训练时生成companions/padding并周期运行真实integer graph。formal evaluator永远batch1，所以batch-composition flip必须为0且是vectorized alternate资格化测试。指标仍分别报告continuous error、batch/quant/process/device flips与erasure migration；RegionV1 boundary直接erasure，Gate阈值中间band erasure。

若reference两clean processes/replicas不能做到所有integer tensors exact，本路线System NO-GO。未资格化CPU/GPU/client返回`UNAVAILABLE_BACKEND`，不得近似检测。

## 26. Watermark embedding protocol

### 26.1 Target stream

没有document nonce、salt、caller identity或codeword target。从master key用`HKDF-Extract(salt=protocol_id,IKM=K_master)`，再以length-prefixed info `"SAGE-C/K-unit/v3"`做HKDF-Expand得到32-byte `K_unit`；不派生`K_msg/K_mask`。每个formal unit唯一target为：

\[
b_t=LSB\,HMAC_{K_{unit}}(
LP(\texttt{"SAGE-C/unit/v3"})\|LP(protocol\_id)\|LP(document\_identity)\|LP(anchor_t)).
\]

`LP(field)=u32be(len(field))||field`，四个fields长度分别固定为14、32、32、32 bytes；任何不符fail closed。`LSB`固定取HMAC digest第31 byte的bit 0。`document_identity=DocumentIdentityV1(D0)=DocumentIdentityV1(D1)`已在key-neutral commitment中冻结，target module只能读取这32 bytes与已提交anchor；它不能计算/修改identity或访问code。anchor不含current candidate text，identity对所有registered candidate alternatives归一，因此pool bytes/order/count与key、target完全无关。不同identity的HMAC preimage必不同（除SHA-256 collision）；观察identity A的公开region仅泄露该PRF input的一位，不能复用于identity B。formal v1的唯一合法状态是`ecc_state=ECC_DISABLED_FORMAL_V1`，全部units都走本公式；`DISABLED_FORMAL_V1`、`DISABLED_BY_PROTOCOL`及任何其他拼写均fail closed。

### 26.2 嵌入伪代码

```text
PLAN_AND_EMBED(prompt, public_protocol P, sealed_key_handle K):
  REQUIRE_EXACT_ARTIFACT_AND_CONFORMANCE(P)
  SEAL(K)                                         # target API inaccessible
  D0 = GENERATE_BASELINE_WITHOUT_KEY(prompt, P.baseline_rng)
  T0 = STRICT_PARSE_LINEARIZE_ANCHOR(D0, P)
  if T0 fails: return NO_FORMAL_ARTIFACT
  if not KEY_FREE_STATIC_EMBEDDING_ADMISSION(T0,D0,P):
      return D0_WITH_NO_PROTOCOL_METADATA, OUT_OF_BAND_STATUS_NOT_WATERMARKED
  doc_id = DOCUMENT_IDENTITY_V1(D0,T0,P)           # no key; full-code binding
  if doc_id unavailable: return NO_FORMAL_ARTIFACT

  candidates = [u for u in T0 if u.origin==GENERATED
                 and ELIGIBILITY(u)==FORMAL_ELIGIBLE]
  q = {u: PRE_INT(BUILD_CTX(T0,u,P.width)) for u in candidates}
  allocated = KEY_FREE_ALLOCATOR(candidates,q,P.B_doc,P.reservation_rule)

  plan = []
  for u in candidates in source order:
    if u not in allocated:
       plan.append(KEY_NEUTRAL_SKIP(u)); continue
    r = DERIVE_POOL_VARIANT(protocol_id,u.anchor)       # 0..3, no nonce
    pool = [BASELINE_RECORD(D0,u)]                 # C0
    for j in 1 .. P.A-1:                          # exactly A-1 rewrites/sentinels
       pool.append(RANKED_CERTIFIED_OR_SENTINEL(D0,u,r,j,P.rewrite_catalog))
    scored = SCORE_ALL_WITHOUT_KEY(pool,
              rewrite_pair_and_blind_certificate=True,
              RegionV1=True, PostG_with_frozen_D0_context=True,
              PostX_current_only=True)
    has0 = EXISTS_ADMISSIBLE(scored, region=0)
    has1 = EXISTS_ADMISSIBLE(scored, region=1)
    state = DUAL_COMPLETE if (has0 and has1) else DUAL_INCOMPLETE_SKIP
    w0 = ARGMIN_PUBLIC_KEY(scored,region=0) if has0 else NULL
    w1 = ARGMIN_PUBLIC_KEY(scored,region=1) if has1 else NULL
    plan.append(COMMITTABLE(u,r,state,scored,w0,w1))

  for scope in LEXICAL_CODE_OBJECT_SCOPES(T0) in structural order:
    entries = SORT_BY_ANCHOR(DUAL_COMPLETE entries in scope)  # m <= 4
    cert = ENUMERATE_ALL_2_POW_M_WINNER_COMBINATIONS(D0,entries)
    if not cert.all_operational_v2_digests_and_document_identities_equal_baseline:
       MARK_ALL(entries,COMPOSITION_INCOMPLETE_SKIP)          # no reallocation
    ATTACH_SCOPE_CERTIFICATE(plan,scope,cert)

  guard = PLAN_SIGNAL_GUARD(plan,min_slots=16,min_clusters=8,min_neff=12,
             max_final_replay_shapes=512,
             final_canonical_token_range=[4096,65536])
  commitment = HASH_ORDERED_FULL_PLAN(P,doc_id,plan,P.B_doc,guard)
  ASSERT_TARGET_MODULE_WAS_SEALED_DURING_ALL_PLANNING()
  if not guard:
      return D0_WITH_NO_PROTOCOL_METADATA, OUT_OF_BAND_STATUS_NOT_WATERMARKED
  target_api = UNLOCK_TARGET_ONLY_INTERFACE(K,commitment,doc_id)

  replacements = []
  for entry in plan:
    if entry.state != DUAL_COMPLETE: continue      # target never derived
    b = TARGET_V3(target_api,P.protocol_id,doc_id,entry.anchor)
    selected = CONSTANT_TIME_MUX(entry.winner_0,entry.winner_1,b)
    ASSERT(selected hash was included in commitment)
    replacements.append(entry.span, selected)

  D1 = APPLY_NONOVERLAPPING_SIMPLE_REPLACEMENTS(D0,replacements)
  if DOCUMENT_IDENTITY_V1(D1,P) != doc_id:
      RECORD_POST_TARGET_SECURITY_INCIDENT_AND_SUSPEND(P,K)
      return NO_FORMAL_ARTIFACT
  blind = BLIND_REPLAY(D1,P)                       # no functionality claim
  pair  = PAIR_VERIFY_DETERMINISTIC(D0,D1,P)       # exact evaluator/digest/composition only
  target_check = CHECK_SELECTED_REGIONS(D1,plan,target_api)
  if blind != PASS or pair != PASS or target_check != PASS:
      RECORD_POST_TARGET_SECURITY_INCIDENT_AND_SUSPEND(P,K)
      return NO_FORMAL_ARTIFACT                    # no D0 fallback/retry/replan
  EMIT_FINAL_CODE_WITH_PROTOCOL_ID_ONLY(D1,P)
  WRITE_OPTIONAL_PRIVATE_AUDIT_OUT_OF_BAND()
  return D1
```

pair verifier必须在target前逐candidate运行；combined`PAIR_VERIFY_DETERMINISTIC(D0,D1)`只作release defense且绝不重跑动态tests。由于certified disjoint constant rewrites预期可组合，combined failure是implementation incident而非可利用fallback。这里generation-side `BLIND_REPLAY(D1,P)`只断言plan内每个selected replacement在final \(D_1\) 仍通过final-observable hard path，并另对全document记录自然non-identity forms的诊断状态；自然form的REJECT/ERASURE不会错误触发incident。selector唯一key-dependent动作是从actual dual-complete fixed pool的目标side选一项。v1只改非重叠simple spans，所有headers不改。

### 26.3 Finalization 与 artifact

生成结束后对完整final code重新strict parse/linearize/anchors/DocumentIdentity/Post-X/RegionV1；任何generation“success”不升级formal evidence。artifact默认只有code。self-describing模式在shebang/UTF-8 cookie后只允许一行`# sage-c-protocol: sha256=<64 lowercase hex>`；否则API唯一指定protocol ID。重复/malformed/conflict/unknown metadata fail closed。**不存在nonce、salt或document-identity metadata/API字段**；identity只能从code重建。不写per-unit state、plan、pool、target、retry或key。

嵌入端的 `INTERNAL_PRE_SKIP/RULE_SKIP` 不是盲提取可见状态。最终 detector 会对所有registered non-identity final forms应用同一Post-Gate-X；因此原代码中预先存在、从未尝试的显式registered forms可能形成null-like match/mismatch/erasure，必须包含在calibration。未分配或dual-incomplete slot保留的canonical identity baseline则必为rule erasure，绝不会偶然成为证据。报告中的embedded-unit rate以实际target-selected positions为分母，detector evidence density只由final replay定义，二者不混用。

## 27. Watermark extraction protocol

### 27.1 输入与伪代码

提取器唯一输入是`(final_code bytes, protocol_id, public registry/artifacts, K)`；protocol ID可由唯一leading comment恢复，不能有nonce/identity/subset/anchor参数：

```text
EXTRACT(final_code D, protocol_id, public registry, secret K):
  REQUIRE_EXACT_PROTOCOL_EVALUATOR_AND_CONFORMANCE()
  if request has nonce/salt/document_identity/subset/anchor/unknown field:
      return UNAVAILABLE_REQUEST
  T = STRICT_PARSE_LINEARIZE_ANCHOR(D,P)
  if parse fails: return INVALID_DOCUMENT
  if not STATIC_DOCUMENT_ADMISSION(T,D,P): return UNAVAILABLE_OR_LENGTH_OUT_OF_SCOPE
  doc_id, blind_cache = RECONSTRUCT_DOCUMENT_IDENTITY_V1(D,T,P)
  if doc_id unavailable: return UNAVAILABLE_DOCUMENT_IDENTITY
  evidence = []
  for t in T.all_syntactic_units:
    if FINAL_REPLAY_SHAPE(t) != REGISTERED_NONIDENTITY_FORM or not t.anchor_unique:
       evidence.append(ERASURE_RULE_OR_ANCHOR); continue
    view = BUILD_POST_X_CURRENT_ONLY(t)            # 3 BOS slots + current
    hard = FINAL_OBSERVABLE_HARD_VERIFY(t)
    blind_cert = blind_cache[t]                     # inverse template + alias digest
    post = POST_X_INTEGER_REFERENCE(view)
    region,margin = REGION_V1(REGION_CURRENT_V2(t.canonical_current))
    if not hard or blind_cert != BLIND_CERTIFIED_EVIDENCE \
       or post != ACCEPT or region == BOUNDARY:
       evidence.append(ERASURE_WITH_REASON); continue
    target = TARGET_V3(K,P.protocol_id,doc_id,t.anchor)
    evidence.append(MATCH if region==target else MISMATCH)
  clusters,counts,n_eff = BUILD_PUBLIC_CLUSTERS(evidence,T)
  if not EVIDENCE_SUFFICIENCY(counts,n_eff,T,P): return INSUFFICIENT_EVIDENCE
  score = SCORE_CLUSTERS_EXACT_Q30(clusters)
  stratum = PUBLIC_STRATUM_MAP(T,evidence_without_match_signs,P)
  if stratum == OOD: return UNAVAILABLE_OOD
  p = STRATIFIED_EMPIRICAL_PVALUE(score,stratum,P.null_release)
  return RESULT_FOR_FAMILY_CORRECTION(p,counts,n_eff,stratum,
                                      ecc_state=ECC_DISABLED_FORMAL_V1)
```

`STATIC_DOCUMENT_ADMISSION`只检查protocol/artifact/parser/encoding、`CanonicalDocumentTokenCountV1`和与sign无关的静态resource limits；它不读取尚未计算的Post-X/evidence count。`DetectorResourceContractV1`进一步要求registered non-identity replay-shaped units≤512；先编译final D一次并缓存OperationalDigestV2 graph，每个shape至多再编译一个identity-reconstruction，DocumentIdentity复用同一blind cache，故最多513次fresh compile和512次Post-X/Region calls，单compile CPU quota 1 second、document总CPU quota 600 seconds、RSS 2 GiB。任一quota/overflow失败使整个document `UNAVAILABLE_RESOURCE`，不能只取前512个units形成有利subset。定义`anchor_ambiguity_fraction = ambiguous_registered_nonidentity_replay_shapes / max(1, all_registered_nonidentity_replay_shapes)`；分子/分母都在Post-X和target前由parser/anchor规则确定。`EVIDENCE_SUFFICIENCY`在全unit hard/Post-X/Region replay及duplicate suppression之后检查16–512 non-erasure、≥8 clusters、exact \(n_{eff}\ge12\) 与该fraction≤20%；public stratum也在此后路由。这样没有“先用accepted evidence admission、后计算accepted evidence”的循环，也不会在看match sign后选择是否检测。

Pre与Post-G不在extraction。Post-X、hard verifier和RegionV1只读final stream。pair semantics不能blind重放，也不由Post-X冒充；system functionality独立约束。提取端绝不读repository/task/generator ID、prompt、logits、hidden states、candidates、attempts、audit或sidecars。

### 27.2 解析对偶与失败路径

提取端使用与嵌入 finalization 相同的 parser conformance corpus，明确识别 individual semicolon small statements、inline suite bodies、multi-line headers、async variants、decorators-as-auxiliary、`match/case`、`except*` 及 owner-specific `else`。最终 parser 没有 token recovery fallback；strict parse failure 是 document-level invalid，不猜 bit。

不存在protocol registry、输入超出正式长度、未知Python feature、unsupported encoding/token/surrogate、runtime mismatch、integer overflow、DocumentIdentity歧义、anchor ambiguity或Post-X/RegionV1 boundary时fail closed为unavailable/erasure。不会换相近模型、接受metadata identity或比较continuous embeddings。

## 28. Match/mismatch/erasure semantics

每个 syntactic slot 先有 diagnostic state，再归并到 primary statistic：

| 原始状态 | 条件 | Primary contribution |
|---|---|---:|
| `MATCH` | rule eligible、local anchor unique、BlindCertificate/hard/Post-X accept、RegionV1=target | +1 |
| `MISMATCH` | 同上，但RegionV1与target相反 | -1 |
| `ERASURE` | rule excluded/oracle-only/identity form、BlindCertificate fail、anchor duplicate、Region/Post-X boundary、低质量、intrinsic unknown | 0 |
| `INVALID_UNIT` | span/structure/hard invariant 失败但 document可解析 | 0；记录原因 |
| `UNAVAILABLE_UNIT` | artifact/input limit/operator failure | 0；若系统性则 document unavailable |
| `INVALID_DOCUMENT` | final strict parse失败 | 不做检测 |
| `UNAVAILABLE_DOCUMENT` | protocol/key/artifact/runtime不兼容 | 不做检测 |

只有一个被Post-X接受的opposite region才是mismatch；候选耗尽/skip不属于提取状态。mismatch负权防止只保留matches；erasure不支持也不反对。`INSUFFICIENT_EVIDENCE`不是negative：formal v1还要求canonical token length 4,096–65,536、non-erasure≥16、clusters≥8、\(n_{eff}\ge12\)。短函数/HumanEval/MBPP多数会insufficient，必须计入overall coverage，不能只报conditional TPR。

`formal evidence`只表示final-code keyed signal，不表示提取器证明相对baseline语义等价。pairwise safety只约束embedding/release；Post-X FAR仅针对intrinsic labels，pair-invalid release计入separate generation safety/system correctness failure。

公开报告默认只给counts、aggregate statistic/p-value、`ecc_state=ECC_DISABLED_FORMAL_V1`和failure histogram；per-anchor expected bit/contribution/document identity不发布，以降低chosen-query detector oracle风险。

## 29. Evidence aggregation

### 29.1 Cluster

formal v1没有ECC blocks。excluded/oracle-only/noncarrier/identity positions保持erasure，不能压缩重编号或把后续target移入空位。用**公开、key-independent**规则形成dependence clusters：在同lexical code-object scope且同nearest compound-owner suite内，以全部syntactic-unit的suite-local`ast_ordinal`为坐标，对两个accepted evidence在ordinal gap≤3时连无向边，cluster取该图的transitive connected components；不同suite永不连边。相同`current_text_identity`复制品先只让raw-byte start最小者保留accepted state，其余强制erasure，再建图。raw-byte tie不可能发生，若发生parser span collision则document unavailable。cluster边界不能看match/mismatch sign。

令 \(e_i\in\{-1,0,+1\}\)，cluster \(c\) 的非 erasure 数 \(n_c\) 和均值：

\[
n_c=\sum_{i\in c}\mathbf1[e_i\ne0],\qquad
G_c=\frac{\sum_{i\in c}e_i}{\max(1,n_c)}\in[-1,1].
\]

解释性的实数document statistic为：

\[
T(D)=\frac{\sum_{c=1}^{C}G_c}{\sqrt C},\qquad
n_{eff}=\frac{(\sum_cn_c)^2}{\sum_cn_c^2}.
\]

formal实现没有float。定义对\(d>0\)的exact half-away quotient

\[
RoundDiv(z,d)=sign(z)\left\lfloor\frac{|z|+\lfloor d/2\rfloor}{d}\right\rfloor,
\]

其中`sign(0)=0`、`sign(z)=1`当`z>0`、`sign(z)=-1`当`z<0`；`d≤0`是protocol failure，不进入除法。

并令

\[
g_c=RoundDiv\!\left(2^{30}\sum_{i\in c}e_i,n_c\right),\quad
z=\sum_{c=1}^{C}g_c,\quad q_C=isqrt(C\,2^{40}),
\]

\[
\boxed{T_K^{Q30}(D)=RoundDiv(z\,2^{20},q_C)}.
\]

`isqrt`是非负整数floor square root；\(C\le512\)时所有中间量静态证明在signed INT64内。null tails、排序、p-value和conformance vectors只存/比较这个INT64 Q30值；第31节的\(T_K\)均指它。`cluster_weight_q30=RoundDiv(2^30,n_c)`，单元`contribution_q30=RoundDiv(e_i·2^30,n_c)`只用于audit，primary以cluster sum公式计算，避免逐项rounding drift。\(n_{eff}\ge12\)的guard用exact交叉乘法\((\sum n_c)^2\ge12\sum n_c^2\)，报告值保存为reduced rational numerator/denominator。

cluster mean让一个复杂clause的大量相似units不能线性放大证据。\(n_{eff}\)只作报告/guard，不解释为iid Bernoulli数。HMAC targets也不使units或Gate selection独立；尾概率完全依赖第31节分层empirical null。

### 29.2 层级输出

每个scope报`match/mismatch/erasure` counts与cluster statistic；全局明确报`ecc_state=ECC_DISABLED_FORMAL_V1`；一个formal v1 request只接受一个Python module，并且只产生一个预注册primary \(T_K\)。多文件project不是v1检测对象：调用方必须把每个module作为独立family member提交，并按第31.3节的Holm/Bonferroni规则校正；任何project-level pooling都需要新的major protocol、key-independent file clusters、长度normalization与独立project-null calibration，不能事后挑分数最高的函数或文件。

正式detection object是单个UTF-8 Python module，canonical tokens须4,096–65,536。初始guard：accepted non-erasure≥16、clusters≥8、\(n_{eff}\ge12\)。未达到返回`INSUFFICIENT_EVIDENCE`。多文件project不在v1；若未来聚合，必须新protocol/null。报告unconditional coverage、conditional TPR和overall TPR。

## 30. Erasure-aware ECC

**Formal v1明确没有ECC。** 这是审查后删除，而不是尚待选择：`m_scope≤4`限制一个lexical code-object scope最多4个target-selected carriers；旧Hamming block完全位于一个suite，suite又属于一个scope，因此由水印自身提供的observed positions最多4，严格小于旧decoder要求的5。natural markers不是按该codeword嵌入，不能充当第五个可靠symbol。故任何旧`Hamming(7,4)` conditional success都会有空分母或依赖非水印偶然项，不能称纠错。

v1的erasure-aware mechanism是`NoCodeV1 + TriStateClusterAggregateV2`：

1. 每个candidate-invariant local anchor独立使用第26.1节`K_unit` target；没有block position、parity、message或mask。
2. rule/identity/boundary/low-confidence/edit-invalid单元保持坐标上的erasure；不impute、不压缩、不把后续单元重编号。
3. accepted units按第29节公开cluster降权，`n_eff`与最小evidence guard在看match sign前裁决。
4. statistic在完整natural/random-side/wrong-key/edit-error/sample-aware-mosaic null上校准；这处理selection、erasure和相关性，而不是假设独立二项。
5. schema、manifest与输出必须逐byte断言`ECC_DISABLED_FORMAL_V1`。旧`block_anchor/block_position/K_msg/K_mask/ecc_verified`字段出现即artifact incompatible，不能静默忽略。

因此用户要求的“erasure-aware ECC或其他擦除处理”由后一种**其他擦除处理**满足；它不增加payload或纠错容量。指标仍保留`ECC decode success`字段作兼容审计，但formal v1唯一合法值同样是`ECC_DISABLED_FORMAL_V1`，无分子/分母且绝不参与GO判断；任何其他拼写或numeric值都是artifact/protocol failure。任何跨scope grouping、提高composition cap、Hamming/BCH/LDPC或把decode flag加入primary score，都会改变target相关性与null，必须注册major protocol、证明非空水印自身occupancy并接受fresh review；不属于v1 ablation开关。

## 31. Detector statistic and FPR calibration

### 31.1 Primary statistic与public admission

v1没有alignment max；唯一primary statistic就是第29节的exact INT64 Q30值

\[
T_K(D)\equiv T_K^{Q30}(D)=RoundDiv\!\left(2^{20}\sum_c RoundDiv(2^{30}\sum_{i\in c}e_i,n_c),\ isqrt(C2^{40})\right).
\]

仍不能用二项/正态tail，因为rule/Post-X/Region erasure、clusters和natural units相关。正式admission先按§27.1检查：UTF-8/Python/protocol、DocumentIdentity、`CanonicalDocumentTokenCountV1∈[4096,65536]`、registered non-identity replay-shaped units≤512、16–512 accepted evidence、≥8 clusters、\(n_{eff}\ge12\)、第27.1节`anchor_ambiguity_fraction≤0.20`。`accepted evidence`在比较expected bit前计算，所以不含match sign。再构造只含public/key-independent covariates的fixed Q15 vector：log2 length/syntactic/accepted/cluster bins各1维、formal-replay-shaped fraction与上述ambiguity fraction各1维、suite-depth bins\([0,1,2,3,4,5+]\) histogram 6维、unit-type mix`Assign/AnnAssign/header/excluded/other` 5维，共17维；count bins和每维Q15 denominator/rounding写入manifest。

strata只在独立`null-admission-fit`上拟合且不看key、region sign或\(T_K\)。固定\(K_{strata}=8\)，用公开deterministic PAM k-medoids和L1距离。将\(n\)个Q15 vectors按17-tuple lexicographic排序；初始medoid \(k=1…8\) 取1-indexed位置\(\min(n,\max(1,\lceil(2k-1)n/16\rceil))\)，位置重复则顺次取尚未使用的最小vector，若distinct vectors<8则protocol unavailable。每轮先把sample分配到最近medoid（距离tie取medoid vector lexicographic最小者）；再枚举全部`(old_medoid,new_nonmedoid)` swaps，计算exact INT64 total L1 cost，选择严格降低cost最大的swap，tie按`(old,new)` vectors lexicographic最小者；无严格改善即收敛，100 rounds仍未收敛则unavailable。最终medoid ID按vector lexicographic重编号0…7。对medoid\(s\)，component envelope是该cluster每维[0.5%,99.5%] empirical order statistics，\(\tau_{OOD,s}\)是cluster内L1 distance的99th-percentile order statistic；quantile index固定\(\min(n,\max(1,\lceil p n\rceil))\)，1-indexed。query先取L1最近medoid（tie最小ID），再要求全部维落envelope且distance≤\(\tau_{OOD,s}\)，否则`UNAVAILABLE_OOD`；通过者的medoid ID就是唯一`stratum_id`。任一stratum若后续无法提供第31.2节样本量则整个protocol NO-GO，不允许看tail后合并或pooled兜底。

generator/repository/task/domain不在API也不在routing中；它们只作hidden worst-group audit。所有17维spec、8 medoids/envelopes/thresholds与admission-fit hash公开并绑定protocol。

### 31.2 Empirical null

对每个supported stratum \(s\)和null construction group
\(g\in\mathcal G=\{natural,random\_side,wrong\_key,edit\_error,sample\_aware\_mosaic\}\)，从independent audit-key distribution与完全分离null documents取得\((K^0_{jg},D^0_{jg})\)。完整执行parser、DocumentIdentity、eligibility、local anchor/ambiguity、Post-X、RegionV1、erasure、duplicates/clusters、guards与stratum routing，形成冻结artifact \(\mathcal S^0_{P,s,g}=\{T_{K_{jg}^0}(D_{jg}^0)\}_{j=1}^{N_{s,g}}\)：

\[
\hat p_{P,s,g}(D,K)=\frac{1+\sum_{j=1}^{N_{s,g}}\mathbf1[T_{K_{jg}^0}(D_{jg}^0)\ge T_K(D)]}{N_{s,g}+1},
\qquad
\boxed{\hat p_{P,s}(D,K)=\max_{g\in\mathcal G}\hat p_{P,s,g}(D,K)}.
\]

取maximum是保守intersection decision：真实input construction group即使不可观察，其正确group p-value也不大于返回值；不能从五组中挑最小p。每个\((s,g)\)初始\(N_{s,g}\ge10,000\)，另有每组≥10,000 held-out null-test；真实sample size按目标order statistic/CI增加。risk-cert/null-test要求每个supported stratum、每个construction/hidden group与总体single-query FPR point≤1%、one-sided simultaneous95% upper≤1.25%；任一失败关闭该stratum或整个protocol。至少128 audit keys检查dispersion；production key均匀生成，qualification不能改public threshold。

`sample_aware_mosaic`按audit key分离donor/target components：攻击构造器可读取该same key的大量合法watermarked donor final codes、全部public artifacts与每个公开observed region，但不能调用detector或读取key；它按重复local skeleton/anchor建立字典，再在**不同DocumentIdentity**的held-out target modules中选择registered realizations并拼接。完整identity-equivalent copy从该组排除并单列provenance non-goal。target preimage inequality做byte assertion，attack output进入与其他组相同的independent tail/test，不允许从donor/target交叉调threshold。另做≥128 audit keys的target-reuse bit-correlation、donor-count曲线与mosaic family-wise FPR；PRF design expectation是correlation 0，但只有held-out上界能支持经验claim。

null corpus覆盖registered长度/结构/evidence strata，并在hidden repository/task/generator/domain及clean/edit/error/mosaic groups做worst-group audit。threshold不使用positive test。detector-side Gate/erasure selection通过current-only Post-X完整重放，不假设独立；embedding-side Pre plan不是证据，只在第24.3节的key-free causal audit中运行且不得condition tail。没有nonce/salt/identity override，null/API也不生成或接受这些字段。

exchangeability target是“registered deployment population内、通过public admission、给定stratum的null artifact与随机key”。这不是任意adversarial code的密码学false-positive guarantee。unseen/OOD与detector-oracle-crafted code只做stress；失败则收缩admission或NO-GO。v1仍不提供逐固定key finite-sample proof。

### 31.3 判决与多重测试

正式family API要求**在任何结果释放前atomic预提交**至多\(Q_{max}=32\)个不同`(canonical_code_hash,protocol_id,key_id)`tests。这里

\[
canonical\_code\_hash=SHA256(CBOR_D([\texttt{"query-code/v2"},protocol\_id,
DocumentIdentityV1(D)])),
\]

所以format/comment/registered-carrier-alternative变化不能把同一normalized artifact伪装成新identity；真正AST/ownership/identifier/literal变化是新test。`key_id`是key registry为一个secret handle一次性分配的128-bit random public identifier并由服务器唯一映射，调用者不能给同一key创建aliases；它不是key material的hash。服务器先提交family commitment，重复三元组去重，然后一次性返回结果。p-values做Holm step-down、family alpha=1%；Holm允许任意test dependence，但不修复根据已看结果选择后续hypothesis的selection bias。跨keys、protocols或documents都计入Q；不允许函数/window/subset扫描。v1不提供online unknown-future或adaptive chosen-code FPR保证：收到已有结果后追加query、超过32或试图跨identity重开family时，formal service返回`UNAVAILABLE_ADAPTIVE_FAMILY`；若仅作red-team stress，输出必须标`NO_FORMAL_FPR_GUARANTEE`。离线library无法强制family bookkeeping，只能输出`SINGLE_QUERY_CALIBRATION_ONLY`；always-valid e-values/alpha-investing需要新的protocol与独立null研究，不能用`0.01/32`一句话冒充解决。

formal比较不把p转float：每个group tail保存`(exceedances+1,N+1)` reduced rational，五组maximum用cross multiplication，Holm排序也用INT128 checked cross products（overflow则big-integer reference；不允许wrap），\(\alpha=1/100\)与\(\alpha/(Q-k+1)\)均按整数交叉乘法判定；tie按canonical test triple。adjusted p作为rational输出。实现间必须在boundary numerators上byte-exact。

held-out null-test原样组成1–32大小atomic families，覆盖同/跨strata和**commit前**预注册的stratum choice；要求family-wise empirical FPR point≤1%、simultaneous one-sided95% upper≤1.25%，且每stratum仍达标。另做中间反馈可见的adaptive code/query stress，但其结果只用于收缩threat model/admission，不转写成v1 FPR证明。报告single raw p、adjusted p、AUROC、TPR、coverage。clean registered population是primary claim；public-region rewrite、dead code与query search是stress，不用于调test threshold。

## 32. Deterministic deployment contract

### 32.1 Protocol manifest

一个`protocol_id`唯一绑定：Python/bytes/parser/CBOR/role table、eligibility/rewrite certificate、OperationalDigestV2 alias graph、UnitSkeleton/k-gram anchor、DocumentIdentityV1、context width、RegionV1、Gate weights与integer evaluator/opset/rounding/fixed shapes、threshold/LUT/bands、AllocatorArithmeticV2/RGE、generator/pool config、HMAC/HKDF domains、`ECC_DISABLED_FORMAL_V1`、cluster/statistic/public strata/OOD/五类null/query-family releases与conformance vectors。registry canonical JSON的SHA-256为ID；任一normative field改变都产生新ID。

manifest release由离线protocol-signing key签名并进入append-only transparency log；detector binary/out-of-band config钉住root public key、最低允许epoch与revocation list。`protocol_id`仍是canonical manifest content hash，签名不进入hash。embedding与extraction启动时都验证signature/log inclusion、rollback/revocation、manifest chain和artifact content hashes。unknown/missing/mixed version、runtime feature不符、conformance mismatch、tokenizer/parser disagreement均fail closed。不能把v1.1 weights与v1.0 thresholds混用，不能接受旧epoch rollback，也不能在artifact不兼容时“容差继续”。registry trust只认证public protocol，不证明provider曾key-neutrally planning。

### 32.2 公共与私有状态

public：protocol manifest、parser/tokenizer/Gate/integer evaluator、RegionV1、DocumentIdentity/OperationalDigest/allocator specs、thresholds/LUT、stratum/OOD medoids、五类null tails、cluster/statistic spec、formal-ECC-disabled flag和可选单行protocol metadata。private：master/`K_unit`、per-anchor expected bits、production-key qualification与generation audits；private audit不改public threshold。没有nonce、salt、identity sidecar或其他sidecar。缺少唯一protocol ID时unavailable；unknown API fields fail closed。

### 32.3 与当前 WFCLLM 的兼容性

当前仓库 official detector rows 包含 `prompt`，现行 generation boundary 只覆盖 `if/for/while` 部分 compound 层，现行 no-quality-gate contract 与本方案 Post-Gate/hard validator相冲突，历史 SAWR 又是 multi-statement window carrier。因此 SAGE-C 不是现有 artifact 的兼容升级：

- 旧 final-code rows 可在外层保留 prompt 作 dataset bookkeeping，但 SAGE-C detector API 必须在调用边界投影为 `final_code + protocol_id`，并用测试证明 prompt byte changes 不改变结果；
- 旧 boundary/anchor/candidate sidecars、semantic LSH weights、calibration thresholds 和 run IDs 全部标 incompatible；
- 需新 protocol namespace、parser conformance suite、artifact registry、null calibration pipeline 和 audit规则；
- 不修改旧 official method 的含义，也不把 diagnostic selector升级成正式 Gate。

迁移成本是中到高：parser/linearizer、data pipeline、two-stage Gate、reference runtime 与 detector calibration均需新实现；可复用的只有 final-code sanitization、run layout、部分 semantic feature 与 audit infrastructure。本文没有实施这些生产变更。

## 33. Security and key-separation analysis

### 33.1 Key schedule 与 dataflow

master\(K\)必须是256-bit uniform secret。固定HKDF-SHA256：

\[
PRK=HKDF\text{-}Extract(salt=protocol\_id,IKM=K),
\]

\[
K_{unit}=HKDF\text{-}Expand(PRK,LP(\texttt{"SAGE-C/K-unit/v3"}),32).
\]

`LP(z)=u32be(len(z))||z`，长度按bytes。第26.1节target HMAC同时绑定完整protocol ID、DocumentIdentity与local anchor；不同protocol/document/key rotation不复用PRF input。public identity/anchor hash不是MAC，不能反推key。master/`K_unit`永不序列化到public schema、model、logs或final code；formal v1不存在`K_msg/K_mask`。

攻击者可以从每个显式non-identity marker计算observed RegionV1；对真正selected carrier，这等于`(protocol,document identity,anchor)` PRF input的一位输出，因此协议必须把它视为chosen-message泄露，而不是声称target不可见。安全假设是HMAC-SHA256作为PRF在这些公开input/output-bit pairs下仍隐藏key，且不同DocumentIdentity的inputs相互独立。same-key donor samples的local anchor字典不再地址相同；第31.2节仍以sample-aware mosaic做经验攻击验证。per-anchor expected-bit sidecar、selected-position map与完整HMAC bytes仍禁止发布。

| 组件 | 可读信息 | 明确禁止 | key-dependent? |
|---|---|---|---:|
| baseline/parser/eligibility/Pre-Gate | frozen \(D_0\)、public structure | key/target/key-selected prefix/pool outcome/generator ID | 否 |
| pool builder | frozen \(D_0\) prefix/current span、public generator config/seeds | key/target、future units、side-specific prompt/early stop | 否 |
| pair/hard/region/Post-G/X scoring | 全planning pools、baseline/tests/public artifacts | target/key、candidate count as feature | 否 |
| document commitment | DocumentIdentity、ordered allocator/skips/pools/scores/config | key/target | 否；完成后才解锁key |
| target derivation | committed protocol/document identity/local anchor、key | code、candidate generation/control、caller identity | 是 |
| selector | precommitted winner_0/winner_1、target | 扫描pool、追加attempt、改温度、重写siblings | 是，仅constant-time二选一 |
| extractor | final code、public artifacts、key | prompt/logs/pool/retry | 是，仅期望 bit |

attempt timeout也产生固定位置sentinel；全document完成planning、actual dual status和two-winner commitment后才解锁target。selector不再扫描A records。服务只公开aggregate detector result与family bookkeeping，不公开per-attempt/side/incident trace。协议依赖honest provider：final code不能密码学证明planning key-neutral；静态taint、separate process/HSM、reproducible audit和可选TEE attestation是控制措施，但attestation/signature绝不计watermark evidence。

### 33.2 Causal/shortcut audits

1. 对同一\(D_0\)/planning RNG改变1,000个random keys，DocumentIdentity、全document Pre/allocator/Post-G/X envelopes、commitment、pool bytes/order/count/seeds/dual-status/latency bins必须byte-identical；只有targets/selected candidates/D1 surface可变，且所有D1重建identity必须等于D0。
2. 用target/key ID/region side probe所有pre-target planning fields、candidate count、retry/timeout、rank和commitment length；硬dataflow/byte invariance必须100%。最终selected source/length/region是授权target output，不纳入独立性判据；它只接受public selection key与side-conditioned质量/长度审计，禁止另加未声明载荷。统计probe按MDE AUROC=0.52做≥90% power并要求one-sided95% AUROC upper≤0.52；未拒绝零假设不作因果证明。
3. gradient-reversal generator/repository heads、style-balanced sampling和 leave-one-generator/repository/task tests共同检查 nuisance shortcut；probe失败即 Gate feasibility NO-GO。
4. actual/opposite target复用同pool；`qboth`是同pool joint label，actual dual-completeness是target前hard condition。它不保证balance/semantics/generalization/FPR，后者独立测试。
5. generator artifact/version 更新必须重跑 Oracle、Gate calibration/leak probes 和 end-to-end null；不能假设 Gate labels跨 generator不变。未重新资格化的 generator不进入正式 embedding protocol。
6. 构造相同local anchor但不同DocumentIdentity的pairs，断言HMAC preimage bytes不同且identity无法由API覆盖；在≥128 keys和不同donor counts下运行sample-aware mosaic，报告bit correlation/FWER UCB。任何identity B能够复用identity A的PRF preimage、或mosaic FPR超第40.3门槛，都是critical System NO-GO。

### 33.3 Prompt boundary 与 detector oracle

盲 detector 无法恢复 prompt/generated boundary，所以 origin label不进入 anchor/Gate/evidence。v1 对所有 def/class headers普遍 rule-exclude，因而满足“已有 top-level header默认不候选”但更保守；其他 prompt-owned eligible units在 embedding 中未改写，在 extraction 中可能贡献随机 match/mismatch/erasure，作为 null-like dilution纳入实际 prompt-length matched calibration。任何声称只检测 generated span 都需要把可信 span写入 final artifact，超出本 protocol。

per-anchor输出形成chosen-query oracle，因此只在atomic family封存后返回aggregate结果；第31节Qmax/FWER只覆盖commit-before-feedback的registered population，rate limit只是执行手段，不能使adaptive selection自动有效。知道public Region/anchor仍不足在未知key下定向match，依赖HMAC/HKDF安全；公开优化可大幅擦除signal。

删除nonce消除了“同一code枚举metadata得到新target”的额外自由度：请求nonce/identity override一律拒绝，重复artifact完全确定。不同DocumentIdentity即使共享local anchors也不共享PRF inputs；完整`IdentityProjectionV1`等价artifact的copy或registered-resurfacing仍无法归属唯一generation event。解决这种完整normalized-copy provenance需外部signature/registry，属于non-goal且不得计watermark evidence。

## 34. Robustness and attack analysis

### 34.1 编辑/攻击结果

| 编辑或攻击 | Anchor/region处理 | 预期失败模式 | 必测指标 |
|---|---|---|---|
| comments/blank lines/formatting | ordinary comments非identity；canonical/anchors/document identity保持 | protocol comment删后API无ID则unavailable；否则Region/Post-X/targets应exact | formatting survival |
| identifier alpha-rename | local anchor与Region/GateProjection通常保持，但full identity保留spelling | document identity改变，全部expected targets重绑定；不承诺检测生存 | projection replay与end-to-end rename availability分别报告 |
| registered carrier alternative/formatter rewrite | BlindCertificate归一回同identity；local anchor保持 | 合法alternative应identity/target exact；normalizer折回identity则formal evidence被擦除 | identity conformance/marker-removal TPR |
| 其他equivalent expression rewrite | local slot anchor可能保持；full identity改变 | 全部targets重绑定，通常降到null/insufficient | rewrite attack TPR/FPR，不设survival门槛 |
| unit deletion | local windows仅影响边界后3个，但full identity改变 | document-wide target reset；local resync只作结构诊断 | untouched-anchor replay与end-to-end availability分开 |
| unit insertion/dead code | 同上；duplicate k-gram全组erasure | document-wide target reset/erase | insertion availability/FPR |
| same-suite reorder | local边界变化，且full identity改变 | document-wide target reset | reorder availability/FPR |
| control-flow rewrite | role path/subtree与full identity改变 | document-wide target reset或invalid | attack TPR/FPR |
| header/body edit | owner local signature与full identity改变 | child anchors变化且全部targets重绑定；syntax坏则invalid | per-clause availability/FPR |
| copy a matching unit | text-identity duplicate cluster | 只有首个有权重 | copy amplification test |
| cross-document mosaic/transplant | target还绑定full DocumentIdentity | 不同identity donor bit不可定向；完整identity-equivalent copy仍非归属证明 | sample-aware mosaic FWER/bit correlation/copy disclosure |
| public Gate/region probing | rejection band、aggregate detector API | 可定向擦除，不能无key定向match | adaptive attack FPR |
| artifact substitution | hash/conformance failure | unavailable，不误判 | fail-closed rate |

local self-synchronization只保证同suite单edit点之前以及编辑后第4个未改原unit起的**structural windows**不变；DocumentIdentity security binding有意使任意AST-content编辑全局重绑定targets。无界编辑可清除水印。只有format/comments/registered candidate alternatives属于identity-preserving channel；rename/rewrite/insertion survival仍测量，但失败不违反v1安全合同，只表明availability有限。

### 34.2 核心理论问题的明确结论

| 问题 | 结论 |
|---|---|
| Gate提高容量还是只筛掉？ | 固定全位置无限预算时Gate不能创造carrier entropy，且可能降低绝对尝试数；只有在固定**全document search/verification RGE**下，Gate排序优于source/rule baseline时才可能提高有效证据/成本。有限catalog的静态side census很便宜，所以学习Gate必须额外战胜规则Gate，主要潜在价值只在昂贵dynamic/quality失败预测；否则删除学习Gate。 |
| Gate能预测SHA region reach吗？ | 对单个anchor的RegionV1 sides由公开SHA-256精确确定，masked小模型既不应也基本不能学习其preimage；Pre的可学信号只能是population-level catalog/type/size与昂贵dynamic/quality survival/cost。正式Pre仍在actual pool生成前运行，不读取raw sides。作为强规则baseline，可先廉价枚举catalog/pool variant/raw sides再决定是否投入compile/tests；learned policy必须在相同RGE下战胜这个baseline。若survival也由template规则完全决定，learned Pre/Post没有可识别增益，Layer-2采用规则Gate。任何把raw region digest、candidate sides或pool count送入Pre tensor的做法都视为shortcut。 |
| 双侧 \(q_0/q_1\) 是否足够？ | 不足；还必须在每个实际pool上target前硬检查`has0&&has1`。q只是预算预测，不能替代actual dual completeness。 |
| Gate会否学style/type/generator？ | 会；type可能是真因也可能shortcut。用stratified labels、nuisance probes、counterfactuals和leave-one-group-out证伪，失败则不用学习Gate。 |
| labels是否绑定generator？ | formal candidates由固定catalog产生，但baseline generator改变D0 unit/value/style分布，所以feasibility仍条件于baseline-generator+budget+protocol。任何generator更新必须重做Oracle/校准，不能宣称通用。 |
| Post-Gate是否新神经瓶颈？ | Post-X仍可能造成accept/erasure漂移，所以使用integer-only reference、risk-cert和bands；它不再预测不可识别pairwise semantics，也不能改变RegionV1 bit。 |
| compound header有容量吗？ | 对v1无关：全部永久noncarrier。零自由度与two-phase chronology使其不能靠threshold promotion。 |
| header改写风险？ | 高且与完整D0 planning结构冲突；v1不改header，未来需独立major protocol。 |
| “未排除都候选”是否可接受？ | 不可接受；v1改为正向allowlist，yield/await/calls/control-sensitive表达式禁用或Oracle-only。 |
| 稳定近均衡二分存在吗？ | RegionV1 hash在random-oracle heuristic下平衡，但有限rewrite classes可能单侧；只有static census/Oracle能成立。 |
| key-independent pool是否太大？ | formal catalog最多9 canonical templates，成本有界；若A=9双侧仍低，更多sentinels/更大Gate无效。 |
| natural side不平衡怎么办？ | actual dual-incomplete在target前skip；gap>0.10触发region/carrier redesign。没有target-side fallback或balance联训。 |
| current unit是否含足够语义？ | 对短/低熵units可能不足；以Oracle evidence density判定。context不能替它创造carrier entropy。 |
| context能否进入RegionV1？ | 正式禁止；RegionV1只hash当前`RegionCurrentV2`。context仅进Pre/Post-G。 |
| 短header表示退化怎么办？ | 保持Oracle-only/erasure，不能把context注入bit encoder补救。 |
| Gate selection后的FPR如何可信？ | 不能用二项模型；按public strata完整重放Post-X/Region/erasure/cluster/guard，并做worst-group及family null。 |
| selection是否data-dependent stopping？ | 是；虽已删除keyed alignment max，Post-X/erasure/admission仍data-dependent，因此用stratified empirical tails。 |
| null必须重放什么？ | parser、DocumentIdentity、eligibility、local anchor ambiguity、Post-X、RegionV1、rejection、duplicates、clusters、guards、stratum/OOD、五类construction groups和query-family correction。 |
| ECC编码什么？ | formal v1明确禁用ECC；擦除由tri-state clustered empirical aggregation处理。任何codeword是新major protocol。 |
| 系统类型？ | 统一定义为 **keyed statistical zero-bit watermark**；不是payload watermark。 |
| 能否证明可行？ | 只能条件性证明协议对偶/风险控制方式；容量、语义、重放和power必须由三层实验门槛验证。 |

## 35. Failure handling

| 失败 | 嵌入端动作 | 提取端动作 | 是否继续/泄漏 |
|---|---|---|---|
| rule excluded/oracle-only/zero-entropy | 不开pool，保留baseline | unit erasure，仍在syntactic stream | 继续；无key调用 |
| identity-form final unit | C0只计attempt，不可作为target winner | `BLIND_CERTIFIED_NON_EVIDENCE`→erasure | 防隐藏selected-position噪声 |
| provisional parse不唯一 | 保留baseline | final strict parse决定 | 继续；内部skip不公开 |
| Pre-Gate低置信/高成本 | 不开pool，保留canonical identity baseline | 不知道skip；identity rule-erasure；其他自然non-identity forms照常blind replay | 继续；无隐藏selected-position状态 |
| pool为空/全重复/timeout | 写满A records/sentinels，target前`DUAL_INCOMPLETE_SKIP` | 该slot的identity baseline erasure；其他自然non-identity forms仍是null-like | 不派生target、不重分配 |
| 任一side候选耗尽 | target前dual-incomplete skip | 同上 | 不存在target-side fallback |
| key-neutral signal guard不足 | 不解锁key；返回无protocol metadata的ordinary D0或no artifact | 若误送检测则缺protocol unavailable | 计overall coverage；不按target重试 |
| candidate跨span/破坏结构 | hard reject | final hard verify erasure | 不重写siblings |
| Post-X/RegionV1边界 | candidate非admissible | erasure | 禁强判bit |
| 任一header | 永不生成formal pool | syntactic/context only | v1 permanent noncarrier |
| final strict parse失败 | 不声称嵌入成功 | `INVALID_DOCUMENT` | 全文无formal evidence |
| surrogate或DocumentIdentity不可重建 | target前停止/no artifact | `UNAVAILABLE_DOCUMENT_IDENTITY` | 不降级local-only、不接受metadata override |
| duplicate local k-gram | 相关units不分配formal pool | ambiguous units/subtree erasure | 不用key破tie |
| target解锁后的blind/pair/target assertion失败 | withhold artifact、立即停用protocol/key并登记`POST_TARGET_SECURITY_INCIDENT` | 没有formal input | 不fallback/retry/replan；一次即Layer-3 Hard失败，因为availability可能泄漏 |
| artifact/runtime/hash不兼容 | 停止或输出无水印baseline | `UNAVAILABLE_DOCUMENT` | fail closed，不近似回退 |
| key缺失/错误 | 无正式target/停止 | unavailable；错误key应呈null | 不猜key |
| accepted evidence不足 | generation完成 | `INSUFFICIENT_EVIDENCE` | 不是negative/positive |
| project tests不可运行 | dynamic label unknown；certificate层另报 | Post-X不声称pair semantics | 计入功能风险/coverage |
| code formatting | 不适用 | 重建相同identity/anchors/targets；不同时fail/attack metric | exact conformance required |
| rename/AST edit/insertion/deletion | 不适用 | local anchors可诊断resync，但DocumentIdentity改变并全局重绑定target | availability/FPR实测；不宣称局部match survival |
| input nonce/salt/identity/subset/unknown API field | 无 | `UNAVAILABLE_REQUEST` | 防metadata/identity/query重采样 |
| short module/evidence不足 | generation可完成 | `INSUFFICIENT_EVIDENCE` | 必须计overall coverage |
| adversary擦除证据 | 无恢复承诺 | erasure/insufficient | availability failure，不虚报positive |

所有生成时 failure records 都是可选 private audit；丢失它们不影响 extraction。系统从不把“曾经成功选择”当证据，也不因候选耗尽把一个 bit-conditioned sidecar写进 final artifact。

## 36. Training curriculum

训练按不可跳级的 stage gates 执行：

1. **Parser/contract qualification。** 两个独立实现验证bytes/Unicode/surrogate fail-closed/semicolon/inline/multiline/async/header/decorator/match/exception、Canonical/Region/Gate projections、skeleton/k-gram anchor与DocumentIdentity；所有gold exact 100%，否则停止。
2. **Rewrite/Region static census。** 不训练模型；验证Rewrite/BlindCertificate、OperationalDigestV2 `DEF/REF` alias graph、template inversion、non-identity canonical classes、scope-composition combinations、candidate-invariant DocumentIdentity与RegionV1 sides。若zero-entropy/density上界失败即NO-GO。
3. **Oracle feasibility。** 运行formal\(A=2/4/6/9\)与隔离oracle-only larger budgets。Layer-1 NO-GO时不生成正式Gate dataset、不训练Gate。
4. **Dataset/materialization。** 按 split-before-expand 对每anchor枚举一个base catalog并产生共享它的 \(R=4\) deterministic pool projections、counterfactual labels、paired views、hard negatives；冻结 schema/data-card/clone audit。
5. **Post-X observable identifiability gate。** 在model-validation按14.4节exact tensor bytes计算collision mass、Bayes error floor与exact-knapsack `C_oracle@0.2%`及component bootstrap；known-label coverage<70%、ceiling point<50%或lower<40%时，learned Post-X路线立即NO-GO并只能另开rule-only major protocol，不能开始Stage-J。
6. **Stage-J joint representation。** 五个public seeds在train上用\(V^{oracle}\)辅助目标训练shared backbone、PRE/POST-G/POST-X；current-only/simple loss、CTX-1/2/3和rule/no-Gate同时保留为baselines，不在test选结构。
7. **Hard-negative mining。** 最多三轮，只从train false accepts选样，保留原prevalence权重；每轮用model-validation评估，test不参与。G只读pair+冻结D0 context；X始终current-only，pair truth不监督X。
8. **QAT/integer curriculum。** FP32稳定后加入IntGateV1 fake quant、batch companions/padding，周期执行真实integer reference；任何conformance mismatch的seed无资格进入选择。
9. **Post map/select/freeze。** 按第40.2节锁定的lower-median rank从有效public seeds中选一个artifact；仅用Post calibration-map/threshold-selection冻结backbone、POST-G/X LUT与thresholds，不看Pre partitions/risk/test。
10. **Stage-P Pre fitting。** 在train用冻结Post pipeline重算\(V^{dep}\)，只更新Pre head；随后只用Pre calibration-map/threshold-selection冻结Pre LUT/bins/thresholds。feature/backbone/Post不得回开。
11. **Joint risk certification。** 用joint risk-cert一次认证Pre/Post所有selected thresholds、hidden groups和alpha ledger；失败不可回调模型。另用null-admission-fit与null-tail releases冻结strata/tails。
12. **System evaluation。** 一次性运行held-out repository/task/generator、null-test、sample-aware mosaic、HumanEval/MBPP、edit/attack、latency/cost矩阵；按第40节裁决。

训练recipe本身是有限且可重放的。Stage-J使用FP32 master weights的AdamW（`betas=(0.9,0.999), eps=1e-8, grad_norm_clip=1.0`），每batch含32个anchor groups及每组全部CTX-1/2/3；Post任务每anchor至多按固定public sampler取8个records并用inverse sampling weight恢复目标prevalence。只枚举`lr∈{1e-4,3e-4,1e-3}`、`weight_decay∈{0,1e-4}`，以及23.6节base recipe或“一次只改变一个loss group”的乘数`{0,0.5,2}`；最多80 epochs，epoch 41开始fake-integer QAT，epoch 50后按constraint-first model-validation utility patience 10 early-stop，完全相同utility时取`(architecture,lr,weight_decay,loss-group,multiplier,epoch)`字典序最小tuple。Stage-P只更新Pre head，AdamW参数相同，`lr∈{1e-4,3e-4}`、weight decay 0、最多30 epochs、patience 5。定义`protocol_candidate_hash=SHA256(CBOR_D(all normative protocol fields except learned weights, fitted LUT/thresholds/null tails and their descendant hashes))`；五个public seeds为`low64(SHA256(CBOR_D(["train-seed/v1",protocol_candidate_hash,u8(i)])))`,`i=0…4`。有效seed少于3即失败；其余按第40.2节的exact lower-median rank选择，不存在偶数seed时取平均或另挑较好seed。任何另加optimizer、epoch、sampler或组合搜索都产生新training-recipe hash并重做calibration/risk-cert。

同anchor paired views不跨split。header heads仅diagnostic，永不v1 formal。privileged labels只能进对应G loss；export graph按17.12 deny-by-default检查。由于formal catalog/Region可被廉价公开枚举，learned Pre的主要可检验价值只剩预测昂贵dynamic/quality survival；若规则Gate在同RGE下不劣，Layer-2明确删除learned Pre，而不是用更多hyperparameter search制造增益。

## 37. Evaluation metrics

所有比例同时报告 numerator/denominator、micro/macro、repository/task cluster-bootstrap 95% CI，以及 conservative simple 与各 header class分层。下面是锁定的 metric dictionary：

| # | 指标 | 严格定义/主分母 |
|---:|---|---|
| 1 | Oracle双侧可达率 | 同一个实际A-pool `Y0rY1r=1` / all Oracle pools；另按anchor聚合 |
| 2 | 单元候选覆盖率 | ≥2 canonical-distinct certified **non-identity** forms的generated FORMAL_ELIGIBLE anchors / syntactically shaped generated anchors |
| 3 | region 0成功率 | ≥1 region-0 non-identity formal-valid record的pool / all fixed pools |
| 4 | region 1成功率 | 同上region 1 |
| 5 | min-side success rate | `min(metric 3, metric 4)`，另报每anchor四个formal variants的`min(successes_0/R,successes_1/R)`有限总体比例；仅另行随机抽样的oracle-only pools可标Jeffreys posterior |
| 6 | Pre-Gate AUROC/AUPRC | 对fixed-budget region-specific与both-side hard labels；AUPRC带prevalence |
| 7 | Pre-Gate calibration error | ECE-15、Brier、NLL、worst-group ECE及LCB coverage |
| 8 | Post-Gate false accept rate | Post-X在final-replay intrinsic negatives上ACCEPT / known negatives；另报Post-G pair-invalid acceptance |
| 9 | Post-Gate false reject rate | non-identity replay-shaped Post-X known intrinsic positives未ACCEPT / positives，含erasure；identity rule erasure另报100% |
| 10 | 有效嵌入率 | target-selected non-identity winners / all generated-span FORMAL_ELIGIBLE(D0)；另报/pre-selected、attempted、dual-complete/composition-certified |
| 11 | 每成功单元平均attempt数 | selected candidate record index+1；每pool始终生成A records，另报RGE |
| 12 | 每千token有效证据数 | final blind non-erasure evidence / `CanonicalDocumentTokenCountV1(final_code)` ×1000；generator BPE/candidate tokens只进成本指标 |
| 13 | batch-composition flip rate | 与batch-1 reference离散state不同的sample-env pairs / pairs |
| 14 | quantization flip rate | FP32 training graph与exported integer state不同 / compared samples |
| 15 | cross-process replay rate | 两clean reference processes state和integer outputs exact相同 / samples |
| 16 | cross-device replay rate | qualified device与reference同state / samples；未资格化单列 |
| 17 | formatting survival rate | format后DocumentIdentity、anchor、RegionV1、Post-X/evidence state与expected target全部保持 / pre-edit formal evidence |
| 18 | rename survival rate | 分开报current projection/anchor/Post-X exact率，以及DocumentIdentity改变后end-to-end matches / pre-edit matches；后者是availability attack metric，无v1成功门槛且不得称同target survival |
| 19 | statement insertion/deletion survival | local structural metric：编辑边界后第4个未改original unit起exact anchor / 理论不受local-window影响units；另报identity reset后的end-to-end attack TPR，不把anchor survival等同evidence survival |
| 20 | local resynchronization rate | edited pairs中理论unaffected local anchors exact保持 / all theoretical unaffected；changed anchors不算“恢复”，DocumentIdentity changed单独为false |
| 21 | 检测AUROC | watermarked vs stratum-matched null的locked \(T_K\)，insufficient另报coverage |
| 22 | 固定FPR下TPR | TPR@1% empirical FPR，报sufficient-conditional与overall |
| 23 | 经验FPR | 五类null（含same-key sample-aware mosaic）的single-query per stratum/worst-group及Q≤32 family-wise positives，simultaneous CP 95% UCB |
| 24 | 平均erasure rate | erasure / all non-identity replay-shaped slots；identity/excluded rule erasures另以all-syntactic denominator报告 |
| 25 | mismatch rate | mismatches / non-erasure formal evidence；同时/eligible denominator |
| 26 | 有效样本量 | 第29节 \(n_{eff}\)，报median/quantiles和guard coverage |
| 27 | ECC解码成功率 | formal v1固定输出`ECC_DISABLED_FORMAL_V1`，无分子/分母且不参与GO；其他拼写或numeric值即artifact/protocol failure |
| 28 | 代码编译/解析成功率 | final strict Python 3.12 parse successes / generated documents |
| 29 | 功能正确率/pass@k变化 | 与同seed、同budget no-watermark baseline的paired pass rate difference/CI |
| 30 | 代码质量变化 | blinded human rubric、lint/complexity delta与semantic-test regressions，分开报告 |
| 31 | 生成延迟 | end-to-end wall time p50/p95与相对baseline增幅 |
| 32 | Gate推理成本 | Pre/Post CPU ms、memory、energy proxy、calls/unit |
| 33 | 总retry成本 | candidate tokens、attempts、timeouts / successful document和formal evidence |
| 34 | Key/candidate leakage probe | planning byte invariance exact rate、target-probe AUROC one-sided UCB与pre-registered power |
| 35 | Evidence sufficiency coverage | 满足length 4,096–65,536、16 evidence、8 clusters、\(n_{eff}\ge12\) / all target-population documents |
| 36 | Anchor uniqueness/collision | unique local k-gram anchors / formal-shaped units；ambiguous units必须erasure，zero-edit replay 100% |
| 37 | Scope composition certification | target前all-\(2^m\) recursive-digest pass scopes / allocated scopes；任何fail均key-neutral skip，post-target incident必须为0 |
| 38 | Plan signal-guard coverage | target前满足16 slots/8 clusters/\(n_{eff}\ge12\)的documents / all generated D0；失败仍计overall系统coverage |
| 39 | Post-X observable collision | exact 1,216-byte tensor cells的collision mass与`Σ_h min(n_h+,n_h-)/N` Bayes error floor |
| 40 | Post-X information ceiling | 14.4节exact-knapsack `C_oracle@0.2%` point与repository-component bootstrap lower |
| 41 | Post-X known-label coverage | quality/format/rename三项全known rows / all registered non-identity replay-shaped rows，按stratum/missingness报 |
| 42 | DocumentIdentity conformance | D0与全部registered single/all-\(2^m\) candidate combinations的identity byte-exact率；不同identity HMAC preimages必须100%不同 |
| 43 | sample-aware mosaic resistance | donor-count×audit-key下bit correlation、single/FWER positive率及simultaneous UCB；targets必须是different-identity modules |
| 44 | allocator arithmetic conformance | 两实现对boundary utilities、order、commitment byte-exact / vectors；任一overflow/wrap为0容忍Hard failure |
| 45 | alias-graph conformance | 两fresh-process serializers对OperationalDigestV2/DEF-REF golds exact / cases；identity-sensitive counterexample漏检为Hard failure |

`code quality` 不压成单一主观分数；功能、lint/complexity和blind human preferences分别报告。任何 unknown/timeout denominator处理在 evaluation plan预注册。

## 38. Baselines

所有baselines复用same splits、RegionV1、catalog pools、每code-object最多4个allocated units、ScopeCompositionCertificate、RGE accounting和各自锁定null calibration。主比较使用相同`B_doc`；另外报告unconstrained all-unit diagnostic，但后者明确nonformal且不能用于Gate增益主张。

| # | Baseline | 变化 |
|---:|---|---|
| 1 | 无Gate | 在同B_doc下按source-order+anchor tie选择FORMAL_ELIGIBLE units，每个A records |
| 2 | 规则Gate | 同一总B_doc；按allowlist、static entropy/length及key-free catalog/pool-variant raw-side enumeration评分，不运行learned Pre/Post；对所有units的cheap enumeration RGE也从B_doc扣除 |
| 3 | current-unit-only Gate | shared Gate不读predecessors |
| 4 | CTX-1 Gate | 注册width=1 |
| 5 | CTX-2 Gate | 注册width=2 |
| 6 | CTX-3 Gate | 注册width=3 |
| 7 | 只用Pre-Gate | learned allocator + hard/certificate verifier，无Post-X |
| 8 | 只用Post-Gate | 同B_doc source-order allocator，Post-X过滤，无learned Pre |
| 9 | 不量化 | FP32 training graph；只作research/nonformal |
| 10 | INT8/QAT | IntGateV1 QAT+integer reference（用户要求的INT8/QAT对照） |
| 11 | 无rejection band | argmax/单阈值强制state，用来测flip/FAR恶化 |
| 12 | 单总体成功率 | 一个q预测actual/overall success，不输出counterfactual q0/q1 |
| 13 | 简单loss | 第23.6节 \(L_{base}\) |

baseline 11/12若违反安全规则只作为诊断，不可成为部署候选。Gate增益主比较是推荐设计 vs no-Gate、规则Gate和Post-only在相同global budget下的formal matches、mismatches、evidence density、功能与FPR。

## 39. Ablation plan

每个 ablation只改一项，复用 paired pool和locked validation protocol；最终test上的ablation表预先固定，不根据结果增删：

| # | Ablation | 要检验的因果问题 |
|---:|---|---|
| 1 | 删除结构路径特征 | 结构是否提供真实跨repo可行性而非style |
| 2 | 删除unit type | type head是否只是shortcut或必要risk separator |
| 3 | 删除header候选 | formal v1本来即删除；仅与oracle-only历史方向作诊断，不是formal ablation |
| 4 | 删除分号拆分 | **仅parser错误对照**，量化边界/anchor污染，不可部署 |
| 5 | 删除context consistency loss | paired predictions稳定性和context增益 |
| 6 | 删除batch consistency loss | batch flip与margin变化 |
| 7 | 删除quantization-aware loss | integer export flip/FAR/coverage变化；不影响integer semantics定义 |
| 8 | 删除calibration loss | ECE、LCB selection、FAR变化 |
| 9 | 删除ranking loss | pool内first-choice validity和cost |
| 10 | 删除robustness label/head | edit survival与clean FAR trade-off |
| 11 | 改attempt budget | formal \(A=2,4,6,9\)；16/32/64仅oracle-only stress |
| 12 | 改candidate pool unique size | 固定A下调dedup/采样diversity，区别数量与预算 |
| 13 | 改region rejection interval | \(\Delta_R\)的balance/coverage/dual-reach trade-off；每项新protocol |
| 14 | 改Gate erasure/rejection threshold | FAR、FRR、evidence density、flip |
| 15 | 改ECC强度 | formal v1固定`none`；Hamming/BCH/LDPC只能作为隔离future-major-protocol study，必须先证明非空水印自身occupancy并完整重校准/复审，不能与v1 winner同表择优 |
| 16 | shared heads vs unit-specific heads vs split models | generalization、稀疏class FAR、size/latency |
| 17 | global source-order vs same-suite context | 跨suite邻接是否带来可泛化信息，还是只增加结构噪声；formal winner只在validation选一次 |

附加安全诊断包括移除nuisance adversary、允许target-conditioned early stop、把context送入RegionV1 preimage；后两项是预期失败的red-team controls，绝不进入formal候选。效应报告paired repository bootstrap CI与multiplicity-corrected p-values；“无显著差异”不自动证明组件无用，还要看hard constraints。

## 40. Go/no-go criteria

所有数值在看held-out test之前冻结。`Hard`不能降低；`Initial`是当前工程建议，若失败只能以新protocol、新calibration和新test重启，不能在同一次test后调低。`Required report`表示scope明确不承诺该attack的availability数值：必须运行并公开完整分母/CI，遗漏报告是Hard process failure，但低survival本身不构成v1 clean-system失败，也不得被包装成鲁棒成功。

### 40.1 Layer 1 — Oracle feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| Rewrite/BlindCertificate/static census | 两实现枚举、template inversion、non-identity forms、OperationalDigestV2 alias graphs与CompositionalityLemma property tests 100%一致；identity永不计side，zero-entropy share不使density上界<4/1000 | Hard |
| DocumentIdentity candidate invariance | D0、所有registered single候选与每scope全部\(2^m\)组合100% byte-exact；surrogate/歧义一致fail closed | Hard |
| simple formal \(A=9\) macro min-side | ≥0.35且95% CI lower≥0.30 | Initial |
| same-pool dual-complete anchors | ≥50%且95% CI lower≥45% | Initial |
| region side absolute gap | ≤0.10 | Initial |
| label-oracle evidence density | ≥4/1000 generated tokens | Initial |
| RegionV1/zero-edit anchor replay | gold与两实现100% byte-exact | Hard |
| formatting region replay | ≥99.9% | Initial |
| paired functionality delta | ≥-2 percentage points | Initial |

任一核心条件失败：停止Gate training，重新设计RegionV1/carrier，Layer-1 NO-GO。headers不进入formal补分母，也没有promotion豁免。

### 40.2 Layer 2 — Gate feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| unseen repo/task/generator Pre calibration | ECE≤0.05，LCB coverage≥95% | Initial |
| Post-X deployable-input identifiability | \(N^+\ge1000,N^-\ge5000\)、≥20 repos、单repo≤10%；known-label coverage≥70%；`C_oracle@0.2%` point≥50%、component-bootstrap95% lower≥40%；否则删除learned X并判当前路线NO-GO | Hard |
| Post-X final-stream intrinsic FAR | point≤0.2%，simultaneous95% upper≤0.5%，每registered stratum | Hard |
| Post-X accepted-positive coverage | point≥50%，component-bootstrap95% lower≥40%，每核心stratum另报 | Initial |
| Post-G pair-invalid acceptance after hard verifier | 0；hard verifier不得被G覆盖 | Hard |
| direct key/target pool invariance | bytes/order/count/Gate logits 100% identical | Hard |
| target probe | one-sided95% AUROC upper≤0.52，MDE 0.52 power≥90%；并有100% byte/dataflow invariance | Hard audit |
| AllocatorArithmeticV2 | INT64 boundary utility/order/commitment两实现100% exact；overflow/wrap事件=0 | Hard |
| same-global-budget evidence/1000 | ≥no-Gate ×1.15 | Initial |
| absolute formal evidence | ≥no-Gate的95% | Initial |
| search cost per formal evidence | RGE/formal evidence与actual verifier-ms/evidence均≤no-Gate ×0.80；同B_doc total reservation不得偷减 | Initial |
| Gate latency/artifact | ≤5 ms/unit，≤8 MiB | Initial |
| leave-one-generator/repo degradation | evidence与AUPRC相对held-in下降≤10%且仍校准 | Initial |

若learned Gate不能在unseen groups校准，或增益不足以覆盖erasure/latency，就不采用学习Gate；保留规则/no-Gate科研baseline，不能宣称Layer-2成功。双侧AUROC高但FAR或budget utility失败仍是NO-GO。

训练run protocol固定5个public optimization seeds。令通过export/conformance且无NaN/overflow的有效seed数为\(n_v\)；若\(n_v<3\)即training failure。只用model-validation把有效seeds按`(U(π) ascending, seed_integer ascending)`稳定排序，选择1-indexed rank \(k=\lfloor(n_v+1)/2\rfloor\) 的**lower-median** seed（\(n_v=3,4,5\)时分别为rank 2,2,3），绝不取两个中位值平均、upper median或test-best。Post/Pre calibration只对该预选artifact运行。Layer-2 `TRAINING SUCCESS`要求这一个artifact通过表中全部Hard与Initial条件；任一risk-cert、unseen-group、same-budget或integer export条件失败即`TRAINING FAILED/LEARNED GATE NO-GO`，不能换seed、看test、扩大模型或降低threshold救回。其余seeds只报告stability，不接触held-out test。

### 40.3 Layer 3 — System feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| held-out null FPR | 五类null（含same-key sample-aware mosaic）每public stratum、hidden worst-group、single-query及Q≤32 family-wise均point≤1%、simultaneous95% upper≤1.25% | Hard |
| TPR@1% family FPR | sufficient supported modules≥80%；target-population end-to-end coverage（plan guard通过且detector sufficient）≥70% | Initial |
| effective embedding/evidence density | target-selected/all eligible≥20%；formal evidence≥4/1000 tokens | Initial |
| erasure/mismatch | erasure≤40%；mismatch/non-erasure≤15% | Initial |
| normative replay | Canonical/DocumentIdentity/anchor/OperationalDigestV2 alias graph/RegionV1/all integer tensors/Post-X/evidence 100%，cross-process byte-exact | Hard |
| formal alternate backend | all conformance/fuzz tensors exact、0 flips；否则nonformal | Hard contract |
| final parse/function | parse≥99.5%；pass/function delta≥-2pp | Initial |
| post-target release assertions | `POST_TARGET_SECURITY_INCIDENT` count=0 across qualification/system test；production任一事件立即suspend | Hard |
| registered normalization conformance | formatter/comment与所有registered carrier alternatives的DocumentIdentity/anchor/Region/Gate/expected-target replay=100%；identity-form仍rule-erasure | Hard |
| broader edit analysis | 一般rename、insert/delete/reorder分别报告local-anchor resync、DocumentIdentity reset和attack TPR/FPR；v1不设end-to-end survival门槛，不得以此补clean TPR | Required report |
| anti-transplant binding | different-identity local-anchor pairs的HMAC preimages 100%不同；sample-aware mosaic FWER满足held-out null FPR门槛 | Hard |
| ECC disabled | manifest/schema/output均为`ECC_DISABLED_FORMAL_V1`且不存在block/key/message字段 | Hard contract |
| cost | same-B_doc RGE accounting exact；candidate tokens≤unwatermarked 2×；end-to-end wall time≤unwatermarked 2×；Gate≤5ms/unit | Initial |

任何Hard失败直接System NO-GO。Initial核心条件（TPR/coverage、density、function、erasure、cost）任一CI未达标也不得宣称系统成功；必须注册新版本后重测。通过三层只支持特定Python/version/generator/threat model，不外推任意代码或硬件。

## 41. Known limitations

### 41.1 证据状态

| 陈述 | 状态 | 可声称范围/关闭证据 |
|---|---|---|
| 一个current-unit-only、DocumentIdentity-bound target协议可定义 | design construction | 只说明规范自洽；需parser/identity fixtures与mosaic test证明实现/经验安全 |
| fixed pool先于target可阻止直接key-conditioned retry | design assumption + dataflow requirement | 需代码级taint/invariance审计，不仅统计probe |
| conservative simple units存在双侧region容量 | **hypothesis** | 第19/40.1节formal Oracle结果 |
| compound headers可作为v1载体 | **明确false / scope exclusion** | two-phase chronology未解决；新major protocol才可重开 |
| Gate在unseen generator/repo提高evidence-per-cost | **hypothesis** | 同global budget paired baselines与Layer-2 criteria |
| Post-X intrinsic FAR≤0.5% upper | **hypothesis** | final-stream strata risk-cert/test；不含pair semantics |
| QAT提升训练后integer margin | **hypothesis** | no-QAT/QAT矩阵；determinism来自integer evaluator |
| normative backend离散重放 | required experiment | 10k vectors+boundary fuzz+independent process exact tensors |
| per-stratum/family-wise FPR≤1% | required experiment | ≥10k/stratum null tails与independent family null-test |
| formatting survival达标 | required experiment | locked identity-preserving edit suite；未运行 |
| rename/insertion后的检测可保持 | **不作v1保证** | DocumentIdentity有意重绑定全部targets；只报告availability attacks |
| 系统检测power、功能与成本达标 | required experiment | Layer-3完整评估；未运行 |

### 41.2 不可消除或不可由更多Gate数据单独修复的限制

1. 一般Python语义等价不可由局部Gate或final-only extractor证明。v1的BlindCertificate只证明final unit属于固定CPython、closed-pure constant rewrite class，并明确排除trace/source/code-object introspection；generation pair/dynamic tests是额外release证据，Post-X不作semantic proxy。
2. current-unit-only region在短、低熵header/statement上可能根本没有双侧自由度。若Oracle失败，增加Gate数据或模型大小无效。
3. blind final code不含prompt boundary；无法精确剔除prompt-owned simple units，只能普遍规则排除某些classes并在null中吸收随机证据。
4. k-gram self-synchronization只恢复local structural addresses，不恢复旧slots或keyed evidence；任何identifier/literal/AST/ownership edit都会改变DocumentIdentity并全局重绑定targets。该安全选择阻断cross-document mosaic，却牺牲rename/insert/delete/reorder availability，不能由更多Gate数据修复。
5. integer semantics提升可移植性，但formal仍要求reference evaluator及exact qualification；任意客户端/GPU未经资格化不满足。
6. public region/Gate允许攻击者优化擦除；无key只能使forge概率受经验FPR约束，不能保证抗重写availability。
7. formal evidence必然是`v+0`、`v*1`、double-negation等公开registered non-identity marker。攻击者不需要key即可把所有这类RHS canonical-fold回identity，令证据全部erasure；v1对这种语义保持normalizer没有availability保证。这是载体语言的结构性弱点，不可由Gate训练修复。
8. 显式marker也可能罕见、可读性差或容易被lint/formatter识别。identity为避免隐藏selected-position噪声又必须non-evidence，因此positive power依赖足够多可见marker；marker-matched `random_side` null、public admission和Post-X质量筛选必须共同覆盖这种distribution shift。若Oracle/Post-X/held-out null不能同时达标，路线NO-GO，不能用pooled natural null掩盖。
9. formal v1没有ECC：`m_scope≤4`与旧Hamming最少5 observations结构性冲突。tri-state clustered empirical aggregation只处理擦除/相关性，不纠正payload。若业务硬要求可解码ECC，本v1为NO-GO，必须设计跨scope major protocol。
10. Gate labels条件于candidate generator和attempt budget；model/generalization不是永久属性，generator更新需重新资格化。
11. license allowlist、依赖可执行性、private tests缺失和real-repo sandbox限制label规模/代表性。
12. 8个conditional strata×5个null construction groups的tails/test各需要至少一万、通常更多独立`(document,key)`pairs，最低即约800,000次tail+test检测，尚不含hidden groups/donor-count mosaic曲线；不能保证任意固定key或无限adaptive查询的有限样本FPR。
13. 当前只规范Python 3.12 source，不覆盖notebook cell order、generated AST bytecode、C extensions或跨文件semantic rewrites。
14. 没有document nonce；不同DocumentIdentity不共享PRF inputs，但完整`IdentityProjectionV1`等价artifact（包括改换registered surface realization）仍共享targets且不可归属唯一generation event。外部signature/registry是另一个provenance机制，不能计watermark evidence。
15. `OperationalDigestV2`只在固定fresh CPython 3.12正常执行观察下保留constant alias topology；source/trace/position introspection、外部interpreter预热、C-extension地址与跨进程absolute `id()`仍被排除。若业务需要这些观察，本certificate不足。
16. 全document commitment要求buffer baseline/pools/two-side winners，不能直接发布online stream；增加内存与首个可发布artifact延迟。尽管CompositionalityLemma应在解锁前排除正常post-target失败，任何实现incident仍可能通过“无artifact”泄漏availability，因此一次事件即停用protocol/key，而不是可恢复fallback。
17. 正式支持对象是≥4,096 canonical tokens的单Python module；常见短函数/HumanEval/MBPP预计多数insufficient。若业务目标是函数级检测，本v1在scope上NO-GO。

这些限制不能用“模型更大”关闭；第1、3–9、13–17属于scope/合同或载体限制，第2属于可能的根本容量瓶颈。

## 42. Implementation roadmap

本任务不改生产代码。后续实现应在独立research protocol namespace中分八个可停止work packages（WP0–WP7）：

| WP | 产物 | 必须验证后才进入下一步 |
|---:|---|---|
| 0 | `protocol.json`、exhaustive grammar/CBOR/role tables、≥1,000 gold vectors | 两实现span/CBOR/surrogate-state/skeleton/anchor/DocumentIdentity/RegionV1 100% exact |
| 1 | parser/linearizer/k-gram anchor/DocumentIdentity library与property tests | edge grammar/Unicode/CRLF/fuzz零未分类；single-edit local impact与identity reset均exact |
| 2 | RewriteCertificateV1、OperationalDigestV2 alias serializer、static census、Oracle pilot | 两fresh-process alias/identity conformance与低成本carrier-entropy falsification；失败停止 |
| 3 | provenance/split graph、10 schemas+final-replay schema、formal Oracle(A≤9) | Layer-1 thresholds/CI与clone audit |
| 4 | Post-X tensor-collision ceiling、shared PRE/POST-G/POST-X、INT64 allocator、baselines/ablations、integer QAT export | identifiability gate、Layer-2、deny-by-default feature audit、final-stream FAR |
| 5 | integer reference evaluator/registry/conformance vectors | ≥10k exact tensors+boundary fuzz+fail-closed matrix |
| 6 | embedding/extraction、ECC-disabled tri-state clusters、stratified statistic/query-family null | ≥10k per supported stratum×5 null groups for tails and again for independent tests、sample-aware mosaic、Layer-3/attacks |
| 7 | external benchmark/data/model cards和安全release review | 三层通过后才讨论production integration |

最低成本下一项不是训练Gate，而是WP0+static census/pilot：从≥10个license-cleared、互不clone repositories抽每个ClosedValue family 300 anchors，加入≥100 parser stress snippets（含surrogates）与≥100 alias-topology cases；运行catalog exact enumeration、OperationalDigestV2、DocumentIdentity candidate-combination conformance、RegionV1和AllocatorArithmeticV2 boundary golds，\(A=6,9,R=2\)。它只能falsify，不能给Layer-1 GO。若parser/identity/alias/allocator/zero-edit exact<100%、format replay<99.9%、pilot min-side<0.20或static density upper<4/1000，立即停止并修协议/判NO-GO。

formal Oracle至少需要20个held-out repositories、3个baseline-generator families、5,000 certified simple anchors，\(R=4,A\in\{2,4,6,9\}\)，满足第19节CI；headers只作diagnostics。规模/成本需另获批准。

当前仓库迁移接口建议（尚未实现）：在 `wfcllm.lang` 下建立versioned parser adapter，在 `wfcllm.generation` 新建scheme-C current-span pool而不改旧boundary semantics，在 `wfcllm.method` 新建独立preset/artifact contract，在 `wfcllm.detection` 新建final-code-only projection与empirical-null detector，在 `wfcllm.audit` 添加no-key-to-pool taint/conformance检查。旧协议继续原样可复现。

## 43. Reviewer issue-resolution matrix

第一轮原始意见完整保存在`review-stage/ROUND_01_REVIEW_RAW.md`与`review-stage/ROUND_01_ADVERSARIAL_RAW.md`；第二轮原始意见在`review-stage/ROUND_02_REVIEW_RAW.md`；逐项append-only ledger在`review-stage/ISSUE_RESOLUTION_MATRIX.md`。同一continuing reviewer在Round 2明确关闭全部Round-1 critical/major design issues，但以7.1/10、NOT READY新增`SAGE-R2-C01`、`SAGE-R2-M01..M03`与三个minor。

0.3-rc1只把ROUND_02四项critical/major作为targeted closure对象并全部`ACCEPT`：§§8.5/26/31/33以final-reconstructible `DocumentIdentityV1`绑定HMAC target并明确牺牲任意AST编辑生存；§§26/29/30禁用formal v1 ECC并统一为keyed statistical zero-bit watermark；§§16/17/25固定唯一checked signed-INT64 allocator规范并fail closed；§§3/10/16以`OperationalDigestV2`编码constant/code-object的`DEF/REF` alias/identity topology。ROUND_02 minor只作冻结前编辑性澄清，不属于本轮关闭范围，也不授权机制扩展。原reviewer在rc1 targeted review中关闭C01/M02/M03，但因ECC-disabled状态存在三种规范拼写而保持M01为`OPEN`。0.3-rc2的唯一正文变化是把schema、伪代码、报告、指标、manifest与conformance contract全部统一为`ECC_DISABLED_FORMAL_V1`并显式拒绝其他拼写；它不新增或恢复ECC机制。尚需实验在matrix保留，没有把design修复写成实验结果。

## 44. Review score progression

| 阶段 | Reviewer | 分数 | Verdict | Critical | Unresolved major |
|---|---|---:|---|---:|---:|
| 0.1 initial author draft | continuing reviewer | 5.4/10 | NOT READY | 3 | 11 |
| 0.1 independent adversarial check | adversarial verifier | 3.8/10 | NOT READY | 2 | 10 |
| 0.2 round-1 repair / formal Round 2 | continuing reviewer | 7.1/10 | NOT READY | 1 | 3 |
| 0.3-rc1 targeted closure review | continuing reviewer | 7.9/10 | NOT READY | 0 | 1 |
| 0.3-rc2 enum-normalization patch | continuing reviewer pending | N/A | NOT READY（未复审） | N/A | N/A |

## 45. Final reviewer verdict

`REVIEW PASSED: NO`（0.3-rc2 targeted re-review提交前状态）。原reviewer已关闭C01/M02/M03；M01仅因ECC-disabled规范枚举冲突仍未关闭。0.3-rc2完成唯一机械统一，但在原reviewer达到`score>=8.0, READY, critical=0, unresolved major=0`并完成fresh zero-context final review前，不得标记Goal complete。

## 46. Final recommendation

当前机制建议为 **CONDITIONAL GO，仅批准parser/CBOR/anchor conformance、rewrite static census与最小Oracle**。这不是系统成功。只有formal Oracle达到Layer-1全部阈值才批准小型Gate训练。若formal catalog在\(A=9\)仍大量单侧、static/Oracle evidence density<4/1000，或parser/anchor/RegionV1 replay非100%，路线直接NO-GO；不能退回多窗口混合载荷或降低阈值。

未运行实验，所以不能声称：catalog有足够双侧容量、Gate提高evidence/cost、Post-X达标、integer backend已实现/稳定、format或攻击鲁棒、mosaic resistance经验达标、FPR/TPR/功能/成本达标。可以声称的只有：0.3给出可实现/可证伪的final-code-only protocol candidate及明确stop conditions；经验前提仍待验证。
