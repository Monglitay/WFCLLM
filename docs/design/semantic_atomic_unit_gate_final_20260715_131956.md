# 面向 Python 代码语义水印的原子代码单元门控方案（方案 C）

**协议名：** SAGE-C（Semantic Atomic-unit Gated Evidence, Scheme C）  
**文档版本：** 0.1-initial-review  
**语法目标：** Python 3.12  
**研究状态：** 机制设计完成，等待独立审查；没有运行本方案训练或端到端实验  
**水印类型：** keyed statistical zero-bit watermark  
**权威输入：** final Python code、固定 public artifacts/parameters、private key  

> 证据纪律：本文中的仓库数字只描述历史 WFCLLM/SAWR；所有 SAGE-C 阈值都是预注册用的 initial engineering thresholds 或 required experiments。没有把未运行的 Oracle、Gate 或 system 实验写成结果。

## 1. Executive conclusion

SAGE-C 是一条**理论上条件成立、工程上可验证、当前尚未实证通过**的路线。方案严格保留一个载体对应一个当前 AtomicWatermarkUnit；最多三个前序单元只供 Gate 判断是否投入固定搜索预算和候选可信度，不进入当前单元的 semantic region 或 target bit。正式v1先无密钥生成并冻结baseline document，再完成全document的Pre-Gate decisions、fixed candidate pools和commitment，之后才解锁密钥；Pre-Gate 同时预测 \(q_0,q_1\)，Post-Gate 和最终 strict replay 决定 formal evidence。提取端不读取 prompt、logits、hidden states、candidate pool、retry trace 或 generation logs。

路线成立需要四个不可绕过的条件：

1. 受限 Python 单元类在 fixed public region 下有足够的 region-0/region-1 双侧可达率；
2. blind Post-Gate 在 unseen repository/task/generator 上有极低 false accept 且可校准；
3. reference deployment contract 内离散 region/gate state 可 100% 重放；
4. 完整 Gate/erasure/resynchronization selection 在独立 null corpus 上仍能控制 FPR 并保有检测 power。

Gate 不能创造底层不存在的容量，也不能证明一般 Python 程序等价。若扩大候选池后大量位置仍单侧可达，或 conservative carrier 过少，第一层 Oracle 直接 NO-GO，停止训练 Gate。当前建议是 **CONDITIONAL GO，仅批准最小 Oracle feasibility study**；不批准大规模训练，不声称水印系统已成功。

相对现有仓库，这不是 `evidence_retry_seed7x3` 的参数升级。现行生成边界器只覆盖 `if/for/while` compound layer，现行 detector row 仍含 `prompt`，现行 no-quality-gate 合同排斥 syntax/quality control fields，旧 SAWR 又以多语句窗口为载体。SAGE-C 需要新 parser contract、新 artifact registry、新 gate policy 和新 empirical-null detector；本任务只设计文档，不修改生产代码。

## 2. Problem definition

给定合法 Python 最终代码 \(D\)、public protocol \(P\)、public model artifacts \(M\) 和 private key \(K\)，定义确定性的原子单元流：

\[
\mathcal U_P(D)=(U_1,\ldots,U_T).
\]

嵌入端在生成当前单元 \(U_t\) 时，可读取 \(U_{t-3:t-1}\) 作为 feasibility context，但正式载荷只由当前单元候选 \(x_t\) 的 public region \(R_P(x_t)\) 与 candidate-independent anchor \(a_t\) 的 keyed target 决定。提取端从 \(D,P,M,K\) 重新得到每个单元的 `match / mismatch / erasure`，再做相关性修正和经验校准检测。

需要同时优化：

- functional/semantic preservation；
- effective evidence density；
- fixed-budget search efficiency；
- blind replay determinism；
- low Post-Gate false accept；
- controlled empirical FPR；
- formatting/rename/local-edit survival；
- bounded latency 与 retry cost。

问题不是“给每行打 bit”，也不是“从 generation trace 证明曾成功嵌入”。formal evidence 的唯一权威来源是 final-code replay。

## 3. Threat model

### 3.1 参与者与秘密

- Provider 持有 \(K\)，运行 generator、Gate 和 embedding protocol。
- Detector 持有同一 \(K\) 与 public artifacts，可不访问 generator。
- 用户/攻击者看到 final code 和全部 public protocol/model artifacts，但看不到 \(K\)。
- public region partition 不保密；安全性来自 anchor-specific PRF target，而非隐藏模型。

### 3.2 允许攻击

- formatting、comment/blank-line change；
- identifier rename；
- expression-level equivalent rewrite；
- unit insertion/deletion；
- 同 suite 内有限 reordering；
- dead-code insertion；
- header/body edit；
- 多次提交、选择最弱片段、公开模型 probing。

### 3.3 不保证的攻击

- 任意控制流重构、整函数重写、语义重新实现；
- 攻击者拥有 oracle detector key；
- 攻击者替换/伪造 public model artifact 却要求 detector 静默接受；
- 超出预注册 local edit channel 的无界 insertion/deletion alignment。

这些攻击不会产生错误 positive；最多使 evidence 变 erasure/mismatch 并导致 `INSUFFICIENT`。若 attack 后仍 positive，必须由 held-out attack evaluation 证明，而非由设计宣称。

### 3.4 安全目标

- key secrecy 与 domain separation；
- candidate pool 及 Gate selection key-independent；
- final artifact 不暴露 target bit、retry 或 key-dependent failure；
- artifact/version mismatch fail closed；
- null corpus 上完整重放所有 data-dependent selection；
- 不把 parser recovery、神经边界输出或 generation audit 当 formal evidence。

## 4. Design goals and non-goals

### 4.1 Goals

1. AST/CST/token-aware 严格定义原子单元，嵌入/提取确定性对偶。
2. 一个 shared small Gate 同时支持 CTX-1/2/3 和 simple/header type。
3. 同 anchor、同 pool 的双侧 counterfactual labels。
4. fixed, key-independent attempt budget 与 deterministic candidate selection。
5. current-unit-only public semantic region；context 不进入 bit 编码。
6. match/mismatch/erasure、ECC 和 empirical FPR 全链路。
7. reference backend、QAT、margin、conformance 的离散重放协议。
8. 明确 Oracle/Gate/System 三层 stop/go。

### 4.2 Non-goals

- 不承载任意用户 payload；ECC 只编码 key-derived check message。
- 不证明一般 Python 程序等价。
- 不让 Gate 使用 private key、target bit、candidate count、attempt index 或 retry outcome。
- 不用原 prompt、generation logits/hidden state、candidate sidecar 或 logs 检测。
- 不自动将所有 syntactic units 视为 carriers。
- 不以 QAT 或固定 seed 代替部署契约。
- 不继承旧多窗口混合载荷或 compound-body fallback。

## 5. Terminology

| 术语 | 严格含义 |
|---|---|
| syntactic unit | strict parser在线性化结构中识别的`SimpleUnit`或`ClauseHeaderUnit`，无论是否可水印；“eligible”只属于policy层 |
| candidate unit | 通过公开 rule policy，可进入 key-independent candidate pool 的 syntactic unit |
| embedded decision | 生成端找到 target-region 合格候选的内部结果；不是 detector 权威事实 |
| formal evidence | 最终代码经 strict replay 后通过 parser、hard verifier、Post-Gate、region margin 与 artifact checks 的单元 |
| erased unit | 结构存在但因边界、低置信、版本、对齐或验证失败归为 erasure 的单元 |
| excluded unit | 规则层永不进入正式候选池的 syntactic unit；仍占 ast ordinal |
| oracle-only class | 只做 feasibility 数据收集，未达到 promotion criteria，不能形成 formal carrier 的类 |
| ast_ordinal | suite 内按所有 atomic syntactic units 计数的稳定局部顺序；excluded 也计数 |
| candidate_ordinal | 仅 eligible candidate 的顺序；只用于统计，不用于 key anchor |
| provisional evidence | 生成中增量 parser 认为边界可能闭合的内部状态；最终 strict replay 前无效 |
| region | 当前单元 public encoder/partition 的 `0 / 1 / boundary-erasure` |
| target bit | HMAC/ECC 从 private key 和 candidate-independent anchor 派生的期望 region |
| Pre-Gate | 搜索前判断是否值得投入固定 attempt budget；不产生证据 |
| Post-Gate | 对具体 final candidate 的 blind validity/replay/quality 风险判断 |
| pairwise semantic verifier | 生成/标注时比较 baseline 与 candidate 的静态、测试、differential证据；key-independent，但提取端不可用 |
| blind validity proxy | Post-Gate仅凭final candidate/context预测其offline pairwise-valid标签的风险；不是等价证明 |
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
- decorators 不属于 header span，也不是 AtomicWatermarkUnit；它们作为 ordered `StructuralAuxiliary` fingerprint 附着到随后 def/class header。
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

`selected` 与 `formal` 不保证相等：stop truncation、最终 parse、artifact mismatch 或 Post-Gate 重放都可使 selected 变 erasure；反之，未搜索的自然单元若盲重放恰好有效，可作为 null-like match/mismatch。这一选择避免 detector 依赖隐藏 generation state。

## 7. Deterministic parser and linearization contract

### 7.1 Normative final parser

`parser_contract_id = py312-cpython-ast-libcst-token-v1`：

1. UTF-8 bytes 按 PEP 263/3120 解码；非法编码 fail closed。
2. CPython 3.12 fixed patch/container 运行 `ast.parse(source, feature_version=(3,12), type_comments=True)`；失败时整个 artifact `PARSE_UNAVAILABLE`，不做 positive 判定。
3. pinned LibCST parse 同一 bytes，并用 `MetadataWrapper(PositionProvider)` 建 CST positions。
4. CPython `tokenize` 产生 exact tokens、logical-line、semicolon/colon boundaries。
5. AST 与 CST 按 source ranges、node category、suite ownership 双向对齐；任何 one-to-many ambiguity 使相关 unit erasure，若 ambiguity 影响 >10% units 则 artifact unavailable。

仅 AST 不足以保留 semicolon/header concrete spans，仅 token stream 又不足以可靠恢复 clause ownership；双层 contract 是必要的。[Python language reference](https://docs.python.org/3.12/reference/)，[LibCST](https://libcst.readthedocs.io/)

### 7.2 Source-order linearization

深度优先但严格 source-order：

1. compound 的当前 clause header；
2. 该 clause suite 内 units；
3. continuation clause header（`elif/else/except/finally/case`）；
4. continuation suite。

同 logical line 的 small statements 按 CST `body` 顺序。decorators 在 def/class header fingerprint 中保留次序，但不占 \(U_t\)。每个 unit 记录 global source ordinal 供调试，同时为每个 suite 记录从 0 开始、包含 excluded units 的 `ast_ordinal`。candidate ordinal 另算，不进入 anchor。

### 7.3 Context stream

`U_1,...,U_T` 是全局上述 source-order stream，而非“同 suite 前一个 eligible candidate”。跨 suite 仍可成为 CTX predecessor，但记录 transition：`same_suite / parent_to_child / child_to_continuation / child_to_parent / sibling_suite`、LCA depth 和 depth delta。被排除单元仍进入流，因此 Gate 看得到控制流事实且不会因 eligibility policy 更新而改变上下文索引。

### 7.4 Prompt/generated boundary

generation audit 可根据 byte provenance 标记 `prompt / generated / mixed`，Pre-Gate 对 prompt-owned unit 固定 skip；但 provenance **不进入 shared Gate 输入、anchor、Post-Gate 或 formal eligibility**，因为 blind extractor没有 prompt。public rule 永久排除 top-level def/class headers；嵌套 def/class 也默认 rule-excluded。若 prompt 自带其他可候选 unit，提取端无法识别其来源，它只可能贡献校准过的 null-like evidence。这是 blind setting 的不可消除限制。

### 7.5 Incremental generation parser

- pinned Tree-sitter/CST-incremental parser 只发 `provisional_header`、`provisional_simple_close`、`suite_transition`。
- parser recovery `ERROR` node、未闭合 bracket/string、ambiguous colon 或 mixed prompt span 时不触发搜索。
- simple statement 只有在 bracket depth 0 的 `NEWLINE/SEMI` 且 incremental CST 给出唯一 small-statement node 时才 provisional close。
- header colon 出现后可生成 header candidate pool，但直到 suite 被最终合法解析前都只是 provisional。
- 最终 strict parser 不接受 token-only fallback；parse failure 统一 unavailable/erasure。

### 7.6 Formatting stability

格式化可改变byte spans；若formatter把inline suite展开，线性化仍是header后相同small statements。normative `canonical_current`不是pretty-printed source，而是versioned canonical-CBOR AST/CST semantic serialization：

1. 去除locations、whitespace、redundant parentheses、semicolon/line-layout和普通comments；
2. 保留node/clause/operator/context tags、identifier bytes、keyword/argument order、literal的Python type与decoded value、type comments和pattern/signature fields；string value不做会改变值的Unicode normalization；
3. header只序列化current clause的kind和header-owned AST/CST fields，不含suite；simple unit只序列化该small statement；
4. CST只提供semicolon/header/span/owner facts，不能把body或neighbor text混入current serialization；
5. canonical-CBOR implementation/hash与gold vectors写入protocol，AST/CST映射歧义即erasure。

Gate/region tokenizer读取该canonical serialization的token projection，exact source只用于输出candidate和span audit。因此纯quote/numeric spelling、换行和冗余括号变化会被折叠，不能冒充semantic capacity；rename/等价AST改写的region survival仍是实验指标，不是parser保证。

## 8. Stable anchor and resynchronization design

### 8.1 Slot anchor 与 text identity 分离

target anchor 必须在候选池的所有 alternatives 间不变，所以不能含当前候选的 canonical text hash。定义：

```text
document_nonce     = uniformly random public 128-bit value fixed before generation
scope_fp_path     = sequence(SHA256(masked lexical-scope header + decorator roles))
role_path         = sequence(parent node type, clause type, suite role)
instance_path     = sequence(owner clause kind,
                             owner header's parent-suite sync_block_id,
                             owner header's parent-suite virtual_slot)
sync_block_id     = floor(ast_ordinal / 7) within current suite
virtual_slot      = ast_ordinal mod 7
anchor            = SHA256(protocol_id || document_nonce || scope_fp_path || role_path ||
                           instance_path || sync_block_id || virtual_slot || unit_kind)
text_identity     = SHA256(canonical current unit)  # not used for target bit
```

`document_nonce`在任何Pre-Gate/pool/target操作前生成，公开且不由key派生；null documents同样随机赋nonce。它避免不同documents的同型结构偶然共享target stream，但本身不是evidence。`scope_fp_path`把每层lexical scope的user identifiers/literals替换为role placeholders，使scope rename不必然改变整段anchor；built-in grammar tokens与parameter arity/markers保留。仅有masked header可能让同型sibling functions碰撞，所以`instance_path`还记录每个enclosing suite owner header在其**父suite**的all-unit virtual locator；`if/elif/else`、多个同型loops和sibling def/class因而可区分。current header自身不进入其instance path，仍由自己的parent-suite slot定位，保证候选改写不改变anchor。

在零编辑 strict parse 下，同一 document 的全部 anchor 必须 pairwise unique；发生碰撞时整个 document `ANCHOR_COLLISION_UNAVAILABLE`，不得共享 target。gold/fuzz conformance必须测到零碰撞。编辑后 instance locators由第 8.3 节 DP 的 virtual slots重建；若存在多个等成本实例路径，则相应 blocks erasure。

### 8.2 为什么不用 candidate_ordinal 或当前文本

- candidate policy 更新会改变 candidate ordinal，造成后续级联；因此禁止。
- 当前文本随 embedding alternative 变化；若进入 anchor，同一 pool 没有固定 target bit，违反 key-independent counterfactual。
- exact global ordinal 的一个 insertion 会污染整个文件；因此 ordinal 在 suite 内重置，并每 7 units 形成 sync block。

### 8.3 Bounded local resynchronization

blind extractor 在每个 suite 独立运行 edit-channel DP：状态为 `(observed_index, virtual_slot, insertions, deletions)`，默认每 7-slot block 最多 2 次 insertion/deletion。transition cost：match 0、insert/delete 1、role-type mismatch 2；同 cost 多解且 keyed score 差距 <1.0 时，受影响 block 全部 erasure。suite header 是 hard reset，编辑不会跨 suite 级联。

递归顺序是确定的：先从module root对parent suite做DP，唯一确定owner header的parent-suite virtual locator后，才构造child `instance_path`并启动child-suite DP；continuation clause各用自身header locator。parent locator多解时不让keyed child score反向选择parent，而是整个相关subtree erasure。每个virtual block的edit counter独立，跨block保留current offset；因此一个合法insertion不会在下一block自动把offset归零，但超过任一block budget即停止该hypothesis。

DP 对每个合法 alignment hypothesis 重建 virtual ordinal/anchor，并计算带 `edit_penalty=0.25` 的 detector score；取最大值构成 data-dependent multiple testing，因此 calibration null 上必须做同样 max。没有声称从 final code 唯一恢复原始序列；bounded hypothesis search 只是局部鲁棒协议。

### 8.4 Reordering 与结构编辑

- 同 suite reorder 视为 delete+insert；超过 budget fail/erasure。
- header/body edit 若改变 clause role path，只影响对应 subtree；其他 suites 由 header reset。
- dead-code insertion 消耗 local insertion budget；大量 insertion 可擦除 signal，但不能使 detector 绕过 FPR calibration。
- text identity 用于 clone/duplicate suppression；重复 identity 在同 cluster 只保留首个，避免攻击者复制一个 match 放大统计量。

## 9. Context window definition

对当前 \(U_t\)：

- CTX-1：`[U_{t-1}, U_t]`；
- CTX-2：`[U_{t-2}, U_{t-1}, U_t]`；
- CTX-3：`[U_{t-3}, U_{t-2}, U_{t-1}, U_t]`。

缺失 predecessor 使用 `[BOS_UNIT]`，不是复制当前单元。每个 unit 输入：canonical text tokens、unit/node/clause/parent types、suite depth、role path tokens、transition type、is_header、is_multiline_header、inline-suite flag、decorator role count。`origin`、key、target bit、candidate count、attempt index、retry/failure、generator/repository id 全不作为 model input。

使用一个shared model；训练时对同anchor的三views做paired batches和uniform context dropout。不会把CTX width与attempt budget混合。正式v1的Pre-Gate只读frozen baseline (D_0\) contexts；Post-Gate训练同时含baseline selection views和由public、key-independent candidate-context augmentation组成的assembled views，formal extraction只读最终 (D_1\) contexts。任何一个predecessor编辑最多直接改变随后3个Post acceptance states，且只能造成accept/mismatch/erasure变化，不能改变那些units的region或target anchor。上下文只进Pre/Post-Gate；public region encoder的输入严格为canonical current unit。

## 10. Candidate eligibility rules

### 10.1 Policy states

每个 syntactic unit 被公开 rule engine 分到：

- `FORMAL_ELIGIBLE`：可进入部署 candidate pool；
- `ORACLE_ONLY`：只收集 feasibility/negative labels，不能形成正式载体；
- `RULE_EXCLUDED`：永不生成 candidates；
- `SYNTACTIC_ONLY`：例如零自由度 header，只用于结构/ordinal。

Gate 只在 `FORMAL_ELIGIBLE` 上运行；excluded 不作为普通负样本训练 Gate，以免模型学习本可由规则确定的 trivial shortcut。它们用于 OOD/evaluation suite。

### 10.2 v1 formal simple-statement allowlist

初始只允许 `Assign` 与有 value 的 `AnnAssign`，target 必须是单一 `Name` 或由 `Name` 构成且形状固定的 tuple/list；无 Attribute/Subscript/Starred。RHS 必须属于公开 `SafeExprV1`：

```text
SafeAtom := Name(Load)
          | Constant(None | bool | bounded int | bounded str | bounded bytes)
          | Tuple(SafeAtom*)
SafeExprV1 := SafeAtom | UnaryOp(+|-|~, bounded-int Constant)
```

禁止 float/complex/NaN、f-string、list/set/dict、Attribute、Subscript、Slice、BinOp/BoolOp/Compare/IfExp on Names、Call、Await、Yield/YieldFrom、NamedExpr、Lambda 和任何 comprehension；这些构造可能触发 descriptor、`__getitem__`、operator/truth overload、iteration、allocation/aliasing、exception timing 或 scheduling。`AnnAssign` annotation 在所有 candidates 中 byte-identical且不参与改写。

候选还必须保持：顶层 node category、target binding sequence/shape、free-name sequence（不只multiset）、constant value/type sequence、control/effect/call/exception/await/yield/scope flags。生成时 pairwise relation必须由注册rewrite certificate或E2/E3 tests判PASS。即使满足这些不变量也不证明value equivalence；blind Post-Gate FAR 与功能测试仍是 go/no-go。这个allowlist很可能容量低，正是Oracle必须先验证的根本风险。

### 10.3 Oracle-only simple classes

- AugAssign；
- 所有一般 `Expr` statement；
- assignment/Expr 含 Attribute/Subscript、BinOp/BoolOp/Compare/IfExp on Names、Call、container literal/comprehension、lambda、walrus；
- `await`、`yield/yield from`；
- 可能改变 evaluation order、mutation、I/O、exception timing 或 coroutine scheduling 的表达式。

只有 class-specific Oracle 达到第 40 节阈值并经 protocol version bump，才可 promoted；不能通过一次全局 Gate 分数隐式开放。

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
- `if/elif/while/for/async for/with/async with/match/case/except/except*`：`ORACLE_ONLY`；按 clause class 独立 promotion，不共享一项平均分。

**v1封闭决定：所有ClauseHeaderUnit均不形成formal carrier。** 原因不只语义风险：第16节全document key-neutral planning必须先冻结baseline body，而“header必须在body前选择”禁止事后回写。下面12.2只定义Oracle/provisional research protocol；即使12.3容量阈值达标，也只能支持注册一个新major protocol，其还必须证明header pool在body前生成且不读取任何先前key-selected context。SAGE-C-v1不会隐式promotion header。

### 12.2 Generation protocol

header 生成到 colon 且 incremental CST 唯一识别时：

1. 在 header 起点 checkpoint 克隆固定 \(A\) 个 attempts；此时 body 尚未生成。
2. candidate 只替换当前 header span，不包含 body，也不重写此前单元。
3. pool 完成后才派生 target bit/选 header。
4. 选择后生成 body；header 只标 provisional。
5. final strict parse 必须确认 clause ownership、完整 suite、candidate-independent anchor 与 Post-Gate；否则 erasure。

若 body 已生成，禁止回滚 header 并保留/顺带改写 body。最终失败只变 erasure，不做 post-hoc header retry。这个限制牺牲容量，换取“retry only current unit”和明确语义边界。

### 12.3 Header promotion gate

每个 header class 必须单独满足：A=64 min-side success ≥0.20、dual-reachable anchors ≥30%、Post-Gate FAR 95% upper ≤0.5%、功能 delta ≥-2pp、format replay ≥99.9%，且不是只靠 lexical colon/identifier artifact。未通过保持 oracle-only；short header representation 退化不能通过把 context 塞入 region encoder 修复。

## 13. Data sources

数据集不是把任意公开 GitHub 代码直接抓来训练，而是建立可审计的 provenance ledger。每个 source artifact 在任何派生前必须有 `source_id`、repository URL/commit、SPDX expression、文件路径、license evidence hash、采集日期、删除/opt-out 状态和 legal-use decision。只有经项目法务/数据治理允许的 permissive allowlist（初始为 MIT、BSD-2/3-Clause、Apache-2.0、ISC；MPL/GPL 等另行评估其再分发义务）进入训练；无法确定 license 的代码拒绝。CodeSearchNet 只作为候选 repository 索引，不能把其聚合数据自动等同于“许可清晰”。

数据来源分为六个互斥 provenance strata：

1. **真实仓库源码。** 按 repository 固定 commit 采样可解析 Python 3.12 文件，保留 project tests、type hints 和依赖 manifest；覆盖库、CLI、数据处理、异步服务、科学计算等领域。仓库必须通过 license allowlist、secret scan 和 clone quarantine。
2. **许可明确的教学/合成任务。** 自建、委托创作或明确允许再使用的函数级任务，专门覆盖 semicolon、inline suite、多行 header、nested flow、`match/case`、`except*`、decorator 和 async。合成任务保留生成规范，不能与真实测试集模板近重复。
3. **真实 prompt 上的当前 generator 轨迹。** prompt 来自训练 partition 的 docstring/signature/issue-style specification；收集正确与错误 completion、完整代码和 incremental prefix。prompt 只用于数据生成，不成为 Gate 输入或 detector 依赖。
4. **多 generator 候选。** 至少包含当前目标 generator、一个不同家族的开源 code model 和一个非同架构/非同 tokenizer generator；记录 generator family/version 仅用于分层、nuisance audit 和 leave-one-generator-out 评估，严禁作为 Gate feature。
5. **语义破坏与 hard negatives。** 从正确候选构造 evaluation-order swap、operator/constant mutation、binding change、exception swallowing、await/yield relocation、API-signature change、header predicate inversion 等；每个 mutation 都记录 operator 和是否由 tests/human/static oracle 检出。未被 tests 检出的 mutation不能被标为语义保持。
6. **编辑与解析 stress corpus。** 对合法样本产生 formatter、identifier alpha-renaming、comment change、局部 equivalent rewrite、单 unit insertion/deletion/reorder 和 header/body edit；另保留 truncated/unbalanced/incomplete prefixes，只训练 provisional parser/robustness auxiliary tasks，绝不作为 final formal evidence。

HumanEval 与 MBPP 以及其已知衍生/镜像只进入封存的 external evaluation；不训练 region encoder/Gate，不做 temperature scaling、threshold/ECC 或 parser policy调整。[HumanEval](https://arxiv.org/abs/2107.03374) 与 [MBPP](https://arxiv.org/abs/2108.07732) 的任务文本和 canonical solutions 均进入 contamination signatures。若训练 source 与其 MinHash/AST clone 命中，整个 connected component 从训练删除。

每个 stratum 都同时覆盖：正确/错误代码、可解析/不可解析前缀、simple/header、低/中/高候选熵、候选为空、单侧可达和双侧可达。数据报告按 source、unit class、generator、correctness、parse state、entropy bin 和 edit type 分层；任何一个大 stratum 不得贡献超过训练 anchors 的 35%，防止模型把 repository 或 generator style 当 feasibility。

## 14. Data generation pipeline

### 14.1 Split-before-expand 流程

数据生成必须按以下不可交换顺序执行：

1. ingest provenance 与 license decision；
2. secret/PII scan，固定 source bytes 和 source hash；
3. 建立 repository/task/function/template/clone 关系图，先分配 split；
4. 仅在各 split 内 strict parse、抽取 scope 和 AtomicWatermarkUnit；
5. 生成 anchor、current-unit canonical form 和 structural metadata；
6. 对 train/calibration/test 各用其预注册 seed namespace 生成 key-independent candidate pools；
7. 从同一 pool 计算 region-0 与 region-1 counterfactual labels；
8. 做 syntax/static invariant/unit tests/property tests/sandbox execution 和人工审计抽样；
9. 生成 CTX-1/2/3 paired views、edit variants、hard negatives 与 Post-Gate labels；
10. 写 immutable records、parent hashes 和 data-card counts；任何过滤都产生日志但不写 secret。

禁止先生成 anchors/candidates 再随机拆行。一个 function 的原始版本、所有 anchors、candidate attempts、generator variants、mutations、retry seed blocks 和 CTX views 由同一 `group_component_id` 锁在一个 split。

### 14.2 Anchor 与 candidate 产生

对合法source function或无key baseline \(D_0\)，strict parser按第7节抽取全部syntactic units。每个当前unit由rewrite generator接收：只读冻结的baseline prefix/context、只允许替换exact current span、公开unit policy、保持不变量和固定decoding parameters；正式planner通过access control屏蔽future units和任何key-selected code。生成器不能看到key、target region或`q_0/q_1`；每个attempt从public `pool_seed_namespace || anchor_record_id || seed_block || attempt_index`派生RNG seed。

真实 autoregressive 轨迹的当前 baseline completion 是 attempt 0；attempt 1…\(A-1\) 是同一 current span 的 alternatives。对于 header，在 body 生成前 checkpoint；对于 simple unit，只截取单一 CST node。任何 candidate 跨越当前 span、增加/删除 sibling unit 或重写 prefix，均标 `span_violation` 并不能 formal-valid。

### 14.3 分层语义与功能标签

单元级“语义保持”采用证据层级，绝不把神经相似度当真值：

- `E0_PARSE`：strict parse 且 span/structure 合法；
- `E1_STATIC`：binding、free names、effects、calls、control flags、type/comment constraints 不变量通过；
- `E2_TESTED`：原 project tests、task tests、metamorphic/property tests 在隔离 sandbox 中与 baseline 同结果；
- `E3_DIFFERENTIAL`：多输入 differential execution 的返回值、exception class、stdout/stderr、observable mutations 一致；
- `E4_AUDITED`：高风险/抽样 pair 经双人 blinded review 一致。

`formal_valid_label=1` 初始只允许 conservative class 且 E0–E3 全过；没有可执行测试或依赖缺失时为 `unknown`，不是正例。这里 E1–E3 是 baseline/candidate pairwise label evidence：生成时可作key-independent过滤，提取端没有baseline因而不能重算。失败 candidates 根据首个失败层标 hard negative。动态执行设定 CPU/内存/时间/网络/文件系统限制；timeout、nondeterminism 或 environmental dependency 归 unknown/erasure。该标签只是可证伪的 operational semantic preservation，不是程序等价证明。

### 14.4 Region、Post-Gate 与 erasure 标签

冻结 public current-unit region encoder 后，对 candidate 的 integer score \(s(x)\) 计算 `region ∈ {0,1,boundary}`。region label 与 key 无关。对固定 pool 同时产生：是否存在 region 0 formal-valid candidate、是否存在 region 1 formal-valid candidate、各侧有效比例、最小合格 rank 和 cost。

Post-Gate targets 来自上述 E0–E4、reference-runtime replay、format/rename/edit variants 和 candidate-group dependence：syntax、blind semantic-validity proxy、quality、region margin、reference replay、robustness、redundancy 与 final formal validity。E2/E3等privileged pairwise结果只作监督；模型输入始终是final-code可重建量。Post-Gate学的是 \(P(F_{pair}=1\mid final\ observables)\)，不能复算或证明 \(F_{pair}\)。靠近 region 或 Gate threshold、不同 replay matrix 输出不一致、alignment 多解、测试 unknown 的样本明确标 `erasure_target=1`。

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

Pre-Gate paired views固定来自同一 \(D_0\)。Post-Gate另构造`assembled_context_group`：从前序三个anchors各自已封存pools中按public RNG选择baseline或一个candidate，绝不读取key/target，并对CTX-1/2/3复用同一assembled choices；这让Post训练覆盖 \(D_1\) context shift而不把actual key-selected trajectory泄漏为feature。actual embedded D1只进held-out system evaluation，不回灌训练。

context dropout 独立采样 \(w\in\{1,2,3\}\)，但不删除中间 unit；curriculum 先 current-only/CTX-1，再均匀加入 CTX-2/3。评估同时报告 paired delta、bootstrap CI 和 permutation test，避免把 pool 随机性解释成 context gain。`attempt_budget` 是独立轴 \(A\in\{2,4,6,8,16\}\)，不与 context width 绑定。“retry 5”在本文唯一解释为 \(A=6\)：1 次 initial attempt + 5 次 additional attempts。

## 16. Candidate-pool protocol

### 16.1 Key-independent pool

正式v1采用document-wide two-phase planning。Phase P0用完全无key的generator产生baseline final code \(D_0\)，strict parse后冻结unit stream、document nonce和baseline contexts。Phase P1仍禁止key模块：Pre-Gate只读 \(D_0\) 中current unit与最多三个baseline predecessors；若接受，pool builder固定产生恰好 \(A\) 个current-unit-only attempts，不得target-conditioned early stop。generator artifact/hash、rewrite template、temperature/top-p、seed derivation、attempt count/order、max tokens、dedup、syntax filter、timeout和fallback全部注册且不读key。

对所有eligible units完成Pre decisions、全部pools、pairwise/blind scoring后，写`document_pool_commitment = SHA256(protocol || nonce || ordered(slot status || ordered candidate hashes || config hash))`；skip/timeout也有固定sentinel。只有该全document commitment封存后，key模块才可解锁并逐slot派生targets。commitment不随final code发布、不是detector输入，但private audit必须保存。由此对固定`(prompt, baseline RNG, nonce, pool RNG)`改变key时，所有Pre decisions、pool bytes/order/count和scores必须逐字节相同。

在线逐unit选择被排除为formal v1：先前key-selected code会进入后续context，使后续Pre decision产生间接key依赖。v1先生成baseline再局部改写，candidate generator虽可访问已冻结baseline prefix，但被访问控制禁止读取future units；pairwise tests可在 \(D_0\) 的单unit替换副本上运行。该改变增加一次baseline pass，但真正满足全计划key-independence。

### 16.2 Dedup、过滤与排序

先按 exact normalized token bytes 去重，再按 current-unit canonical AST hash 去重；保留最小 attempt index 作为代表。重复仍计生成成本，但不增加覆盖率。非水印 hard filters（span、parse、binding/effect invariants、resource limits）对所有 candidates 一视同仁。不能因某侧 region 缺候选而调整 temperature、追加 attempt 或换 prompt。

合格 target-side candidates 按公开 total order 排序：

\[
(\text{hard-validator tier desc},\ p^F_{\rm int}\text{ desc},\ |s(x)|\text{ desc},
\text{quality}_{\rm int}\text{ desc},\text{canonical bytes asc},\text{attempt index asc}).
\]

整数 outputs 与 canonical bytes 消除 backend/tie ambiguity。若 pool 为空、只有 duplicates、只有 boundary candidates 或目标侧没有 formal-valid candidate，选择预先确定的 attempt-0 safe fallback；记录内部 `embedding_outcome=ERASURE_FALLBACK`。fallback 不根据缺失的目标侧换形态，也不追加搜索。最终 detector 仍只看 final code，可能将 fallback 判 match、mismatch 或 erasure；统计 null 会完整吸收这种分布。

### 16.3 泄漏审计

在同一frozen \(D_0\)、nonce和public RNG下对1,000个keys重放planning，要求document commitment、Pre decisions、candidate IDs/bytes/order/count/config、all scores与timeouts逐字节相同。用permuted key/target做two-sample probes，检查pool size、duplicate count、attempt latency、seed/order、token count、generator log fields和fallback rate是否可预测target；AUC upper tolerance 0.52且permutation \(p\ge0.01\)。统计screen不替代key-module taint/dataflow audit。candidate count、attempt index、generator ID、seed block和retry outcome只作监督分层/审计，不能进入Gate tensors。若实现让任何key-selected prefix回流到P0/P1，直接违反协议，不以“probe未检出”豁免。

## 17. Dataset schema

### 17.1 Schema 约定

所有记录使用 UTF-8 JSON/Parquet logical schema；ID 是 canonical CBOR 字段序列的 SHA-256，不依赖数据库行号。`schema_version` 控制结构，`protocol_id` 控制 parser/canonicalizer/model/threshold 语义，`data_release_id` 控制数据快照。表中“公”表示可进入 public dataset/registry；“训/嵌/提”表示该阶段可作为可用信息，`监督`表示只作 target 不作 model input，`审计`表示可记录但不进决策。任何标“敏感”的字段不得出现在 public final-code artifact。

以下common envelope物理存在于后续十种record（不是隐含数据库列），且全部禁止进入model tensor，除非对象表另有明确输入字段：

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| schema_version | semver | 是 | schema check | schema check | schema check | 无 | record schema |
| protocol_id | string | 是 | join | registry | registry | 无 | immutable registry key |
| data_release_id | sha256 | 是 | join/audit | 否 | 否 | 无key；可识别release | data manifest |
| split | enum(train,model_val,calibration,test,external)/null | 条件 | sampler only | 否 | 否 | test leakage；禁模型 | split manifest |
| group_component_id | sha256/null | 条件 | grouping only | 否 | 否 | repo/task identity shortcut；禁模型 | split-graph version |
| provenance_record_id | sha256/null | 条件 | audit only | 否 | 否 | 可能识别source；禁模型 | provenance release |
| parent_record_hashes | list[sha256] | 条件 | lineage | audit | audit | 无key；generation trace不进detector | schema version |
| record_hash | sha256 | 是 | integrity | integrity | integrity | 无 | canonical-CBOR/hash version |

### 17.2 AnchorRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| schema_version | semver | 是 | 是 | 是 | 是 | 无 | record schema |
| protocol_id | string | 是 | 是 | 是 | 是 | 无 | immutable registry key |
| anchor_record_id | sha256 | 是 | 是 | 审计 | 重建 | 无；非 keyed | schema+protocol |
| source_artifact_id | sha256 | 条件 | 分组 | 否 | 否 | 可能识别仓库，非 bit | data release |
| document_nonce | bytes16 | 是 | join only，禁Gate tensor | anchor输入 | 从leading metadata/API恢复 | 公开随机值；不含key/bit | anchor protocol |
| scope_fp_path | list[sha256] | 是 | 是 | 是 | 重建 | masked lexical scopes；非secret | canonicalizer version |
| role_path | list[RoleToken] | 是 | 输入 | 输入 | 重建 | 不得含 origin/key | parser version |
| structural_instance_path | list[(role,block,slot)] | 是 | 输入 | 输入 | DP重建 | 候选独立；防同型siblings碰撞 | anchor/sync version |
| suite_sync_id | uint64 | 是 | 输入 | 输入 | 重建 | 无 | sync algorithm version |
| all_unit_local_ordinal | uint32 | 是 | 输入 | 输入 | 重建 | excluded 也计数 | linearizer version |
| clause_owner_role | enum/null | 是 | 输入 | 输入 | 重建 | 无 | grammar version |
| decorator_fingerprint | sha256/null | 是 | 输入 | 输入 | 重建 | 仅结构，不含 secret | canonicalizer version |
| anchor_preimage | bytes | 是 | 审计 | PRF输入 | 重建 | 候选独立；公开无妨 | protocol_id |
| anchor_hash | sha256 | 是 | join | join | 重建 | 无 | hash suite version |
| document_anchor_unique | bool | 是 | hard check | hard check | hard check | false时全文fail closed | anchor contract |
| current_text_identity | sha256 | 是 | 输入/分层 | 质量检查 | 重建 | **禁止进入 target anchor** | canonicalizer version |

### 17.3 AtomicUnitRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| unit_record_id | sha256 | 是 | 是 | 审计 | 重建 | 无 | schema+protocol |
| anchor_record_id | sha256 | 是 | join | join | join | 无 | AnchorRecord version |
| exact_span | (byte_start,byte_end) | 条件 | 抽取 | 当前 prefix | 重建 | source location only | tokenizer/parser |
| token_span | (tok_start,tok_end) | 条件 | 抽取 | 当前 prefix | 重建 | 无 | tokenizer version |
| exact_text | string | 条件 | 输入 | 输入 | final code | 可能含 source secrets，入库前 scan | data release |
| canonical_current | bytes | 是 | 输入 | 输入 | 输入 | 当前单元公开 | canonicalizer version |
| unit_kind | enum(simple,header) | 是 | 输入 | 输入 | 重建 | 无 | grammar version |
| node_type | enum | 是 | 输入 | 输入 | 重建 | 无 | grammar version |
| clause_type | enum/null | 是 | 输入 | 输入 | 重建 | 无 | grammar version |
| owner_clause_type | enum/null | 是 | 输入 | 输入 | 重建 | 区分各类else | grammar version |
| parent_node_type | enum | 是 | 输入 | 输入 | 重建 | 无 | linearizer version |
| suite_depth | uint16 | 是 | 输入 | 输入 | 重建 | 无 | linearizer version |
| is_multiline_header | bool | 是 | 输入 | 输入 | 重建 | 无 | CST contract |
| is_inline_suite | bool | 是 | 输入 | 输入 | 重建 | 无 | CST contract |
| syntactic_ordinal | uint32 | 是 | 分层 | 审计 | 重建 | 无 | linearizer version |
| candidate_ordinal | uint32/null | 是 | 统计 | 审计 | 重建 | 不得进入 anchor | eligibility policy |
| eligibility_state | enum | 是 | rule mask | rule gate | rule gate | policy 可观察，非 target | policy version |
| exclusion_reason | enum/null | 是 | OOD分层 | rule gate | rule gate | 无 | policy version |
| origin_label | enum(prompt,generated,unknown) | 否 | **仅分层** | rule-only skip audit | 不可用 | detector 不可重建；禁止模型/anchor | data release |
| parse_status | enum | 是 | 辅助监督 | provisional | strict重建 | 无 | parser version |
| recovery_ambiguity_count | uint16 | 是 | 辅助监督 | provisional | strict重建 | 无 | parser version |

### 17.4 CandidatePoolRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| pool_record_id | sha256 | 条件 | join | 审计 | 否 | 无 | schema+protocol |
| anchor_record_id | sha256 | 条件 | join | 审计 | 否 | 无 | AnchorRecord version |
| generator_artifact_id | sha256 | 条件 | **分层非输入** | 固定配置 | 否 | generator shortcut | generator registry |
| rewrite_template_hash | sha256 | 是 | 审计 | 固定 | 否 | 无 target | pool protocol |
| pool_seed_namespace | string | 条件 | 分层非输入 | 固定 | 否 | 公开且 key-independent | data release |
| seed_block_count | uint16 | 是 | 标签 | 1 | 否 | 不能进模型 | pool protocol |
| attempt_budget | uint16 | 是 | 标签/成本 | 固定 | 否 | 不能进模型 | pool protocol |
| ordered_candidate_ids | list[sha256] | 条件 | labels | 审计 | 否 | 若公开可能泄漏生成轨迹；非 key | data release |
| unique_count | uint16 | 条件 | label | 审计 | 否 | candidate-count shortcut，禁输入 | pool protocol |
| duplicate_count | uint16 | 条件 | label | 审计 | 否 | candidate-count shortcut，禁输入 | pool protocol |
| pool_commitment | sha256 | 条件 | audit | 在 target 前封存 | 否 | 无 bit（若流程合规） | hash suite |
| document_plan_commitment | sha256 | 条件 | audit | 全文唯一、target前封存 | 否 | 无bit；绑定所有skip/pools/scores | hash suite |
| target_assignment_relation | enum(after_document_commit) | 条件 | leak audit | assert | 否 | 违反即critical leak | protocol_id |
| pool_status | enum | 条件 | label | fallback | 否 | 不发布到 final artifact | schema version |

### 17.5 CandidateRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| candidate_record_id | sha256 | 条件 | join | 审计 | 否 | 无 | schema+protocol |
| pool_record_id | sha256 | 条件 | join | 审计 | 否 | 无 | CandidatePool version |
| attempt_index | uint16 | 条件 | **分层非输入** | order/tie-break | 否 | retry shortcut；禁 Gate input | pool protocol |
| seed_block | uint16 | 条件 | **分层非输入** | seed lookup | 否 | seed shortcut；禁 Gate input | pool protocol |
| candidate_exact | string | 条件 | Post输入 | Post输入 | final candidate重建 | source content | data release |
| candidate_canonical | bytes | 条件 | Post输入 | Post输入 | final candidate重建 | public current content | canonicalizer version |
| content_hash | sha256 | 是 | dedup | dedup | 重建 | 无 | hash suite |
| ast_hash | sha256 | 是 | dedup | dedup | 重建 | 无 | parser version |
| span_valid | bool | 是 | 监督 | hard filter | 重建 | 无 | verifier version |
| parse_valid | bool | 是 | 监督 | hard filter | 重建 | 无 | verifier version |
| invariant_vector | bitset | 是 | 监督/Post输入中仅可部署项 | hard filter | 重建 | 不含 tests/key | verifier version |
| region_score_int | int16 | 是 | 监督 | 选择 | 重建 | public region；不是 target | region artifact |
| observed_region | enum(0,1,boundary) | 是 | 监督 | 选择 | 重建 | 无 key | margin policy |
| evidence_tier | enum(E0..E4,unknown) | 条件 | 监督 | 仅部署可重放子集 | 否 | tests/human 不能作输入 | labeler version |
| semantic_valid_label | bool/unknown | 否 | 监督 | 不可用 | 不可用 | 无 key；privileged label | label release |
| mutation_type | enum/null | 条件 | hard-negative分层 | 否 | 否 | shortcut，禁输入 | mutation suite |
| execution_outcome_hashes | list[sha256] | 否 | 监督 | 不可用 | 不可用 | 可能含 private tests | sandbox version |
| quality_label | float/unknown | 条件 | 监督 | 不可用真值 | 不可用真值 | privileged label | rubric version |
| robustness_labels | map[edit,bool/unknown] | 条件 | 监督 | 不可用真值 | 不可用真值 | privileged labels | edit-suite version |
| formal_valid_label | bool/unknown | 否 | 监督 | 不可用 | 不可用 | 不含 target；禁作部署输入 | policy+label version |
| erasure_target | bool | 否 | 监督 | 不可用 | 不可用 | 不含 target；禁作部署输入 | policy+label version |

### 17.6 ContextViewRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| context_view_id | sha256 | 条件 | join | 构造 | 可重建 | 无 | schema+protocol |
| anchor_record_id | sha256 | 条件 | join | 构造 | 可重建 | 无 | AnchorRecord version |
| pool_record_id | sha256 | 条件 | join | audit | 否 | generation pool；不进模型 | CandidatePool version |
| width | enum(1,2,3) | 是 | 输入 | 输入 | Post重放可用 | 不含 bit | model contract |
| context_source | enum(baseline,public_assembled,final) | 条件 | sampler/分层，禁模型token | baseline/final | final | actual target trajectory不得进train | data protocol |
| context_augmentation_seed | uint64/null | 否 | sampler only | 否 | 否 | 禁模型；必须key-independent | data release |
| predecessor_unit_ids | fixed list[sha256/null;3] | 条件 | join | prefix | 重建 | 无 | linearizer version |
| predecessor_canonical | fixed list[bytes;3] | 条件 | 输入 | 输入 | final code | 代码内容 | canonicalizer version |
| current_unit_id | sha256 | 条件 | join | 构造 | 重建 | 无 target | AtomicUnitRecord version |
| current_canonical | bytes | 条件 | 输入 | 输入 | 输入 | 无 target | canonicalizer version |
| structural_feature_tensor | fixed int tensor | 是 | 输入 | 输入 | 重建 | 禁 origin/generator | feature schema |
| BOS_mask | bitset | 是 | 输入 | 输入 | 重建 | 无 | feature schema |
| transition_mask | bitset | 是 | 输入 | 输入 | 重建 | 无 | feature schema |
| paired_group_id | sha256 | 是 | sampler | audit | 否 | 无 | data release |
| pool_label_hash | sha256 | 是 | parity assert | audit | 否 | 保证三 view 同 label | label release |

### 17.7 PreGateLabelRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| label_id | sha256 | 条件 | join | 否 | 否 | 无 | schema+label release |
| context_view_id | sha256 | 条件 | join | 否 | 否 | 无 | ContextView version |
| trials_R | uint16 | 是 | loss | 否 | 否 | candidate-count，不作输入 | pool protocol |
| successes_region0 | uint16 | 条件 | supervision | 否 | 否 | 反事实、无 actual target | labeler version |
| successes_region1 | uint16 | 条件 | supervision | 否 | 否 | 反事实、无 actual target | labeler version |
| beta_posterior0 | (alpha:float,beta:float) | 是 | soft label | 否 | 否 | 无 | prior+labeler version |
| beta_posterior1 | (alpha:float,beta:float) | 是 | soft label | 否 | 否 | 无 | prior+labeler version |
| y0_soft | float | 是 | supervision | 否 | 否 | 无 | labeler version |
| y1_soft | float | 是 | supervision | 否 | 否 | 无 | labeler version |
| both_side_success | float | 是 | supervision | 否 | 否 | 无 | pool protocol |
| min_valid_rank_0 | uint16/null | 条件 | cost supervision | 否 | 否 | 不能作输入 | pool protocol |
| min_valid_rank_1 | uint16/null | 条件 | cost supervision | 否 | 否 | 不能作输入 | pool protocol |
| expected_attempt_cost | float | 是 | supervision | 否 | 否 | 不能作 input | cost model version |
| class_weight | float | 是 | loss only | 否 | 否 | 无 | training recipe |
| confidence_weight | float | 是 | loss only | 否 | 否 | 无 | training recipe |

### 17.8 PostGateLabelRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| label_id | sha256 | 条件 | join | 否 | 否 | 无 | schema+label release |
| candidate_record_id | sha256 | 条件 | join | 否 | 否 | 无 | CandidateRecord version |
| syntax_valid | bool | 是 | supervision | hard verifier重算 | 重算 | 无 | verifier version |
| structure_valid | bool | 是 | supervision | hard verifier重算 | 重算 | 无 | verifier version |
| pairwise_semantic_preservation | bool/unknown | 否 | supervision | 不可用真值 | 不可用真值 | privileged baseline/tests | labeler version |
| quality_score | float/unknown | 条件 | supervision | 预测 | 预测 | 无 key | rubric version |
| region_margin_int | int16 | 是 | supervision | 重算 | 重算 | public | region artifact |
| replay_matrix | map[env,bool] | 条件 | supervision | conformance | conformance | 环境信息非 target | deployment matrix version |
| formatting_survival | bool/unknown | 条件 | supervision | 预测/部分重算 | 预测/重算 | 无 key | edit-suite version |
| rename_survival | bool/unknown | 条件 | supervision | 预测/部分重算 | 预测/重算 | 无 key | edit-suite version |
| local_edit_survival | map[edit,bool/unknown] | 条件 | supervision | 预测/部分重算 | 预测/重算 | 无 key | edit-suite version |
| redundancy_cluster_id | sha256 | 条件 | supervision | 预测 | 聚合 | 不含 target | clustering version |
| correlation_label | float | 条件 | supervision | 预测 | 聚合 | 不含 target | clustering version |
| formal_valid | bool/unknown | 否 | supervision | 不可用真值 | 不可用真值 | privileged label | policy version |
| false_accept_cost | float | 否 | loss weight | 不可用真值 | 不可用真值 | asymmetric risk only | cost version |
| erasure_reason_target | enum/null | 条件 | supervision | 预测 | 重算 | 无 | policy version |

### 17.9 EmbeddingDecisionRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| decision_id | sha256 | 否 | audit | 是 | 否 | decision row非detector input | schema+protocol |
| anchor_hash | sha256 | 否 | audit | 是 | 重建 | 单独非secret | AnchorRecord version |
| protocol_id | string | 否 | audit | 是 | 是 | 无 | immutable registry key |
| pregate_q0_int | int16 | 否 | diagnostics | 是 | 可重算但不必 | selection trace shortcut | Gate artifact |
| pregate_q1_int | int16 | 否 | diagnostics | 是 | 可重算但不必 | selection trace shortcut | Gate artifact |
| pregate_q_safe_int | int16 | 否 | diagnostics | 是 | 可重算但不必 | selection trace shortcut | Gate artifact |
| pregate_decision | enum | 否 | diagnostics | 是 | 可重算 | 不得写 final artifact | threshold registry |
| pregate_reason | enum | 否 | diagnostics | 是 | 可重算 | 不得写 final artifact | threshold registry |
| pool_commitment | sha256 | 否 | diagnostics | 是 | 否 | unit generation trace | hash suite |
| document_plan_commitment | sha256 | 否 | diagnostics | 解锁key前assert | 否 | generation trace；非detector input | hash suite |
| attempt_budget | uint16 | 否 | diagnostics | 是 | 否 | generation trace | pool protocol |
| target_bit | secret bit | **否** | **否** | PRF后内存 | 重派生不持久化 | 直接泄漏 key-derived bit | PRF suite |
| selected_candidate_id | sha256/null | 否 | diagnostics | 是 | 否 | 与 target 联合可泄漏 | selection policy |
| embedding_outcome | enum(success,skip,fallback,provisional) | 否 | diagnostics | 是 | final replay覆盖 | retry/selection side channel | protocol |
| final_content_hash | sha256 | 条件 | audit | 是 | final code hash | 单独不泄 key | canonicalizer |
| provisional_to_formal_status | enum | 否 | diagnostics | header finalize | final replay | 不得被 detector信任 | parser protocol |

### 17.10 ExtractionEvidenceRecord

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| evidence_id | sha256 | 报告可脱敏 | 否 | 否 | 是 | 多次查询可能泄 keyed comparisons | schema+protocol |
| document_run_id | sha256 | 报告可脱敏 | 否 | 否 | 是 | 可关联查询 | detector-run schema |
| anchor_hash | sha256 | 默认不公开逐项 | 否 | 否 | 重建 | 与per-anchor state联合可成oracle | AnchorRecord version |
| parser_status | enum | 可 | 否 | 否 | 重建 | 无 key | parser version |
| alignment_status | enum | 可 | 否 | 否 | 重建 | 无 key | sync version |
| observed_region | enum(0,1,boundary) | 可 | 否 | 否 | 重建 | public code property | region artifact |
| expected_bit | secret bit | **否** | 否 | 否 | PRF内存 | 直接 key-derived | PRF suite |
| gate_state | enum(accept,reject,erasure) | 可 | 否 | 否 | 重建 | public model result | Gate artifact |
| evidence_state | enum(match,mismatch,erasure,invalid,unavailable) | 默认只聚合 | 否 | 否 | 是 | per-anchor release may aid oracle attack | evidence policy |
| erasure_reason | enum/null | 可聚合 | 否 | 否 | 是 | 无 direct key | policy version |
| cluster_id | sha256 | 可 | 否 | 否 | 是 | 无 key | clustering version |
| cluster_weight | float | 可 | 否 | 否 | 是 | 无 key | clustering version |
| ecc_block_id | uint32 | 默认不公开逐项 | 否 | 否 | 是 | repeated keyed oracle risk | ECC version |
| ecc_position | uint8 | 默认不公开逐项 | 否 | 否 | 是 | repeated keyed oracle risk | ECC version |
| ecc_syndrome | uint8 | 默认不公开逐项 | 否 | 否 | 是 | repeated keyed oracle risk | ECC version |
| contribution_int | int16 | 默认只聚合 | 否 | 否 | 是 | keyed sign information | statistic version |
| artifact_conformance | bitset | 可 | 否 | 否 | 是 | 无 key | deployment contract |

### 17.11 ModelArtifactMetadata

| 字段 | 类型 | 公 | 训 | 嵌 | 提 | 泄漏风险 | 版本化 |
|---|---|---:|---:|---:|---:|---|---|
| artifact_id | sha256 | 是 | 固定 | 校验 | 校验 | 无 | content-addressed registry |
| protocol_id | string | 是 | 固定 | 校验 | 校验 | 无 | immutable registry key |
| schema_version | semver | 是 | 固定 | 校验 | 校验 | 无 | metadata schema |
| role | enum(region_encoder,gate,tokenizer) | 是 | 是 | 是 | 是 | 无 | artifact registry |
| weights_hash | sha256 | 是 | 校验 | 校验 | 校验 | 无 key | immutable content |
| tokenizer_hash | sha256 | 是 | 校验 | 校验 | 校验 | 无 key | immutable content |
| architecture | object | 是 | export | reference runtime | reference runtime | 无 | artifact version |
| opset | uint16 | 是 | export | reference runtime | reference runtime | 无 | runtime contract |
| operator_allowlist | list[string] | 是 | export | reference runtime | reference runtime | 无 | runtime contract |
| quantization_format | enum | 是 | QAT/export | integer inference | integer inference | public model | quant contract |
| quantization_scales | tensor/map | 是 | QAT/export | integer inference | integer inference | public model | quant contract |
| quantization_zero_points | tensor/map | 是 | QAT/export | integer inference | integer inference | public model | quant contract |
| input_shape | fixed list[uint16] | 是 | 训练模拟 | 固定 | 固定 | 无 | input contract |
| padding_rule | object | 是 | 训练模拟 | 固定 | 固定 | 无 | input contract |
| truncation_rule | object | 是 | 训练模拟 | 固定 | 固定 | 无 | input contract |
| attention_mask_rule | object | 是 | 训练模拟 | 固定 | 固定 | 无 | input contract |
| canonicalizer_hash | sha256 | 是 | 固定 | 校验 | 校验 | 无 | protocol registry |
| parser_contract_hash | sha256 | 是 | 固定 | 校验 | 校验 | 无 | protocol registry |
| feature_schema_hash | sha256 | 是 | 固定 | 校验 | 校验 | 无 | protocol registry |
| reference_backend | string | 是 | conformance | normative | normative | 限制可移植性，非secret | deployment contract |
| reference_backend_version | string | 是 | conformance | normative | normative | 无 | deployment contract |
| reference_device_contract | object | 是 | conformance | normative | normative | 限制可移植性，非secret | deployment contract |
| thresholds_int | map[string,int16] | 是 | calibration后冻结 | 使用 | 使用 | 无 key | calibration release |
| rejection_bands | map[string,(int16,int16)] | 是 | calibration后冻结 | 使用 | 使用 | 无 key | calibration release |
| conformance_vector_hash | sha256 | 是 | export gate | 启动校验 | 启动校验 | 无 | conformance suite |
| conformance_expected_results_hash | sha256 | 是 | export gate | 启动校验 | 启动校验 | 无 | conformance suite |
| training_data_card_hash | sha256 | 是 | provenance | audit | audit | 无 secret | release |
| model_card_hash | sha256 | 是 | provenance | audit | audit | 无 secret | release |
| calibration_release_hash | sha256 | 是 | 阈值来源 | audit | audit | 不发布私有代码内容 | calibration release |
| null_corpus_manifest_hash | sha256 | 是 | 阈值来源 | audit | audit | 不发布私有代码内容 | calibration release |
| compatibility_rules | object | 是 | 是 | 是 | 是 | 无 | protocol version |
| fail_closed_rules | object | 是 | 是 | 是 | 是 | 无 | protocol version |

`private_key` 永不属于任何 schema。public final artifact 默认只有 code 本身；如文件外协议参数不可由部署环境唯一确定，可附一个不含 per-unit state 的 `protocol_id`，或由 detector invocation 显式指定。缺少/未知 protocol 时 fail closed。

## 18. Train/validation/test split

### 18.1 分组与 clone graph

先构建无向关系图：同 repository/fork、同 task/spec、同 function ancestry、同 synthetic template、token MinHash 命中、AST subtree similarity 命中、identifier-normalized clone 命中、同 prompt 的多 generator outputs 均连边。connected component 是最小 split unit；repository 连接优先于函数级随机化。所有 descendants、anchors、pools、attempts、mutations、edits 和 paired views 继承 component split。

主数据按 component/repository stratified 分为：train 60%、model-validation 10%、calibration 10%、held-out test 20%。比例是初始规划，最终按 repository 数而非行数报告。calibration 只拟合 temperature/conformal bins/decision thresholds；test 只做一次锁定评估。另构造：

- leave-one-generator-family-out test；
- leave-one-domain/repository-family-out test；
- sealed HumanEval/MBPP external test；
- null-calibration \(N\ge10{,}000\) documents 与独立 null-test \(N\ge20{,}000\) documents；
- attack test，与 clean threshold selection 分离。

null documents 来自未加水印、同任务/长度/generator/quality strata 的 final code；同一 prompt/task 的 positive 与 null variants 必须同属一个 test family，防止近重复跨 calibration/test。

### 18.2 泄漏检查

在 expansion 前和最终 records 后各做一次 exact hash、normalized token MinHash、AST path fingerprint、docstring/spec embedding nearest-neighbor 和 known benchmark signature scan。任何跨 split clone 把整个 connected component移动到 quarantine，而不是只删一个 candidate。报告每种 detector 的阈值、命中数量、人工抽样 precision 和被删除 component 数；generator/version 只能用于 stratification，不作为 features。

### 18.3 Balance、weights 与 hard negatives

训练 sampler 先按 unit class × source stratum × generator family × min-side bin 分层，再在 anchor 内保留完整 paired views/pools。`y0/y1` 单侧、双侧、皆不可达均显式采样；对 rare false-accept hard negatives 采用最高 5× weight，但 validation/test 保持自然 prevalence。hard-negative mining 每轮只从 train predictions 选择，固定 mining model hash 和 round；不得从 test 错例回灌。class balance 通过 loss weights/sampler 实现，不复制同一 anchor 到不同 split。

excluded statements 的选择是明确的：规则层直接排除，不作为普通 Gate 负样本；另建 policy-OOD suite，期望 rule engine 在 Gate 前 100% fail closed。oracle-only classes 进入 feasibility 数据和独立 class head 训练研究，但在 promotion 前不参与 formal embedding denominator。

## 19. Oracle feasibility study

### 19.1 目的与阶段

Oracle study 在训练正式 Pre/Post-Gate 前执行，回答“固定 current-unit region 与候选 generator 是否产生双侧容量”。它不是用更大的模型掩盖不可达。冻结 parser、allowlist、current-unit region encoder、candidate generator 和 hard verifier，在 held-out repositories/tasks/generators 上运行 \(A\in\{6,16,32,64\}\)。

同时报告两个上界/下界：

1. `hard-valid capacity`：只计 strict parse、结构不变量、可用 tests/differential checks 全过的 candidates；unknown 不计成功，是保守下界。
2. `label-oracle formal ceiling`：假设完美 Post-Gate 能从现有 pool 选出所有 `formal_valid_label=1` candidates；这是 learned Post-Gate 的上界，不把 privileged label 部署化。

因此“Post-Gate 后有效容量”先以 label-oracle ceiling 和 public rule-verifier baseline 夹逼；若二者都低，无需训练 Gate。只有 Oracle GO 后才训练 learned Post-Gate，并在第二层检验其 FAR/容量。

### 19.2 测量协议

对每个 anchor、seed block \(r\) 和 region \(b\) 计算 pool success \(z_{tbr}\)、有效 candidate 比例、最小成功 rank、dual-reachable 指示、region imbalance、quality/function survival 和 format/rename/edit replay。对 class/generator/repository 分层给出 beta-binomial posterior、95% bootstrap/cluster CI，且同时给 micro 与 macro average。

主要指标：Oracle 双侧可达率、region-0/1 success、

\[
S_{\min}=\min(S_0,S_1),\qquad
D=\Pr(\exists\ valid\ r0\land\exists\ valid\ r1),
\]

formal evidence/1000 generated tokens、attempt-success curve、marginal gain \(A\to2A\)、格式化/rename/local-edit region+anchor replay。header 按 clause type 单独测量，绝不汇成一个平均 header 数掩盖 `else/try/finally` 的零容量。

### 19.3 Oracle go/no-go

预注册 initial engineering thresholds（test 前冻结）：

- \(A=64\) conservative simple classes：macro min-side success ≥0.35，dual-reachable anchors ≥50%；
- 每个拟 promotion header class：macro min-side ≥0.20，dual-reachable ≥30%；
- 任一 side 与另一 side 的绝对 success gap ≤0.10；
- label-oracle formal evidence density ≥4/1000 generated tokens；
- reference-runtime region replay 100%，format replay ≥99.9%；
- 功能正确率相对 no-watermark attempt-0 下降不超过 2 percentage points；
- 95% CI 下界也必须超过相应阈值，不能只看 point estimate。

若 conservative simple 总体失败，停止 Gate training，重新设计 public region 或载体，路线为第一层 NO-GO。某一 header class 失败只会维持 `ORACLE_ONLY`，不能影响 simple 路线的分母；若关闭 headers 后 evidence density 仍不足 4/1000 tokens，则系统路线仍 NO-GO。若 \(A=64\) 的增长曲线尚未饱和但成本不可接受，不能把“更多 candidates 可能成功”作为通过依据。Oracle GO 只授权小规模 Gate training，不等于系统 GO。

## 20. Model architecture

### 20.1 冻结的 public region encoder

region encoder \(E_R\) 与 Gate 分开训练、先冻结。它只接收当前 unit 的 canonical tokens、current unit type 和 current-unit-internal AST role sequence，不接收 predecessor、anchor、key、target、candidate pool 或 generator identity。一个冻结的 signed projection 输出 quantized score \(s_R(x)\in\mathbb Z\)，region 按第 23 节 margin 规则离散化。

初始模型上限为 4 层、hidden 192、4 attention heads、FFN 512、≤6M parameters；INT8 artifact ≤8 MiB。这个大小只是实现 budget，不是已测性能。formatter/quote/numeric spelling variants必须先canonicalize为完全相同bytes，只作conformance vectors而非学习信号；region encoder使用受控alpha-rename positives、非等价hard negatives、balance和margin训练。不同semantic rewrites不自动当positive，否则会抹掉carrier的region自由度。冻结其 tokenizer、weights、projection、quant scales 与 margin 后才能生成 Gate labels，防止 region 与 Gate/candidate generator 联训共谋。

### 20.2 Shared Gate encoder

Gate \(E_G\) 使用 4 层 Transformer encoder、hidden 256、4 heads、FFN 768、16k frozen code tokenizer，parameter cap 12M、INT8 artifact cap 16 MiB。输入固定为 4 个 unit slots × 96 tokens：三个 predecessors 和 current/candidate。每 slot 同时加入 unit/node/clause/parent、depth、role path、transition、header/multiline/inline flags 的 public embeddings。current canonical token 超过 96 时 policy erasure；predecessor 超长保留 canonical tail 96 tokens，并设置 public truncation flag。

训练时 shared backbone 有两条 task tokens：

- `PRE`：current 是 baseline unit，输出是否值得生成 fixed pool；
- `POST`：current 替换为某个具体 candidate，输出 blind formal-valid risk。

formal deployment 的 `context_width_formal` 从 CTX-1/2/3 中仅在 validation 预注册选择一次；默认候选是 CTX-3，随后写入 artifact。嵌入和提取必须使用同一 width；不能逐样本挑最有利 width。Pre-Gate 可为研究同时报告三 view，正式决策只使用注册 width。

### 20.3 统一、type-specific 与分模型比较

最终推荐 **shared backbone + unit-family-specific final heads/calibrators**：simple 与每个被 promotion 的 header family 共享 tokenizer/backbone，但 formal-valid、calibration threshold 和 nuisance audit 分头；未 promotion header head 永远 rule-disabled。理由是结构/语义风险可共享表示，而 header 样本稀少且 FAR/容量不同，不能用全局平均阈值掩盖。

必须在同一 split、pool 和 compute budget 下比较：完全统一 heads、推荐结构、simple/header 完全分模型。若分模型在 leave-one-generator performance、FAR 或 latency 没有显著优势，不采用；若任一 header 专用模型仍不达 promotion threshold，关闭该 class。v1 正式协议允许最终只部署 simple head，这不是删除 syntactic header 定义，而是保守 carrier policy。

### 20.4 成本预算

部署目标（均为 initial engineering thresholds）：batch-1 reference CPU Gate ≤10 ms/unit，Gate artifact 总计 ≤24 MiB，Gate 总延迟使 generation wall time 增幅 ≤20%，candidate generation token cost ≤无水印的 2×。这些必须实测；当前没有 latency 结果。

## 21. Pre-Gate outputs

对 anchor \(i\) 和注册 context view \(v\)，Pre-Gate 输出固定点值：

\[
(q_{i0}^{(v)},q_{i1}^{(v)},q_i^{both,(v)},\hat c_{i0}^{(v)},\hat c_{i1}^{(v)},p_i^{parse},p_i^{budget}).
\]

- \(q_0,q_1\)：fixed \(A\) 下各 region 至少一个 formal-valid candidate 的校准概率；
- \(q^{both}\)：同 anchor 在 counterfactual two-side sense 可达的概率；
- \(\hat c_0,\hat c_1\)：各侧 first-success cost，归一化到 \([0,1]\)；
- \(p^{parse}\)：provisional prefix 最终 strict-parse 的辅助概率；
- \(p^{budget}\)：给定 public cost cap 后值得分配预算的辅助输出。

部署分数不是 `max(q0,q1)`，而是 calibration lower bounds 的

\[
q_{safe,i}=\min(q^-_{i0},q^-_{i1}),\qquad c_i^+=\max(\hat c_{i0},\hat c_{i1}),
\]

\[
G_i^{pre}=\mathbf1[q_{safe,i}\ge\tau_{pre}\land c_i^+\le C_{max}].
\]

正式v1的baseline已strict parse，所以 \(p^{parse}\) 只作incremental/header Oracle辅助输出，不进入simple-unit部署决策。初始 \(\tau_{pre}=0.65,C_{max}=0.75\)，只在calibration前的validation utility上选定。Pre-Gate不读取actual pool、不选择bit、不产生证据。若rule-ineligible或reject，planner写固定skip sentinel且不开pool；该unit仍占structural ordinal，但不计正式candidate denominator。统计报告同时给出“all syntactic”“policy eligible”“Pre-Gate selected”三个分母，禁止选择性呈现容量。

## 22. Post-Gate outputs

Post-Gate 对 final candidate 与相同 public context 输出整数 logits/概率：

| Head | 意义 | formal decision 用法 |
|---|---|---|
| `p_syntax`, `p_structure` | strict syntax/span/binding/effect invariants | hard verifier 先否决；head 只作冗余审计 |
| `p_semantic_proxy` | 仅凭final observable预测offline pairwise-valid的风险 | formal-valid必要风险过滤；不证明相对baseline等价 |
| `p_quality` | 可读性/局部质量不退化 | 低于 reject threshold 则 erasure |
| `margin_region` | public region integer distance | 必须越过 region rejection band |
| `p_replay` | reference/conformance 离散重放概率 | 低置信 erasure |
| `p_robust` | format/rename/local-edit survival | 排序与辅助报告；不证明攻击鲁棒 |
| `p_redundancy` | 与邻近证据的相关/重复风险 | 用于 cluster/downweight，不得提高单元分 |
| `p_formal` | 综合 final formal validity | asymmetric threshold 决定 accept/erasure/reject |

formal state 取 hard validator 与 neural state 的交：`ACCEPT` 需要所有 hard checks、`p_formal` integer logit ≥\(T_A\)、quality/replay thresholds 和 region margin；logit ≤\(T_R\) 或 hard violation 为 `REJECT`，中间或任一不确定为 `ERASURE`。`REJECT` 在 final evidence 中通常形成 erasure；只有 unit 被 ACCEPT 且 observed region 可判时才与 expected bit 比较产生 match/mismatch。这样 Post-Gate 假阳性是高代价，假阴性只损失容量。

训练输入与部署输入一致：final code 可重建的 canonical current/context 和 public structure/hard-verifier features。project tests、baseline/candidate pair、execution traces、candidate alternatives、attempt/seed、human labels、generator/repository ID、original prompt 与 key 都只是监督/分层或完全不可用。因而 Post-Gate false accept的严格定义是“blind ACCEPT但offline (F_{pair}=0\)”；这项held-out FAR是系统能否采用该proxy的硬门槛。

## 23. Loss functions with formulas

### 23.1 双侧 soft labels

固定 pool 中 candidate \(x_{ijr}\) 对 region \(b\) 的 offline validity：

\[
V_{ijr,b}=H_{ijr}F_{ijr}\mathbf1[R(x_{ijr})=b],\quad
Y_{ibr}=\mathbf1[\sum_{j=1}^{A}V_{ijr,b}>0].
\]

对 \(R=4\) independent seed blocks，\(S_{ib}=\sum_rY_{ibr}\)。Jeffreys prior 的 Beta-Binomial posterior mean：

\[
\tilde y_{ib}=\frac{S_{ib}+1/2}{R+1}.
\]

同时保存 \(y_i^{both}=\mathbf1[S_{i0}>0\land S_{i1}>0]\) 与 first-success cost

\[
c_{ibr}=\frac{\min\{j:V_{ijr,b}=1\}\wedge(A+1)}{A+1},\qquad
\tilde c_{ib}=R^{-1}\sum_rc_{ibr}.
\]

若只能 \(R=1\)，必须标 `single_pool` low-precision，不把一次 Bernoulli 当真概率。

### 23.2 Pre-Gate losses

\[
\mathcal L_{region}=\frac1{3N}\sum_{i,v,b}\operatorname{BCE}(q_{ib}^{(v)},\tilde y_{ib}),
\]

\[
\mathcal L_{both}=\frac1{3N}\sum_{i,v}\operatorname{BCE}(q_i^{both,(v)},y_i^{both}),
\quad
\mathcal L_{cost}=\frac1{6N}\sum_{i,v,b}\operatorname{Huber}(\hat c_{ib}^{(v)}-\tilde c_{ib};0.1).
\]

不强制 \(q_{safe}(CTX3)\ge q_{safe}(CTX1)\)：额外 context 可能揭露副作用，使可行性下降。使用容差一致性：

\[
\mathcal L_{ctx}=\frac1N\sum_{i,b,v=1}^{2}\omega_{ibv}
[|q_{ib}^{(v+1)}-q_{ib}^{(v)}|-0.15]_+^2,
\]

其中只有两预测离 threshold ≥0.1 且方向一致时 (omega=1)；边界样本不被错误拉平。

### 23.3 Post-Gate asymmetric、ranking 与 margin losses

辅助任务集合 \(\mathcal M=\{syn,str,sem,qual,replay,rob,redund\}\)：

\[
\mathcal L_{aux}=|\mathcal M|^{-1}\sum_{m\in\mathcal M}\operatorname{WBCE}(p^m,y^m),
\]

每项prevalence weight截断在 \([1/5,5]\)。所有`unknown` targets在对应loss中mask掉并单独报告coverage，绝不自动当negative。formal-valid使用false-accept-sensitive BCE：

\[
\mathcal L_{formal}^{FA}=-M^{-1}\sum_k
[y_k^F\log p_k^F+20(1-y_k^F)\log(1-p_k^F)].
\]

同 anchor 内 \(u_k=p_k^F+0.25p_k^{rob}+0.25p_k^{replay}\)，有效/无效 pair：

\[
\mathcal L_{rank}=\mathbb E\log(1+\exp[-(u_{k^+}-u_{k^-})]).
\]

对 integer accept/reject logits \(T_R<T_A\) 设 rejection band：

\[
\mathcal L_{decision}=\mathbb E\begin{cases}
[m_F-(\ell_F-T_A)]_+^2,&y^F=1,\\
[m_F+(\ell_F-T_R)]_+^2,&y^F=0,
\end{cases}
\]

初始 \(m_F\) 对应校准概率距离 0.05；靠边样本在部署归 erasure。

### 23.4 Calibration、batch、quantization 与 nuisance losses

\[
\mathcal L_{cal}=\operatorname{Brier}(p,y)+0.5\operatorname{SoftECE}_{15}(p,y).
\]

对 \(T_R<T_A\) 定义“保持reference state \(s\)”的margin：

\[
\mu(\ell;s)=\begin{cases}
\ell-T_A,&s=ACCEPT,\\
T_R-\ell,&s=REJECT,\\
\min(\ell-T_R,T_A-\ell),&s=ERASURE.
\end{cases}
\]

同一样本的batch-1 reference state \(s_*\) 与随机companions/padding变体：

\[
\mathcal L_{batch}=\mathbb E\|\ell_*-\ell_{B,L}\|_2^2+
5\,[m_D-\mu(\ell_{B,L};s_*)]_+^2\mathbf1[\mu(\ell_*;s_*)\ge m_D].
\]

reference自身在 \(m_D\) 内则监督为erasure，不强迫一个不稳定bit。该定义逐head使用；region score把 \(T_R=-m_R,T_A=m_R\) 代入，middle state即boundary erasure。QAT使用最终QDQ graph fake quantization：

\[
\mathcal L_{quant}=\mathbb E\|\ell_{fp32}-\ell_{fakeINT8}\|_2^2+
5\,[m_D-\mu(\ell_{fakeINT8};s_{fp})]_+^2
\mathbf1[\mu(\ell_{fp32};s_{fp})\ge m_D].
\]

shared representation \(h\) 接 gradient-reversal heads 预测 generator/repository family：

\[
\mathcal L_{nuis}=CE(d_{gen}(GRL(h)),g)+CE(d_{repo}(GRL(h)),r).
\]

它只降低 shortcut pressure，不替代 leave-one-group-out 测试；attempt、pool size、target/key/retry 根本不进入 input graph。

### 23.5 Region encoder loss

region encoder 在 Gate dataset 之前独立训练：

\[
\mathcal L_E=\mathcal L_{triplet}+0.5\mathcal L_{invariance}
+0.2\mathcal L_{balance}+0.2\mathcal L_{QAT-E}+0.1\mathcal L_{region-margin},
\]

\[
\mathcal L_{balance}=\sum_u\left(\mathbb E_{x\mid u}\sigma(s_R(x)/T)-0.5\right)^2.
\]

balance 只在 train 优化；test 必须报告 per-unit-class natural imbalance。Oracle 失败后不得与 Gate 联训移动 partition 来制造可达率。

### 23.6 最终总损失、权重和 cost-aware selection

\[
\begin{aligned}
\mathcal L_{pre}={}&1.0L_{region}+0.5L_{both}+0.25L_{cost}+0.10L_{ctx}\\
&+0.10L_{batch}+0.25L_{cal}+0.20L_{nuis},\\
\mathcal L_{post}={}&2.0L_{formal}^{FA}+0.5L_{aux}+0.5L_{rank}+0.25L_{decision}\\
&+0.10L_{batch}+0.10L_{quant}+0.25L_{cal}+0.20L_{nuis},\\
\boxed{\mathcal L_{total}={}&\mathcal L_{pre}+\mathcal L_{post}.}
\end{aligned}
\]

这是 shared Gate 的 joint supervised multi-task training，不是 generator-to-detector 的不可微“纯端到端”。初始权重只在 validation 做有限网格 \(\{0,0.5,1,2\}\) 和 Pareto selection，首要 hard constraint 是 Post-Gate FAR；选定后冻结，不在 test 动态调权，也不默认用 GradNorm。

validation model selection 的 cost-aware utility 为：

\[
U(\pi)=\frac{\mathbb E[N_{formal,match}-20N_{mismatch}]}
{\mathbb E[C_{gen}+\lambda_g C_{gate}]},
\]

且必须同时满足 functionality、FAR、calibration、replay、FPR 和 evidence-density hard constraints；高 \(U\) 不能补偿违反任何 hard constraint。RL 的 action 会是 candidate-or-skip，reward 是 \(U\) 的噪声代理，而 sampling、parser、execution 与 region threshold 不可微，且 generator-style reward hacking 风险高。因此 v1 不使用 RL；只有 supervised baseline、Oracle 与 system tests 都通过后，contextual bandit 才可作为隔离 ablation，不能替代 validators。

简单 baseline：

\[
\mathcal L_{base}=\mathcal L_{region}+BCE(p^F,y^F),
\]

没有双侧 head、calibration、ranking、context/batch consistency、QAT、nuisance 或 auxiliary heads，用于辨认复杂目标的真实增益。

## 24. Calibration strategy

### 24.1 Pre-Gate conservative lower bounds

先在 model-validation 选择 temperature scaling 或 isotonic mapping（只按 Brier/NLL 的预注册规则二选一），再在独立 calibration split 按 unit family × generator-held-in/out × repository-domain 形成 20 个 equal-mass bins。每个 calibration anchor使用一个**不参与训练label的独立deployment-seed pool**，产生anchor-level Bernoulli success；bin 有 \(n\) anchors、\(s\) successes时，region-specific Clopper–Pearson lower bound：

\[
LCB_{CP}(s,n;\delta)=\begin{cases}
0,&s=0,\\
\operatorname{Beta}^{-1}(\delta;s,n-s+1),&s>0.
\end{cases}
\]

因为同repository anchors仍相关，同时做repository/task connected-component cluster bootstrap；最终lower bound取 `min(CP LCB, cluster-bootstrap 5th percentile)`。样本 \(n<200\) 或独立repositories<20 时按预注册层次回退到更粗 unit family，再不足回退全局；仍不足则 lower bound=0。\(q^-_{ib}\) 是样本所在 bin 的保守LCB，`q_safe=min(q0-,q1-)`。报告 ECE、Brier、reliability diagram、worst-group calibration 和 coverage；initial success criterion ECE ≤0.05 on unseen repository/task/generator。

### 24.2 Post-Gate risk control

在 calibration hard negatives 上，对每个 deployable unit-family head 枚举 integer \(T_A,T_R\)。候选相关性先按anchor取“该anchor是否至少一次false accept”，再计算 one-sided 95% CP upper bound和repository/task cluster-bootstrap 95th percentile；FAR UCB取二者较大值。选择满足UCB≤0.5%的最高-coverage threshold；candidate-level point target仍≤0.2%。每个 family 至少1,000 negative anchors、20 repositories且任一repo占比≤10%，稀疏 family 不 promotion。\(T_R<T_A\) 中间是 erasure；不对 unknown labels 当 negative 或 positive，而是单独报告 coverage。

temperature/isotonic mapping、integer thresholds、region margin、quality/replay thresholds 和 fallback hierarchy 全部写进 `protocol_id`。test 只应用一次；test failure 不可回头改 calibration。conformal risk control 可作为替代 calibration baseline，但若选择，nonconformity score、exchangeability strata 和 finite-sample bound 必须预注册；不能与 CP 方法在 test 上择优。

### 24.3 Selection-aware calibration

Gate-selected anchors 的 prevalence 是 data-dependent；因此 calibration/test 同时报告 all-eligible 与 selected conditional performance，并在 null detection 中重放 rule eligibility、Post-Gate、erasure 和 alignment。Pre-Gate 本身在 extraction 不决定 evidence——final Post-Gate 可接受任何 rule-eligible final unit——避免 detector 需要 generation selection trace；但 null corpus 必须采用最终同一盲 evidence rule，不能只在嵌入成功 rows 上校准。

## 25. Quantization-aware and batch-consistency training

### 25.1 Normative deterministic contract

神经 continuous vectors 无跨硬件逐元素相同承诺；规范只要求离散 `region / Post-Gate state / evidence state` 在注册 contract 中重放。normative v1 contract 是：

- CPython 3.12 patch version、LibCST/tokenize versions 与 canonicalizer hashes 固定；
- tokenizer vocab/normalization/hash 固定；未知 token、UTF-8、newline、tab 和 truncation rule 固定；
- tensor shape 固定为 `[batch=1, units=4, tokens=96]`，masked slots 仍为固定 BOS bytes；
- no BatchNorm；只用 LayerNorm/RMSNorm；inference dropout=0，`eval` mode；
- static INT8 QDQ region/Gate artifacts、quant scales/zero-points、operator allowlist、opset 固定；
- exact container image、ONNX Runtime CPU build、x86-64 AVX2 feature floor、single-thread settings 和 reference device class 固定；
- integer projection/logits/threshold comparison；NaN/overflow/unsupported op 一律 fail closed；
- artifact、runtime、operator 或 conformance hash 不匹配即 `UNAVAILABLE_ARTIFACT`，不降级到近似 FP 模型。

PyTorch 官方文档明确不保证不同 release/platform 或 CPU/GPU 完全可复现，batch 与逐样本计算也可能不同；因此 formal correctness 不能靠“固定 seed”。[PyTorch numerical accuracy](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html) [PyTorch reproducibility](https://docs.pytorch.org/docs/stable/notes/randomness.html)

### 25.2 QAT 的边界

QAT 用与 export 同一 QDQ graph 的 fake quantization、threshold-distance margin 和 float/INT8 consistency loss，目标是减少 decision flips，而非证明跨 backend bit-exact。量化本身是 lossy transform，且 graph/operator/hardware 会改变结果；只有 reference artifact + runtime + conformance vectors 构成证明义务。[ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)

每次发布至少 10,000 个无 secret conformance vectors，覆盖边界、长度、unit class、Unicode、padding、hard negatives；包含 expected canonical bytes、integer region score、all Post logits 和 discrete states。reference backend 必须 100% exact；任何 mismatch 阻止启动。另一个 backend 只有在 qualification set 0 flips 且 stress-set flip rate ≤\(10^{-4}\)、所有 flips 都落入 expanded erasure band 时才标 `CONFORMANT_NONNORMATIVE`；否则只可诊断，不能出 formal evidence。

### 25.3 必测矩阵

| 轴 | 水平 | 比较 | formal acceptance |
|---|---|---|---|
| batch size | 1,2,4,8 | 同样本独立 vs 混合 companions | normative B1 100%；batched backend 需上述 conformance |
| batch composition | all-short/all-long/mixed/adversarial boundary | integer logits 与 state | state flip 超标即不合格 |
| padding/shape | effective 64/128/256/384 tokens；固定与动态 export | mask、position、state | formal 只接受固定 384-token tensor |
| precision | FP32, FP16, BF16, INT8-QDQ | continuous delta 与 discrete flip 分开 | INT8 reference 100%；其余诊断/资格化 |
| device | reference CPU、第二 CPU、GPU | score/state | 未资格化设备不得 formal detect |
| process | 两次 clean process、两 container replicas | bytes/logits/state | reference outputs byte-identical |
| mode | single sample 与 mixed batch | state | 见 batch criterion |

训练时每个 minibatch 随机生成 companions 和 padding 变体，并周期性用真正 exported INT8 graph 做 holdout replay；只用 fake-QAT 不足。指标分别是 continuous max/mean error、batch-composition flip、quantization flip、cross-process replay、cross-device replay 和 erasure migration。靠近任一 region/Gate threshold 的样本扩大 rejection band；不能靠四舍五入强判 0/1。

若无法在 reference CPU/container 达到 100% discrete replay，本路线在 System 层 NO-GO。限制到固定 CPU backend 降低 portability，但仍可用于受控 provider-side detector；它不能被表述为任意 GPU/客户端可移植的盲检测。

## 26. Watermark embedding protocol

### 26.1 Target stream

document generation session在baseline/pool之前通过与key模块隔离的CSPRNG均匀采样并冻结public 128-bit `document_nonce`，所有anchor都包含它，且`scope_anchor=SHA256(protocol_id || document_nonce || scope_fp_path || structural_instance_path)`。从 master key 经 HKDF-SHA256 和 `protocol_id` 派生互异子钥 `K_msg` 与 `K_mask`。每个 suite-local 7-slot block 的 4-bit check message：

\[
m_s=\operatorname{Trunc}_4 HMAC_{K_{msg}}(d_M\|scope\_anchor\|block\_id),
\]

经第 30 节 Hamming(7,4) 得 codeword \(c_{s,0:6}\)。slot \(j\) 的正式 target：

\[
b_t=c_{s,j}\oplus\operatorname{LSB}HMAC_{K_{mask}}(d_B\|anchor_t).
\]

若关闭 ECC baseline，则 \(c_{s,j}=0\)，仍由 per-anchor HMAC mask 产生 pseudorandom target。所有 domain separators 是 length-prefixed ASCII constants；anchor 不含 current candidate text。

### 26.2 嵌入伪代码

```text
PLAN_AND_EMBED(prompt, public_protocol P, sealed_key_handle K):
  assert all P artifact/runtime hashes and conformance vectors pass
  nonce = PUBLIC_CSPRNG_128_ONCE()               # K handle is still sealed
  D0 = GENERATE_BASELINE_WITHOUT_KEY(prompt, P.baseline_rng)
  T0 = STRICT_PARSE_AND_LINEARIZE(D0, nonce)
  if T0 fails or anchors collide: emit_with_metadata(D0); return DOCUMENT_FALLBACK

  plan = []
  for slot in T0.all_syntactic_slots:            # source order, frozen D0
    if slot.origin != GENERATED or
       public_eligibility(slot.unit) != FORMAL_ELIGIBLE:
        plan.append(slot, FIXED_SKIP_SENTINEL); continue
    view0 = BUILD_BASELINE_CONTEXT(T0, slot, P.context_width_formal)
    q = PRE_GATE_INT8(view0)
    if conservative_calibrated_decision(q) == REJECT:
        plan.append(slot, FIXED_PRE_SKIP_SENTINEL); continue

    pool = []
    for j in 0 .. P.attempt_budget-1:             # exactly A attempts
        xj = GENERATE_CURRENT_ONLY_FROM_D0_PREFIX(
                 P.generator, D0, slot, P.public_seed(nonce,slot,j))
        pool.append(xj)                            # K/target module inaccessible
    scored = []
    for xj in pool:                                # every entry, including failures
        hard = PUBLIC_BLIND_HARD_VERIFY_CURRENT(xj, slot)
        pair = KEY_INDEPENDENT_PAIR_VERIFY(D0, slot, xj,
                                           registered_tests_or_certificates)
        region = REGION_INT8_CURRENT_ONLY(xj)
        post0 = POST_GATE_INT8(BUILD_BASELINE_CONTEXT(T0, slot, xj))
        scored.append(hard, pair, region, post0, quality, canonical_bytes)
    plan.append(slot, COMMITTABLE_POOL(scored))

  doc_commitment = HASH_PROTOCOL_NONCE_AND_ORDERED_PLAN(P, nonce, plan)
  assert AUDIT_KEY_TAINT_FREE(plan, doc_commitment)
  UNLOCK_TARGET_ONLY_INTERFACE(K, doc_commitment)  # first permitted K access

  replacements = []
  for slot, entry in plan:
    if entry is a skip sentinel: continue
    target = ECC_MASKED_TARGET(K, nonce, slot.anchor, slot.block, slot.position)
    admissible = [x for x in entry
                  if x.hard_pass and x.pair == PASS and x.region == target
                  and x.region_margin >= P.region_margin
                  and x.post0_state == ACCEPT]
    if admissible is not empty:
        replacements.append(slot, min(admissible, key=P.public_total_order))
    # empty target side means retain D0 current unit; never add attempts

  D1 = APPLY_NONOVERLAPPING_SIMPLE_UNIT_REPLACEMENTS(D0, replacements)
  final = STRICT_FINAL_BLIND_REPLAY(D1, P, nonce, K)
  if final.parse/anchor/functionality qualification fails:
      emit_with_metadata(D0, P, nonce)             # no retry/resample/replan
      return INTERNAL_DOCUMENT_FALLBACK
  emit_with_metadata(D1, P, nonce)
  return INTERNAL_PLAN_RESULT                    # audit-only, not detector input
```

pairwise verifier可使用baseline、rewrite certificate和注册tests，但必须对全部document pools在target前运行；unknown不准入。它是生成质量保护，不进入detector。Phase P1的Post-Gate使用frozen baseline context作selection prefilter；D1的final context可能不同，因此只有final blind replay有formal authority，差异变erasure而不重试。selection的唯一key-dependent operation是从已封存、已评分pool中选择expected region；空侧保留D0 current unit。最终组合功能失败时整文档退回D0但保留公开metadata，不能换nonce/key/seed或再搜索。

v1 replacement集合只含非重叠simple-unit spans；所有headers保持Oracle-only，所以不会在body已生成后回写header。该two-pass协议仍满足“retry只改当前unit”，但生成成本和interaction erasure必须在Layer-3实测。

### 26.3 Finalization 与 artifact

生成结束后对**完整 final code**重新 strict parse/linearize/Post-Gate/region。header 只有 suite ownership、body 和整个 document 都合法才从 provisional 转为 final replay state；生成日志中的 “success” 不升级 formal evidence。final artifact默认只写final code。self-describing模式在可选shebang/PEP-263 encoding cookie之后按固定次序写两行：`# sage-c-protocol: sha256=<64 lowercase hex>`与`# sage-c-document: nonce128=<32 lowercase hex>`。二者不是unit；nonce在生成开始前均匀采样、不含key。否则detector API必须显式接收protocol ID与document nonce。model/tokenizer/parser hashes和thresholds由registry ID恢复；metadata缺失、重复、malformed或与API参数冲突均fail closed。不写per-unit anchor、candidate count、target、selected ID、retry、Gate score、pool commitment或key。

嵌入端的 `INTERNAL_PRE_SKIP/RULE_SKIP` 不是盲提取可见状态。最终 detector 会对所有 public-rule-eligible final units 应用同一 Post-Gate；因此未尝试的 baseline 偶然形成 match/mismatch/erasure 是 null-like evidence，必须包含在 calibration。报告中的 embedded-unit rate 以实际 target-selected positions 为分母，detector evidence density 则只由 final replay定义，二者不混用。

## 27. Watermark extraction protocol

### 27.1 输入与伪代码

提取器唯一输入是 `(final_code, protocol_id, document_nonce, public registry/artifacts, K)`；protocol ID与nonce可由上述leading comments恢复，不能来自生成日志：

```text
EXTRACT(final_code D, protocol_id, document_nonce, public registry, secret K):
  if protocol unknown, artifact hash mismatch, runtime not conformant,
     or conformance vectors fail: return UNAVAILABLE_DOCUMENT
  T = CPYTHON_STRICT_PARSE(D)
  if T fails: return INVALID_DOCUMENT
  units = CST_TOKEN_LINEARIZE(T, D)              # semicolon/inline/multiline included
  hypotheses = BOUNDED_SUITE_RESYNC(units, P)    # all units, excluded included

  for each alignment hypothesis h:
    evidence_h = []
    for each reconstructed syntactic slot t in h:
      if public_eligibility(t.unit) != FORMAL_ELIGIBLE:
          evidence_h.append(ERASURE_RULE); continue
      view = BUILD_FIXED_FINAL_CONTEXT(units, t, P.context_width_formal)
      hard = PUBLIC_BLIND_HARD_VERIFY_FINAL(t.unit, t.structure)
      post = POST_GATE_INT8(view with current=t.unit)
      region = REGION_INT8_CURRENT_ONLY(t.unit)
      if not hard or post != ACCEPT or region is boundary:
          evidence_h.append(ERASURE_WITH_REASON); continue
      target = ECC_MASKED_TARGET(K, t.anchor, t.block, t.position)
      evidence_h.append(MATCH if region == target else MISMATCH)
    score_h = CLUSTER_AND_SCORE(evidence_h) - P.edit_penalty * h.edits

  h_star = public_tie_break(argmax_h score_h)
  if top alignments are ambiguous under P.ambiguity_rule:
      erase affected blocks and recompute
  primary = EMPIRICAL_NULL_PVALUE(score_h_star, P, K)
  ecc = ERASURE_AWARE_ECC_REPORT(evidence_h_star, K)
  return DETECTION_RESULT(primary, ecc, counts, effective_n, failure_states)
```

Pre-Gate 不在 extraction 路径：它只控制生成预算，且其 generation-time decision 未写入 final code。Post-Gate、blind hard verifier 和 region 均盲重放；Post-Gate context 来自 final stream，同一注册 width。pairwise semantic verifier明确不重放，因为baseline/tests不在final artifact；其风险由blind proxy的held-out FAR和系统功能评估约束。提取端绝不读取 repository/task/generator ID、prompt、logits、hidden states、candidates、attempts、generation audit 或 sidecars。

### 27.2 解析对偶与失败路径

提取端使用与嵌入 finalization 相同的 parser conformance corpus，明确识别 individual semicolon small statements、inline suite bodies、multi-line headers、async variants、decorators-as-auxiliary、`match/case`、`except*` 及 owner-specific `else`。最终 parser 没有 token recovery fallback；strict parse failure 是 document-level invalid，不猜 bit。

不存在 protocol registry、输入过长、未知 Python feature、unsupported Unicode/token、runtime mismatch、integer overflow、alignment 超预算或 Post-Gate/region boundary 时都 fail closed 为 unavailable/erasure。不会静默换相近 tokenizer/model，也不会用 continuous tolerance 比较两个 embedding vectors。

## 28. Match/mismatch/erasure semantics

每个 syntactic slot 先有 diagnostic state，再归并到 primary statistic：

| 原始状态 | 条件 | Primary contribution |
|---|---|---:|
| `MATCH` | rule eligible、alignment unique、hard/Post accept、region stable且等于 target | +1 |
| `MISMATCH` | 同上，但 stable region 与 target相反 | -1 |
| `ERASURE` | rule excluded/oracle-only、region/Post boundary、低质量、低 replay、unknown semantics、edit ambiguity | 0 |
| `INVALID_UNIT` | span/structure/hard invariant 失败但 document可解析 | 0；记录原因 |
| `UNAVAILABLE_UNIT` | artifact/input limit/operator failure | 0；若系统性则 document unavailable |
| `INVALID_DOCUMENT` | final strict parse失败 | 不做检测 |
| `UNAVAILABLE_DOCUMENT` | protocol/key/artifact/runtime不兼容 | 不做检测 |

只有一个**已被 blind Post-Gate 接受**的稳定 opposite region 才是 mismatch；候选耗尽、generation skip 或低置信都不能伪装成 mismatch。mismatch 的负权防止只保留方便 matches；erasure 不支持也不反对水印。`INSUFFICIENT_EVIDENCE` 是聚合输出而非 negative：非 erasure <16、clusters <8 或 \(n_{eff}<12\) 时禁止 positive。

`formal evidence` 在这里严格表示“最终代码中可盲重放的keyed watermark evidence”，不表示提取器证明了该unit相对生成前版本语义等价。pairwise semantics只约束embedding admission和训练标签；若blind proxy仍接受pairwise-invalid代码，就是Post-Gate false accept，并由第24/40节硬门槛否决方案。

公开报告默认只给 counts、aggregate statistic/p-value、ECC block summary 和 failure histogram；per-anchor expected bit/contribution 不发布，以降低 chosen-query detector oracle 风险。

## 29. Evidence aggregation

### 29.1 Block 与 cluster

先按 7 个 all-syntactic slots 形成 suite-local ECC/sync blocks；excluded、oracle-only 和不存在的 virtual slots是 erasure，不能压缩后重编号。再用**公开、key-independent**规则形成 dependence clusters：同 enclosing lexical scope、同 nearest compound clause owner、source-order gap ≤3 的连续 accepted evidence 为一 cluster；相同 `current_text_identity` 的复制品无论位置，只让最早一项有权重，其余 erasure/duplicate。cluster 边界不能看 match/mismatch sign。

令 \(e_i\in\{-1,0,+1\}\)，cluster \(c\) 的非 erasure 数 \(n_c\) 和均值：

\[
n_c=\sum_{i\in c}\mathbf1[e_i\ne0],\qquad
G_c=\frac{\sum_{i\in c}e_i}{\max(1,n_c)}\in[-1,1].
\]

主 document statistic：

\[
T(D)=\frac{\sum_{c=1}^{C}G_c}{\sqrt C},\qquad
n_{eff}=\frac{(\sum_cn_c)^2}{\sum_cn_c^2}.
\]

cluster mean 让一个复杂 clause 的大量相似 units 不能线性放大证据。\(n_{eff}\) 只是报告/最低证据 guard，不被解释为严格独立 Bernoulli 数。HMAC targets 近似 pseudorandom 也不使 code units、Gate selection、ECC slots 或 alignment hypotheses 独立；尾概率完全依赖第 31 节 empirical null。

### 29.2 层级输出

每个 block 报 `match/mismatch/erasure` counts 与 `ecc_verified`；每个 scope 报 cluster statistic；document 只允许一个预注册 primary \(T^*\)。多文件 project 的默认输出是逐文件并做 Holm family-wise correction；若要 project-level pooling，必须预注册 key-independent file clusters、长度 normalization 和 project-null calibration，不能事后挑分数最高函数/文件。

初始 evidence guard：accepted non-erasure ≥16、clusters ≥8、\(n_{eff}\ge12\)。未达到只返回 `INSUFFICIENT_EVIDENCE`，不纳入 positive。报告 unconditional detection coverage、在 sufficient documents 上的 conditional TPR 以及 overall TPR，避免通过大量 abstention 美化性能。

## 30. Erasure-aware ECC

SAGE-C 不承载外部 payload。ECC 编码的是 detector 可从 key 重新生成的 4-bit check message，不是用户消息。systematic Hamming(7,4,3)：

\[
p_1=d_1\oplus d_2\oplus d_4,\quad
p_2=d_1\oplus d_3\oplus d_4,\quad
p_3=d_2\oplus d_3\oplus d_4,
\]

\[
c=(p_1,p_2,d_1,p_3,d_2,d_3,d_4).
\]

all-syntactic virtual slot \(j\) 的 embedded target 是 \(b_j=c_j\oplus HMAC_{K_{mask}}(anchor_j)_{LSB}\)。提取时对 stable observed region 先 XOR 同一 mask，erasure 保留未知，再 syndrome decode。只有满足 minimum-distance condition

\[
2e+s<3
\]

且解出的 data bits 等于重新派生 \(m_s\) 时，block 才 `ecc_verified=true`；少于 5 个 non-erasure bits 必定 fail closed。由于 v1 有大量 rule-excluded slots，许多 block 可能无法解码，这正是必须实测的限制，不能跳过 erasures 紧凑重排。

ECC 是 secondary robustness/quality diagnostic。primary statistic 仍使用所有单元的 masked match/mismatch；不能以 `primary positive OR ECC verified` 合并两个未经联合校准的 tests。若实验选择 stronger shortened BCH/LDPC 或把 ECC flag 加入 primary score，必须作为第 39 节强度 ablation，在完整 null 上重新注册 joint statistic/FPR。Hamming 是简单 baseline，不声称最优。

## 31. Detector statistic and FPR calibration

### 31.1 Alignment-aware primary statistic

每个 suite 的 bounded edit DP 产生预注册假设集合 \(\mathcal H(D)\)，默认每 7-slot block最多 2 insertions/deletions。对假设 \(h\)：

\[
T_h(D)=\frac{\sum_cG_{c,h}}{\sqrt{C_h}},\qquad
T^*(D)=\max_{h\in\mathcal H(D)}[T_h(D)-0.25\,edits(h)].
\]

最大化使用 key-dependent match signs，构成 multiple/data-dependent selection，经典二项/正态 tail 不适用。alignment 数、tie handling、edit budget 和 penalty 都是 protocol parameters；若假设数超过 cap 10,000 或同分歧义影响 block，相关 block erasure。

### 31.2 Empirical null

从协议注册的independent audit-key distribution采样 \(K^0_1,\ldots,K^0_{N_0}\)，并与训练、model-validation、positive test 全部分离的matched null documents \(D^0_1,\ldots,D^0_{N_0}\) 一一配对。对每对执行**完全相同**的 parser、eligibility、Post-Gate、region boundary、erasure、duplicate clustering、resync max、minimum-evidence guard 和（若进入 statistic）ECC，形成公开冻结的null-score artifact \(\mathcal S^0_P=\{T^*_{K_j^0}(D_j^0)\}\)：

\[
\hat p_P(D,K)=\frac{1+\sum_{j=1}^{N_0}\mathbf1[T^*_{K_j^0}(D_j^0)\ge T^*_K(D)]}{N_0+1}.
\]

初始 \(N_0\ge10{,}000\)，\(\alpha=1\%\)；统一公开阈值 \(\tau_{P,\alpha}\) 取该artifact的有限样本 upper-tail order statistic。另用≥20,000个held-out `(null document, independent key)` pairs一次性测经验FPR与one-sided Clopper–Pearson 95% CI，要求point≤1%、upper≤1.25%。protocol development跨至少128个audit keys检查key-stratified dispersion；production key必须均匀随机生成，并可在不改阈值的独立null corpus上做predeployment qualification。若actual-key audit异常，只能拒绝/轮换key或重注册协议，不能产生私有key-specific threshold。

null corpus必须匹配repository/task domain、generator family/version、长度、parse/quality、unit-type/evidence-count strata，并包含未水印baseline、generation error和code edits。每个null也按正式格式获得独立random nonce与protocol metadata；metadata存在本身不作为label或statistic。threshold不使用positive test。Gate selection的影响通过final-code blind Post-Gate/erasure完整重放，而不是假设selected evidence独立；这也是对data-dependent stopping的答案。

正式FPR claim是对“按协议均匀生成的key (K\) 与matched null document (D^0\)”联合分布的marginal guarantee；有限样本不能保证任一人为选择的固定key都有同一条件FPR。生产key必须随机生成并通过不调阈值的qualification audit；若业务要求逐key finite-sample guarantee，则需要把可公开重放的key-specific calibration proof纳入artifact，v1不提供。

### 31.3 判决与多重测试

只有 `sufficient=true` 且 \(\hat p_P(D,K)\le0.01\) 才是 primary positive；否则分别返回 negative 或 insufficient。报告 detection AUROC、TPR@1% FPR、经验 FPR/CI、coverage 和 score distributions。一个 document/protocol/key 是一个 primary test。扫描多个 keys、protocol versions、functions 或 files 时，预注册 Holm correction；如果选择最大 window/project subset，则把该 max 原样加入 null statistic。绝不发布“尝试多个方案后的最小 p-value”。

clean null FPR 是主 claim；另用 public-region-aware rewrite、dead-code、alignment search 等 adversarial null 做 false-positive stress，任何显著超标触发 NO-GO 或另设保守 threshold。positive edit attacks只测 survival/TPR，不能用来调 clean threshold。

## 32. Deterministic deployment contract

### 32.1 Protocol manifest

一个可接受的 `protocol_id` 唯一绑定：Python grammar/patch、CPython AST/LibCST/tokenizer/canonicalizer/linearizer、eligibility policy、anchor/sync、context width、region/Gate artifacts、QDQ graph/opset/runtime/container、input shape、integer thresholds/rejection bands、candidate generator/pool config（嵌入端）、HMAC/HKDF domains、ECC、cluster/statistic/null release 与 conformance vectors。registry entry 是 canonical JSON，ID 是其 SHA-256；任何 normative field 改变产生新 ID。

embedding 与 extraction 启动时都验证 manifest chain 和 artifact content hashes。unknown/missing/mixed version、runtime feature不符、conformance mismatch、tokenizer/parser disagreement 均 fail closed。不能把 v1.1 weights 与 v1.0 thresholds 混用；不能在 artifact 不兼容时“容差继续”。

### 32.2 公共与私有状态

public：protocol manifest、model/tokenizer/parser artifacts、QAT scales、统一thresholds、公开null-score artifact、canonicalization、ECC/statistic spec、document nonce和上述可选两行leading metadata。private：master/subkeys、per-anchor expected bits、production-key qualification audit、generation decisions/audits；private audit不改变公开threshold。final code不需要sidecar；调用者若未提供protocol ID/document nonce且代码没有合法leading metadata，detector返回unavailable，而非自动猜版本。

### 32.3 与当前 WFCLLM 的兼容性

当前仓库 official detector rows 包含 `prompt`，现行 generation boundary 只覆盖 `if/for/while` 部分 compound 层，现行 no-quality-gate contract 与本方案 Post-Gate/hard validator相冲突，历史 SAWR 又是 multi-statement window carrier。因此 SAGE-C 不是现有 artifact 的兼容升级：

- 旧 final-code rows 可在外层保留 prompt 作 dataset bookkeeping，但 SAGE-C detector API 必须在调用边界投影为 `final_code + protocol_id`，并用测试证明 prompt byte changes 不改变结果；
- 旧 boundary/anchor/candidate sidecars、semantic LSH weights、calibration thresholds 和 run IDs 全部标 incompatible；
- 需新 protocol namespace、parser conformance suite、artifact registry、null calibration pipeline 和 audit规则；
- 不修改旧 official method 的含义，也不把 diagnostic selector升级成正式 Gate。

迁移成本是中到高：parser/linearizer、data pipeline、two-stage Gate、reference runtime 与 detector calibration均需新实现；可复用的只有 final-code sanitization、run layout、部分 semantic feature 与 audit infrastructure。本文没有实施这些生产变更。

## 33. Security and key-separation analysis

### 33.1 Key schedule 与 dataflow

\[
K_{msg}=HKDF(K,protocol\_id,\texttt{SAGE-C/msg}),\quad
K_{mask}=HKDF(K,protocol\_id,\texttt{SAGE-C/mask}).
\]

所有 HMAC preimages 采用 length-prefix canonical encoding；不同 protocol/key rotation 不复用 domain。public anchor hash 不是 MAC，不能反推 key。master/subkeys永不序列化到 public schema、model、logs 或 final code。

| 组件 | 可读信息 | 明确禁止 | key-dependent? |
|---|---|---|---:|
| baseline/parser/eligibility/Pre-Gate | frozen \(D_0\)、public structure | key/target/key-selected prefix/pool outcome/generator ID | 否 |
| pool builder | frozen \(D_0\) prefix/current span、public generator config/seeds | key/target、future units、side-specific prompt/early stop | 否 |
| pair/hard/region/Post0 scoring | 全planning pools、baseline/tests/public artifacts | target/key、candidate count as feature | 否 |
| document commitment | nonce、ordered skips/pools/scores/config | key/target | 否；完成后才解锁key |
| target derivation | nonce/anchor/block/slot、key | candidate generation/control | 是 |
| selector | scored fixed pool、target | 追加 attempt/改温度/重写 siblings | 是，仅选择 |
| extractor | final code、public artifacts、key | prompt/logs/pool/retry | 是，仅期望 bit |

attempt timeout也产生固定位置的public failure sentinel；全document始终完成注册planning后才解锁target接口。所有candidates先运行相同validators/heads，selector固定扫描 \(A\) entries，降低timing/cache中的target side channel。服务只公开final result与粗粒度总延迟，不公开per-attempt trace或document fallback reason。

### 33.2 Causal/shortcut audits

1. 对同一 \(D_0\)/nonce/planning RNG改变1,000个random keys，全document Pre/Post0 logits、commitment、pool bytes/order/count/seeds/latency bins必须byte-identical；只有targets/selected candidates/D1可变。
2. 用 target、key ID、region side probe candidate count、retry/timeout、score rank和 output length；pool-stage AUC需≤0.52且 permutation \(p\ge0.01\)，并检查 source code dataflow。
3. gradient-reversal generator/repository heads、style-balanced sampling和 leave-one-generator/repository/task tests共同检查 nuisance shortcut；probe失败即 Gate feasibility NO-GO。
4. actual target 与 counterfactual opposite target复用同 pool；q0/q1 标签不读取一次抽到的 bit。双侧监督只关闭直接 selection bias，不保证 natural region balance、semantic validity、generator shift或FPR，后者各有独立硬测试。
5. generator artifact/version 更新必须重跑 Oracle、Gate calibration/leak probes 和 end-to-end null；不能假设 Gate labels跨 generator不变。未重新资格化的 generator不进入正式 embedding protocol。

### 33.3 Prompt boundary 与 detector oracle

盲 detector 无法恢复 prompt/generated boundary，所以 origin label不进入 anchor/Gate/evidence。v1 对所有 def/class headers普遍 rule-exclude，因而满足“已有 top-level header默认不候选”但更保守；其他 prompt-owned eligible units在 embedding 中未改写，在 extraction 中可能贡献随机 match/mismatch/erasure，作为 null-like dilution纳入实际 prompt-length matched calibration。任何声称只检测 generated span 都需要把可信 span写入 final artifact，超出本 protocol。

per-anchor detector 输出会形成 chosen-query key oracle，因此外部接口只返回 aggregate result、rate-limit、审计查询并支持 key rotation。知道 public region/anchor仍不足以伪造 target，假设 HMAC/HKDF安全；攻击者可通过大幅重写擦除信号，这是 availability攻击，不是 key forgery。

public `document_nonce`阻止未携带metadata的跨文档结构碰撞和朴素unit transplant；null calibration也随机化nonce。但攻击者可以复制整段watermarked code并连同nonce一起复制，形成watermark-copy/provenance攻击。final-code-only detector无法区分“原始生成”与“完整复制”；解决它需要外部签名/provenance registry，属于v1 non-goal。nonce存在本身绝不计positive evidence。

## 34. Robustness and attack analysis

### 34.1 编辑/攻击结果

| 编辑或攻击 | Anchor/region处理 | 预期失败模式 | 必测指标 |
|---|---|---|---|
| comments/blank lines/formatting | ordinary comments非unit；canonical bytes保持 | leading metadata被删且API无ID/nonce则unavailable；否则region/Post应稳定 | formatting survival |
| identifier rename | masked scope anchor保持；current region可能变 | mismatch/erasure | rename survival ≥70% initial |
| equivalent expression rewrite | slot anchor保持；current-only region重新算 | 可翻 region，不能保证 | rewrite survival/quality |
| unit deletion | bounded DP保留后续virtual slots | deleted evidence erasure | deletion/resync |
| unit insertion/dead code | DP insert transition；duplicates降权 | budget耗尽或 ambiguity | insertion/resync/FPR |
| same-suite reorder | delete+insert计费 | 超2 edits/block erasure | reorder survival |
| control-flow rewrite | role path/subtree改变 | subtree evidence erasure | attack TPR/FPR |
| header/body edit | suite reset限制传播 | header/subtree erasure；syntax坏则invalid document | per-clause survival |
| copy a matching unit | text-identity duplicate cluster | 只有首个有权重 | copy amplification test |
| cross-document transplant | nonce改变target；复制原nonce则可转移provenance | 无nonce时mismatch；完整copy无法区分 | transplant/copy disclosure |
| public Gate/region probing | rejection band、aggregate detector API | 可定向擦除，不能无key定向match | adaptive attack FPR |
| artifact substitution | hash/conformance failure | unavailable，不误判 | fail-closed rate |

local sync 只对预注册 bounded channel有效；无界 insertion、全函数重写或重实现可清除水印。格式稳定主要由 parser/canonicalizer定义，rename/semantic rewrite生存率是 required experiment，不是形式保证。

### 34.2 核心理论问题的明确结论

| 问题 | 结论 |
|---|---|
| Gate提高容量还是只筛掉？ | 固定全位置无限预算时只会降低绝对尝试数；固定**总生成预算**可重分配且Gate排序 \(p_i/c_i\) 优于随机时，才可能提高有效证据/成本。需no-Gate同预算验证。 |
| 双侧 \(q_0/q_1\) 是否足够？ | 只消除target-conditioned gate的直接偏差；不解决region imbalance、错误labels、generator shift、Post-Gate/FPR或alignment selection。 |
| Gate会否学style/type/generator？ | 会；type可能是真因也可能shortcut。用stratified labels、nuisance probes、counterfactuals和leave-one-group-out证伪，失败则不用学习Gate。 |
| labels是否绑定generator？ | 是，feasibility是generator+budget+protocol条件概率。任何generator更新必须重做Oracle/校准，不能宣称通用。 |
| Post-Gate是否新神经瓶颈？ | 是，且final-only输入不能复算pairwise semantics；它只是blind proxy。因此asymmetric held-out FAR硬门槛、INT8 reference/rejection band/hard verifier。达不到FAR或replay即NO-GO。 |
| compound header有容量吗？ | 未知且很可能class-dependent；`else/try/finally`零容量，其他全为Oracle-only直至逐class promotion。 |
| header改写风险？ | 高于conservative assignment，尤其predicate/evaluation order/API；body前选择、final suite验证仍不证明语义，因此阈值更严。 |
| “未排除都候选”是否可接受？ | 不可接受；v1改为正向allowlist，yield/await/calls/control-sensitive表达式禁用或Oracle-only。 |
| 稳定近均衡二分存在吗？ | 是核心未验证假设；current-unit LSH/encoder文献只提供动机，Oracle min-side/balance/replay决定是否存在。 |
| key-independent pool是否太大？ | 可能；\(A\)曲线与min-side决定。若A=64仍低或部署A=6容量不足，Gate无法挽救。 |
| natural side不平衡怎么办？ | 用min-side LCB筛选、双侧labels、balance训练和target-independent fallback；gap>0.10触发region redesign，而非偏向易侧。 |
| current unit是否含足够语义？ | 对短/低熵units可能不足；以Oracle evidence density判定。context不能替它创造carrier entropy。 |
| context能否进入region encoder？ | 正式禁止；会使编辑一个predecessor改变后续bits并破坏单元独立对偶。只允许进入Gate。 |
| 短header表示退化怎么办？ | 保持Oracle-only/erasure，不能把context注入bit encoder补救。 |
| Gate selection后的FPR如何可信？ | 不能用未选择样本的二项模型；在matched null上完整盲重放Post-Gate/erasure/resync/max。 |
| selection是否data-dependent stopping？ | 是，尤其alignment max和erasure；因此只用empirical null order statistic。 |
| null必须重放什么？ | parser、eligibility、Post-Gate、region、rejection、duplicates、clusters、minimum evidence、resync max及任何ECC joint rule。 |
| ECC编码什么？ | key-derived check message；不是外部payload。primary仍为keyed statistical evidence。 |
| 系统类型？ | 统一定义为 **keyed statistical zero-bit watermark**；不是payload watermark。 |
| 能否证明可行？ | 只能条件性证明协议对偶/风险控制方式；容量、语义、重放和power必须由三层实验门槛验证。 |

## 35. Failure handling

| 失败 | 嵌入端动作 | 提取端动作 | 是否继续/泄漏 |
|---|---|---|---|
| rule excluded/oracle-only | 不开pool，保留baseline | unit erasure，仍占ordinal | 继续；无key调用 |
| provisional parse不唯一 | 保留baseline | final strict parse决定 | 继续；内部skip不公开 |
| Pre-Gate低置信/高成本 | 不开pool，保留baseline | 不知道skip；照常blind replay | 继续；null-like evidence |
| pool为空/全重复/timeout | 完成固定A slots，target后用safe fallback | final code blind state | 不追加retry，不公开轨迹 |
| 目标侧候选耗尽 | target-independent fallback | match/mismatch/erasure均可能 | 不改变预算/温度 |
| candidate跨span/破坏结构 | hard reject | final hard verify erasure | 不重写siblings |
| Post/region边界 | 不选为target candidate | erasure | 禁强判bit |
| header body尚未完成 | provisional only | 不存在generation state | body完成后finalize |
| header/suite final不合法 | generation formal status失败 | strict parse若失败则invalid document | 不回滚已生成body/前文 |
| final strict parse失败 | 不声称嵌入成功 | `INVALID_DOCUMENT` | 全文无formal evidence |
| zero-edit anchor碰撞 | finalization拒绝formal status | `ANCHOR_COLLISION_UNAVAILABLE` | 全文fail closed；不得共享target |
| alignment多解/超预算 | 不适用 | affected blocks erasure/insufficient | max已在null校准 |
| artifact/runtime/hash不兼容 | 停止或输出无水印baseline | `UNAVAILABLE_DOCUMENT` | fail closed，不近似回退 |
| key缺失/错误 | 无正式target/停止 | unavailable；错误key应呈null | 不猜key |
| accepted evidence不足 | generation完成 | `INSUFFICIENT_EVIDENCE` | 不是negative/positive |
| project tests不可运行 | label unknown；不作为formal-positive训练标签 | 部署只能依赖blind Gate/hard checks | 计入局限/FAR test |
| code formatting/rename/edit | 不适用 | canonicalize、bounded resync、boundary erasure | 生存率实测 |
| adversary擦除证据 | 无恢复承诺 | erasure/insufficient | availability failure，不虚报positive |

所有生成时 failure records 都是可选 private audit；丢失它们不影响 extraction。系统从不把“曾经成功选择”当证据，也不因候选耗尽把一个 bit-conditioned sidecar写进 final artifact。

## 36. Training curriculum

训练按不可跳级的 stage gates 执行：

1. **Parser/contract qualification。** 不训练模型；在人工 gold fixtures 上验证 semicolon、inline suite、多行/async/header/decorator/match/exception linearization、anchor round-trip 和 formatting stability。要求 exact unit/span/role/anchor 100%，否则停止。
2. **Region encoder。** 在 train-only invariance/hard-negative pairs 上训练 \(E_R\)，做 QAT/export；在 validation 检查 per-class balance、margin、format/rename replay。冻结 artifact 后不再因 Gate label 调整。
3. **Oracle feasibility。** 运行第 19 节 \(A=6/16/32/64\)。第一层 NO-GO 时不生成正式 Gate dataset、不做大训练。
4. **Dataset/materialization。** 按 split-before-expand 产生 \(R=4\) pool blocks、counterfactual labels、paired views、hard negatives；冻结 schema/data-card/clone audit。
5. **Pre-Gate baseline。** 先 current-only 和 simple loss，确认优于 rule/no-Gate random ranking；再加入 CTX-1、paired CTX-2/3、soft labels、cost/calibration 与 nuisance heads。
6. **Post-Gate。** 先 hard-verifier + formal asymmetric head，再加入 auxiliary/ranking/margin/robustness。每 epoch 按 repository cluster bootstrap监测 hard-negative FAR；FAR恶化时不以容量补偿。
7. **Hard-negative mining。** 最多三轮，只从 train false accepts选样，保留原 prevalence权重；每轮用固定 validation评估，test不参与。
8. **QAT/batch curriculum。** FP32稳定后加入 fake-QDQ、random batch companions/padding，最后周期性执行真实 exported graph；reference flip非零则不进入 calibration。
9. **Calibration/freeze。** 在独立 calibration split拟合映射、LCB和integer thresholds，生成 immutable model/protocol manifest；之后禁止改 feature、weight、threshold。
10. **System evaluation。** 一次性运行 held-out repository/task/generator、null-test、HumanEval/MBPP、edit/attack、latency/cost矩阵；按第 40 节裁决。

optimizer、learning rate 与 epoch数由 model-validation NLL/FAR early stopping选；同 anchor 的 paired views不能跨 optimizer split。class-specific header heads只在对应 Oracle promotion后训练。训练中 privileged labels只能进 loss，export graph用 schema allowlist检查确保没有测试结果、origin、generator/repository、pool/retry、target/key字段。

## 37. Evaluation metrics

所有比例同时报告 numerator/denominator、micro/macro、repository/task cluster-bootstrap 95% CI，以及 conservative simple 与各 header class分层。下面是锁定的 metric dictionary：

| # | 指标 | 严格定义/主分母 |
|---:|---|---|
| 1 | Oracle双侧可达率 | \(A\)内两侧各≥1 formal-valid candidate的anchors / Oracle anchors |
| 2 | 单元候选覆盖率 | ≥1 unique span/parse-valid candidate的policy-eligible anchors / policy-eligible anchors |
| 3 | region 0成功率 | ≥1 region-0 formal-valid pool / all fixed pools |
| 4 | region 1成功率 | 同上region 1 |
| 5 | min-side success rate | `min(metric 3, metric 4)`，另报per-anchor posterior min |
| 6 | Pre-Gate AUROC/AUPRC | 对fixed-budget region-specific与both-side hard labels；AUPRC带prevalence |
| 7 | Pre-Gate calibration error | ECE-15、Brier、NLL、worst-group ECE及LCB coverage |
| 8 | Post-Gate false accept rate | `formal_invalid`且非unknown candidates被ACCEPT / 全invalid；CP CI |
| 9 | Post-Gate false reject rate | formal-valid candidates未ACCEPT / 全formal-valid，含erasure |
| 10 | 有效嵌入率 | `INTERNAL_TARGET_SELECTED` / all policy-eligible generated units；另报/Pre-selected |
| 11 | 每成功单元平均attempt数 | first successful attempt index+1；失败成本另报A |
| 12 | 每千token有效证据数 | final blind non-erasure evidence / generated code tokens ×1000 |
| 13 | batch-composition flip rate | 与batch-1 reference离散state不同的sample-env pairs / pairs |
| 14 | quantization flip rate | FP32与exported INT8离散state不同 / compared samples |
| 15 | cross-process replay rate | 两clean reference processes state和integer outputs exact相同 / samples |
| 16 | cross-device replay rate | qualified device与reference同state / samples；未资格化单列 |
| 17 | formatting survival rate | format后anchor、region、Post/evidence state均保持 / pre-edit formal evidence |
| 18 | rename survival rate |受控alpha-rename后仍同expected target且match / pre-edit matches |
| 19 | statement insertion/deletion survival |单unit edit后原未编辑 evidence仍正确对齐 / 可追踪原evidence |
| 20 | local resynchronization rate | bounded-channel documents恢复正确virtual slots / edited documents |
| 21 | 检测AUROC | watermarked vs matched null的locked \(T^*\)，insufficient按预注册最低分处理并另报coverage |
| 22 | 固定FPR下TPR | TPR@1% empirical FPR，报sufficient-conditional与overall |
| 23 | 经验FPR | held-out null-test positives / all null documents，CP 95% CI |
| 24 | 平均erasure rate | erasure policy-eligible syntactic slots / policy-eligible slots |
| 25 | mismatch rate | mismatches / non-erasure formal evidence；同时/eligible denominator |
| 26 | 有效样本量 | 第29节 \(n_{eff}\)，报median/quantiles和guard coverage |
| 27 | ECC解码成功率 | uniquely verified blocks / blocks with 7 defined virtual slots；另报≥5-observed条件率 |
| 28 | 代码编译/解析成功率 | final strict Python 3.12 parse successes / generated documents |
| 29 | 功能正确率/pass@k变化 | 与同seed、同budget no-watermark baseline的paired pass rate difference/CI |
| 30 | 代码质量变化 | blinded human rubric、lint/complexity delta与semantic-test regressions，分开报告 |
| 31 | 生成延迟 | end-to-end wall time p50/p95与相对baseline增幅 |
| 32 | Gate推理成本 | Pre/Post CPU ms、memory、energy proxy、calls/unit |
| 33 | 总retry成本 | candidate tokens、attempts、timeouts / successful document和formal evidence |
| 34 | Key/candidate leakage probe | pool/Gate invariance exact rate、target probe AUROC/permutation p |
| 35 | Evidence sufficiency coverage | 满足16 evidence/8 clusters/\(n_{eff}\ge12\)的documents / all documents |
| 36 | Anchor uniqueness/collision | unique anchors / all strict-parsed documents；normative requirement 100% |

`code quality` 不压成单一主观分数；功能、lint/complexity和blind human preferences分别报告。任何 unknown/timeout denominator处理在 evaluation plan预注册。

## 38. Baselines

所有 baselines 复用同一 split、region artifact、candidate pools、attempt token accounting和detector calibration；既报告固定per-unit \(A\)，也报告相同global generation-token budget，避免Gate通过少做工作取胜。

| # | Baseline | 变化 |
|---:|---|---|
| 1 | 无Gate | 每个FORMAL_ELIGIBLE unit固定尝试A，保留hard validator |
| 2 | 规则Gate | 仅第10/11节allowlist与静态熵/长度阈值，无learned Pre/Post |
| 3 | current-unit-only Gate | shared Gate不读predecessors |
| 4 | CTX-1 Gate | 注册width=1 |
| 5 | CTX-2 Gate | 注册width=2 |
| 6 | CTX-3 Gate | 注册width=3 |
| 7 | 只用Pre-Gate | learned budget selection + public hard verifier，无neural Post |
| 8 | 只用Post-Gate | 所有eligible units分配pool，learned Post过滤 |
| 9 | 不量化 | FP32 reference-like graph；只作研究，不能替代formal contract |
| 10 | INT8/QAT | 最终候选设计 |
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
| 3 | 删除header候选 | headers对容量、FAR、cost的净贡献；v1可能等于正式配置 |
| 4 | 删除分号拆分 | **仅parser错误对照**，量化边界/anchor污染，不可部署 |
| 5 | 删除context consistency loss | paired predictions稳定性和context增益 |
| 6 | 删除batch consistency loss | batch flip与margin变化 |
| 7 | 删除quantization-aware loss | INT8 flip/FAR/coverage变化 |
| 8 | 删除calibration loss | ECE、LCB selection、FAR变化 |
| 9 | 删除ranking loss | pool内first-choice validity和cost |
| 10 | 删除robustness label/head | edit survival与clean FAR trade-off |
| 11 | 改attempt budget | \(A=2,4,6,8,16,64\) capacity-cost曲线；64仅Oracle |
| 12 | 改candidate pool unique size | 固定A下调dedup/采样diversity，区别数量与预算 |
| 13 | 改region margin threshold | balance/coverage/replay trade-off |
| 14 | 改Gate erasure/rejection threshold | FAR、FRR、evidence density、flip |
| 15 | 改ECC强度 | none、Hamming(7,4)、预注册stronger code；完整重校准 |
| 16 | shared heads vs unit-specific heads vs split models | generalization、稀疏class FAR、size/latency |

附加安全诊断包括移除nuisance adversary、允许target-conditioned early stop、把context送入region encoder；后两项是预期失败的red-team controls，绝不进入正式候选。效应报告paired repository bootstrap CI与multiplicity-corrected p-values；“无显著差异”不自动证明组件无用，还要看hard constraints。

## 40. Go/no-go criteria

所有数值在看held-out test之前冻结。`Hard` 不能降低；`Initial` 是当前工程建议，若失败只能以新protocol、新calibration和新test重启，不能在同一次test后调低。

### 40.1 Layer 1 — Oracle feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| simple \(A=64\) macro min-side | ≥0.35且95% CI lower达标 | Initial |
| simple dual-reachable anchors | ≥50%且CI lower达标 | Initial |
| promoted header class min-side/dual | ≥0.20 / ≥30%，逐class | Initial |
| region side absolute gap | ≤0.10 | Initial |
| label-oracle evidence density | ≥4/1000 generated tokens | Initial |
| reference region replay | 100% | Hard |
| formatting region replay | ≥99.9% | Initial |
| paired functionality delta | ≥-2 percentage points | Initial |

任一 simple 核心条件失败：停止Gate training，重新设计region/carrier，Layer-1 NO-GO。只失败某header则该class不promotion；关闭全部headers后density仍失败则整体NO-GO。

### 40.2 Layer 2 — Gate feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| unseen repo/task/generator Pre calibration | ECE≤0.05，LCB coverage≥95% | Initial |
| Post hard-negative FAR | point≤0.2%，95% upper≤0.5% | Hard-for-promotion |
| direct key/target pool invariance | bytes/order/count/Gate logits 100% identical | Hard |
| target probe | AUROC≤0.52且permutation \(p\ge0.01\) | Initial audit |
| same-global-budget evidence/1000 | ≥no-Gate ×1.15 | Initial |
| absolute formal evidence | ≥no-Gate的95% | Initial |
| retry candidate tokens | ≤no-Gate ×0.80 | Initial |
| Gate latency/artifact | ≤10 ms/unit，≤24 MiB | Initial |
| leave-one-generator/repo degradation | evidence与AUPRC相对held-in下降≤10%且仍校准 | Initial |

若learned Gate不能在unseen groups校准，或增益不足以覆盖erasure/latency，就不采用学习Gate；保留规则/no-Gate科研baseline，不能宣称Layer-2成功。双侧AUROC高但FAR或budget utility失败仍是NO-GO。

### 40.3 Layer 3 — System feasibility

| 条件 | 门槛 | 类型 |
|---|---:|---|
| held-out null empirical FPR | point≤1%，95% upper≤1.25% | Hard |
| TPR@1% FPR | sufficient docs≥80%；sufficiency coverage≥70% | Initial |
| effective embedding/evidence density | target-selected/all eligible≥20%；formal evidence≥4/1000 tokens | Initial |
| erasure/mismatch | erasure≤40%；mismatch/non-erasure≤15% | Initial |
| normative replay | region/Post/evidence 100%，cross-process byte-exact | Hard |
| qualified nonnormative flip | ≤\(10^{-4}\)，flips仅入expanded erasure；否则不资格化 | Hard contract |
| final parse/function | parse≥99.5%；pass/function delta≥-2pp | Initial |
| edit survival | format≥95%，rename≥70%，单insert/delete local resync≥90% | Initial |
| ECC | ≥5 observed的blocks中unique verified≥80%；all blocks另报 | Initial secondary |
| cost | total generated tokens≤2×；wall time增幅≤20%；Gate≤10ms/unit | Initial |

任何Hard失败直接System NO-GO。Initial核心条件（TPR/coverage、density、function、erasure、cost）任一CI未达标也不得宣称系统成功；必须注册新版本后重测。通过三层只支持特定Python/version/generator/threat model，不外推任意代码或硬件。

## 41. Known limitations

### 41.1 证据状态

| 陈述 | 状态 | 可声称范围/关闭证据 |
|---|---|---|
| 一个current-unit-only、candidate-independent anchor/target协议可定义 | design construction | 只说明规范自洽；需parser fixtures证明实现对偶 |
| fixed pool先于target可阻止直接key-conditioned retry | design assumption + dataflow requirement | 需代码级taint/invariance审计，不仅统计probe |
| conservative simple units存在双侧region容量 | **hypothesis** | 第19/40.1节formal Oracle结果 |
| compound headers有可用容量 | **hypothesis，风险更高** | 每clause独立Oracle/FAR/function测试；当前全oracle-only |
| Gate在unseen generator/repo提高evidence-per-cost | **hypothesis** | 同global budget paired baselines与Layer-2 criteria |
| Post-Gate可盲识别语义破坏且FAR≤0.5% upper | **hypothesis** | 自然/hard-negative calibration+held-out tests |
| QAT提升稳定性 | **hypothesis** | no-QAT/QAT export矩阵；它不证明determinism |
| normative backend离散重放 | required experiment | 10k conformance +独立process exact replay |
| empirical FPR≤1% | required experiment | 10k actual-key calibration +20k held-out null test |
| formatting/rename/edit survival达标 | required experiment | locked edit suite；未运行 |
| 系统检测power、功能与成本达标 | required experiment | Layer-3完整评估；未运行 |

### 41.2 不可消除或不可由更多Gate数据单独修复的限制

1. 一般Python语义等价不可由局部神经Gate证明；final-only extractor还缺少baseline，原则上不能复算pairwise preservation。动态反射、I/O、并发、exceptions和environment让blind验证存在根本不完备性。只能缩小carrier policy、生成时pairwise过滤，并以blind-proxy FAR/functional tests限制风险。
2. current-unit-only region在短、低熵header/statement上可能根本没有双侧自由度。若Oracle失败，增加Gate数据或模型大小无效。
3. blind final code不含prompt boundary；无法精确剔除prompt-owned simple units，只能普遍规则排除某些classes并在null中吸收随机证据。
4. bounded resynchronization没有不可变外部marker；大规模insert/delete/reorder会导致歧义或erasure。max-over-alignments增加校准样本和计算成本。
5. 固定CPU/container contract牺牲跨硬件可移植性。若业务要求任意客户端/GPU本地检测，本v1不满足。
6. public region/Gate允许攻击者优化擦除；无key只能使forge概率受经验FPR约束，不能保证抗重写availability。
7. Hamming(7,4)面对大量rule erasures可能很少可解；ECC不增加底层carrier容量。
8. Gate labels条件于candidate generator和attempt budget；model/generalization不是永久属性，generator更新需重新资格化。
9. license allowlist、依赖可执行性、private tests缺失和real-repo sandbox限制label规模/代表性。
10. protocol-level 1% tail在不同keys上的稳定估计需要至少数万独立 `(document,key)` null pairs；多protocol/project扫描会进一步增加calibration成本，且不能保证每个固定key的有限样本条件FPR完全相同。
11. 当前只规范Python 3.12 source，不覆盖notebook cell order、generated AST bytecode、C extensions或跨文件semantic rewrites。
12. `document_nonce`不构成不可复制provenance；完整复制code+nonce可转移检测结果，需外部签名/registry才可归属到唯一生成事件。
13. 全document key-neutral commitment要求buffer baseline与全部candidate plans，不能把未经承诺的online token stream直接作为最终输出；它增加内存、首token后延迟和document fallback成本。

这些限制不能用“模型更大”自动关闭；第1、3、4、5、6属于scope/工程合同限制，第2属于可能的根本容量瓶颈。

## 42. Implementation roadmap

本任务不改生产代码。后续实现应在独立research protocol namespace中分七个可停止work packages：

| WP | 产物 | 必须验证后才进入下一步 |
|---:|---|---|
| 0 | `protocol.json`、grammar/role tables、250+人工gold snippets | embed/final parse unit stream、span、owner、anchor 100%一致；format round-trip |
| 1 | parser/linearizer/anchor research library与property tests | semicolon/inline/multiline/async/decorator/match/except* coverage；fuzz无未分类node |
| 2 | frozen region encoder/heuristic baselines和Oracle pilot records | 先做低成本falsification；pilot差则停止，不训练Gate |
| 3 | license/provenance ledger、split graph、10 schemas、formal Oracle (A≤64) | Layer-1全部threshold/CI，data clone/leak audit |
| 4 | shared Pre/Post Gate、baselines/ablations、QAT export | Layer-2、privileged-input export audit、hard-negative FAR |
| 5 | reference runtime/container、artifact registry/conformance vectors | 10k normative exact replay、fail-closed matrix |
| 6 | embedding/extraction prototype、ECC/resync/statistic、null pipeline | ≥10k calibration/≥20k null-test、Layer-3和attacks |
| 7 | external benchmark/data/model cards和安全release review | 三层通过后才讨论production integration |

最低成本的下一项不是训练Gate，而是 WP0+Oracle falsification pilot：从至少10个license-cleared、互不clone的repositories抽取每个conservative simple class 300 anchors，加入100个parser stress snippets；固定一个现有public current-unit region baseline和两个generator families，在相同pools上运行 \(A=6,16\)、每anchor \(R=2\)。它只能快速发现单侧容量/解析失败，**不能**给Layer-1 GO。若pilot min-side point <0.20或parser/format anchor replay <99.9%，立即停止并修region/parser。

formal Oracle至少需要20个held-out repositories、3个generator families、5,000 conservative simple anchors及每个拟promotion header class 1,000 anchors，\(R=4,A\in\{6,16,32,64\}\)，并满足第19节CI门槛；规模/成本需另获资源批准。

当前仓库迁移接口建议（尚未实现）：在 `wfcllm.lang` 下建立versioned parser adapter，在 `wfcllm.generation` 新建scheme-C current-span pool而不改旧boundary semantics，在 `wfcllm.method` 新建独立preset/artifact contract，在 `wfcllm.detection` 新建final-code-only projection与empirical-null detector，在 `wfcllm.audit` 添加no-key-to-pool taint/conformance检查。旧协议继续原样可复现。

## 43. Reviewer issue-resolution matrix

初始作者稿尚无独立reviewer issue，不能把作者自检记为关闭。正式审查后，本节将逐issue记录 `issue_id / severity / disposition / 修改章节 / closure evidence / remaining experiment / reviewer recheck`；在原始review落盘前，所有review criteria均处于未验收状态。

## 44. Review score progression

| 阶段 | Reviewer | 分数 | Verdict | Critical | Unresolved major |
|---|---|---:|---|---:|---:|
| 0.1 initial author draft | none | N/A | NOT READY（尚未独立审查） | N/A | N/A |

## 45. Final reviewer verdict

`REVIEW PASSED: NO`（初始稿状态）。尚未执行continuing reviewer多轮审查和fresh final reviewer验收，因此不得标记Goal complete或声称方案通过。

## 46. Final recommendation

当前机制建议为 **CONDITIONAL GO，仅批准Oracle feasibility与parser/replay原型**。这不是系统成功结论。先完成WP0和低成本Oracle falsification；只有formal Oracle达到Layer-1全部阈值，才批准小型Gate训练。若current-unit region在\(A=64\)仍大量单侧、formal evidence density <4/1000 tokens，或reference replay非100%，路线直接NO-GO；不能退回多窗口混合载荷来掩盖失败。

在尚未运行实验的当前状态，不能声称：Gate提升容量、header可承载bit、Post-Gate达到目标FAR、量化跨设备稳定、编辑鲁棒、FPR受控、TPR/功能/成本达标。可以声称的只有：SAGE-C给出了可实现、可证伪、final-code-only的协议设计和明确stop conditions；其经验前提仍待验证。
