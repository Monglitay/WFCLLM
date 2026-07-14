# Watermark Mechanism V3：动态语义水印文献综述

检索截止：2026-07-14（Asia/Shanghai）  
正式基线：`eac59bfbdc80a91bd5eb5d1332dcace75442d9ad`  
正式分支：`codex/watermark-mechanism-v3-dynamic-semantic-reset`

## 1. 结论

文献支持保留 frozen semantic encoder 作为正式水印 symbol 的来源，但不支持上一轮“20 个完整 final code 全部生成完毕后，再逐候选解析和重编码”的主循环。最接近本任务约束的共同结构是：在生成过程到达稳定语义边界时，对当前闭合 unit 的 final-observable bounded causal context 批量编码；在 encoder 输出后施加 secret-keyed projection/LSH 与 target；把证据增量写入 trajectory state；检测时从 final code 一次解析、一次收集、microbatch 编码全部 context 并重放同一整数信道。

正式主臂选择 **Streaming-SharedPool-V3-R20**：Current 与 V3 共享同一 task 的 20 条 raw trajectories；每条 trajectory 在 statement/block closure 时动态累计 semantic evidence；EOS 只对已经累计的 score 做 quality-first、key-conditioned ranking，不重新对 20 份完整代码做全量语义重评分；被选中的 final code 允许一次独立 R3 replay 用于 contract 与 detector。它是 generation-time dynamic embedding，因为 encoder 在 unit commit 时运行，secret 决定 generation-time final candidate ranking；它同时保留 exact shared-pool 科学公平性。

Dynamic block rejection/rollback 与 semantic beam state 是更强 online steering 机制，但第一处 key-conditioned block 选择后就与 Current 的后续 prefix 分叉。本轮将它们作为机制族比较和未来 matched-budget arm，而不偷偷称为 exact same-pool。纯 AST/CST spelling、变量改名、字面量/引号/括号、dead code、formatting 和不依赖 encoder 输出的 category codebook 均被排除。

## 2. 检索方法与证据等级

检索覆盖 arXiv、OpenReview、ACL Anthology、AAAI proceedings、ACM/IEEE 元数据、作者项目页、官方 GitHub 和本地 `papers/`。搜索结果只用于发现；标题、作者、年份、机制、detector 输入和限制回到论文首页/方法正文或官方代码核验。核心元数据见 `literature_matrix_v3.json`。

证据等级按以下顺序使用：正式 proceedings/出版页 > arXiv/OpenReview 最新版本 > 作者官方代码/项目页 > 本地 PDF。2026 年尚未正式出版或撤稿的工作明确标为 preprint/submission，不把预印本结果写成已复现事实。

## 3. 直接语义水印工作

### SemStamp 与 k-SemStamp

SemStamp（Hou et al., NAACL 2024，[paper](https://aclanthology.org/2024.naacl-long.226/)）把闭合 sentence 编码为 embedding，以 random-hyperplane LSH 形成 signature，再按前一单位确定的 valid semantic regions 做 sentence-level rejection sampling。它证明 encoder 输出可以是正式水印 symbol，generation 与 final-output detector 都能重算该 symbol。其原始第一单位依赖 prompt hash，previous-signature chaining 在 insertion/deletion 后会级联失步，且报告的 rejection 成本约为普通生成的数量级倍增，因此不能原样迁移。

k-SemStamp（Hou et al., Findings ACL 2024，[arXiv:2402.11399](https://arxiv.org/abs/2402.11399)）以 k-means regions 代替任意 hyperplanes，改善采样效率和 semantic-manifold occupancy。它说明 clean bucket bias 必须实测，也说明 learned centroids 会引入 domain/data provenance 与 hash closure；本项目因此只允许 calibration-only public whitening，不从 pilot/held-out positives 学 regions。

### SeqMark、SimMark 与 SemanticShift

SeqMark（Le, Ritter, Goyal, 2026 preprint，[arXiv:2601.11629](https://arxiv.org/abs/2601.11629)）指出低熵 constrained generation 会发生 region collapse：高质量候选彼此太相似，LSH 恰好把它们送入同一区域，retry=20 也未必提供 20 个有效 symbol。其 prompt-specific candidate mean 与 detector-side resampling 违反本 R3 合同，但 region entropy、pairwise cosine、attainable best-of-20 gain 和 outlier analysis 必须纳入 debug/pilot。

SimMark（Dabiriaghdam and Wang, EMNLP 2025，[arXiv:2502.02787](https://arxiv.org/abs/2502.02787)）用相邻句 embedding similarity 区间和 soft counting 提高 paraphrase robustness。它支持 bounded neighboring context 和 soft evidence，但固定区间本身不形成独立 wrong-key null，不能直接成为 keyed channel。

SemanticShift（Li and Tan, AAAI SSS 2026，[paper](https://ojs.aaai.org/index.php/AAAI-SS/article/download/42577/50137/46678)）在 token generation 时计算 bounded semantic prefix 加入 candidate token 前后的 embedding shift，并用 secret preferred directions 引导 sampling。它直接支持“generation-time semantic shift + key”的方向，但逐 token/逐 candidate encoder 调用成本高，且 token-level prefix/truncation 对 code boundary replay 的稳定性不足。本项目把它提升到 deterministic closed statement/block boundary，并把 active contexts 合并成 batch。

### SAEMark、black-box chunk selection 与 PostMark

SAEMark（Yu et al., NeurIPS 2025，[OpenReview](https://openreview.net/forum?id=tXnyVPNOfa)）按 sentence/function unit 计算 deterministic semantic features，再从多个候选中选择最接近 key-derived target 的 unit。它表明 feature-based rejection sampling 可以让 key 真正改变 generation decision；其 SAE 依赖 anchor-LM hidden activations，不满足本项目“public frozen code encoder + final code”最小边界，但 selection/compute-budget 分析可迁移。

Bahri and Wieting 的 black-box watermark（TMLR 2026，[arXiv:2410.02099](https://arxiv.org/abs/2410.02099)）反复生成 bounded chunks，并在每轮用 secret score 选择 chunk；它为 block rejection/rollback 提供一般 black-box祖型。其正式 symbol 是 token n-gram PRF，不是 semantic encoder，因此只能影响动态控制流设计，不能充当本项目 evidence。

PostMark（Chang et al., EMNLP 2024，[paper](https://aclanthology.org/2024.emnlp-main.506/)）从 whole-text embedding 导出 secret word list，再重写插入目标词。它证明 content-conditioned semantic target 可从 output 重算，但 whole-output rewrite、secret table 和 online embedding API 都不适合功能敏感的代码，也属于生成后路线。

## 4. Code watermark 与攻击研究

SWEET（Lee et al., ACL 2024，[paper](https://aclanthology.org/2024.acl-long.268/)）通过 entropy threshold 避开低熵 code tokens；STONE（Kim et al., Findings EACL 2026，[arXiv:2502.18851](https://arxiv.org/abs/2502.18851)）进一步指出 syntax-critical token 也可能高熵并用 grammar list 过滤。两者对 quality/unsafe filtering 有价值，但正式 evidence 仍是 token green-list hit，不是 semantic encoder。

ACW（Guo and Cheng, NeurIPS 2025，[OpenReview](https://openreview.net/forum?id=RpE4HeuX69)）和 CodeTracer（Guo et al., ICML 2026，[OpenReview](https://openreview.net/forum?id=4xjq3iR4aK)）分别以 AST-aware learned partitioner 与 RL policy 改善 code quality/detectability；它们支持未来 learned proposer，却不能替代 frozen semantic detector。CodeGenGuard 用 semantic-preserving transformations、dead-code augmentation 与 trigger/LoRA 验证 model ownership，威胁模型也不同。

Suresh et al.（ICLR 2024 Tiny Paper，[OpenReview](https://openreview.net/forum?id=8PhI1PzSYY)）展示 rename、format、dead code 与结构改写会移除 code watermark。它要求本项目报告 formatting、conservative rename、AST unparse、insertion/deletion diagnostics；这些变换不能被假定为普遍语义等价，也不能反过来成为载体。

## 5. Code semantic encoder

CodeT5（Wang et al., EMNLP 2021，[paper](https://aclanthology.org/2021.emnlp-main.685/)）以 identifier-aware objectives 学习 code representation，服务器已有完整 checkpoint，适合低成本 frozen prototype。其基础模型不是专门的 sentence embedding 模型，因此必须冻结 pooling、normalization、precision、token budget 与 truncation/erasure 行为，并用 constant/bypass、batch/scalar 和 transformation diagnostics证明 observed bits 依赖 encoder。

GraphCodeBERT（Guo et al., ICLR 2021，[OpenReview](https://openreview.net/forum?id=jLoC4ez43PZ)）通过 data-flow graph 强化 code semantics；UniXcoder（Guo et al., ACL 2022，[paper](https://aclanthology.org/2022.acl-long.499/)）以 AST/comment cross-modal pretraining 与 contrastive code-fragment representation 更接近 semantic LSH。两者会扩大 parser/artifact surface，且服务器未缓存模型，所以本轮不在 pilot outcome 后临时切换。它们只作为预先记录的 future encoder ablation。

现有 LoRA/projection checkpoint 的训练数据含 HumanEval/MBPP canonical solutions，属于 benchmark-distribution leakage 风险。它可以作为冻结 development evidence，但必须记录 SHA-256、training provenance，并禁止使用 hidden tests 或 pilot outcome 选择 encoder。本轮主 encoder/是否使用 projection 必须在 preregistration 前由 transformation/parity debug 冻结。

## 6. Boundary、incremental parsing 与 constrained decoding

PICARD（Scholak et al., EMNLP 2021，[arXiv:2109.05093](https://arxiv.org/abs/2109.05093)）在 autoregressive decoding 中增量解析并拒绝 grammar-invalid tokens；SynCode（Ugare et al., 2024，[arXiv:2403.01632](https://arxiv.org/abs/2403.01632)）用 CFG/DFA mask store 高效保证格式 soundness/completeness。这些工作支持 public parser 找 boundary、过滤 unsafe candidates 和同步，但不支持把 grammar category 当主水印 evidence。

本项目冻结 Python AST statement/basic-block closure。当前 unit 先 canonicalize，context 只含 masked function signature、public role/path、current canonical unit 和至多 `K=1` 个 preceding committed unit。该窗口在 generation 可见、从 final code 可重放、token cost 有界；删除一个 block 至多使被删 unit 与后继一个 context 失效。

## 7. LSH、quantization、ECC 与同步

Charikar 的 random-hyperplane LSH（STOC 2002，DOI [10.1145/509907.509965](https://doi.org/10.1145/509907.509965)）给出 cosine/angle similarity 的 compact bits。V3 在 public whitening/L2 normalization 后做 deterministic int16 quantization，再以 domain-separated HMAC-SHA256 runtime-derived int8 hyperplanes 和 int64 dot product 得到 observed bits。Secret 只在 encoder 输出后生效；private planes、targets、key fingerprint 均不序列化。

Chao et al. 的 RBC/ECC watermark（2024，[arXiv:2406.10281](https://arxiv.org/abs/2406.10281)）提供 error/erasure、sliding synchronization 与 p-value 模板。本项目冻结 per-unit Hamming(7,4) target comparison和跨 unit keyed interleaving，但不把重叠 blocks 当独立样本。ECC 不能创造短代码中不存在的容量；不足 unit 必须返回 insufficient evidence。

## 8. GPU 与 serving 文献的工程含义

PagedAttention/vLLM（Kwon et al., SOSP 2023，[paper](https://arxiv.org/abs/2309.06180)）和 continuous batching 说明共享 KV 与动态 batch 可以提高 causal generation throughput；本服务器当前未安装 vLLM/FlashAttention，因此正式 profile 先用 PyTorch SDPA 与 existing KV cache，避免为了优化更换 sampling distribution。Encoder 侧的首要优化不是复杂 serving，而是：收集同一 closure wave 的所有 active contexts、batch tokenizer、单次/microbatch frozen encoder forward、`torch.inference_mode()`、BF16 parity、duplicate-context cache 与 persistent model。

旧 complete-final route 的复杂度约为 `O(R * B)` contexts/task 且若逐 block forward 则是 `O(R * B)` GPU calls；dynamic shared-pool 仍编码相同数量级的闭合 contexts，但按 closure waves batch，并且不在 EOS 重新解析/tokenize/encode 20 份完整 final code。Detector 对一个 final code 的全部 contexts 只做 batch/microbatch forward。成本门冻结为 median semantic encoder wall time/task 至少下降 30%。

## 9. 三个动态机制族比较

| 维度 | A Streaming shared pool | B Block rejection/rollback | C Semantic beam state |
|---|---|---|---|
| encoder 正式 evidence | 是 | 是 | 是 |
| generation dynamic | unit closure 增量累计 | unit closure 即时选择并改 prefix | closure 时 batch prune |
| secret 改变决策 | EOS generation-time ranking | accept/reject/rollback | beam retention/pruning |
| final-code R3 | 最小风险 | rollback 污染风险 | state/final 对齐风险 |
| exact shared pool | 是 | 否 | 否 |
| encoder batching | 20 trajectories 同波次 | proposals 同波次 | all new beam states |
| Pass 风险 | 最低 | 中高 | 中高 |
| channel capacity | 中 | 高 | 高 |
| implementation | 中 | 高 | 最高 |
| 本轮决策 | **正式主臂** | matched-budget future arm | future arm |

可选 D（learned proposer + frozen detector）只在未来新 preregistration 中研究。Learned component只能提高安全 candidate diversity，不能读取 prompt/task ID/hidden state来检测，也不能替代 semantic encoder symbol。

## 10. 对关键问题的回答

1. **稳定 symbol**：只在 public AST statement/basic-block closure 后 canonicalize unit，编码 `signature + role/path + previous committed unit + current unit`。
2. **final-code deterministic context**：masked target-function signature、public role/path、current canonical unit、最多一个 preceding committed canonical unit。
3. **prefix mismatch**：generation 与 detector 调用同一 serializer；rejected/rolled-back unit 不进入 committed state。
4. **避免 O(B²)**：固定 `K=1`，每个 context 长度上限固定；不使用 growing whole prefix。
5. **20-way batch**：closure scheduler 聚合同一 wave 所有 ready contexts，一次 tokenizer 和一个或数个 frozen microbatches。
6. **key 真正生效**：observed bits 由 encoder+keyed planes得到，target由 content-derived unit ID+key得到，正式 EOS ranking使用累计 exact score。
7. **detector batch**：先完成一次 parse/context collection，再 batch tokenizer/encoder；禁止 per-block GPU forward。
8. **插删影响上界**：`K=1` 时删除一个 unit最多使该 unit和紧随 unit成为 erasure；content-derived IDs避免全局 ordinal shift。
9. **bit-identical**：context text/token IDs/int quantization/LSH bits/statistic必须相同；floating embedding仅在冻结 tolerance 内比较。
10. **occupancy bias**：calibration negatives、fresh held-out、wrong-key 与 per-bit/bucket occupancy共同审计；不假设理论 0.5。
11. **ECC位置**：Hamming code在 unit内定义 target codeword，跨 unit只做 interleaving/erasure-aware aggregation。
12. **Pass风险**：主臂不改 raw trajectories，只在 quality hard gate 后 ranking；hidden HumanEval tests只在选择完成后评价。

## 11. 可迁移与禁止迁移

可迁移：SemStamp 的 encoder→LSH symbol；SeqMark 的 region-collapse diagnostic；SAEMark 的 feature-guided selection；SemanticShift 的 bounded dynamic semantic difference；PICARD/SynCode 的 boundary/safety；RBC 的 erasure/ECC；ACW/CodeTracer 的 proposer思想；vLLM 的 batch/KV工程原则。

禁止迁移：prompt-dependent detector、generation ledger、saved embeddings、candidate index、retry index target、global ordinal target、whole-final candidate semantic rescore主循环、syntax spelling codebook、token green-list证据、private serialized hyperplanes、key fingerprint、hidden-test selection。

## 12. 文献驱动的正式实现约束

- schema 为 `wfcllm-dynamic-semantic-code-watermark/v3`，与 V1/V2/旧 V3 fail fast。
- encoder 是 observed symbol 的必要因果来源；constant/bypass ablation必须失败或显著改变 evidence。
- parser只决定 boundary/context/safety/sync；AST category不计 hit。
- generation score在 unit commit 时累计；EOS 不准对 20 full finals重新编码。
- detector输入仅 `{ "final_code": string }`；public config/artifacts与secret通过独立边界提供。
- public artifacts不含 secret、fingerprint、targets、private planes、raw ledger、prompt、task ID、candidate list或generation embeddings。
- opening held-out 前冻结 code/hash/threshold/arms/report/R3/profile；held-out只打开一次。

## 13. 局限

没有现成论文同时满足 code、dynamic block embedding、exact final-code replay、public frozen encoder、post-encoder secret、strict R3 和 legal held-out FPR。本方法是文献组件在更严格合同下的组合，必须由 contract prototype 和 pilot证伪。单 seed `20260713` 只能支持冻结配置结论，不能声称跨 seed 稳定；若 R3、FPR、TPR、Pass或成本门失败，将以负结果闭合，而不回退 syntax codebook。
