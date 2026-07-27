---
status: accepted
---

# Semantic encoder 按数据集在每次 run 中训练

每个 Full Reproduction Profile 必须先执行 `encoder`，使用该语言/数据设置的
full Gate source catalog 训练 semantic projection。后续 Gate、generation
和 detection 都绑定这一 checkpoint 的身份。

Python 正样本来自 AST 正变换；C++、Java 来自公开语句等价变换；JavaScript
使用 dependency-light lexical-trivia 等价变换，不要求 JavaScript
tree-sitter wheel。语言语料不足时应补充本地 source catalog，不能降低 full
profile 的训练合同或复用历史 checkpoint。
