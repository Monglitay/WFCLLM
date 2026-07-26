---
status: accepted
---

# 不归档 Gate Bundle，靠重跑 gate-train 复现

服务器 A 上有 32 个 Gate Bundle 共 4.4G（每个约 138M，主体是 `gate_float.pt`）。Gate Bundle 由配置、gate 数据、基座模型与随机种子确定性生成 —— `wfcllm/gate/trainer.py` 的 `seed_gate_training` 对 `random`/`numpy`/`torch`/`cuda` 全部设种（默认 `seed=7`，每 epoch 用 `seed+epoch`），训练子集也按 sha256 确定性排序。既然它是代码的产物而非不可再生的输入，我们一个都不取，改为在复现时重跑 gate-train 阶段。

## Consequences

- 重训出的 bundle 树哈希与历史 `gate_bundle_sha256` 不会相同，`resolve_validated_gate_bundle` 与 `wfcllm/cli/runners.py` 里的一致性核对会对不上历史值；历史报告里记的那些哈希只能当作已消失产物的指纹，不能当作可验证的锚点。
- `trainer.py` 未设 `cudnn.deterministic` / `use_deterministic_algorithms`，即使同种子，CUDA kernel 层面的不确定性仍可能让权重逐字节不同，进而让 detection 指标漂移。这落在「复现出差不多的指标」这一验收口径内；若漂移离谱，应当作代码逻辑缺陷来查，而不是补档 bundle。
- 有几个历史 run 跳过自己的 gate-train、直接复用别的 run 产出的 bundle 与 gate-cache（例如 Python gamma25 目标复用 `attempt14` 的）。这是仓库明文支持的一等模式 —— 带 hash 绑定的外部已验证 bundle 时，实验链从 `generate` 起跑并保留五个主阶段。因此丢弃 Gate Bundle 之后，这些目标必须改为各自跑完 gate-train，而非视作修正一处偷懒。
- 前一条有个前提需要单独查证：该模式要求被复用的 bundle 是「hash 绑定的外部已验证」bundle，而 `attempt14` 的 bundle 的 [[Suitable False Positive Rate]] 为 0.9886（准入上限 0.05），其 `validated: true` 仅表示混淆矩阵非退化、并不检查准入阈值。它是否真正满足该模式的前提，不在本轮处置范围内，但必须记录。
