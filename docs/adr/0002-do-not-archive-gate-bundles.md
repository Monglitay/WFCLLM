---
status: accepted
---

# 不归档 Gate Bundle，靠重跑 gate-train 复现

服务器 A 上有 32 个 Gate Bundle 共 4.4G（每个约 138M，主体是 `gate_float.pt`）。Gate Bundle 由配置、gate 数据、基座模型与随机种子确定性生成 —— `wfcllm/gate/trainer.py` 的 `seed_gate_training` 对 `random`/`numpy`/`torch`/`cuda` 全部设种（默认 `seed=7`，每 epoch 用 `seed+epoch`），训练子集也按 sha256 确定性排序。既然它是代码的产物而非不可再生的输入，我们一个都不取，改为在复现时重跑 gate-train 阶段。

## Consequences

- 重训出的 bundle 树哈希与历史 `gate_bundle_sha256` 不会相同，`resolve_validated_gate_bundle` 与 `wfcllm/cli/runners.py` 里的一致性核对会对不上历史值；历史报告里记的那些哈希只能当作已消失产物的指纹，不能当作可验证的锚点。
- `trainer.py` 未设 `cudnn.deterministic` / `use_deterministic_algorithms`，即使同种子，CUDA kernel 层面的不确定性仍可能让权重逐字节不同，进而让 detection 指标漂移。这落在「复现出差不多的指标」这一验收口径内；若漂移离谱，应当作代码逻辑缺陷来查，而不是补档 bundle。
- 有几个历史 run 的启动脚本跳过了自己的 gate-train、直接指向别的 run 的输出目录（例如 `/tmp/run_gate_on_fixed_base.py` 复用 `attempt14` 的 bundle 与 gate-cache）。这是 run 脚本的偷懒而非 Gate Bundle 的固有性质；复现这些目标时必须让它们各自跑完 gate-train。
