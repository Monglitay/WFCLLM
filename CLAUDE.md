# CLAUDE.md

WFCLLM 当前工作树是 Gate-only Full Reproduction Core。

## 不可变边界

1. 唯一方法是 `gated_semantic_window_v1`。
2. 唯一公开 profile 是 full；公开矩阵恰好五项。
3. 唯一阶段链是
   `encoder → gate-data → gate-train → generate → calibrate → detect → report → audit`。
4. 每次执行从新的 root 开始，并训练本 run 的 encoder 与 Gate Bundle。
5. 不接受旧状态、外部历史 Bundle、旧 candidate/schema/report 或兼容参数。
6. 缺少本地资源必须失败，不得静默替代。
7. detector positive input 必须严格为
   `id,dataset,prompt,final_code` 四字段。
8. Pass@1/Pass@k、test 和 correctness 只能在生成后使用，不得反馈到生成、
   重试、选择、校准或检测；Metric Contract 只消费 Pass@1。
9. 私有 key material 永不进入公开 artifact、状态、日志、配置或文档。
10. 历史实现只从 Git 历史和既有 Archival Tag 恢复。

## 开发约定

- 使用 `WFCLLM` conda 环境，常规测试默认离线。
- 不安装新依赖来绕过本地资源问题。
- 优先测试 Public Execution Surface，再补无法经济覆盖的合同单测。
- 测试可注入轻量依赖，但不能进入生产入口；测试产物保持非正式身份。
- 修改后运行 compileall、离线 collect-only、相关目标套件、完整 `tests/`、
  shell syntax、状态命令和 removed-concept residue scan。
- 保留用户已有工作树改动。未经明确授权不执行 Git 发布操作。

当前结构、命令和 artifact 细节见 `README.md`、`AGENTS.md` 与 `docs/`。
