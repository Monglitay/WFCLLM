# C++ / Java 语言支持实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 为 WFCLLM 的语言注册层和 HumanEvalPack 数据层增加可离线验证的 C++、Java 支持。

**架构：** 复用现有 `LanguageAdapter` / `DatasetAdapter` 边界。语言侧用两个官方 Tree-sitter grammar 和共享 AST 遍历器；数据侧按 `bigcode/humanevalpack` 的语言 config 归一化为 `CodeSample`。

**技术栈：** Python 3.11、tree-sitter 0.25、tree-sitter-cpp、tree-sitter-java、datasets、pytest。

---

### 任务 1：锁定语言适配器契约

**文件：**
- 创建：`tests/lang/test_cpp_adapter.py`
- 创建：`tests/lang/test_java_adapter.py`
- 修改：`tests/lang/test_registry.py`

- [x] 编写注册、语句类型、嵌套 block、空规则列表测试。
- [x] 运行 `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang -q`，确认因 `cpp` / `java` 未注册而失败。

### 任务 2：实现 Tree-sitter 语言适配

**文件：**
- 创建：`wfcllm/lang/tree_sitter.py`
- 创建：`wfcllm/lang/cpp/parser.py`
- 创建：`wfcllm/lang/cpp/adapter.py`
- 创建：`wfcllm/lang/java/parser.py`
- 创建：`wfcllm/lang/java/adapter.py`
- 修改：`wfcllm/lang/cpp/__init__.py`
- 修改：`wfcllm/lang/java/__init__.py`
- 修改：`wfcllm/lang/__init__.py`
- 修改：`requirements.txt`

- [x] 实现 `SourceBlock` 和共享递归抽取器。
- [x] 为 C++ / Java 声明各自 simple / compound 节点集合和 parser。
- [x] 注册两个适配器，暂以空列表明确表示没有跨语言 transform rules。
- [x] 运行 `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang -q`，确认通过。

### 任务 3：实现 HumanEvalPack 加载

**文件：**
- 修改：`tests/datasets/test_adapters.py`
- 修改：`wfcllm/datasets/humanevalpack.py`

- [x] mock `datasets.load_dataset`，先写 config、cache、字段归一化和错误路径测试。
- [x] 运行数据集测试，确认 skeleton 的缺失实现失败。
- [x] 实现 `cpp` / `java` test split 加载、迭代和按语言查找。
- [x] 运行 `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/datasets -q`，确认通过。

### 任务 4：回归验证

**文件：**
- 验证：`wfcllm/`、`tests/lang/`、`tests/datasets/`

- [x] 运行 `HF_HUB_OFFLINE=1 conda run -n WFCLLM pytest tests/lang tests/datasets -q`。
- [x] 运行 `conda run -n WFCLLM python -m compileall wfcllm`。
- [x] 检查 `git diff --check` 和 `git status --short`，确认未触碰无关用户文件。
