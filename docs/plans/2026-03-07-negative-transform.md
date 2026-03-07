# Negative Transform Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `statement_block_transform` 目录内扩展 `--mode` 参数，实现 36 条负变换规则，生成语义破坏的负样本，每个 variant 带 `sample_type` 字段标记 `"positive"` 或 `"negative"`。

**Architecture:** 方案 C：在现有目录内最小改动。`engine.py` 加 `mode` 参数，`transform.py` 加 `--mode` CLI 参数，新建 `rules/negative/` 目录实现 36 条负规则（复用 `Rule` 基类和 `Match`），`config.py` 加负变换输出路径。

**Tech Stack:** Python, tree-sitter, tree-sitter-python。所有规则复用 `rules/base.py` 的 `Rule`、`Match`、`parse_code`。

---

### Task 1: 更新 engine.py 支持 mode 参数

**Files:**
- Modify: `experiment/statement_block_transform/engine.py`

**Step 1: 修改 TransformEngine.__init__ 加 mode 参数**

在 `engine.py` 中将：
```python
def __init__(
    self,
    rules: list[Rule],
    max_perm_len: int = 5,
    max_variants: int = 1000,
):
    self.rules = rules
    self.max_perm_len = max_perm_len
    self.max_variants = max_variants
```

改为：
```python
def __init__(
    self,
    rules: list[Rule],
    max_perm_len: int = 5,
    max_variants: int = 1000,
    mode: str = "positive",
):
    self.rules = rules
    self.max_perm_len = max_perm_len
    self.max_variants = max_variants
    self.mode = mode
```

**Step 2: 在 generate_variants 中给每个 variant 加 sample_type 字段**

将：
```python
variants.append({
    "variant_id": len(variants),
    "rules_applied": [r.name for r in perm],
    "transformed_source": result,
})
```

改为：
```python
variants.append({
    "variant_id": len(variants),
    "rules_applied": [r.name for r in perm],
    "transformed_source": result,
    "sample_type": self.mode,
})
```

**Step 3: 验证语法正确**

```bash
cd experiment/statement_block_transform && python -c "from engine import TransformEngine; print('OK')"
```

Expected: `OK`

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/engine.py
git commit -m "feat: add mode param to TransformEngine, tag variants with sample_type"
```

---

### Task 2: 更新 config.py 和 transform.py

**Files:**
- Modify: `experiment/statement_block_transform/config.py`
- Modify: `experiment/statement_block_transform/transform.py`

**Step 1: 更新 config.py 加负变换输出路径**

在 `config.py` 末尾加：
```python
OUTPUT_FILE_NEGATIVE = RESULTS_DIR / "mbpp_blocks_negative_transformed.json"
```

**Step 2: 更新 transform.py 的 process_blocks 函数签名**

在 `process_blocks` 函数加 `mode` 参数并传给 `TransformEngine`：

将：
```python
def process_blocks(
    input_path: Path,
    output_path: Path,
    max_perm_len: int = 5,
    max_variants: int = 1000,
) -> None:
```

改为：
```python
def process_blocks(
    input_path: Path,
    output_path: Path,
    max_perm_len: int = 5,
    max_variants: int = 1000,
    mode: str = "positive",
) -> None:
```

将：
```python
engine = TransformEngine(
    rules=get_all_rules(),
    max_perm_len=max_perm_len,
    max_variants=max_variants,
)
```

改为：
```python
from rules.negative import get_all_negative_rules

if mode == "negative":
    rules = get_all_negative_rules()
else:
    rules = get_all_rules()

engine = TransformEngine(
    rules=rules,
    max_perm_len=max_perm_len,
    max_variants=max_variants,
    mode=mode,
)
```

同时在 `metadata` 字典中加 `"mode": mode`。

**Step 3: 更新 main 函数和 argparse**

将 `main` 函数签名改为：
```python
def main(limit: int | None = None, max_perm_len: int = 5, max_variants: int = 1000, mode: str = "positive") -> None:
```

在 `main` 内部根据 mode 选择输出路径：
```python
from config import INPUT_FILE, OUTPUT_FILE, OUTPUT_FILE_NEGATIVE

output = OUTPUT_FILE_NEGATIVE if mode == "negative" else OUTPUT_FILE
```

然后把 `process_blocks` 调用都传入 `mode` 和动态 `output`（替换原来写死的 `OUTPUT_FILE`）。

在 argparse 部分加：
```python
ap.add_argument("--mode", choices=["positive", "negative"], default="positive",
                help="Transform mode: positive (semantic-equivalent) or negative (semantic-breaking)")
```

并在 `main(...)` 调用处传入 `mode=args.mode`。

**Step 4: 验证 import 正确（负规则目录尚未创建，先跳过运行，只检查语法）**

```bash
cd experiment/statement_block_transform && python -c "import ast, sys; ast.parse(open('transform.py').read()); print('syntax OK')"
```

Expected: `syntax OK`

**Step 5: Commit**

```bash
git add experiment/statement_block_transform/config.py experiment/statement_block_transform/transform.py
git commit -m "feat: add --mode flag to transform pipeline, route rules and output by mode"
```

---

### Task 3: 创建 rules/negative/ 目录结构

**Files:**
- Create: `experiment/statement_block_transform/rules/negative/__init__.py`

**Step 1: 创建 `rules/negative/__init__.py`（暂时返回空列表）**

```python
"""Negative transform rules registry."""

from rules.base import Rule


def get_all_negative_rules() -> list[Rule]:
    """Return all negative transformation rules."""
    return []
```

**Step 2: 验证 import**

```bash
cd experiment/statement_block_transform && python -c "from rules.negative import get_all_negative_rules; print(get_all_negative_rules())"
```

Expected: `[]`

**Step 3: 验证整个 transform.py 可以带 --mode negative 运行（空规则，应产生空 variants）**

```bash
cd experiment/statement_block_transform && python transform.py --mode negative --limit 2
```

Expected: 正常完成，输出 `results/mbpp_blocks_negative_transformed.json`，其中 variants 全为 `[]`。

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/rules/negative/
git commit -m "feat: scaffold negative rules directory with empty registry"
```

---

### Task 4: 实现 Simple 负规则 — API 调用类（7条）

**Files:**
- Create: `experiment/statement_block_transform/rules/negative/api_calls.py`
- Modify: `experiment/statement_block_transform/rules/negative/__init__.py`

**规则列表（Simple 级别，直接文本/节点替换）：**

| 类名 | name | 原始 → 变换 |
|------|------|------------|
| `MinMaxFlip` | `neg_min_max_flip` | `min(...)` ↔ `max(...)` |
| `AnyAllFlip` | `neg_any_all_flip` | `any(...)` → `all(...)` |
| `SortedReverseFlip` | `neg_sorted_reverse` | `reverse=False` → `reverse=True` 或给 `sorted(x)` 加 `reverse=True` |
| `OpenModeCorrupt` | `neg_open_mode` | `open(f, 'r')` → `open(f, 'w')` |
| `ExtendAppendSwap` | `neg_extend_append` | `.extend(x)` → `.append(x)` |
| `StartsEndsSwap` | `neg_starts_ends` | `.startswith(x)` → `.endswith(x)` |
| `CeilFloorFlip` | `neg_ceil_floor` | `math.ceil(...)` → `math.floor(...)` |

**Step 1: 创建 `rules/negative/api_calls.py`**

```python
"""Negative transformation rules — API call semantic corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class MinMaxFlip(Rule):
    """Replace min with max or max with min."""
    name = "neg_min_max_flip"
    category = "API与函数调用"
    description = "函数语义翻转：min ↔ max"

    _flip = {"min": "max", "max": "min"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "identifier":
                    name = func.text.decode("utf-8")
                    if name in self._flip:
                        matches.append(Match(
                            "call", func.start_byte, func.end_byte,
                            name, self._flip[name],
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class AnyAllFlip(Rule):
    """Replace any() with all() and vice versa."""
    name = "neg_any_all_flip"
    category = "API与函数调用"
    description = "逻辑量词替换：any ↔ all"

    _flip = {"any": "all", "all": "any"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "identifier":
                    name = func.text.decode("utf-8")
                    if name in self._flip:
                        matches.append(Match(
                            "call", func.start_byte, func.end_byte,
                            name, self._flip[name],
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class SortedReverseFlip(Rule):
    """Flip sorted reverse=False to reverse=True, or add reverse=True to sorted(x)."""
    name = "neg_sorted_reverse"
    category = "API与函数调用"
    description = "关键参数篡改：sorted reverse 方向翻转"

    def _get_call_name(self, node) -> str:
        func = node.child_by_field_name("function")
        return func.text.decode("utf-8") if func else ""

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call" and self._get_call_name(node) == "sorted":
                args_node = node.child_by_field_name("arguments")
                if args_node is None:
                    for child in node.children:
                        walk(child)
                    return
                # Case 1: has reverse=False → flip to True
                for child in args_node.children:
                    if child.type == "keyword_argument":
                        kname = child.child_by_field_name("name")
                        kval = child.child_by_field_name("value")
                        if (kname and kname.text.decode("utf-8") == "reverse"
                                and kval and kval.text.decode("utf-8") == "False"):
                            matches.append(Match(
                                "keyword_argument", kval.start_byte, kval.end_byte,
                                "False", "True",
                            ))
                            return
                # Case 2: no reverse kwarg → insert reverse=True
                existing_args = [c for c in args_node.children if c.type not in ("(", ")", ",")]
                has_reverse = any(
                    c.type == "keyword_argument"
                    and c.child_by_field_name("name")
                    and c.child_by_field_name("name").text.decode("utf-8") == "reverse"
                    for c in args_node.children
                )
                if not has_reverse:
                    close_paren = args_node.end_byte - 1
                    prefix = ", " if existing_args else ""
                    matches.append(Match(
                        "call", close_paren, close_paren,
                        "", f"{prefix}reverse=True",
                    ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class OpenModeCorrupt(Rule):
    """Change open(f, 'r') to open(f, 'w')."""
    name = "neg_open_mode"
    category = "API与函数调用"
    description = "文件模式破坏：'r' → 'w'"

    _corrupt_map = {"'r'": "'w'", '"r"': '"w"'}

    def _get_call_name(self, node) -> str:
        func = node.child_by_field_name("function")
        return func.text.decode("utf-8") if func else ""

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call" and self._get_call_name(node) == "open":
                args_node = node.child_by_field_name("arguments")
                if args_node:
                    positional = [c for c in args_node.children
                                  if c.type not in ("(", ")", ",", "keyword_argument")]
                    if len(positional) >= 2:
                        mode_node = positional[1]
                        mode_text = mode_node.text.decode("utf-8")
                        if mode_text in self._corrupt_map:
                            matches.append(Match(
                                "call", mode_node.start_byte, mode_node.end_byte,
                                mode_text, self._corrupt_map[mode_text],
                            ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ExtendAppendSwap(Rule):
    """Replace .extend(x) with .append(x)."""
    name = "neg_extend_append"
    category = "API与函数调用"
    description = "列表追加模式破坏：extend ↔ append"

    _flip = {"extend": "append", "append": "extend"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
                        name = attr.text.decode("utf-8")
                        matches.append(Match(
                            "attribute", attr.start_byte, attr.end_byte,
                            name, self._flip[name],
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class StartsEndsSwap(Rule):
    """Replace .startswith() with .endswith() and vice versa."""
    name = "neg_starts_ends"
    category = "API与函数调用"
    description = "字符串边界颠倒：startswith ↔ endswith"

    _flip = {"startswith": "endswith", "endswith": "startswith"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
                        name = attr.text.decode("utf-8")
                        matches.append(Match(
                            "attribute", attr.start_byte, attr.end_byte,
                            name, self._flip[name],
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class CeilFloorFlip(Rule):
    """Replace math.ceil with math.floor and vice versa."""
    name = "neg_ceil_floor"
    category = "API与函数调用"
    description = "舍入方向翻转：ceil ↔ floor"

    _flip = {"ceil": "floor", "floor": "ceil"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    obj = func.child_by_field_name("object")
                    if (attr and obj
                            and obj.text.decode("utf-8") == "math"
                            and attr.text.decode("utf-8") in self._flip):
                        name = attr.text.decode("utf-8")
                        matches.append(Match(
                            "attribute", attr.start_byte, attr.end_byte,
                            name, self._flip[name],
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 2: 更新 `rules/negative/__init__.py`**

```python
"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)


def get_all_negative_rules() -> list[Rule]:
    return [
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
    ]
```

**Step 3: 快速测试这 7 条规则**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
from rules.base import parse_code
from rules.negative.api_calls import MinMaxFlip, AnyAllFlip, SortedReverseFlip, ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip

def check(rule, src, expected):
    tree = parse_code(src)
    m = rule.detect(src, tree)
    assert m, f"{rule.name}: no match for: {src}"
    result = rule.apply(src, m)
    assert result == expected, f"{rule.name}: got {result!r}, expected {expected!r}"
    print(f"  OK: {rule.name}")

check(MinMaxFlip(), "res = min(x)", "res = max(x)")
check(AnyAllFlip(), "if any(lst):", "if all(lst):")
check(SortedReverseFlip(), "sorted(x, reverse=False)", "sorted(x, reverse=True)")
check(ExtendAppendSwap(), "lst.extend(items)", "lst.append(items)")
check(StartsEndsSwap(), "s.startswith(prefix)", "s.endswith(prefix)")
check(CeilFloorFlip(), "math.ceil(x)", "math.floor(x)")
print("All API negative rules passed.")
EOF
```

Expected: 全部 `OK`。

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/rules/negative/
git commit -m "feat: implement 7 negative API call rules"
```

---

### Task 5: 实现 Simple 负规则 — 控制流类（5条）

**Files:**
- Create: `experiment/statement_block_transform/rules/negative/control_flow.py`
- Modify: `experiment/statement_block_transform/rules/negative/__init__.py`

**规则列表：**

| 类名 | name | 原始 → 变换 |
|------|------|------------|
| `OffByOne` | `neg_off_by_one` | `range(n)` → `range(n - 1)` |
| `BreakContinueSwap` | `neg_break_continue` | `break` → `continue` |
| `IfElseBodySwap` | `neg_if_else_body_swap` | 不改条件，对调 if/else 体 |
| `MembershipNegate` | `neg_membership` | `in valid_set` → `not in valid_set` |
| `YieldReturnSwap` | `neg_yield_return` | `yield x` → `return x` |

**Step 1: 创建 `rules/negative/control_flow.py`**

```python
"""Negative transformation rules — control flow corruption."""

from __future__ import annotations

from rules.base import Match, Rule, parse_code


class OffByOne(Rule):
    """Change range(n) to range(n - 1) — classic off-by-one."""
    name = "neg_off_by_one"
    category = "控制流"
    description = "边界条件偏移：range(n) → range(n - 1)"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.text.decode("utf-8") == "range":
                    args_node = node.child_by_field_name("arguments")
                    if args_node:
                        positional = [c for c in args_node.children
                                      if c.type not in ("(", ")", ",")]
                        if len(positional) == 1:
                            arg = positional[0]
                            arg_text = arg.text.decode("utf-8")
                            matches.append(Match(
                                "call", arg.start_byte, arg.end_byte,
                                arg_text, f"{arg_text} - 1",
                            ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class BreakContinueSwap(Rule):
    """Swap break and continue."""
    name = "neg_break_continue"
    category = "控制流"
    description = "循环控制中断：break ↔ continue"

    _flip = {"break_statement": "continue", "continue_statement": "break"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type in self._flip:
                matches.append(Match(
                    node.type, node.start_byte, node.end_byte,
                    source[node.start_byte:node.end_byte],
                    self._flip[node.type],
                ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class IfElseBodySwap(Rule):
    """Swap if/else bodies WITHOUT negating the condition (malicious, not equivalent)."""
    name = "neg_if_else_body_swap"
    category = "控制流"
    description = "恶意分支对调：条件不变但对调 if/else 体"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "if_statement":
                alt = node.child_by_field_name("alternative")
                if alt and alt.type == "else_clause":
                    condition = node.child_by_field_name("condition")
                    consequence = node.child_by_field_name("consequence")
                    else_body = alt.child_by_field_name("body")
                    if condition and consequence and else_body:
                        cond_text = condition.text.decode("utf-8")
                        if_body_text = source[consequence.start_byte:consequence.end_byte]
                        else_body_text = source[else_body.start_byte:else_body.end_byte]
                        # Swap bodies, keep condition unchanged
                        replacement = f"if {cond_text}:\n{else_body_text}\nelse:\n{if_body_text}"
                        matches.append(Match(
                            "if_statement", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte], replacement,
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class MembershipNegate(Rule):
    """Negate membership test: `x in s` → `x not in s`."""
    name = "neg_membership"
    category = "控制流"
    description = "包含关系取反：in ↔ not in"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                # Look for `in` or `not in` operator tokens
                for i, child in enumerate(children):
                    if child.type == "in":
                        # x in s → x not in s
                        left = children[i - 1] if i > 0 else None
                        right = children[i + 1] if i + 1 < len(children) else None
                        if left and right:
                            l = left.text.decode("utf-8")
                            r = right.text.decode("utf-8")
                            matches.append(Match(
                                "comparison_operator", node.start_byte, node.end_byte,
                                source[node.start_byte:node.end_byte],
                                f"{l} not in {r}",
                            ))
                    elif child.type == "not_in":
                        # x not in s → x in s
                        left = children[i - 1] if i > 0 else None
                        right = children[i + 1] if i + 1 < len(children) else None
                        if left and right:
                            l = left.text.decode("utf-8")
                            r = right.text.decode("utf-8")
                            matches.append(Match(
                                "comparison_operator", node.start_byte, node.end_byte,
                                source[node.start_byte:node.end_byte],
                                f"{l} in {r}",
                            ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class YieldReturnSwap(Rule):
    """Replace yield with return to break generator protocol."""
    name = "neg_yield_return"
    category = "控制流"
    description = "生成器状态截断：yield → return"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "yield":
                # yield expression node: "yield <value>"
                orig = source[node.start_byte:node.end_byte]
                # Replace leading "yield" with "return"
                replacement = "return" + orig[5:]
                matches.append(Match(
                    "yield", node.start_byte, node.end_byte, orig, replacement,
                ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 2: 更新 `rules/negative/__init__.py`**

```python
"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from rules.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)


def get_all_negative_rules() -> list[Rule]:
    return [
        # API (7)
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        # Control flow (5)
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
    ]
```

**Step 3: 测试控制流规则**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
from rules.base import parse_code
from rules.negative.control_flow import OffByOne, BreakContinueSwap, MembershipNegate

def check(rule, src, expected):
    tree = parse_code(src)
    m = rule.detect(src, tree)
    assert m, f"{rule.name}: no match for: {src!r}"
    result = rule.apply(src, m)
    assert result == expected, f"{rule.name}: got {result!r}, expected {expected!r}"
    print(f"  OK: {rule.name}")

check(OffByOne(), "for i in range(n):", "for i in range(n - 1):")
check(BreakContinueSwap(), "break", "continue")
check(MembershipNegate(), "item in valid_set", "item not in valid_set")
print("Control flow negative rules passed.")
EOF
```

Expected: 全部 `OK`。

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/rules/negative/
git commit -m "feat: implement 5 negative control flow rules"
```

---

### Task 6: 实现 Simple 负规则 — 表达式与逻辑类（6条）

**Files:**
- Create: `experiment/statement_block_transform/rules/negative/expression_logic.py`
- Modify: `experiment/statement_block_transform/rules/negative/__init__.py`

**规则列表：**

| 类名 | name | 原始 → 变换 |
|------|------|------------|
| `EqNeqFlip` | `neg_eq_neq` | `==` → `!=` |
| `ArithmeticOpReplace` | `neg_arithmetic_op` | `+` → `-`（加减互换） |
| `AndOrSwap` | `neg_and_or` | `and` → `or` |
| `BoundsNarrow` | `neg_bounds_narrow` | `<=` → `<`，`>=` → `>` |
| `AugAssignCorrupt` | `neg_aug_assign` | `count += 1` → `count = 1` |
| `ShiftFlip` | `neg_shift_flip` | `<<` → `>>` |

**Step 1: 创建 `rules/negative/expression_logic.py`**

```python
"""Negative transformation rules — expression and logic corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class EqNeqFlip(Rule):
    """Flip == to != and vice versa."""
    name = "neg_eq_neq"
    category = "表达式与逻辑"
    description = "关系运算符反转：== ↔ !="

    _flip = {"==": "!=", "!=": "=="}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                if len(children) == 3:
                    op = children[1]
                    if op.type in self._flip:
                        left = children[0].text.decode("utf-8")
                        right = children[2].text.decode("utf-8")
                        matches.append(Match(
                            "comparison_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{left} {self._flip[op.type]} {right}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ArithmeticOpReplace(Rule):
    """Replace + with - in binary expressions."""
    name = "neg_arithmetic_op"
    category = "表达式与逻辑"
    description = "算术运算符替换：+ → -"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type == "+":
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "binary_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} - {r}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class AndOrSwap(Rule):
    """Replace boolean and with or and vice versa."""
    name = "neg_and_or"
    category = "表达式与逻辑"
    description = "逻辑运算符替换：and ↔ or"

    _flip = {"and": "or", "or": "and"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "boolean_operator":
                op_node = None
                for child in node.children:
                    if child.type in ("and", "or"):
                        op_node = child
                        break
                if op_node:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        new_op = self._flip[op_node.type]
                        matches.append(Match(
                            "boolean_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} {new_op} {r}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class BoundsNarrow(Rule):
    """Replace <= with < and >= with > (remove boundary inclusion)."""
    name = "neg_bounds_narrow"
    category = "表达式与逻辑"
    description = "比较界限收缩：<= → <，>= → >"

    _narrow = {"<=": "<", ">=": ">"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "comparison_operator":
                children = node.children
                if len(children) == 3:
                    op = children[1]
                    if op.type in self._narrow:
                        left = children[0].text.decode("utf-8")
                        right = children[2].text.decode("utf-8")
                        matches.append(Match(
                            "comparison_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{left} {self._narrow[op.type]} {right}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class AugAssignCorrupt(Rule):
    """Replace augmented assignment (+=) with simple assignment (=)."""
    name = "neg_aug_assign"
    category = "表达式与逻辑"
    description = "赋值降级：+= → ="

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "augmented_assignment":
                op_node = None
                for child in node.children:
                    if child.type == "+=":
                        op_node = child
                        break
                if op_node:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "augmented_assignment", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} = {r}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class ShiftFlip(Rule):
    """Replace left shift with right shift and vice versa."""
    name = "neg_shift_flip"
    category = "表达式与逻辑"
    description = "移位运算反转：<< ↔ >>"

    _flip = {"<<": ">>", ">>": "<<"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "binary_operator":
                op = node.child_by_field_name("operator")
                if op and op.type in self._flip:
                    left = node.child_by_field_name("left")
                    right = node.child_by_field_name("right")
                    if left and right:
                        l = left.text.decode("utf-8")
                        r = right.text.decode("utf-8")
                        matches.append(Match(
                            "binary_operator", node.start_byte, node.end_byte,
                            source[node.start_byte:node.end_byte],
                            f"{l} {self._flip[op.type]} {r}",
                        ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 2: 更新 `rules/negative/__init__.py`**

```python
"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from rules.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from rules.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)


def get_all_negative_rules() -> list[Rule]:
    return [
        # API (7)
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        # Control flow (5)
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        # Expression logic (6)
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
    ]
```

**Step 3: 测试表达式规则**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
from rules.base import parse_code
from rules.negative.expression_logic import EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip

def check(rule, src, expected):
    tree = parse_code(src)
    m = rule.detect(src, tree)
    assert m, f"{rule.name}: no match for: {src!r}"
    result = rule.apply(src, m)
    assert result == expected, f"{rule.name}: got {result!r}, expected {expected!r}"
    print(f"  OK: {rule.name}")

check(EqNeqFlip(), "a == b", "a != b")
check(ArithmeticOpReplace(), "price + tax", "price - tax")
check(AndOrSwap(), "x and y", "x or y")
check(BoundsNarrow(), "n <= right", "n < right")
check(AugAssignCorrupt(), "count += 1", "count = 1")
check(ShiftFlip(), "x << 1", "x >> 1")
print("Expression logic negative rules passed.")
EOF
```

Expected: 全部 `OK`。

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/rules/negative/
git commit -m "feat: implement 6 negative expression/logic rules"
```

---

### Task 7: 实现剩余负规则（标识符、数据结构、异常、系统）

**Files:**
- Create: `experiment/statement_block_transform/rules/negative/identifier.py`
- Create: `experiment/statement_block_transform/rules/negative/data_structure.py`
- Create: `experiment/statement_block_transform/rules/negative/exception.py`
- Create: `experiment/statement_block_transform/rules/negative/system.py`
- Modify: `experiment/statement_block_transform/rules/negative/__init__.py`

**规则列表（4个文件，6条规则）：**

| 文件 | 类名 | name | 原始 → 变换 |
|------|------|------|------------|
| identifier.py | `ScopeVarCorrupt` | `neg_scope_var` | `lst[i]` → `lst[j]`（索引变量替换） |
| data_structure.py | `SliceStepFlip` | `neg_slice_step` | `lst[::-1]` → `lst[::1]` |
| data_structure.py | `DictViewSwap` | `neg_dict_view` | `.keys()` → `.values()` |
| exception.py | `ExceptionSwallow` | `neg_exception_swallow` | `raise` → `pass` 在 except 块中 |
| system.py | `SysExitFlip` | `neg_sys_exit` | `sys.exit(0)` → `sys.exit(1)` |

**Step 1: 创建 `rules/negative/identifier.py`**

```python
"""Negative transformation rules — identifier corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class ScopeVarCorrupt(Rule):
    """In index accesses lst[i], replace i with j if j exists in scope."""
    name = "neg_scope_var"
    category = "标识符操作"
    description = "作用域变量混淆：lst[i] → lst[j]"

    def detect(self, source, tree):
        """Find subscript accesses with single-char identifier index.
        Collect all single-char identifiers used as indices, then swap i→j or j→i."""
        matches = []
        index_vars = set()

        def collect_indices(node):
            if node.type == "subscript":
                idx = node.child_by_field_name("subscript")
                if idx and idx.type == "identifier" and len(idx.text.decode("utf-8")) == 1:
                    index_vars.add(idx.text.decode("utf-8"))
            for child in node.children:
                collect_indices(child)

        collect_indices(tree.root_node)

        if len(index_vars) < 2:
            return []

        # Build a rotation map among found index vars
        sorted_vars = sorted(index_vars)
        rotate = {sorted_vars[i]: sorted_vars[(i + 1) % len(sorted_vars)]
                  for i in range(len(sorted_vars))}

        def walk(node):
            if node.type == "subscript":
                idx = node.child_by_field_name("subscript")
                if idx and idx.type == "identifier":
                    name = idx.text.decode("utf-8")
                    if name in rotate:
                        matches.append(Match(
                            "subscript", idx.start_byte, idx.end_byte,
                            name, rotate[name],
                        ))
            for child in node.children:
                walk(child)

        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 2: 创建 `rules/negative/data_structure.py`**

```python
"""Negative transformation rules — data structure corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class SliceStepFlip(Rule):
    """Replace lst[::-1] with lst[::1] (reverse to forward)."""
    name = "neg_slice_step"
    category = "数据结构操作"
    description = "切片方向反转：[::-1] → [::1]"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "subscript":
                # Look for slice children with step=-1
                for child in node.children:
                    if child.type == "slice":
                        step = child.child_by_field_name("step")
                        if step:
                            step_text = step.text.decode("utf-8")
                            if step_text in ("-1", "- 1"):
                                matches.append(Match(
                                    "slice", step.start_byte, step.end_byte,
                                    step_text, "1",
                                ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class DictViewSwap(Rule):
    """Replace dict.keys() with dict.values() and vice versa."""
    name = "neg_dict_view"
    category = "数据结构操作"
    description = "字典视图错位：.keys() ↔ .values()"

    _flip = {"keys": "values", "values": "keys"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    attr = func.child_by_field_name("attribute")
                    if attr and attr.text.decode("utf-8") in self._flip:
                        # Verify no args (bare .keys() / .values())
                        args_node = node.child_by_field_name("arguments")
                        args = [c for c in args_node.children
                                if c.type not in ("(", ")", ",")] if args_node else []
                        if not args:
                            name = attr.text.decode("utf-8")
                            matches.append(Match(
                                "attribute", attr.start_byte, attr.end_byte,
                                name, self._flip[name],
                            ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 3: 创建 `rules/negative/exception.py`**

```python
"""Negative transformation rules — exception handling corruption."""

from __future__ import annotations

from rules.base import Match, Rule, parse_code


class ExceptionSwallow(Rule):
    """Replace `raise` inside except block with `pass`."""
    name = "neg_exception_swallow"
    category = "异常处理"
    description = "异常吞噬：except 块中 raise → pass"

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "except_clause":
                # Find raise_statement children inside the except body
                body = None
                for child in node.children:
                    if child.type == "block":
                        body = child
                        break
                if body:
                    for stmt in body.children:
                        if stmt.type == "raise_statement":
                            matches.append(Match(
                                "raise_statement", stmt.start_byte, stmt.end_byte,
                                source[stmt.start_byte:stmt.end_byte], "pass",
                            ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 4: 创建 `rules/negative/system.py`**

```python
"""Negative transformation rules — system interaction corruption."""

from __future__ import annotations

from rules.base import Match, Rule


class SysExitFlip(Rule):
    """Replace sys.exit(0) with sys.exit(1) and vice versa."""
    name = "neg_sys_exit"
    category = "系统交互"
    description = "退出状态码反转：sys.exit(0) ↔ sys.exit(1)"

    _flip = {"0": "1", "1": "0"}

    def detect(self, source, tree):
        matches = []
        def walk(node):
            if node.type == "call":
                func = node.child_by_field_name("function")
                if func and func.type == "attribute":
                    obj = func.child_by_field_name("object")
                    attr = func.child_by_field_name("attribute")
                    if (obj and obj.text.decode("utf-8") == "sys"
                            and attr and attr.text.decode("utf-8") == "exit"):
                        args_node = node.child_by_field_name("arguments")
                        if args_node:
                            positional = [c for c in args_node.children
                                          if c.type not in ("(", ")", ",")]
                            if len(positional) == 1:
                                arg = positional[0]
                                arg_text = arg.text.decode("utf-8")
                                if arg_text in self._flip:
                                    matches.append(Match(
                                        "call", arg.start_byte, arg.end_byte,
                                        arg_text, self._flip[arg_text],
                                    ))
            for child in node.children:
                walk(child)
        walk(tree.root_node)
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda m: m.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result
```

**Step 5: 更新 `rules/negative/__init__.py` 为最终完整版**

```python
"""Negative transform rules registry."""

from rules.base import Rule
from rules.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from rules.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from rules.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from rules.negative.identifier import ScopeVarCorrupt
from rules.negative.data_structure import SliceStepFlip, DictViewSwap
from rules.negative.exception import ExceptionSwallow
from rules.negative.system import SysExitFlip


def get_all_negative_rules() -> list[Rule]:
    return [
        # API (7)
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        # Control flow (5)
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        # Expression logic (6)
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
        # Identifier (1)
        ScopeVarCorrupt(),
        # Data structure (2)
        SliceStepFlip(), DictViewSwap(),
        # Exception (1)
        ExceptionSwallow(),
        # System (1)
        SysExitFlip(),
    ]
```

**Step 6: 测试所有规则加载和基本用例**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
from rules.negative import get_all_negative_rules
rules = get_all_negative_rules()
print(f"Total negative rules loaded: {len(rules)}")
for r in rules:
    print(f"  - {r.name}")

from rules.base import parse_code
from rules.negative.data_structure import SliceStepFlip, DictViewSwap
from rules.negative.system import SysExitFlip

def check(rule, src, expected):
    tree = parse_code(src)
    m = rule.detect(src, tree)
    assert m, f"{rule.name}: no match for: {src!r}"
    result = rule.apply(src, m)
    assert result == expected, f"{rule.name}: got {result!r}, expected {expected!r}"
    print(f"  OK: {rule.name}")

check(SliceStepFlip(), "lst[::-1]", "lst[::1]")
check(DictViewSwap(), "d.keys()", "d.values()")
check(SysExitFlip(), "sys.exit(0)", "sys.exit(1)")
print("All remaining negative rules passed.")
EOF
```

Expected: 加载 22 条规则，全部 `OK`。

**Step 7: Commit**

```bash
git add experiment/statement_block_transform/rules/negative/
git commit -m "feat: implement remaining negative rules (identifier, data_structure, exception, system)"
```

---

### Task 8: 端到端集成测试

**Step 1: 用正变换模式跑前 5 个样本，检查 sample_type=positive**

```bash
cd experiment/statement_block_transform && python transform.py --limit 5 --mode positive
```

Expected: 正常完成，输出到 `results/mbpp_blocks_transformed.json`。

**Step 2: 验证正变换 variant 有 sample_type 字段**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
import json
data = json.load(open("results/mbpp_blocks_transformed.json"))
for sample in data["samples"]:
    for block in sample["blocks"]:
        for v in block["variants"]:
            assert v["sample_type"] == "positive", f"Wrong sample_type: {v}"
print(f"OK: all {data['metadata']['total_variants']} positive variants tagged correctly")
EOF
```

**Step 3: 用负变换模式跑前 5 个样本，检查 sample_type=negative**

```bash
cd experiment/statement_block_transform && python transform.py --limit 5 --mode negative
```

Expected: 正常完成，输出到 `results/mbpp_blocks_negative_transformed.json`。

**Step 4: 验证负变换 variant 有 sample_type=negative**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
import json
data = json.load(open("results/mbpp_blocks_negative_transformed.json"))
for sample in data["samples"]:
    for block in sample["blocks"]:
        for v in block["variants"]:
            assert v["sample_type"] == "negative", f"Wrong sample_type: {v}"
print(f"OK: all {data['metadata']['total_variants']} negative variants tagged correctly")
print(f"Metadata mode: {data['metadata']['mode']}")
EOF
```

**Step 5: 验证负变换代码语法正确性**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
import json
data = json.load(open("results/mbpp_blocks_negative_transformed.json"))
errors = 0
for sample in data["samples"]:
    for block in sample["blocks"]:
        for v in block["variants"]:
            try:
                compile(v["transformed_source"], "<string>", "exec")
            except SyntaxError as e:
                print(f"SyntaxError in task {sample['task_id']} block {block['block_id']}: {e}")
                errors += 1
print(f"Syntax check done. Errors: {errors}")
EOF
```

Expected: `Errors: 0`（负变换改变语义但不破坏语法）。

**Step 6: Commit**

```bash
git add experiment/statement_block_transform/results/
git commit -m "test: end-to-end integration test for positive and negative transform modes"
```

---

### Task 9: 全量运行

**Step 1: 全量正变换（更新 sample_type 标记）**

```bash
cd experiment/statement_block_transform && python transform.py --mode positive
```

Expected: 完成，覆盖原有 `mbpp_blocks_transformed.json`，所有 variant 带 `"sample_type": "positive"`。

**Step 2: 全量负变换**

```bash
cd experiment/statement_block_transform && python transform.py --mode negative
```

Expected: 完成，生成 `mbpp_blocks_negative_transformed.json`。

**Step 3: 打印统计**

```bash
cd experiment/statement_block_transform && python - <<'EOF'
import json
for fname, label in [
    ("results/mbpp_blocks_transformed.json", "positive"),
    ("results/mbpp_blocks_negative_transformed.json", "negative"),
]:
    data = json.load(open(fname))
    m = data["metadata"]
    print(f"[{label}] blocks={m['total_blocks']}, transformed={m['transformed_blocks']}, variants={m['total_variants']}")
EOF
```

**Step 4: Commit**

```bash
git add experiment/statement_block_transform/results/
git commit -m "feat: complete positive and negative transform full runs with sample_type tags"
```
