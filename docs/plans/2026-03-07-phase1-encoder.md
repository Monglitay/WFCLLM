# Phase 1: 鲁棒语义编码器预训练 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Train a robust semantic encoder E that maps code blocks to vectors invariant under semantic-equivalent transformations but distinct for semantic-breaking mutations.

**Architecture:** Migrate experiment code to `wfcllm/common/` (AST parser + transform engine), then build `wfcllm/encoder/` (dataset, model, trainer, evaluator) on top. CodeT5-base encoder with projection head, trained via triplet margin loss on MBPP+HumanEval.

**Tech Stack:** tree-sitter, PyTorch, transformers (T5EncoderModel), datasets (HuggingFace)

**Design doc:** `docs/plans/2026-03-07-phase1-encoder-design.md`

---

### Task 1: `wfcllm/common/ast_parser.py` — Tree-sitter Python 解析封装

**Files:**
- Create: `wfcllm/common/ast_parser.py`
- Create: `tests/common/test_ast_parser.py`

**Step 1: Write the failing tests**

```python
# tests/common/test_ast_parser.py
"""Tests for wfcllm.common.ast_parser."""

import pytest
from wfcllm.common.ast_parser import PythonParser, StatementBlock, extract_statement_blocks


class TestPythonParser:
    def test_singleton(self):
        p1 = PythonParser()
        p2 = PythonParser()
        assert p1 is p2

    def test_parse_returns_tree(self):
        p = PythonParser()
        tree = p.parse("x = 1")
        assert tree.root_node.type == "module"

    def test_parse_empty(self):
        p = PythonParser()
        tree = p.parse("")
        assert tree.root_node.type == "module"


class TestExtractStatementBlocks:
    def test_single_simple_statement(self):
        blocks = extract_statement_blocks("x = 1")
        assert len(blocks) == 1
        b = blocks[0]
        assert b.block_type == "simple"
        assert b.node_type == "expression_statement"
        assert b.source == "x = 1"
        assert b.depth == 0
        assert b.parent_id is None

    def test_compound_statement(self):
        code = "for i in range(10):\n    print(i)"
        blocks = extract_statement_blocks(code)
        # Should have for_statement (compound) + expression_statement (simple child)
        compound = [b for b in blocks if b.block_type == "compound"]
        simple = [b for b in blocks if b.block_type == "simple"]
        assert len(compound) == 1
        assert compound[0].node_type == "for_statement"
        assert len(simple) == 1
        assert simple[0].parent_id == compound[0].block_id

    def test_nested_depth(self):
        code = "if True:\n    for i in range(3):\n        print(i)"
        blocks = extract_statement_blocks(code)
        depths = [b.depth for b in blocks]
        assert 0 in depths
        assert 1 in depths
        assert 2 in depths

    def test_children_ids(self):
        code = "if True:\n    x = 1\n    y = 2"
        blocks = extract_statement_blocks(code)
        parent = [b for b in blocks if b.block_type == "compound"][0]
        assert len(parent.children_ids) == 2

    def test_multiple_simple_statements(self):
        code = "x = 1\ny = 2\nz = 3"
        blocks = extract_statement_blocks(code)
        assert len(blocks) == 3
        assert all(b.block_type == "simple" for b in blocks)
        assert all(b.depth == 0 for b in blocks)

    def test_function_definition(self):
        code = "def foo(x):\n    return x + 1"
        blocks = extract_statement_blocks(code)
        func_blocks = [b for b in blocks if b.node_type == "function_definition"]
        assert len(func_blocks) == 1
        assert func_blocks[0].block_type == "compound"

    def test_empty_code(self):
        blocks = extract_statement_blocks("")
        assert blocks == []

    def test_line_numbers(self):
        code = "x = 1\ny = 2"
        blocks = extract_statement_blocks(code)
        assert blocks[0].start_line == 1
        assert blocks[1].start_line == 2
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/common/test_ast_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'wfcllm.common.ast_parser'`

**Step 3: Write implementation**

Reference: `experiment/statement_block_split/split.py` and `experiment/statement_block_split/config.py`

```python
# wfcllm/common/ast_parser.py
"""Tree-sitter Python parsing utilities.

Provides a singleton parser and statement block extraction for Python source code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language(tspython.language())

# Statement block node type classification
SIMPLE_STATEMENT_TYPES = frozenset({
    "expression_statement",
    "return_statement",
    "assert_statement",
    "import_statement",
    "import_from_statement",
    "pass_statement",
    "break_statement",
    "continue_statement",
    "raise_statement",
    "delete_statement",
    "global_statement",
    "nonlocal_statement",
})

COMPOUND_STATEMENT_TYPES = frozenset({
    "function_definition",
    "class_definition",
    "if_statement",
    "for_statement",
    "while_statement",
    "try_statement",
    "with_statement",
    "match_statement",
})

STATEMENT_TYPES = SIMPLE_STATEMENT_TYPES | COMPOUND_STATEMENT_TYPES


class PythonParser:
    """Singleton tree-sitter Python parser."""

    _instance: PythonParser | None = None
    _parser: Parser

    def __new__(cls) -> PythonParser:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._parser = Parser(PY_LANGUAGE)
        return cls._instance

    def parse(self, source: str) -> Tree:
        """Parse Python source code into a tree-sitter Tree."""
        return self._parser.parse(source.encode("utf-8"))


@dataclass
class StatementBlock:
    """A statement block extracted from Python source code."""

    block_id: str
    block_type: Literal["simple", "compound"]
    node_type: str
    source: str
    start_line: int
    end_line: int
    depth: int
    parent_id: str | None
    children_ids: list[str] = field(default_factory=list)


def extract_statement_blocks(source: str) -> list[StatementBlock]:
    """Extract all statement blocks from Python source code.

    Returns a flat list of StatementBlock with parent-child references.
    """
    parser = PythonParser()
    tree = parser.parse(source)
    blocks: list[StatementBlock] = []
    _extract_recursive(tree.root_node, source, blocks, depth=0, parent_id=None)
    return blocks


def _extract_recursive(
    node,
    source: str,
    blocks: list[StatementBlock],
    depth: int,
    parent_id: str | None,
) -> None:
    """Recursively walk AST, extracting statement blocks."""
    for child in node.children:
        if child.type not in STATEMENT_TYPES:
            _extract_recursive(child, source, blocks, depth, parent_id)
            continue

        block_id = str(len(blocks))
        is_compound = child.type in COMPOUND_STATEMENT_TYPES

        block = StatementBlock(
            block_id=block_id,
            block_type="compound" if is_compound else "simple",
            node_type=child.type,
            source=child.text.decode("utf-8"),
            start_line=child.start_point[0] + 1,
            end_line=child.end_point[0] + 1,
            depth=depth,
            parent_id=parent_id,
        )
        blocks.append(block)

        if is_compound:
            child_count_before = len(blocks)
            _extract_recursive(child, source, blocks, depth + 1, block_id)
            block.children_ids = [str(i) for i in range(child_count_before, len(blocks))]
```

Also create `tests/common/__init__.py`:
```python
# tests/common/__init__.py
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/common/test_ast_parser.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/common/ast_parser.py tests/common/__init__.py tests/common/test_ast_parser.py
git commit -m "feat: add tree-sitter Python parser and statement block extraction"
```

---

### Task 2: `wfcllm/common/transform/base.py` + `engine.py` — 变换框架

**Files:**
- Create: `wfcllm/common/transform/__init__.py`
- Create: `wfcllm/common/transform/base.py`
- Create: `wfcllm/common/transform/engine.py`
- Create: `tests/common/transform/__init__.py`
- Create: `tests/common/transform/test_base.py`
- Create: `tests/common/transform/test_engine.py`

**Step 1: Write the failing tests**

```python
# tests/common/transform/test_base.py
"""Tests for wfcllm.common.transform.base."""

from wfcllm.common.transform.base import Match, Rule, parse_code


class TestParseCode:
    def test_returns_tree(self):
        tree = parse_code("x = 1")
        assert tree.root_node.type == "module"

    def test_empty_source(self):
        tree = parse_code("")
        assert tree.root_node.type == "module"


class TestMatch:
    def test_fields(self):
        m = Match("call", 0, 5, "print", "print")
        assert m.node_type == "call"
        assert m.start_byte == 0
        assert m.end_byte == 5


class _DummyRule(Rule):
    """A test rule that replaces 'hello' with 'world'."""
    name = "dummy"
    category = "test"
    description = "test rule"

    def detect(self, source, tree):
        matches = []
        idx = source.find("hello")
        if idx >= 0:
            matches.append(Match("identifier", idx, idx + 5, "hello", "world"))
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class TestRule:
    def test_can_apply_true(self):
        r = _DummyRule()
        assert r.can_apply('x = "hello"') is True

    def test_can_apply_false(self):
        r = _DummyRule()
        assert r.can_apply("x = 1") is False

    def test_transform(self):
        r = _DummyRule()
        result = r.transform('x = "hello"')
        assert result == 'x = "world"'

    def test_transform_not_applicable(self):
        r = _DummyRule()
        assert r.transform("x = 1") is None
```

```python
# tests/common/transform/test_engine.py
"""Tests for wfcllm.common.transform.engine."""

from wfcllm.common.transform.base import Match, Rule, parse_code
from wfcllm.common.transform.engine import TransformEngine


class _AddOneRule(Rule):
    """Replace '1' with '2' in source."""
    name = "add_one"
    category = "test"
    description = "test"

    def detect(self, source, tree):
        matches = []
        for i, ch in enumerate(source):
            if ch == "1":
                matches.append(Match("integer", i, i + 1, "1", "2"))
        return matches

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class _AddExcl(Rule):
    """Append '!' to source by replacing last char."""
    name = "add_excl"
    category = "test"
    description = "test"

    def detect(self, source, tree):
        if source.endswith("\n"):
            return []
        return [Match("module", len(source), len(source), "", "!")]

    def apply(self, source, matches):
        result = source
        for m in sorted(matches, key=lambda x: x.start_byte, reverse=True):
            result = result[:m.start_byte] + m.replacement_text + result[m.end_byte:]
        return result


class TestTransformEngine:
    def test_get_applicable_rules(self):
        engine = TransformEngine(rules=[_AddOneRule(), _AddExcl()])
        applicable = engine.get_applicable_rules("x = 1")
        assert len(applicable) == 2

    def test_no_applicable_rules(self):
        engine = TransformEngine(rules=[_AddOneRule()])
        applicable = engine.get_applicable_rules("x = 0")
        assert len(applicable) == 0

    def test_generate_variants(self):
        engine = TransformEngine(
            rules=[_AddOneRule()],
            max_perm_len=1,
            max_variants=10,
        )
        variants = engine.generate_variants("x = 1")
        assert len(variants) >= 1
        assert variants[0]["transformed_source"] == "x = 2"
        assert variants[0]["rules_applied"] == ["add_one"]
        assert variants[0]["sample_type"] == "positive"

    def test_negative_mode(self):
        engine = TransformEngine(
            rules=[_AddOneRule()],
            max_perm_len=1,
            max_variants=10,
            mode="negative",
        )
        variants = engine.generate_variants("x = 1")
        assert variants[0]["sample_type"] == "negative"

    def test_max_variants_limit(self):
        engine = TransformEngine(
            rules=[_AddOneRule(), _AddExcl()],
            max_perm_len=5,
            max_variants=3,
        )
        variants = engine.generate_variants("x = 1")
        assert len(variants) <= 3

    def test_empty_source(self):
        engine = TransformEngine(rules=[_AddOneRule()])
        variants = engine.generate_variants("")
        assert variants == []
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/common/transform/ -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

Reference: `experiment/statement_block_transform/rules/base.py` and `experiment/statement_block_transform/engine.py`

```python
# wfcllm/common/transform/__init__.py
"""Code transformation engine and rules."""

from wfcllm.common.transform.base import Match, Rule, parse_code
from wfcllm.common.transform.engine import TransformEngine

__all__ = ["Match", "Rule", "parse_code", "TransformEngine"]
```

```python
# wfcllm/common/transform/base.py
"""Base class for all transformation rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Tree

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def parse_code(source: str) -> Tree:
    """Parse Python source code into a tree-sitter Tree."""
    return _parser.parse(source.encode("utf-8"))


@dataclass
class Match:
    """A location in source code where a rule can be applied."""

    node_type: str
    start_byte: int
    end_byte: int
    original_text: str
    replacement_text: str


class Rule(ABC):
    """Base class for transformation rules."""

    name: str = ""
    category: str = ""
    description: str = ""

    @abstractmethod
    def detect(self, source: str, tree: Tree) -> list[Match]:
        """Find all positions where this rule can be applied."""
        ...

    @abstractmethod
    def apply(self, source: str, matches: list[Match]) -> str:
        """Apply the transformation, returning new source code."""
        ...

    def can_apply(self, source: str) -> bool:
        """Check if this rule can be applied to the given source."""
        tree = parse_code(source)
        return len(self.detect(source, tree)) > 0

    def transform(self, source: str) -> str | None:
        """Detect and apply in one step. Returns None if not applicable."""
        tree = parse_code(source)
        matches = self.detect(source, tree)
        if not matches:
            return None
        return self.apply(source, matches)
```

```python
# wfcllm/common/transform/engine.py
"""Transform engine: applies rule permutations to code blocks."""

from __future__ import annotations

from itertools import permutations

from wfcllm.common.transform.base import Rule, parse_code


class TransformEngine:
    """Applies all permutations of applicable rules to a code block."""

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

    def get_applicable_rules(self, source: str) -> list[Rule]:
        """Return rules that can be applied to the given source."""
        tree = parse_code(source)
        return [r for r in self.rules if r.detect(source, tree)]

    def generate_variants(self, source: str) -> list[dict]:
        """Generate all permutation variants of applicable rules."""
        applicable = self.get_applicable_rules(source)
        if not applicable:
            return []

        variants: list[dict] = []
        max_len = min(len(applicable), self.max_perm_len)

        for length in range(1, max_len + 1):
            for perm in permutations(applicable, length):
                if len(variants) >= self.max_variants:
                    return variants

                result = self._apply_permutation(source, perm)
                if result is not None:
                    variants.append({
                        "variant_id": len(variants),
                        "rules_applied": [r.name for r in perm],
                        "transformed_source": result,
                        "sample_type": self.mode,
                    })

        return variants

    def _apply_permutation(
        self, source: str, rules: tuple[Rule, ...]
    ) -> str | None:
        """Apply a sequence of rules. Returns None if any step fails."""
        current = source
        for rule in rules:
            tree = parse_code(current)
            matches = rule.detect(current, tree)
            if not matches:
                return None
            current = rule.apply(current, matches)
        return current
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/common/transform/ -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/common/transform/ tests/common/transform/
git commit -m "feat: add transform base classes and engine"
```

---

### Task 3: `wfcllm/common/transform/positive/` — 正变换规则（39条）

**Files:**
- Create: `wfcllm/common/transform/positive/__init__.py`
- Create: `wfcllm/common/transform/positive/api_calls.py`
- Create: `wfcllm/common/transform/positive/syntax_init.py`
- Create: `wfcllm/common/transform/positive/control_flow.py`
- Create: `wfcllm/common/transform/positive/expression_logic.py`
- Create: `wfcllm/common/transform/positive/identifier.py`
- Create: `wfcllm/common/transform/positive/formatting.py`
- Create: `tests/common/transform/test_positive_rules.py`

**Step 1: Write the failing tests**

```python
# tests/common/transform/test_positive_rules.py
"""Tests for positive transformation rules."""

import pytest
from wfcllm.common.transform.positive import get_all_positive_rules


class TestPositiveRulesRegistry:
    def test_all_rules_loaded(self):
        rules = get_all_positive_rules()
        # 12 api_calls + 4 syntax_init + 4 control_flow + 5 expression_logic + 2 identifier + 2 formatting = 29
        # Note: exact count based on experiment code __init__.py
        assert len(rules) >= 25  # at least this many

    def test_all_rules_have_name(self):
        for rule in get_all_positive_rules():
            assert rule.name, f"Rule {rule.__class__.__name__} has no name"

    def test_all_rules_have_category(self):
        for rule in get_all_positive_rules():
            assert rule.category, f"Rule {rule.name} has no category"


class TestApiCallRules:
    def test_explicit_default_print(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultPrint
        rule = ExplicitDefaultPrint()
        result = rule.transform("print(x)")
        assert result is not None
        assert "flush" in result or "end" in result

    def test_explicit_default_range(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultRange
        rule = ExplicitDefaultRange()
        result = rule.transform("range(10)")
        assert result is not None
        assert "0" in result  # range(10) -> range(0, 10)

    def test_min_max_unchanged_semantics(self):
        from wfcllm.common.transform.positive.api_calls import ExplicitDefaultMinMax
        rule = ExplicitDefaultMinMax()
        result = rule.transform("min(a, b)")
        assert result is not None
        assert "key=None" in result


class TestControlFlowRules:
    def test_loop_convert(self):
        from wfcllm.common.transform.positive.control_flow import LoopConvert
        rule = LoopConvert()
        source = "for i in range(n):\n    print(i)"
        assert rule.can_apply(source)

    def test_branch_flip(self):
        from wfcllm.common.transform.positive.control_flow import BranchFlip
        rule = BranchFlip()
        source = "if x > 0:\n    print('yes')\nelse:\n    print('no')"
        assert rule.can_apply(source)


class TestExpressionLogicRules:
    def test_operand_swap(self):
        from wfcllm.common.transform.positive.expression_logic import OperandSwap
        rule = OperandSwap()
        result = rule.transform("x + y")
        assert result is not None

    def test_comparison_flip(self):
        from wfcllm.common.transform.positive.expression_logic import ComparisonFlip
        rule = ComparisonFlip()
        result = rule.transform("a < b")
        assert result is not None
        assert ">" in result  # a < b → b > a


class TestIdentifierRules:
    def test_variable_rename(self):
        from wfcllm.common.transform.positive.identifier import VariableRename
        rule = VariableRename()
        result = rule.transform("my_var = 1")
        # Should convert snake_case to camelCase or vice versa
        assert result is not None


class TestFormattingRules:
    def test_fix_spacing(self):
        from wfcllm.common.transform.positive.formatting import FixSpacing
        rule = FixSpacing()
        # If source has operators without spaces, should add them
        assert rule.name  # at minimum check it loads
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/common/transform/test_positive_rules.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

Migrate all 6 rule files from `experiment/statement_block_transform/rules/` to `wfcllm/common/transform/positive/`. Key changes for each file:

- Change import: `from rules.base import Match, Rule, parse_code` → `from wfcllm.common.transform.base import Match, Rule, parse_code`
- Keep all rule class logic identical to experiment code
- No other changes needed — the rule logic is pure and has no external dependencies beyond base.py

Files to migrate (source → destination):
- `experiment/.../rules/api_calls.py` → `wfcllm/common/transform/positive/api_calls.py`
- `experiment/.../rules/syntax_init.py` → `wfcllm/common/transform/positive/syntax_init.py`
- `experiment/.../rules/control_flow.py` → `wfcllm/common/transform/positive/control_flow.py`
- `experiment/.../rules/expression_logic.py` → `wfcllm/common/transform/positive/expression_logic.py`
- `experiment/.../rules/identifier.py` → `wfcllm/common/transform/positive/identifier.py`
- `experiment/.../rules/formatting.py` → `wfcllm/common/transform/positive/formatting.py`

Registry file:

```python
# wfcllm/common/transform/positive/__init__.py
"""Positive (semantic-equivalent) transformation rules registry."""

from wfcllm.common.transform.base import Rule
from wfcllm.common.transform.positive.api_calls import (
    ExplicitDefaultPrint, ExplicitDefaultRange, ExplicitDefaultOpen,
    ExplicitDefaultSorted, ExplicitDefaultMinMax, ExplicitDefaultZip,
    ExplicitDefaultRandomSeed, ExplicitDefaultHtmlEscape,
    ExplicitDefaultRound, ExplicitDefaultJsonDump,
    LibraryAliasReplace, ThirdPartyFuncReplace,
)
from wfcllm.common.transform.positive.syntax_init import ListInit, DictInit, TypeCheck, StringFormat
from wfcllm.common.transform.positive.control_flow import LoopConvert, IterationConvert, ComprehensionConvert, BranchFlip
from wfcllm.common.transform.positive.expression_logic import (
    OperandSwap, ComparisonFlip, UnarySimplify, DeMorgan, ArithmeticAssociativity,
)
from wfcllm.common.transform.positive.identifier import VariableRename, NameObfuscation
from wfcllm.common.transform.positive.formatting import FixSpacing, FixCommentSymbols

_ALL_POSITIVE_RULES: list[Rule] = [
    ExplicitDefaultPrint(), ExplicitDefaultRange(), ExplicitDefaultOpen(),
    ExplicitDefaultSorted(), ExplicitDefaultMinMax(), ExplicitDefaultZip(),
    ExplicitDefaultRandomSeed(), ExplicitDefaultHtmlEscape(),
    ExplicitDefaultRound(), ExplicitDefaultJsonDump(),
    LibraryAliasReplace(), ThirdPartyFuncReplace(),
    ListInit(), DictInit(), TypeCheck(), StringFormat(),
    LoopConvert(), IterationConvert(), ComprehensionConvert(), BranchFlip(),
    OperandSwap(), ComparisonFlip(), UnarySimplify(), DeMorgan(), ArithmeticAssociativity(),
    VariableRename(), NameObfuscation(),
    FixSpacing(), FixCommentSymbols(),
]


def get_all_positive_rules() -> list[Rule]:
    return list(_ALL_POSITIVE_RULES)
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/common/transform/test_positive_rules.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/common/transform/positive/ tests/common/transform/test_positive_rules.py
git commit -m "feat: add positive transformation rules (39 rules, 6 categories)"
```

---

### Task 4: `wfcllm/common/transform/negative/` — 负变换规则（23条）

**Files:**
- Create: `wfcllm/common/transform/negative/__init__.py`
- Create: `wfcllm/common/transform/negative/api_calls.py`
- Create: `wfcllm/common/transform/negative/control_flow.py`
- Create: `wfcllm/common/transform/negative/expression_logic.py`
- Create: `wfcllm/common/transform/negative/identifier.py`
- Create: `wfcllm/common/transform/negative/data_structure.py`
- Create: `wfcllm/common/transform/negative/exception.py`
- Create: `wfcllm/common/transform/negative/system.py`
- Create: `tests/common/transform/test_negative_rules.py`

**Step 1: Write the failing tests**

```python
# tests/common/transform/test_negative_rules.py
"""Tests for negative transformation rules."""

import pytest
from wfcllm.common.transform.negative import get_all_negative_rules


class TestNegativeRulesRegistry:
    def test_all_rules_loaded(self):
        rules = get_all_negative_rules()
        # 7 api + 5 control_flow + 6 expression + 1 identifier + 2 data_structure + 1 exception + 1 system = 23
        assert len(rules) == 23

    def test_all_rules_have_name(self):
        for rule in get_all_negative_rules():
            assert rule.name, f"Rule {rule.__class__.__name__} has no name"
            assert rule.name.startswith("neg_"), f"Negative rule {rule.name} should start with 'neg_'"


class TestNegativeApiCalls:
    def test_min_max_flip(self):
        from wfcllm.common.transform.negative.api_calls import MinMaxFlip
        rule = MinMaxFlip()
        result = rule.transform("x = min(a, b)")
        assert result is not None
        assert "max" in result

    def test_any_all_flip(self):
        from wfcllm.common.transform.negative.api_calls import AnyAllFlip
        rule = AnyAllFlip()
        result = rule.transform("if any(lst):\n    pass")
        assert result is not None
        assert "all" in result

    def test_ceil_floor_flip(self):
        from wfcllm.common.transform.negative.api_calls import CeilFloorFlip
        rule = CeilFloorFlip()
        result = rule.transform("x = math.ceil(y)")
        assert result is not None
        assert "floor" in result


class TestNegativeControlFlow:
    def test_off_by_one(self):
        from wfcllm.common.transform.negative.control_flow import OffByOne
        rule = OffByOne()
        result = rule.transform("for i in range(n):\n    pass")
        assert result is not None
        assert "n - 1" in result

    def test_break_continue_swap(self):
        from wfcllm.common.transform.negative.control_flow import BreakContinueSwap
        rule = BreakContinueSwap()
        result = rule.transform("for i in range(10):\n    break")
        assert result is not None
        assert "continue" in result


class TestNegativeExpressionLogic:
    def test_eq_neq_flip(self):
        from wfcllm.common.transform.negative.expression_logic import EqNeqFlip
        rule = EqNeqFlip()
        result = rule.transform("x == y")
        assert result is not None
        assert "!=" in result

    def test_and_or_swap(self):
        from wfcllm.common.transform.negative.expression_logic import AndOrSwap
        rule = AndOrSwap()
        result = rule.transform("a and b")
        assert result is not None
        assert "or" in result
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/common/transform/test_negative_rules.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

Migrate all 7 rule files from `experiment/statement_block_transform/rules/negative/` to `wfcllm/common/transform/negative/`. Key changes:

- Change import: `from rules.base import Match, Rule` → `from wfcllm.common.transform.base import Match, Rule`
- For `control_flow.py`: also change `from rules.base import ... parse_code` → `from wfcllm.common.transform.base import ... parse_code`
- For `exception.py`: same parse_code import change
- Keep all rule logic identical

Registry:

```python
# wfcllm/common/transform/negative/__init__.py
"""Negative (semantic-breaking) transformation rules registry."""

from wfcllm.common.transform.base import Rule
from wfcllm.common.transform.negative.api_calls import (
    MinMaxFlip, AnyAllFlip, SortedReverseFlip, OpenModeCorrupt,
    ExtendAppendSwap, StartsEndsSwap, CeilFloorFlip,
)
from wfcllm.common.transform.negative.control_flow import (
    OffByOne, BreakContinueSwap, IfElseBodySwap, MembershipNegate, YieldReturnSwap,
)
from wfcllm.common.transform.negative.expression_logic import (
    EqNeqFlip, ArithmeticOpReplace, AndOrSwap, BoundsNarrow, AugAssignCorrupt, ShiftFlip,
)
from wfcllm.common.transform.negative.identifier import ScopeVarCorrupt
from wfcllm.common.transform.negative.data_structure import SliceStepFlip, DictViewSwap
from wfcllm.common.transform.negative.exception import ExceptionSwallow
from wfcllm.common.transform.negative.system import SysExitFlip


def get_all_negative_rules() -> list[Rule]:
    return [
        MinMaxFlip(), AnyAllFlip(), SortedReverseFlip(), OpenModeCorrupt(),
        ExtendAppendSwap(), StartsEndsSwap(), CeilFloorFlip(),
        OffByOne(), BreakContinueSwap(), IfElseBodySwap(), MembershipNegate(), YieldReturnSwap(),
        EqNeqFlip(), ArithmeticOpReplace(), AndOrSwap(), BoundsNarrow(), AugAssignCorrupt(), ShiftFlip(),
        ScopeVarCorrupt(),
        SliceStepFlip(), DictViewSwap(),
        ExceptionSwallow(),
        SysExitFlip(),
    ]
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/common/transform/test_negative_rules.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/common/transform/negative/ tests/common/transform/test_negative_rules.py
git commit -m "feat: add negative transformation rules (23 rules, 7 categories)"
```

---

### Task 5: `wfcllm/encoder/config.py` — 编码器配置

**Files:**
- Create: `wfcllm/encoder/config.py`
- Create: `tests/encoder/__init__.py`
- Create: `tests/encoder/test_config.py`

**Step 1: Write the failing tests**

```python
# tests/encoder/test_config.py
"""Tests for wfcllm.encoder.config."""

from wfcllm.encoder.config import EncoderConfig


class TestEncoderConfig:
    def test_default_values(self):
        cfg = EncoderConfig()
        assert cfg.model_name == "Salesforce/codet5-base"
        assert cfg.embed_dim == 128
        assert cfg.lr == 2e-5
        assert cfg.batch_size == 32
        assert cfg.epochs == 10
        assert cfg.margin == 0.3
        assert cfg.max_seq_length == 256
        assert cfg.warmup_ratio == 0.1
        assert cfg.early_stopping_patience == 3
        assert cfg.negative_ratio == 0.5

    def test_lora_defaults(self):
        cfg = EncoderConfig()
        assert cfg.use_lora is True
        assert cfg.lora_r == 8
        assert cfg.lora_alpha == 16
        assert cfg.lora_dropout == 0.1

    def test_bf16_default(self):
        cfg = EncoderConfig()
        assert cfg.use_bf16 is True

    def test_disable_lora(self):
        cfg = EncoderConfig(use_lora=False)
        assert cfg.use_lora is False

    def test_disable_bf16(self):
        cfg = EncoderConfig(use_bf16=False)
        assert cfg.use_bf16 is False

    def test_data_sources(self):
        cfg = EncoderConfig()
        assert "mbpp" in cfg.data_sources
        assert "humaneval" in cfg.data_sources

    def test_custom_values(self):
        cfg = EncoderConfig(lr=1e-4, batch_size=16)
        assert cfg.lr == 1e-4
        assert cfg.batch_size == 16

    def test_paths(self):
        cfg = EncoderConfig()
        assert "checkpoints" in cfg.checkpoint_dir
        assert "results" in cfg.results_dir
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_config.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/config.py
"""Configuration for the semantic encoder pretraining pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    """All hyperparameters and paths for encoder pretraining."""

    # Model
    model_name: str = "Salesforce/codet5-base"
    embed_dim: int = 128

    # LoRA (optional, default on)
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.1
    lora_target_modules: list[str] = field(default_factory=lambda: ["q", "v"])

    # Precision (optional, default BF16)
    use_bf16: bool = True

    # Data
    data_sources: list[str] = field(default_factory=lambda: ["mbpp", "humaneval"])
    max_seq_length: int = 256
    negative_ratio: float = 0.5  # fraction of hard negatives vs random negatives

    # Training
    lr: float = 2e-5
    batch_size: int = 32
    epochs: int = 10
    margin: float = 0.3
    warmup_ratio: float = 0.1
    early_stopping_patience: int = 3

    # Paths
    checkpoint_dir: str = "data/checkpoints/encoder"
    results_dir: str = "data/results"
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_config.py -v`
Expected: All 7 tests PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/config.py tests/encoder/__init__.py tests/encoder/test_config.py
git commit -m "feat: add encoder configuration dataclass"
```

---

### Task 6: `wfcllm/encoder/model.py` — SemanticEncoder 模型

**Files:**
- Create: `wfcllm/encoder/model.py`
- Create: `tests/encoder/test_model.py`

**Prerequisite:** `torch`, `transformers`, and `peft` must be installed in the WFCLLM conda env.

**Step 0: Install dependencies**

```bash
conda run -n WFCLLM pip install torch transformers peft
```

**Step 1: Write the failing tests**

```python
# tests/encoder/test_model.py
"""Tests for wfcllm.encoder.model."""

import pytest
import torch
from transformers import AutoTokenizer

from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.config import EncoderConfig


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("Salesforce/codet5-base")


class TestSemanticEncoderFullFinetune:
    """Tests with LoRA disabled (full finetune, FP32)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=False, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_batch_input(self, model, tokenizer):
        texts = ["x = 1", "y = 2", "for i in range(10):\n    print(i)"]
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (3, 128)

    def test_all_params_trainable(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable == total


class TestSemanticEncoderLoRA:
    """Tests with LoRA enabled (default config)."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=True, use_bf16=False, embed_dim=128)
        return SemanticEncoder(config=config)

    def test_output_shape(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 128)

    def test_output_normalized(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        norms = torch.norm(output, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_fewer_trainable_params(self, model):
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert trainable < total, "LoRA should freeze most parameters"
        # LoRA typically trains <5% of params
        ratio = trainable / total
        assert ratio < 0.10, f"Expected <10% trainable params, got {ratio:.2%}"

    def test_deterministic(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        model.eval()
        with torch.no_grad():
            out1 = model(inputs["input_ids"], inputs["attention_mask"])
            out2 = model(inputs["input_ids"], inputs["attention_mask"])
        assert torch.allclose(out1, out2)


class TestSemanticEncoderBF16:
    """Tests with BF16 enabled."""

    @pytest.fixture
    def model(self):
        config = EncoderConfig(use_lora=False, use_bf16=True, embed_dim=64)
        return SemanticEncoder(config=config)

    def test_encoder_dtype(self, model):
        # Encoder weights should be BF16
        param = next(model.encoder.parameters())
        assert param.dtype == torch.bfloat16

    def test_output_is_float32(self, model, tokenizer):
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        # Output should be cast back to float32 for downstream use
        assert output.dtype == torch.float32

    def test_different_embed_dim(self, tokenizer):
        config = EncoderConfig(use_lora=False, use_bf16=False, embed_dim=64)
        model = SemanticEncoder(config=config)
        inputs = tokenizer("x = 1", return_tensors="pt", padding=True, truncation=True, max_length=256)
        with torch.no_grad():
            output = model(inputs["input_ids"], inputs["attention_mask"])
        assert output.shape == (1, 64)
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_model.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/model.py
"""Semantic encoder model for code representation learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import T5EncoderModel

from wfcllm.encoder.config import EncoderConfig


class SemanticEncoder(nn.Module):
    """CodeT5 encoder with projection head for contrastive learning.

    Architecture: CodeT5 Encoder → [CLS] pooling → Linear → L2 normalize

    Supports optional LoRA (via peft) and BF16 precision, both configurable
    via EncoderConfig and enabled by default.
    """

    def __init__(self, config: EncoderConfig | None = None):
        super().__init__()
        if config is None:
            config = EncoderConfig()
        self.config = config

        # Load encoder with optional BF16
        load_kwargs = {}
        if config.use_bf16:
            load_kwargs["torch_dtype"] = torch.bfloat16

        self.encoder = T5EncoderModel.from_pretrained(
            config.model_name, **load_kwargs
        )
        hidden_size = self.encoder.config.d_model

        # Apply LoRA if enabled
        if config.use_lora:
            from peft import LoraConfig, get_peft_model

            lora_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
            )
            self.encoder = get_peft_model(self.encoder, lora_config)

        # Projection head (always float32 for stable cosine similarity)
        self.projection = nn.Linear(hidden_size, config.embed_dim)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Encode input tokens into L2-normalized semantic vectors.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            (batch_size, embed_dim) L2-normalized float32 vectors.
        """
        encoder_output = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        # Use first token ([CLS] equivalent) as sequence representation
        cls_hidden = encoder_output.last_hidden_state[:, 0, :]
        # Cast to float32 before projection for numerical stability
        cls_hidden = cls_hidden.float()
        projected = self.projection(cls_hidden)
        return F.normalize(projected, p=2, dim=1)
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_model.py -v`
Expected: All 11 tests PASS (first run will download model weights ~900MB)

**Step 5: Commit**

```bash
git add wfcllm/encoder/model.py tests/encoder/test_model.py
git commit -m "feat: add SemanticEncoder model with CodeT5 backbone, optional LoRA and BF16"
```

---

### Task 7: `wfcllm/encoder/dataset.py` — Triplet 数据集

**Files:**
- Create: `wfcllm/encoder/dataset.py`
- Create: `tests/encoder/test_dataset.py`

**Step 1: Write the failing tests**

```python
# tests/encoder/test_dataset.py
"""Tests for wfcllm.encoder.dataset."""

import pytest
from unittest.mock import patch, MagicMock
from wfcllm.encoder.dataset import TripletCodeDataset, build_triplets_from_blocks


class TestBuildTripletsFromBlocks:
    """Test triplet construction logic with synthetic data."""

    def _make_block(self, source, positive_variants=None, negative_variants=None):
        return {
            "source": source,
            "positive_variants": positive_variants or [],
            "negative_variants": negative_variants or [],
        }

    def test_basic_triplet(self):
        blocks = [
            self._make_block("x = 1", ["x = 2"], ["x = -1"]),
            self._make_block("y = 3", ["y = 4"], ["y = -3"]),
        ]
        triplets = build_triplets_from_blocks(blocks, negative_ratio=0.5, seed=42)
        assert len(triplets) > 0
        for t in triplets:
            assert "anchor" in t
            assert "positive" in t
            assert "negative" in t

    def test_negative_ratio_hard(self):
        """With ratio=1.0, all negatives should be hard (from negative_variants)."""
        blocks = [
            self._make_block("x = 1", ["x = 2"], ["x = -1"]),
        ]
        # Need other blocks for random negatives
        all_blocks_sources = ["x = 1", "y = 2"]
        triplets = build_triplets_from_blocks(
            blocks, negative_ratio=1.0, seed=42, all_sources=all_blocks_sources
        )
        for t in triplets:
            assert t["negative"] in ["x = -1"]  # only hard negatives

    def test_skip_blocks_without_positives(self):
        blocks = [
            self._make_block("x = 1", [], ["x = -1"]),  # no positives
        ]
        triplets = build_triplets_from_blocks(blocks, negative_ratio=0.5, seed=42)
        assert len(triplets) == 0

    def test_empty_blocks(self):
        triplets = build_triplets_from_blocks([], negative_ratio=0.5, seed=42)
        assert triplets == []


class TestTripletCodeDataset:
    def test_len(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
            {"anchor": "a = 1", "positive": "a = 2", "negative": "b = 3"},
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        ds = TripletCodeDataset(triplets, tokenizer, max_length=64)
        assert len(ds) == 2

    def test_getitem_keys(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        ds = TripletCodeDataset(triplets, tokenizer, max_length=64)
        item = ds[0]
        assert "anchor_input_ids" in item
        assert "anchor_attention_mask" in item
        assert "positive_input_ids" in item
        assert "positive_attention_mask" in item
        assert "negative_input_ids" in item
        assert "negative_attention_mask" in item

    def test_getitem_shapes(self):
        triplets = [
            {"anchor": "x = 1", "positive": "x = 2", "negative": "y = 3"},
        ]
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        max_len = 64
        ds = TripletCodeDataset(triplets, tokenizer, max_length=max_len)
        item = ds[0]
        assert item["anchor_input_ids"].shape[0] == max_len
        assert item["anchor_attention_mask"].shape[0] == max_len
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_dataset.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/dataset.py
"""Triplet code dataset for contrastive learning."""

from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


def build_triplets_from_blocks(
    blocks: list[dict],
    negative_ratio: float = 0.5,
    seed: int = 42,
    all_sources: list[str] | None = None,
) -> list[dict]:
    """Build (anchor, positive, negative) triplets from transformed blocks.

    Args:
        blocks: List of dicts with keys 'source', 'positive_variants', 'negative_variants'.
        negative_ratio: Fraction of negatives that are hard (from negative_variants).
            0.0 = all random, 1.0 = all hard.
        seed: Random seed for reproducibility.
        all_sources: Pool of all block sources for random negative sampling.
            If None, collected from blocks.

    Returns:
        List of {"anchor": str, "positive": str, "negative": str}.
    """
    rng = random.Random(seed)

    if all_sources is None:
        all_sources = [b["source"] for b in blocks]

    triplets: list[dict] = []

    for block in blocks:
        source = block["source"]
        positives = block.get("positive_variants", [])
        negatives = block.get("negative_variants", [])

        if not positives:
            continue

        for pos in positives:
            # Decide negative type
            use_hard = rng.random() < negative_ratio and negatives
            if use_hard:
                neg = rng.choice(negatives)
            else:
                # Random negative from other blocks
                candidates = [s for s in all_sources if s != source]
                if not candidates:
                    continue
                neg = rng.choice(candidates)

            triplets.append({
                "anchor": source,
                "positive": pos,
                "negative": neg,
            })

    return triplets


class TripletCodeDataset(Dataset):
    """PyTorch Dataset yielding tokenized (anchor, positive, negative) triplets."""

    def __init__(
        self,
        triplets: list[dict],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 256,
    ):
        self.triplets = triplets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.triplets)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        triplet = self.triplets[idx]
        result = {}
        for key in ("anchor", "positive", "negative"):
            encoded = self.tokenizer(
                triplet[key],
                max_length=self.max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            result[f"{key}_input_ids"] = encoded["input_ids"].squeeze(0)
            result[f"{key}_attention_mask"] = encoded["attention_mask"].squeeze(0)
        return result
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_dataset.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/dataset.py tests/encoder/test_dataset.py
git commit -m "feat: add triplet code dataset for contrastive learning"
```

---

### Task 8: `wfcllm/encoder/trainer.py` — 对比学习训练器

**Files:**
- Create: `wfcllm/encoder/trainer.py`
- Create: `tests/encoder/test_trainer.py`

**Step 1: Write the failing tests**

```python
# tests/encoder/test_trainer.py
"""Tests for wfcllm.encoder.trainer."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import MagicMock

from wfcllm.encoder.trainer import ContrastiveTrainer, triplet_cosine_loss
from wfcllm.encoder.config import EncoderConfig


class TestTripletCosineLoss:
    def test_zero_loss_when_perfect(self):
        """Loss should be 0 when positive is identical and negative is orthogonal."""
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[1.0, 0.0]])  # cos = 1.0
        negative = torch.tensor([[0.0, 1.0]])  # cos = 0.0
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.item() == 0.0

    def test_positive_loss_when_bad(self):
        """Loss > 0 when positive is far and negative is close."""
        anchor = torch.tensor([[1.0, 0.0]])
        positive = torch.tensor([[0.0, 1.0]])  # cos = 0.0
        negative = torch.tensor([[0.9, 0.1]])  # cos ≈ 0.9
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.item() > 0.0

    def test_batch(self):
        anchor = torch.randn(8, 128)
        positive = torch.randn(8, 128)
        negative = torch.randn(8, 128)
        loss = triplet_cosine_loss(anchor, positive, negative, margin=0.3)
        assert loss.shape == ()  # scalar


class TestContrastiveTrainer:
    @pytest.fixture
    def dummy_setup(self):
        """Create minimal trainer with dummy data for smoke testing."""
        from wfcllm.encoder.model import SemanticEncoder
        config = EncoderConfig(
            embed_dim=32, epochs=1, batch_size=2, lr=1e-4,
            use_lora=False, use_bf16=False,
            checkpoint_dir="/tmp/wfcllm_test_ckpt",
            results_dir="/tmp/wfcllm_test_results",
        )
        model = SemanticEncoder(config=config)

        # Create tiny synthetic dataset
        seq_len = 32
        n_samples = 4

        def make_loader():
            data = {}
            for prefix in ("anchor", "positive", "negative"):
                data[f"{prefix}_input_ids"] = torch.randint(0, 100, (n_samples, seq_len))
                data[f"{prefix}_attention_mask"] = torch.ones(n_samples, seq_len, dtype=torch.long)
            dataset = _DictDataset(data, n_samples)
            return DataLoader(dataset, batch_size=2)

        return model, config, make_loader(), make_loader()

    def test_train_epoch_returns_loss(self, dummy_setup):
        model, config, train_loader, val_loader = dummy_setup
        trainer = ContrastiveTrainer(model, train_loader, val_loader, config)
        metrics = trainer.train_epoch()
        assert "loss" in metrics
        assert metrics["loss"] > 0

    def test_validate_returns_metrics(self, dummy_setup):
        model, config, train_loader, val_loader = dummy_setup
        trainer = ContrastiveTrainer(model, train_loader, val_loader, config)
        metrics = trainer.validate()
        assert "val_loss" in metrics


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, data: dict, n: int):
        self.data = data
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_trainer.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/trainer.py
"""Contrastive learning trainer for the semantic encoder."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder


def triplet_cosine_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 0.3,
) -> torch.Tensor:
    """Triplet margin loss using cosine similarity.

    L = max(0, margin - cos(anchor, positive) + cos(anchor, negative))
    """
    cos_pos = F.cosine_similarity(anchor, positive, dim=1)
    cos_neg = F.cosine_similarity(anchor, negative, dim=1)
    loss = torch.clamp(margin - cos_pos + cos_neg, min=0.0)
    return loss.mean()


class ContrastiveTrainer:
    """Training loop for contrastive encoder pretraining.

    Supports optional BF16 mixed precision via config.use_bf16.
    """

    def __init__(
        self,
        model: SemanticEncoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: EncoderConfig,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Only optimize parameters that require grad (respects LoRA frozen params)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = AdamW(trainable_params, lr=config.lr)
        total_steps = len(train_loader) * config.epochs
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # BF16 autocast context
        self._autocast_ctx = (
            torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)
            if config.use_bf16 and self.device.type == "cuda"
            else nullcontext()
        )

    def _encode_batch(self, batch: dict, prefix: str) -> torch.Tensor:
        input_ids = batch[f"{prefix}_input_ids"].to(self.device)
        attention_mask = batch[f"{prefix}_attention_mask"].to(self.device)
        return self.model(input_ids, attention_mask)

    def train_epoch(self) -> dict:
        """Run one training epoch. Returns {"loss": float}."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in self.train_loader:
            with self._autocast_ctx:
                anchor_emb = self._encode_batch(batch, "anchor")
                positive_emb = self._encode_batch(batch, "positive")
                negative_emb = self._encode_batch(batch, "negative")

                loss = triplet_cosine_loss(
                    anchor_emb, positive_emb, negative_emb, margin=self.config.margin
                )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

        return {"loss": total_loss / max(n_batches, 1)}

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation. Returns {"val_loss": float}."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            with self._autocast_ctx:
                anchor_emb = self._encode_batch(batch, "anchor")
                positive_emb = self._encode_batch(batch, "positive")
                negative_emb = self._encode_batch(batch, "negative")

                loss = triplet_cosine_loss(
                    anchor_emb, positive_emb, negative_emb, margin=self.config.margin
                )
            total_loss += loss.item()
            n_batches += 1

        return {"val_loss": total_loss / max(n_batches, 1)}

    def save_checkpoint(self, epoch: int, metrics: dict) -> Path:
        """Save model checkpoint."""
        ckpt_dir = Path(self.config.checkpoint_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        path = ckpt_dir / f"encoder_epoch{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
        }, path)
        return path

    def train(self) -> dict:
        """Full training loop with early stopping and checkpointing."""
        best_val_loss = float("inf")
        patience_counter = 0
        best_metrics: dict = {}

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics = self.validate()
            metrics = {**train_metrics, **val_metrics, "epoch": epoch}

            print(
                f"Epoch {epoch}/{self.config.epochs} — "
                f"loss: {train_metrics['loss']:.4f}, "
                f"val_loss: {val_metrics['val_loss']:.4f}"
            )

            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                patience_counter = 0
                best_metrics = metrics
                self.save_checkpoint(epoch, metrics)
            else:
                patience_counter += 1
                if patience_counter >= self.config.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        return best_metrics
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_trainer.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/trainer.py tests/encoder/test_trainer.py
git commit -m "feat: add contrastive learning trainer with early stopping and BF16 autocast"
```

---

### Task 9: `wfcllm/encoder/evaluate.py` — 编码器评估

**Files:**
- Create: `wfcllm/encoder/evaluate.py`
- Create: `tests/encoder/test_evaluate.py`

**Step 1: Write the failing tests**

```python
# tests/encoder/test_evaluate.py
"""Tests for wfcllm.encoder.evaluate."""

import pytest
import torch

from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    projection_sign_accuracy,
)


class TestCosineSeparation:
    def test_perfect_separation(self):
        anchor = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        positive = torch.tensor([[1.0, 0.0], [0.0, 1.0]])  # identical
        negative = torch.tensor([[0.0, 1.0], [1.0, 0.0]])  # orthogonal
        result = cosine_separation(anchor, positive, negative)
        assert result["mean_pos_cos"] > result["mean_neg_cos"]
        assert result["separation"] > 0

    def test_keys(self):
        anchor = torch.randn(4, 16)
        positive = torch.randn(4, 16)
        negative = torch.randn(4, 16)
        result = cosine_separation(anchor, positive, negative)
        assert "mean_pos_cos" in result
        assert "mean_neg_cos" in result
        assert "separation" in result


class TestRecallAtK:
    def test_perfect_recall(self):
        # anchor[i] is closest to candidates[i]
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=1)
        assert r == 1.0

    def test_recall_at_5(self):
        embeddings = torch.eye(5)
        candidates = torch.eye(5)
        r = recall_at_k(embeddings, candidates, k=5)
        assert r == 1.0


class TestProjectionSignAccuracy:
    def test_all_correct(self):
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target_bits = torch.tensor([1, 0])  # 1 → positive, 0 → negative
        acc = projection_sign_accuracy(embeddings, directions, target_bits)
        assert acc == 1.0

    def test_all_wrong(self):
        embeddings = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
        directions = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        target_bits = torch.tensor([0, 1])  # reversed
        acc = projection_sign_accuracy(embeddings, directions, target_bits)
        assert acc == 0.0
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_evaluate.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/evaluate.py
"""Evaluation metrics for the semantic encoder."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F


def cosine_separation(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
) -> dict[str, float]:
    """Compute mean cosine similarity for positive and negative pairs.

    Returns dict with mean_pos_cos, mean_neg_cos, separation.
    """
    cos_pos = F.cosine_similarity(anchor, positive, dim=1).mean().item()
    cos_neg = F.cosine_similarity(anchor, negative, dim=1).mean().item()
    return {
        "mean_pos_cos": cos_pos,
        "mean_neg_cos": cos_neg,
        "separation": cos_pos - cos_neg,
    }


def recall_at_k(
    query_embeddings: torch.Tensor,
    candidate_embeddings: torch.Tensor,
    k: int = 1,
) -> float:
    """Compute Recall@K: fraction of queries whose true match is in top-K.

    Assumes query_embeddings[i]'s ground truth match is candidate_embeddings[i].
    """
    # Cosine similarity matrix
    sim_matrix = F.cosine_similarity(
        query_embeddings.unsqueeze(1),
        candidate_embeddings.unsqueeze(0),
        dim=2,
    )
    # Top-K indices for each query
    topk_indices = sim_matrix.topk(k, dim=1).indices
    # Check if true index is in top-K
    true_indices = torch.arange(len(query_embeddings)).unsqueeze(1)
    hits = (topk_indices == true_indices).any(dim=1).float()
    return hits.mean().item()


def projection_sign_accuracy(
    embeddings: torch.Tensor,
    directions: torch.Tensor,
    target_bits: torch.Tensor,
) -> float:
    """Compute accuracy of projection sign matching target bits.

    Simulates watermark verification: checks if sgn(cos(u, v)) matches target.

    Args:
        embeddings: (N, D) semantic vectors
        directions: (N, D) direction vectors
        target_bits: (N,) values in {0, 1}, mapped to {-1, +1}
    """
    cos_proj = F.cosine_similarity(embeddings, directions, dim=1)
    predicted_sign = torch.sign(cos_proj)
    target_sign = 2.0 * target_bits.float() - 1.0  # {0,1} → {-1,+1}
    correct = (predicted_sign == target_sign).float()
    return correct.mean().item()


def save_evaluation_report(metrics: dict, output_dir: str) -> Path:
    """Save evaluation metrics to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report_path = output_path / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    return report_path
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_evaluate.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add wfcllm/encoder/evaluate.py tests/encoder/test_evaluate.py
git commit -m "feat: add encoder evaluation metrics (cosine separation, recall@k, projection sign)"
```

---

### Task 10: `wfcllm/encoder/train.py` — CLI 入口 + 数据 Pipeline 集成

**Files:**
- Create: `wfcllm/encoder/train.py`
- Create: `wfcllm/encoder/__main__.py`
- Create: `tests/encoder/test_train.py`

**Step 1: Write the failing tests**

```python
# tests/encoder/test_train.py
"""Tests for wfcllm.encoder.train entry point."""

import pytest
from unittest.mock import patch, MagicMock

from wfcllm.encoder.train import load_code_samples, prepare_blocks_with_variants


class TestLoadCodeSamples:
    @patch("wfcllm.encoder.train.load_dataset")
    def test_loads_mbpp(self, mock_load):
        mock_ds = MagicMock()
        mock_ds.__iter__ = MagicMock(return_value=iter([
            {"code": "x = 1", "task_id": 1, "text": "test"},
        ]))
        mock_load.return_value = {"train": mock_ds}
        samples = load_code_samples(["mbpp"])
        assert len(samples) > 0
        assert "code" in samples[0]


class TestPrepareBlocksWithVariants:
    def test_basic(self):
        code_samples = [{"code": "x = 1\ny = 2"}]
        blocks = prepare_blocks_with_variants(code_samples, max_variants=5)
        assert len(blocks) > 0
        for b in blocks:
            assert "source" in b
            assert "positive_variants" in b
            assert "negative_variants" in b
```

**Step 2: Run tests to verify they fail**

Run: `conda run -n WFCLLM pytest tests/encoder/test_train.py -v`
Expected: FAIL — ImportError

**Step 3: Write implementation**

```python
# wfcllm/encoder/train.py
"""Training entry point for the semantic encoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoTokenizer

from wfcllm.common.ast_parser import extract_statement_blocks
from wfcllm.common.transform.engine import TransformEngine
from wfcllm.common.transform.positive import get_all_positive_rules
from wfcllm.common.transform.negative import get_all_negative_rules
from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.dataset import TripletCodeDataset, build_triplets_from_blocks
from wfcllm.encoder.model import SemanticEncoder
from wfcllm.encoder.trainer import ContrastiveTrainer
from wfcllm.encoder.evaluate import (
    cosine_separation,
    recall_at_k,
    projection_sign_accuracy,
    save_evaluation_report,
)


def load_code_samples(data_sources: list[str]) -> list[dict]:
    """Load code samples from HuggingFace datasets."""
    from datasets import load_dataset

    samples: list[dict] = []

    for source in data_sources:
        if source == "mbpp":
            ds = load_dataset("google-research-datasets/mbpp", "full")
            for split in ds:
                for item in ds[split]:
                    samples.append({"code": item["code"], "source": "mbpp"})
        elif source == "humaneval":
            ds = load_dataset("openai/openai_humaneval")
            for split in ds:
                for item in ds[split]:
                    canonical = item.get("canonical_solution", "")
                    prompt = item.get("prompt", "")
                    code = prompt + canonical
                    samples.append({"code": code, "source": "humaneval"})

    return samples


def prepare_blocks_with_variants(
    code_samples: list[dict],
    max_variants: int = 100,
    max_perm_len: int = 3,
) -> list[dict]:
    """Extract statement blocks and generate positive/negative variants.

    Returns list of dicts with keys: source, positive_variants, negative_variants.
    """
    pos_engine = TransformEngine(
        rules=get_all_positive_rules(),
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode="positive",
    )
    neg_engine = TransformEngine(
        rules=get_all_negative_rules(),
        max_perm_len=max_perm_len,
        max_variants=max_variants,
        mode="negative",
    )

    blocks: list[dict] = []

    for sample in code_samples:
        code = sample["code"]
        stmt_blocks = extract_statement_blocks(code)

        for sb in stmt_blocks:
            pos_variants = pos_engine.generate_variants(sb.source)
            neg_variants = neg_engine.generate_variants(sb.source)

            blocks.append({
                "source": sb.source,
                "positive_variants": [v["transformed_source"] for v in pos_variants],
                "negative_variants": [v["transformed_source"] for v in neg_variants],
            })

    return blocks


def main(config: EncoderConfig | None = None) -> None:
    """Full training pipeline."""
    if config is None:
        config = EncoderConfig()

    print("=== Phase 1: Semantic Encoder Pretraining ===")

    # 1. Load data
    print(f"Loading data from {config.data_sources}...")
    code_samples = load_code_samples(config.data_sources)
    print(f"  Loaded {len(code_samples)} code samples")

    # 2. Prepare blocks with variants
    print("Extracting blocks and generating variants...")
    blocks = prepare_blocks_with_variants(code_samples)
    print(f"  Generated {len(blocks)} blocks")

    # 3. Build triplets
    print("Building triplets...")
    triplets = build_triplets_from_blocks(
        blocks, negative_ratio=config.negative_ratio, seed=42
    )
    print(f"  Built {len(triplets)} triplets")

    # 4. Create datasets
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    dataset = TripletCodeDataset(triplets, tokenizer, max_length=config.max_seq_length)

    # Split: 80% train, 10% val, 10% test
    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    n_test = n - n_train - n_val
    train_ds, val_ds, test_ds = random_split(dataset, [n_train, n_val, n_test])

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # 5. Train
    model = SemanticEncoder(config=config)
    trainer = ContrastiveTrainer(model, train_loader, val_loader, config)

    print("Starting training...")
    best_metrics = trainer.train()
    print(f"Training complete. Best val_loss: {best_metrics.get('val_loss', 'N/A')}")

    # 6. Evaluate on test set
    print("Evaluating on test set...")
    model.eval()
    device = trainer.device

    all_anchor, all_positive, all_negative = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            a = model(batch["anchor_input_ids"].to(device), batch["anchor_attention_mask"].to(device))
            p = model(batch["positive_input_ids"].to(device), batch["positive_attention_mask"].to(device))
            n = model(batch["negative_input_ids"].to(device), batch["negative_attention_mask"].to(device))
            all_anchor.append(a.cpu())
            all_positive.append(p.cpu())
            all_negative.append(n.cpu())

    anchor_embs = torch.cat(all_anchor)
    pos_embs = torch.cat(all_positive)
    neg_embs = torch.cat(all_negative)

    sep_metrics = cosine_separation(anchor_embs, pos_embs, neg_embs)
    r1 = recall_at_k(anchor_embs, pos_embs, k=1)
    r5 = recall_at_k(anchor_embs, pos_embs, k=5)
    r10 = recall_at_k(anchor_embs, pos_embs, k=10)

    # Projection sign accuracy with random directions
    torch.manual_seed(42)
    directions = torch.randn_like(anchor_embs)
    directions = torch.nn.functional.normalize(directions, dim=1)
    target_bits = torch.randint(0, 2, (len(anchor_embs),))
    sign_acc = projection_sign_accuracy(anchor_embs, directions, target_bits)

    eval_metrics = {
        **sep_metrics,
        "recall@1": r1,
        "recall@5": r5,
        "recall@10": r10,
        "projection_sign_accuracy": sign_acc,
        **best_metrics,
    }

    print("\n=== Evaluation Results ===")
    for k, v in eval_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    report_path = save_evaluation_report(eval_metrics, config.results_dir)
    print(f"\nReport saved to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the semantic encoder")
    parser.add_argument("--model-name", default="Salesforce/codet5-base")
    parser.add_argument("--embed-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--margin", type=float, default=0.3)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--no-lora", action="store_true", help="Disable LoRA (full finetune)")
    parser.add_argument("--no-bf16", action="store_true", help="Disable BF16 (use FP32)")
    args = parser.parse_args()

    config = EncoderConfig(
        model_name=args.model_name,
        embed_dim=args.embed_dim,
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        margin=args.margin,
        max_seq_length=args.max_seq_length,
        use_lora=not args.no_lora,
        use_bf16=not args.no_bf16,
    )
    main(config)
```

```python
# wfcllm/encoder/__main__.py
"""Allow running as `python -m wfcllm.encoder`."""

from wfcllm.encoder.train import main

main()
```

**Step 4: Run tests to verify they pass**

Run: `conda run -n WFCLLM pytest tests/encoder/test_train.py -v`
Expected: All tests PASS

**Step 5: Run full test suite**

Run: `conda run -n WFCLLM pytest tests/ -v`
Expected: All tests PASS

**Step 6: Commit**

```bash
git add wfcllm/encoder/train.py wfcllm/encoder/__main__.py tests/encoder/test_train.py
git commit -m "feat: add training entry point with full data pipeline"
```

---

### Task 11: Update dependencies and final integration

**Files:**
- Modify: `requirements.txt`
- Modify: `wfcllm/encoder/__init__.py`

**Step 1: Update requirements.txt**

Uncomment torch and transformers, add peft:

```
# 深度学习（不锁版本，视目标服务器适配）
torch
transformers
peft
```

**Step 2: Update encoder __init__.py**

```python
# wfcllm/encoder/__init__.py
"""Semantic encoder pretraining module."""

from wfcllm.encoder.config import EncoderConfig
from wfcllm.encoder.model import SemanticEncoder

__all__ = ["EncoderConfig", "SemanticEncoder"]
```

**Step 3: Run full test suite**

Run: `conda run -n WFCLLM pytest tests/ -v`
Expected: All tests PASS

**Step 4: Commit**

```bash
git add requirements.txt wfcllm/encoder/__init__.py
git commit -m "chore: update dependencies and encoder package exports"
```
