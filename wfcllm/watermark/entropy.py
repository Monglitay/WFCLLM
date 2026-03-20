"""Heuristic entropy estimation for AST nodes.

Uses a pre-computed lookup table of mean entropy values per AST node type,
derived from experiment/node_entropy/results/node_entropy_results.json.
"""

from __future__ import annotations

from wfcllm.common.ast_parser import PythonParser
from wfcllm.watermark.config import WatermarkConfig

ENTROPY_SCALE = 10000


class NodeEntropyEstimator:
    """Estimate statement block entropy via AST node type lookup table."""

    DEFAULT_ENTROPY: float = 0.1

    # Pre-computed from experiment/node_entropy — 133 AST node types
    ENTROPY_TABLE: dict[str, float] = {
        "boolean_operator": 0.959,
        "for_statement": 0.7095,
        "function_definition": 0.504,
        "comparison_operator": 0.4944,
        "identifier": 0.4909,
        "conditional_expression": 0.4487,
        "if_statement": 0.4348,
        "while_statement": 0.4266,
        "binary_operator": 0.4026,
        "default_parameter": 0.3584,
        "module": 0.3571,
        "comment": 0.3333,
        "call": 0.3198,
        "block": 0.3116,
        "dictionary_comprehension": 0.3062,
        "elif_clause": 0.2992,
        "assignment": 0.2971,
        "expression_statement": 0.2817,
        "parenthesized_expression": 0.2809,
        "list_comprehension": 0.2779,
        "argument_list": 0.2251,
        "slice": 0.2214,
        "if_clause": 0.1979,
        "string_content": 0.1949,
        "generator_expression": 0.1901,
        "else_clause": 0.168,
        "float": 0.1672,
        "for_in_clause": 0.1556,
        "return_statement": 0.1539,
        "integer": 0.1522,
        "not_operator": 0.1496,
        "subscript": 0.1225,
        "yield": 0.1138,
        "tuple": 0.1105,
        "augmented_assignment": 0.107,
        "set_comprehension": 0.1048,
        "string": 0.0881,
        "expression_list": 0.0749,
        "lambda": 0.0716,
        "attribute": 0.0688,
        "list": 0.0569,
        "set": 0.0467,
        "parameters": 0.0451,
        "unary_operator": 0.0375,
        "class_definition": 0.0317,
        "pattern_list": 0.0259,
        "keyword_argument": 0.0187,
        "string_start": 0.017,
        "pair": 0.0145,
        "string_end": 0.0133,
        "import_from_statement": 0.008,
        "dotted_name": 0.0077,
        "=": 0.0,
        "def": 0.0,
        ":": 0.0,
        "(": 0.0,
        ")": 0.0,
        ",": 0.0,
        ".": 0.0,
        "class": 0.0,
        "return": 0.0,
        "[": 0.0,
        "]": 0.0,
        "if": 0.0,
        "<": 0.0,
        "and": 0.0,
        "in": 0.0,
        ">": 0.0,
        "for": 0.0,
        "+": 0.0,
        "+=": 0.0,
        "!=": 0.0,
        "while": 0.0,
        "-": 0.0,
        "==": 0.0,
        "%": 0.0,
        "//": 0.0,
        "true": 0.0,
        "false": 0.0,
        "else": 0.0,
        ">=": 0.0,
        "import": 0.0,
        "import_statement": 0.0,
        "/": 0.0,
        "*": 0.0,
        "list_splat": 0.0,
        "lambda_parameters": 0.0,
        "or": 0.0,
        ";": 0.0,
        "<=": 0.0,
        "break_statement": 0.0,
        "break": 0.0,
        "elif": 0.0,
        "**": 0.0,
        "-=": 0.0,
        "from": 0.0,
        "not": 0.0,
        "^": 0.0,
        "as": 0.0,
        "aliased_import": 0.0,
        "\\": 0.0,
        "*=": 0.0,
        "not in": 0.0,
        "{": 0.0,
        "}": 0.0,
        "&": 0.0,
        "|": 0.0,
        "<<": 0.0,
        "~": 0.0,
        "continue": 0.0,
        "continue_statement": 0.0,
        "<<=": 0.0,
        "/=": 0.0,
        "tuple_pattern": 0.0,
        ">>": 0.0,
        "|=": 0.0,
        "is": 0.0,
        "none": 0.0,
        "line_continuation": 0.0,
        "dictionary": 0.0,
        "except_clause": 0.0,
        "except": 0.0,
        "try_statement": 0.0,
        "try": 0.0,
        "escape_sequence": 0.0,
        ">>=": 0.0,
        "//=": 0.0,
        "%=": 0.0,
        "pass_statement": 0.0,
        "pass": 0.0,
        "del": 0.0,
        "delete_statement": 0.0,
        "is not": 0.0,
    }

    def estimate_block_entropy(self, block_source: str) -> float:
        """Estimate entropy as float, derived from canonical integer units."""
        return self.estimate_block_entropy_units(block_source) / ENTROPY_SCALE

    def estimate_block_entropy_units(self, block_source: str) -> int:
        """Parse block AST, traverse all sub-nodes, sum entropy in canonical units."""
        if not block_source.strip():
            return 0
        parser = PythonParser()
        tree = parser.parse(block_source)
        entropy = self._sum_entropy(tree.root_node)
        return max(0, int(round(entropy * ENTROPY_SCALE)))

    def _sum_entropy(self, node) -> float:
        total = self.ENTROPY_TABLE.get(node.type, self.DEFAULT_ENTROPY)
        for child in node.children:
            total += self._sum_entropy(child)
        return total

    def compute_margin(self, block_entropy: float, config: WatermarkConfig) -> float:
        """Dynamic margin = m_base + alpha * block_entropy."""
        return config.margin_base + config.margin_alpha * block_entropy
