"""Transform engine: applies rule permutations to code blocks."""

from __future__ import annotations

from itertools import permutations

from rules.base import Rule, parse_code


class TransformEngine:
    """Applies all permutations of applicable rules to a code block."""

    def __init__(
        self,
        rules: list[Rule],
        max_perm_len: int = 5,
        max_variants: int = 1000,
    ):
        self.rules = rules
        self.max_perm_len = max_perm_len
        self.max_variants = max_variants

    def get_applicable_rules(self, source: str) -> list[Rule]:
        """Return rules that can be applied to the given source."""
        tree = parse_code(source)
        return [r for r in self.rules if r.detect(source, tree)]

    def generate_variants(self, source: str) -> list[dict]:
        """Generate all permutation variants of applicable rules.

        Returns list of dicts with keys:
            - variant_id: int
            - rules_applied: list[str]  (rule names in order)
            - transformed_source: str
        """
        applicable = self.get_applicable_rules(source)
        if not applicable:
            return []

        variants = []
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
                return None  # rule no longer applicable after prior transforms
            current = rule.apply(current, matches)
        return current
