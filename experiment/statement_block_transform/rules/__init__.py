"""Transform rules registry."""

from .base import Match, Rule, parse_code

ALL_RULES: list[Rule] = []


def register_rules(rules: list[Rule]) -> None:
    """Register rules into the global registry."""
    ALL_RULES.extend(rules)


def get_all_rules() -> list[Rule]:
    """Return all registered rules."""
    return list(ALL_RULES)
