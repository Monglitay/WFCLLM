"""Canonical text normalization for parser-defined statement units."""

from __future__ import annotations

import io
import token
import tokenize

WINDOW_NORMALIZATION_VERSION = "wfcllm-window-normalization/v1"

_TRAILING_WHITESPACE = " \t\f\v"
_FSTRING_START = getattr(token, "FSTRING_START", None)
_FSTRING_END = getattr(token, "FSTRING_END", None)


def normalize_unit_text(text: str) -> str:
    """Return canonical unit text while preserving token-internal whitespace.

    Inside a complete ordinary string or complete f-string token, only CRLF
    and CR newline spelling is normalized; internal line-end whitespace is not
    removed. Canonical equivalence for f-string expression whitespace is
    intentionally weaker in favor of preserving runtime semantics.

    If tokenization cannot finish, only newline spelling and outer newlines are
    normalized so potential incomplete string content remains unchanged.
    """

    canonical_newlines = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = canonical_newlines.split("\n")
    trailing_starts = [
        len(line.rstrip(_TRAILING_WHITESPACE)) for line in lines
    ]
    protected_lines, tokenization_complete = (
        _string_literal_trailing_whitespace_lines(
            canonical_newlines,
            lines,
            trailing_starts,
        )
    )
    if not tokenization_complete:
        return canonical_newlines.strip("\n")

    normalized_lines = [
        line if line_number in protected_lines else line.rstrip(_TRAILING_WHITESPACE)
        for line_number, line in enumerate(lines, start=1)
    ]
    return "\n".join(normalized_lines).strip(" \t\f\v\n")


def _string_literal_trailing_whitespace_lines(
    text: str,
    lines: list[str],
    trailing_starts: list[int],
) -> tuple[set[int], bool]:
    items: list[tokenize.TokenInfo] = []
    tokens = tokenize.generate_tokens(io.StringIO(text).readline)
    while True:
        try:
            items.append(next(tokens))
        except StopIteration:
            break
        except (IndentationError, tokenize.TokenError):
            return set(), False

    protected: set[int] = set()
    fstring_starts: list[tokenize.TokenInfo] = []
    for item in items:
        if _FSTRING_START is not None and item.type == _FSTRING_START:
            fstring_starts.append(item)
            continue
        if _FSTRING_END is not None and item.type == _FSTRING_END:
            if not fstring_starts:
                return set(), False
            start = fstring_starts.pop()
            _protect_source_span(
                start.start,
                item.end,
                lines,
                trailing_starts,
                protected,
            )
            continue
        if item.type == token.STRING:
            _protect_source_span(
                item.start,
                item.end,
                lines,
                trailing_starts,
                protected,
            )

    if fstring_starts:
        return set(), False
    return protected, True


def _protect_source_span(
    start: tuple[int, int],
    end: tuple[int, int],
    lines: list[str],
    trailing_starts: list[int],
    protected: set[int],
) -> None:
    for line_number in range(start[0], end[0] + 1):
        line = lines[line_number - 1]
        trailing_start = trailing_starts[line_number - 1]
        if trailing_start == len(line):
            continue

        span_start = start[1] if line_number == start[0] else 0
        span_end = end[1] if line_number == end[0] else len(line)
        if span_start < len(line) and span_end > trailing_start:
            protected.add(line_number)
