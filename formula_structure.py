"""Derive lightweight symbol-position supervision from LaTeX captions."""

from __future__ import annotations

from typing import Sequence


POSITION_MAIN = 0
POSITION_UPPER = 1
POSITION_LOWER = 2
POSITION_IGNORE = -100


def derive_position_labels(tokens: Sequence[str]) -> list[int]:
    """Label each target token as main, upper, or lower relative to its structure.

    The labels are weak supervision derived from LaTeX syntax, not image boxes.
    They are an auxiliary signal only; the main decoder still learns the complete
    original caption, including braces and relation markers.
    """
    labels = [POSITION_MAIN] * len(tokens)
    for index, token in enumerate(tokens):
        if token in {"{", "}", "^", "_", r"\limits"}:
            labels[index] = POSITION_IGNORE

    def parse_group(index: int, role: int) -> int:
        if index >= len(tokens):
            return index
        if tokens[index] == "{":
            labels[index] = POSITION_IGNORE
            return parse_sequence(index + 1, role, stop_at_brace=True)
        return parse_atom(index, role)

    def parse_atom(index: int, role: int) -> int:
        token = tokens[index]
        if token == "{":
            labels[index] = POSITION_IGNORE
            return parse_sequence(index + 1, role, stop_at_brace=True)

        if token == r"\frac":
            labels[index] = role
            index = parse_group(index + 1, POSITION_UPPER)
            return parse_group(index, POSITION_LOWER)

        if token == r"\sqrt":
            labels[index] = role
            index += 1
            # In CROHME captions, an optional root index is written as [ ... ].
            if index < len(tokens) and tokens[index] == "[":
                labels[index] = POSITION_IGNORE
                index = parse_sequence(index + 1, POSITION_UPPER, stop_at_bracket=True)
            return parse_group(index, role)

        labels[index] = role
        return index + 1

    # parse_sequence optionally stops at a closing square bracket for sqrt indices.
    # A separate wrapper keeps the normal brace parser easy to inspect.
    def parse_sequence(index: int, role: int, stop_at_brace: bool = False,
                       stop_at_bracket: bool = False) -> int:
        while index < len(tokens):
            token = tokens[index]
            if stop_at_brace and token == "}":
                labels[index] = POSITION_IGNORE
                return index + 1
            if stop_at_bracket and token == "]":
                labels[index] = POSITION_IGNORE
                return index + 1
            if token in ("^", "_"):
                labels[index] = POSITION_IGNORE
                child_role = POSITION_UPPER if token == "^" else POSITION_LOWER
                index = parse_group(index + 1, child_role)
                continue
            if token == r"\limits":
                labels[index] = POSITION_IGNORE
                index += 1
                continue
            index = parse_atom(index, role)
        return index

    parse_sequence(0, POSITION_MAIN, stop_at_brace=False)
    return labels


def align_position_labels(tokens: Sequence[str], max_target_length: int) -> list[int]:
    """Align token roles with target tokens, EOS, and fixed-length padding."""
    labels = derive_position_labels(tokens)
    if len(tokens) < max_target_length:
        labels.append(POSITION_IGNORE)  # EOS has no spatial role.
    labels = labels[:max_target_length]
    return labels + [POSITION_IGNORE] * (max_target_length - len(labels))
