"""Convert CROHME LaTeX-like captions into flat visible-symbol sequences."""
from typing import Iterable, List, Set


STRUCTURAL_TOKENS = {"{", "}", "^", "_", r"\limits"}
FUNCTION_WORDS = {
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
    r"\log": "log",
    r"\lim": "lim",
}
FRAC_BAR_TOKEN = "<FRAC_BAR>"
RADICAL_TOKEN = "<RADICAL>"


def _sqrt_optional_index_brackets(tokens: List[str]) -> Set[int]:
    """Return only the square-bracket delimiters used for a root index."""
    skipped: Set[int] = set()
    for index, token in enumerate(tokens[:-1]):
        if token != r"\sqrt" or tokens[index + 1] != "[":
            continue
        depth = 0
        for cursor in range(index + 1, len(tokens)):
            if tokens[cursor] == "[":
                depth += 1
            elif tokens[cursor] == "]":
                depth -= 1
                if depth == 0:
                    skipped.update((index + 1, cursor))
                    break
    return skipped


def visible_symbol_tokens(caption: str) -> List[str]:
    """Flatten positioning syntax while retaining tokens for visible marks.

    Plain braces, superscript/subscript markers and ``\\limits`` are omitted.
    Escaped braces remain visible symbols. Square brackets remain visible unless
    they delimit an optional root index. Fraction and radical commands map to
    visual primitives; common function commands expand to their handwritten letters.
    """
    tokens = caption.split()
    root_index_delimiters = _sqrt_optional_index_brackets(tokens)
    visible: List[str] = []
    for index, token in enumerate(tokens):
        if index in root_index_delimiters or token in STRUCTURAL_TOKENS:
            continue
        if token == r"\frac":
            visible.append(FRAC_BAR_TOKEN)
        elif token == r"\sqrt":
            visible.append(RADICAL_TOKEN)
        elif token in FUNCTION_WORDS:
            visible.extend(FUNCTION_WORDS[token])
        else:
            visible.append(token)
    return visible


def visible_vocabulary(dictionary_tokens: Iterable[str]) -> List[str]:
    """Build the cleaned vocabulary using the original CROHME token inventory."""
    result = set()
    for token in dictionary_tokens:
        result.update(visible_symbol_tokens(token))
    result.update((FRAC_BAR_TOKEN, RADICAL_TOKEN))
    return sorted(result)
