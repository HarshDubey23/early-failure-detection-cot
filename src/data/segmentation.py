"""
WHAT: Step-boundary detection for segmenting generated text into steps.

WHY:  The paper's Limitations section is explicit: steps are delimited
      heuristically by line breaks, which approximates but does not guarantee
      semantic boundaries. Centralising that heuristic in one module means the
      approximation is stated once and applied consistently - including in the
      activation-extraction path, where hidden states are read at each step's
      final token.

HOW:  `split_steps` strips blank lines. `step_boundary_token_positions` maps
      each step back to its final token index in a tokenised full sequence by
      incremental re-tokenisation, clamped to the hidden-state array bounds.
"""

from __future__ import annotations


def split_steps(generated_text: str) -> list:
    """Split generated text into non-empty line-delimited steps.

    Example:
        >>> split_steps("d = 60\\n\\nv = d * t\\n")
        ['d = 60', 'v = d * t']
    """
    return [line.strip() for line in generated_text.splitlines() if line.strip()]


def step_boundary_token_positions(
    prompt_text: str, steps: list, tokenize, n_tokens: int
) -> list:
    """Map each step to the token index of its final token.

    Args:
        prompt_text: Text preceding the generation.
        steps: Step strings in order.
        tokenize: Callable `str -> list[int]` (tokeniser encode function).
        n_tokens: Length of the hidden-state sequence (for clamping).

    Returns:
        One token position per step (0-indexed, clamped to `n_tokens - 1`).

    HOW: cumulative text `prompt + steps[:i+1]` is re-tokenised; its last
    index is the boundary. Clamping guards against tokeniser mismatch.
    """
    positions = []
    cumulative = prompt_text
    for step in steps:
        cumulative += step + "\n"
        position = len(tokenize(cumulative)) - 1
        position = max(0, min(position, n_tokens - 1))
        positions.append(position)
    return positions
