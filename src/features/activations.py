"""
WHAT: Hidden-state extraction from an open model at reasoning step boundaries.

WHY:  The research hypothesis is that internal representations carry
      information about eventual failure before the failure is visible in the
      output text. Testing that requires activations captured at the moment
      each step completes - not at the end of generation, which would defeat
      the purpose.

HOW:  A single forward pass over the full generated sequence yields hidden
      states for every token. We map each step's final token to its index in
      that sequence (see `src/data/segmentation.py`) and take the activation
      at that position, for each requested layer.
"""

from __future__ import annotations

import numpy as np


def needs_gpu() -> bool:
    """Return True when CUDA is available for activation extraction."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def extract_step_states(
    model,
    tokenizer,
    prompt_text: str,
    generated_text: str,
    steps: list,
    layers: list,
) -> dict:
    """Extract hidden states at each step boundary for the requested layers.

    Args:
        model: Causal LM (hidden states are requested per call).
        tokenizer: Its tokeniser.
        prompt_text: Text preceding the generation.
        generated_text: Full generated continuation.
        steps: Step strings (from `split_steps`).
        layers: Model layer indices to capture.

    Returns:
        Dict `layer -> (n_steps, hidden_size)` numpy array.
    """
    import torch

    from src.data.segmentation import step_boundary_token_positions

    full = tokenizer(prompt_text + generated_text, return_tensors="pt")
    full = {k: v.to(model.device) for k, v in full.items()}
    with torch.no_grad():
        outputs = model(**full, output_hidden_states=True)
    # hidden_states[0] is the embedding output; hidden_states[L] is the output of
    # transformer layer L, so valid layer indices are 1..num_hidden_layers.
    hidden_states = outputs.hidden_states
    n_tokens = hidden_states[0].shape[1]
    positions = step_boundary_token_positions(
        prompt_text, steps, lambda s: tokenizer(s).input_ids, n_tokens
    )
    result = {}
    for layer in layers:
        states = hidden_states[layer][0].float().cpu().numpy()
        result[layer] = np.stack([states[pos] for pos in positions])
    return result


def check_gpu_or_raise() -> None:
    """Raise a named error when no GPU is present (callers catch -> skip)."""
    if not needs_gpu():
        raise RuntimeError(
            "Activation extraction requires a GPU runtime (CUDA unavailable). "
            "Run 'make activations' on a GPU machine, or continue with "
            "non-activation representations; downstream stages mark E4 NOT_RUN."
        )
