"""
WHAT: Seeded synthetic trace corpus for the 5-minute smoke test (`make smoke`).

WHY:  The smoke test must exercise every stage with no network and no GPU. A
      tiny labelled corpus with a KNOWN construction lets the pipeline run end
      to end in minutes. Its numbers mean nothing about real models, which is
      why every smoke output is written under results/smoke/ and labelled
      SYNTHETIC.

HOW:  Each question gets 3 traces. A failing trace is 2 steps longer and has
      the token "error marker xyz" at its first erroneous step, so lexical and
      length detectors can separate classes by construction. Two tiers
      (tierA / tierB) exist so the Kruskal-Wallis and transfer code paths run.
"""

from __future__ import annotations

import numpy as np

from .loaders import Trace


def synthetic_traces(seed: int, n_questions: int, steps_range: list,
                     dataset: str = "synthetic") -> list:
    """Generate the smoke corpus.

    Args:
        seed: Config seed.
        n_questions: Number of distinct questions (3 traces each).
        steps_range: [min, max] steps for a non-failing trace.
        dataset: Dataset name recorded on each trace (a second name gives a
            separate "domain" for the transfer code path).

    Returns:
        List of `Trace` with 1-indexed first_error_idx and final_correct.
    """
    rng = np.random.default_rng(seed)
    traces = []
    for q in range(n_questions):
        for sample in range(3):
            n_steps = int(rng.integers(steps_range[0], steps_range[1] + 1))
            fails = bool(rng.random() < 0.5)
            first_error = None
            if fails:
                n_steps += 2
                first_error = int(rng.integers(1, n_steps + 1))
            steps = [
                f"synthetic step {i + 1} with "
                + ("error marker xyz " if fails and i + 1 == first_error else "")
                + "reasoning text filler"
                for i in range(n_steps)
            ]
            traces.append(Trace(
                trace_id=f"{dataset}:q{q}:{sample}", question_id=f"{dataset}-q{q}",
                dataset=dataset, subset="tierA" if q % 2 == 0 else "tierB",
                question=f"{dataset} smoke question {q}", steps=steps, n_steps=n_steps,
                first_error_idx=first_error, final_correct=not fails, generator="synthetic",
            ))
    return traces
