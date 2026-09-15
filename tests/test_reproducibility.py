"""Reproducibility: same seed + same config -> identical outputs."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.predictability import make_split, run_surface_curve, run_tfidf_curve
from src.analysis.statistics import auroc_with_ci
from src.config import load_config
from src.data.loaders import traces_to_frame
from src.data.synthetic import synthetic_traces
from src.intervention.simulate import InterventionAssumptions, compare_policies
from src.utils.seeds import grouped_split, seed_everything


def test_same_seed_same_split():
    groups = [f"q{i // 2}" for i in range(40)]
    seed_everything(7)
    tr1, te1 = grouped_split(groups, 0.25, 7, context="test")
    seed_everything(7)
    tr2, te2 = grouped_split(groups, 0.25, 7, context="test")
    assert list(tr1) == list(tr2) and list(te1) == list(te2)


def test_bootstrap_ci_is_seeded():
    rng = np.random.default_rng(3)
    y, scores = rng.integers(0, 2, 200), rng.random(200)
    assert auroc_with_ci(y, scores, n_bootstrap=50, seed=11) == auroc_with_ci(y, scores, n_bootstrap=50, seed=11)


def test_prefix_curves_identical_across_runs():
    frames = []
    for _ in range(2):
        df = traces_to_frame(synthetic_traces(seed=5, n_questions=30, steps_range=[3, 8]))
        split = make_split(df, 42, 0.25, "test")
        surface, _ = run_surface_curve(df, "answer_incorrect", [0.5, 1.0], split, 42, 50)
        lexical, _ = run_tfidf_curve(df, "answer_incorrect", [0.5, 1.0], split, 42, 50, min_df=1)
        frames.append(pd.concat([surface, lexical], ignore_index=True))
    pd.testing.assert_frame_equal(frames[0], frames[1])


def test_intervention_simulation_identical_across_runs():
    df = traces_to_frame(synthetic_traces(seed=5, n_questions=10, steps_range=[3, 5]))
    risks = {t: list(np.linspace(0.2, 0.9, n)) for t, n in zip(df["trace_id"], df["n_steps"])}
    params = InterventionAssumptions(0.4, 0.05, 0.6, 0.1, 0.5, 32, 4.0)
    run = lambda: compare_policies(df, risks, [0.5, 0.7], 0.8, 256, params, [2], "answer_incorrect", 42)  # noqa: E731
    pd.testing.assert_frame_equal(run(), run())


def test_both_configs_load_and_validate():
    for name in ("default.yaml", "quick.yaml"):
        cfg = load_config(ROOT / "configs" / name)
        assert cfg.split.group_field == "question_id"
        assert cfg.prefix_grid[-1] == 1.0
