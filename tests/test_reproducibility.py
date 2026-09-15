"""Reproducibility: same seed + same config -> identical outputs."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.statistics import auroc_with_ci
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
    y = rng.integers(0, 2, 200)
    scores = rng.random(200)
    first = auroc_with_ci(y, scores, n_bootstrap=50, seed=11)
    second = auroc_with_ci(y, scores, n_bootstrap=50, seed=11)
    assert first == second
