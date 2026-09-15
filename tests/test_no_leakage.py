"""The critical test: every split in this repo must be question-disjoint."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.seeds import assert_no_leakage, grouped_split


def test_grouped_split_is_disjoint():
    groups = [f"q{i // 4}" for i in range(100)]  # 4 correlated rows per question
    tr, te = grouped_split(groups, test_size=0.25, seed=42, context="test")
    assert_no_leakage([groups[i] for i in tr], [groups[i] for i in te], "test")


def test_assert_no_leakage_catches_overlap():
    try:
        assert_no_leakage(["q1", "q2"], ["q2", "q3"], "test")
    except AssertionError:
        return
    raise AssertionError("assert_no_leakage failed to flag overlapping question IDs")


def test_repeated_seeds_stay_disjoint():
    groups = [f"q{i // 3}" for i in range(60)]
    for seed in (0, 1, 42):
        tr, te = grouped_split(groups, test_size=0.3, seed=seed, context="test")
        assert_no_leakage([groups[i] for i in tr], [groups[i] for i in te], "test")
