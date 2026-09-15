"""Tests for preprocessing rules P1-P6 on synthetic fixtures with known answers."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.loaders import Trace
from src.data.preprocessing import (
    apply_all,
    apply_p1_merge_tiers,
    apply_p2_count_steps,
    apply_p3_error_free_code,
    apply_p4_relative_onset,
    apply_p5_min_steps,
)


def make_trace(first_error_idx=2, n=5, subset="gsm8k"):
    return Trace(
        trace_id="t", question_id="q", dataset="d", subset=subset,
        question="q?", steps=[f"s{i}" for i in range(n)], n_steps=n,
        first_error_idx=first_error_idx, final_correct=first_error_idx is None,
        generator="test",
    )


def test_p1_keeps_subset_covariate():
    traces = [make_trace(subset="gsm8k"), make_trace(subset="math")]
    merged = apply_p1_merge_tiers(traces)
    assert {t.subset for t in merged} == {"gsm8k", "math"}


def test_p2_counts_steps():
    trace = make_trace(n=4)
    trace.n_steps = 0
    assert apply_p2_count_steps(trace).n_steps == 4


def test_p3_error_free_code():
    assert apply_p3_error_free_code(make_trace(first_error_idx=-1)).first_error_idx is None
    assert apply_p3_error_free_code(make_trace(first_error_idx=3)).first_error_idx == 3


def test_p4_relative_onset():
    assert apply_p4_relative_onset(2, 10) == 0.2
    assert apply_p4_relative_onset(None, 10) is None


def test_p5_drops_single_step():
    assert apply_p5_min_steps(make_trace(n=1), 2) is False
    assert apply_p5_min_steps(make_trace(n=2), 2) is True


def test_apply_all_end_to_end():
    traces = [make_trace(first_error_idx=-1, n=4), make_trace(first_error_idx=2, n=4),
              make_trace(first_error_idx=1, n=1)]
    cleaned = apply_all(traces, min_steps=2, error_free_code=-1)
    assert len(cleaned) == 2  # single-step trace dropped
    assert cleaned[0].first_error_idx is None
