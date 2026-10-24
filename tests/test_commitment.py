"""Tests for the choke-point decomposition (stage 11) and commitment analysis (stages 12-13).

The planted-effect test runs the full stage 12 -> 13 pipeline with the MOCK
generator, whose success probability collapses exactly at the annotated first
error, and checks that the analysis recovers that planted structure.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis import commitment as cm  # noqa: E402
from src.analysis import decomposition as dec  # noqa: E402


def test_last_boxed_nested_and_missing():
    assert cm.last_boxed(r"a \boxed{1} b \boxed{\frac{3}{4}}") == r"\frac{3}{4}"
    assert cm.last_boxed("nothing here") is None


def test_answers_match_fallback_numeric():
    assert cm.answers_match("1,000", "1000", use_math_verify=False)
    assert not cm.answers_match("999", "1000", use_math_verify=False)
    assert not cm.answers_match(None, "1", use_math_verify=False)


def test_commitment_and_decisive_step():
    v = np.array([0.8, 0.7, 0.1, 0.0, 0.0])
    assert cm.decisive_step(v) == 2
    assert cm.commitment_step(v, 0.125) == 2
    assert cm.classify_failing_trace(v, 2, 0.125) == "commits-at-error"
    assert cm.classify_failing_trace(v, 4, 0.125) == "latent-commitment"
    assert cm.classify_failing_trace(v, 1, 0.125) == "recoverable-error"
    assert cm.classify_failing_trace(np.array([0.0, 0.0, 0.0]), 1, 0.125) == "capability-limited"


def test_grouped_folds_never_split_a_question():
    groups = np.array(["a", "a", "b", "c", "c", "c", "d", "e"])
    folds = dec.grouped_folds(groups, 3, seed=7)
    for g in set(groups):
        assert len(set(folds[groups == g])) == 1


def test_concordance_ignores_cross_problem_pairs():
    groups = np.array(["p", "p", "q", "q"])
    y = np.array([1, 0, 1, 0])
    pairs = dec.within_problem_pairs(groups, y)
    assert sorted(pairs) == [(0, 1), (2, 3)]
    scores = np.array([0.9, 0.1, 0.2, 0.8])
    assert dec.concordance(scores, pairs) == 0.5


def test_correction_marker():
    assert dec.has_correction_marker("Wait, I made a mistake above.")
    assert not dec.has_correction_marker("The correct answer is 5.")


def test_planted_effect_recovered_by_pipeline(tmp_path):
    traces = ROOT / "data" / "processed" / "processed_traces.jsonl"
    if not traces.exists():
        return  # needs preprocessed ProcessBench (stage 02)
    out = tmp_path / "mock.jsonl"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "12_counterfactual_rollouts.py"),
                    "--generator", "Qwen2.5-1.5B-Instruct", "--backend", "mock", "--no-source-gold",
                    "--m", "32", "--out", str(out), "--traces", str(traces)], check=True, cwd=ROOT)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "13_commitment_analysis.py"),
                    "--rollouts", str(out), "--out", str(tmp_path / "tables")], check=True, cwd=ROOT)
    summary = json.loads((tmp_path / "tables" / "summary.json").read_text())
    share, _, _ = summary["first_error_is_decisive"]["share"]
    assert share > 0.9  # the mock collapses success exactly at the first error
