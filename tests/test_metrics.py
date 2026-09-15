"""Metric tests against hand-computed examples and constructed fixtures."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.lead_time import first_alarm
from src.analysis.statistics import (
    auroc_with_ci,
    expected_calibration_error,
)
from src.intervention.simulate import simulate


def test_auroc_perfect_separation():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    result = auroc_with_ci(y, scores, n_bootstrap=50, seed=0)
    assert result["auroc"] == 1.0


def test_auroc_random_is_half():
    y = np.array([0, 0, 1, 1])
    scores = np.array([0.5, 0.5, 0.5, 0.5])
    result = auroc_with_ci(y, scores, n_bootstrap=10, seed=0)
    assert result["auroc"] == 0.5


def test_ece_perfect_calibration():
    y = np.array([0, 0, 1, 1])
    probs = np.array([0.0, 0.0, 1.0, 1.0])
    assert expected_calibration_error(y, probs, n_bins=2)["ece"] == 0.0


def test_first_alarm():
    assert first_alarm([0.1, 0.6, 0.4], 0.5) == 2
    assert first_alarm([0.1, 0.2], 0.5) is None


def test_recovery_corruption_known_fixture():
    # Two traces: one failing with high risk (repaired), one succeeding with
    # low risk (never alarmed). Recovery=1.0, corruption=0.0 by construction.
    df = pd.DataFrame([
        {"trace_id": "fail", "has_error": True},
        {"trace_id": "ok", "has_error": False},
    ])
    risks = {"fail": [0.9, 0.9], "ok": [0.1, 0.1]}
    out = simulate(df, risks, threshold=0.5, escalation_threshold=0.8,
                   token_budget=256, p_verify=1.0, q_verify=0.0,
                   p_regen=1.0, q_regen=0.0, seed=0)
    assert out["recovery_rate"] == 1.0
    assert out["corruption_rate"] == 0.0
