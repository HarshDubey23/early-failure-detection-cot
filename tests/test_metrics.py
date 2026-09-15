"""Metric tests against hand-computed examples and fixtures whose answer is known by construction."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.lead_time import first_alarm, lead_time_summary, lead_times
from src.analysis.onset import uniform_step_expected_rho
from src.analysis.statistics import (
    auroc_with_ci,
    expected_calibration_error,
    ks_discrete_uniform_null,
    mann_whitney_length,
)
from src.intervention.simulate import (
    InterventionAssumptions,
    majority_vote_accuracy,
    simulate_oracle,
    simulate_policy,
    suffix_tokens,
)


def assumptions(**overrides) -> InterventionAssumptions:
    base = dict(p_verify_repair=1.0, p_verify_corrupt=0.0, p_regen_repair=1.0, p_regen_corrupt=0.0,
                p_revert_catches_corruption=0.0, verify_tokens=10, chars_per_token=1.0)
    return InterventionAssumptions(**{**base, **overrides})


def test_auroc_perfect_separation():
    result = auroc_with_ci(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]), n_bootstrap=50, seed=0)
    assert result["auroc"] == 1.0


def test_auroc_hand_computed():
    # Positive-negative pairs: (0.35>0.1) (0.35<0.4) (0.8>0.1) (0.8>0.4) -> 3 of 4 correctly ordered.
    result = auroc_with_ci(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8]), n_bootstrap=10, seed=0)
    assert result["auroc"] == pytest.approx(0.75)


def test_auroc_constant_score_is_half():
    assert auroc_with_ci(np.array([0, 0, 1, 1]), np.full(4, 0.5), n_bootstrap=10, seed=0)["auroc"] == 0.5


def test_ece_perfect_calibration():
    assert expected_calibration_error(np.array([0, 0, 1, 1]), np.array([0.0, 0.0, 1.0, 1.0]), 2)["ece"] == 0.0


def test_ece_hand_computed():
    # One bin (0.5, 1.0]: mean confidence 0.8, observed rate 0.5 -> ECE = |0.5 - 0.8| = 0.3.
    ece = expected_calibration_error(np.array([1, 0]), np.array([0.8, 0.8]), n_bins=2)["ece"]
    assert ece == pytest.approx(0.3)


def test_rank_biserial_sign():
    longer = mann_whitney_length(np.array([5, 6, 7]), np.array([1, 2, 3]))
    shorter = mann_whitney_length(np.array([1, 2, 3]), np.array([5, 6, 7]))
    assert longer["rank_biserial_r"] == pytest.approx(1.0)
    assert shorter["rank_biserial_r"] == pytest.approx(-1.0)


def test_uniform_step_reference_value():
    # n=2: rho in {0.5, 1} -> mean 0.75; n=4: {.25,.5,.75,1} -> 0.625; average 0.6875.
    assert uniform_step_expected_rho(np.array([2, 4])) == pytest.approx(0.6875)


def test_discrete_null_does_not_reject_uniform_steps():
    rng = np.random.default_rng(3)
    n_steps = rng.integers(2, 6, size=400)
    first_error = np.floor(rng.random(400) * n_steps) + 1  # truly uniform over steps
    result = ks_discrete_uniform_null(first_error, n_steps, n_simulations=200, seed=1)
    assert result["p_value"] > 0.05


def test_discrete_null_rejects_early_concentration():
    n_steps = np.full(300, 10)
    first_error = np.ones(300)  # every error at step 1
    assert ks_discrete_uniform_null(first_error, n_steps, n_simulations=200, seed=1)["p_value"] < 0.01


def test_first_alarm_and_lead_time():
    assert first_alarm([0.1, 0.6, 0.4], 0.5) == 2
    assert first_alarm([0.1, 0.2], 0.5) is None
    df = pd.DataFrame([{"trace_id": "a", "has_error": True, "first_error_idx": 3},
                       {"trace_id": "b", "has_error": True, "first_error_idx": 1}])
    lead = lead_times(df, {"a": [0.1, 0.9, 0.9], "b": [0.1, 0.1]}, [0.5])
    assert lead.set_index("trace_id").loc["a", "delta"] == 1.0  # alarm at 2, error at 3
    summary = lead_time_summary(lead)
    assert summary["frac_warned_before_error"].iloc[0] == 0.5  # b never alarms -> not warned


def test_majority_vote_accuracy():
    assert majority_vote_accuracy(0.6, 3) == pytest.approx(0.648)
    assert majority_vote_accuracy(0.6, 2) == pytest.approx(0.36 + 0.5 * 0.48)  # tie broken at random


def test_suffix_tokens():
    assert suffix_tokens(["aaaa", "bbbbbbbb"], 2, 4.0) == 2.0


def test_recovery_corruption_known_fixture():
    # Failing trace alarms and is always repaired; succeeding trace never alarms.
    df = pd.DataFrame([{"trace_id": "fail", "answer_incorrect": True, "steps": ["x", "y"]},
                       {"trace_id": "ok", "answer_incorrect": False, "steps": ["x", "y"]}])
    out = simulate_policy(df, {"fail": [0.9, 0.9], "ok": [0.1, 0.1]}, 0.5, 0.8, 256, assumptions(),
                          "answer_incorrect", seed=0)
    assert out["recovery_rate"] == 1.0 and out["corruption_rate"] == 0.0
    assert out["net_accuracy_change"] == 0.5


def test_revert_safeguard_prevents_corruption():
    df = pd.DataFrame([{"trace_id": "ok", "answer_incorrect": False, "steps": ["x"]}])
    risks = {"ok": [0.6]}  # alarms -> verify
    always_breaks = assumptions(p_verify_corrupt=1.0, p_revert_catches_corruption=0.0)
    always_caught = assumptions(p_verify_corrupt=1.0, p_revert_catches_corruption=1.0)
    assert simulate_policy(df, risks, 0.5, 0.8, 256, always_breaks, "answer_incorrect", 0)["corruption_rate"] == 1.0
    assert simulate_policy(df, risks, 0.5, 0.8, 256, always_caught, "answer_incorrect", 0)["corruption_rate"] == 0.0


def test_oracle_regenerates_from_true_error():
    df = pd.DataFrame([{"trace_id": "f", "answer_incorrect": True, "first_error_idx": 2,
                        "steps": ["aa", "bbb", "c"]},
                       {"trace_id": "s", "answer_incorrect": False, "first_error_idx": None,
                        "steps": ["aa"]}])
    out = simulate_oracle(df, {"f": [0.0] * 3, "s": [0.0]}, assumptions(), "answer_incorrect")
    assert out["recovery_rate"] == 1.0 and out["corruption_rate"] == 0.0
    assert out["extra_tokens_per_question"] == pytest.approx(4 / 2)  # "bbb"+"c" = 4 chars over 2 traces
