"""ChokeBench localisers name the intended step (1-based) on a toy trace."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("s49", Path(__file__).resolve().parents[1] / "scripts" / "49_chokebench.py")
s49 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s49)

PRM = [0.9, 0.8, 0.3, 0.6, 0.2]
GEN = {"nll_max": [1.0, 5.0, 2.0, 0.5, 1.0], "ent_max": [0.1, 0.2, 0.3, 4.0, 0.1]}


def test_localisers_on_toy_trace():
    L = s49.localise
    assert L("human_first_error", 5, 3, PRM, GEN) == 3
    assert L("prm_first_below_0.5", 5, 3, PRM, GEN) == 3
    assert L("prm_min", 5, 3, PRM, GEN) == 5
    assert L("prm_largest_drop", 5, 3, PRM, GEN) == 3      # drops: .1 .1 .5 -.3 .4
    assert L("gen_max_token_nll", 5, 3, PRM, GEN) == 2
    assert L("gen_max_token_entropy", 5, 3, PRM, GEN) == 4
    assert L("first_step", 5, 3, PRM, GEN) == 1 and L("last_step", 5, 3, PRM, GEN) == 5


def test_missing_or_mismatched_signal_names_no_step():
    assert s49.localise("prm_min", 5, 3, None, GEN) is None
    assert s49.localise("prm_min", 4, 3, PRM, GEN) is None
    assert s49.localise("prm_first_below_0.5", 2, 1, [0.9, 0.8], GEN) is None


def test_jeffreys_pass_bounds():
    assert s49.jeffreys_pass(8, 8, 0) == 0.0
    assert 0.0 < s49.jeffreys_pass(1, 8, 1) < s49.jeffreys_pass(1, 8, 4) < 1.0


def test_within_trace_regression_removes_trace_effects():
    import numpy as np
    import pandas as pd
    spec54 = importlib.util.spec_from_file_location("s54", Path(__file__).resolve().parents[1] / "scripts" / "54_chokebench_validity.py")
    s54 = importlib.util.module_from_spec(spec54)
    spec54.loader.exec_module(s54)
    rng = np.random.default_rng(0)
    rows = []
    for t in range(200):                                    # gain = trace effect + 0.4 * hit_d, hit_e irrelevant
        base = rng.normal()
        for _ in range(6):
            hd, he = float(rng.random() < 0.3), float(rng.random() < 0.3)
            rows.append({"trace_id": t, "w": 1.0, "hit_d": hd, "hit_e": he, "gain": base + 0.4 * hd + rng.normal(0, 0.01)})
    b = s54.within(pd.DataFrame(rows), ["hit_d", "hit_e"])
    assert abs(b[0] - 0.4) < 0.01 and abs(b[1]) < 0.01
