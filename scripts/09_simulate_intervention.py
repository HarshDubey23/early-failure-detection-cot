"""Stage 09: offline intervention simulation (E8) -> F11, F12."""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features.surface import batch_surface
from src.intervention.simulate import compare_policies
from src.utils.io import Manifest, atomic_write_dataframe
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import grouped_split, seed_everything
from src.viz import figures


def build_risks(df: pd.DataFrame, seed: int) -> dict:
    """Per-trace risk series from per-length surface probes.

    Mirrors stage 08's construction exactly (same features, same seeds), so
    the intervention replay consumes the same detector the timing analysis
    evaluated. Duplicated here deliberately: scripts stay decoupled and each
    remains runnable on its own.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    steps_list = list(df["steps"])
    max_n = max((len(s) for s in steps_list), default=0)
    y = df["has_error"].astype(int).to_numpy()
    groups = df["question_id"].to_numpy()
    tr, _ = grouped_split(groups, 0.25, seed, context="intervention_risks")
    probes = {}
    for k in range(1, max_n + 1):
        X = batch_surface([s[:k] for s in steps_list])
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(scaler.transform(X[tr]), y[tr])
        probes[k] = (scaler, clf)
    risks: dict = {}
    for i in df.index:
        series = []
        for k in range(1, len(df.loc[i, "steps"]) + 1):
            scaler, clf = probes[k]
            x = batch_surface([df.loc[i, "steps"][:k]])
            series.append(float(clf.predict_proba(scaler.transform(x))[0, 1]))
        risks[df.loc[i, "trace_id"]] = series
    return risks


def main() -> None:
    parser = argparse.ArgumentParser(description="09: simulate intervention E8")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("09_simulate_intervention", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)

    prefix = "synthetic_" if cfg.data.synthetic else ""
    df = pd.read_csv(Path(cfg.data.processed_dir) / f"{prefix}processed_traces.csv")
    df["steps"] = df["steps"].apply(
        lambda s: ast.literal_eval(s) if isinstance(s, str) and s.startswith("[") else []
    )
    risks = build_risks(df, cfg.seed)
    policy_df = compare_policies(
        df, risks, list(cfg.intervention.thresholds),
        escalation_threshold=cfg.intervention.escalation_threshold,
        token_budget=cfg.intervention.token_budget, seed=cfg.seed,
    )
    # Self-consistency comparators at matched budgets (resampling model):
    # majority vote over N resampled outcomes; cost = N full trajectories.

    base_acc = float((~df["has_error"]).mean())
    sc_rows = []
    for n in cfg.intervention.self_consistency_N:
        # P(majority correct) under independent resamples with base accuracy.
        from math import comb

        p_maj = sum(comb(n, k) * base_acc**k * (1 - base_acc) ** (n - k)
                    for k in range(n // 2 + 1, n + 1))
        sc_rows.append({"threshold": f"self-consistency-N{n}",
                        "recovery_rate": float(p_maj - base_acc),
                        "corruption_rate": 0.0,
                        "net_benefit": float(p_maj - base_acc),
                        "intervention_rate": 1.0,
                        "extra_tokens_per_question": float(n - 1)})
    policy_df = pd.concat([policy_df, pd.DataFrame(sc_rows)], ignore_index=True)
    atomic_write_dataframe(policy_df,
                           Path(cfg.outputs.tables_dir) / f"{prefix}table_intervention.csv")
    (Path(cfg.outputs.tables_dir) / f"{prefix}table_intervention.tex").write_text(
        policy_df.to_latex(index=False, float_format="%.4f"))
    figures.plot_f11(policy_df, cfg.outputs.figures_dir)
    figures.plot_f12(policy_df, cfg.outputs.figures_dir)

    best = policy_df.loc[policy_df["net_benefit"].idxmax()]
    summary = [
        f"Policies compared : {len(policy_df)} (thresholds + no-op + oracle + SC)",
        f"Best net benefit  : {best['net_benefit']:+.3f} ({best['threshold']})",
        (
            f"  recovery={best['recovery_rate']:.3f} "
            f"corruption={best['corruption_rate']:.3f} "
            f"extra_tokens={best['extra_tokens_per_question']:.1f}/q"
        ),
        "Assumption (stated, not measured): repair/corruption probs in simulate.py.",
        "Figures: F11 (recovery vs corruption), F12 (cost frontier)",
    ]
    log.info("intervention simulation complete")
    manifest.log("09_simulate_intervention", args.config, "OK",
                 {"policies": len(policy_df), "best_net": float(best["net_benefit"])})
    print_summary_block("09_simulate_intervention", summary)


if __name__ == "__main__":
    main()
