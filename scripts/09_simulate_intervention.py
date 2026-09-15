"""Stage 09: offline intervention simulation (E8) -> F11, F12. Thin CLI wrapper."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.intervention.simulate import InterventionAssumptions, compare_policies
from src.utils.io import Manifest, atomic_write_dataframe, atomic_write_text, read_traces_frame
from src.utils.logging import get_logger, print_summary_block
from src.utils.seeds import seed_everything
from src.viz import figures


def main() -> None:
    parser = argparse.ArgumentParser(description="09: simulate intervention E8")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    seed_everything(cfg.seed)
    log = get_logger("09_simulate_intervention", cfg.outputs.logs_dir)
    manifest = Manifest(cfg.outputs.manifest)
    tables = Path(cfg.outputs.tables_dir)

    df = read_traces_frame(Path(cfg.data.processed_dir) / "processed_traces.jsonl")
    series_path = tables / "stepwise_risk_series.json"
    if not series_path.exists():
        manifest.log("09_simulate_intervention", args.config, "FAILED", {"missing": str(series_path)})
        print_summary_block("09_simulate_intervention", [f"FAILED: {series_path} missing. Run make analysis."])
        sys.exit(1)
    # Same detector, same held-out traces as the lead-time analysis (stage 08).
    risks = json.loads(series_path.read_text())["risks"]
    assumptions = InterventionAssumptions(**asdict(cfg.intervention.assumed))
    policy_df = compare_policies(df, risks, list(cfg.intervention.thresholds),
                                 cfg.intervention.escalation_threshold, cfg.intervention.token_budget,
                                 assumptions, list(cfg.intervention.self_consistency_N),
                                 cfg.detector.target, cfg.seed)
    atomic_write_dataframe(policy_df, tables / "table_intervention.csv")
    atomic_write_text(tables / "table_intervention.tex", policy_df.to_latex(index=False, float_format="%.3f"))
    atomic_write_text(tables / "table_intervention_assumptions.json", json.dumps(asdict(assumptions), indent=2))
    figures.plot_f11(policy_df, cfg.outputs.figures_dir)
    figures.plot_f12(policy_df, cfg.outputs.figures_dir)

    summary = ["SIMULATED under ASSUMED repair/corruption parameters (not measured model behaviour):",
               "  " + json.dumps(asdict(assumptions))]
    for _, row in policy_df.iterrows():
        summary.append(f"{row['policy']:<34} recovery {row['recovery_rate']:.3f}  corruption "
                       f"{row['corruption_rate']:.3f}  d-acc {100 * row['net_accuracy_change']:+.1f}pp  "
                       f"interv {row['intervention_rate']:.2f}  +tok/q {row['extra_tokens_per_question']:.0f}")
    for line in summary:
        log.info(line)
    manifest.log("09_simulate_intervention", args.config, "OK", {"policies": int(len(policy_df))})
    print_summary_block("09_simulate_intervention", summary)


if __name__ == "__main__":
    main()
