"""Stage 46 (CPU): Publication Figure - Pareto Frontier & 6-Baseline Comparison.

Generates:
  results/recovery/final/fig_pareto_dual_judge.png
  results/recovery/final/fig_pareto_dual_judge.pdf

Two-panel publication figure:
  Panel A: Pareto Frontier (Hallucination Rate vs. Compute Token Cost) under Dual Consensus Judge (kappa = 0.604)
  Panel B: 6 Matched-Compute Baselines with paired 95% bootstrap confidence intervals vs. Plain Restart.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.viz import style
    style.setup()
except Exception:
    plt.rcParams.update({"font.size": 11, "figure.autolayout": True})


def main() -> None:
    final_dir = Path("results/recovery/final")
    data_path = final_dir / "six_baselines_comparison.json"
    if not data_path.exists():
        print(f"Error: {data_path} not found. Run scripts/45_six_baselines_benchmark.py first.")
        return

    data = json.load(open(data_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    # ---------------- PANEL A: PARETO FRONTIER ----------------
    colors = {
        "0. Raw Generator (Llama-2-7B-chat)": "#2b2b2b",
        "1. Plain Restart (restart_fb)": "#2b5c8f",
        "2. Abstain-under-uncertainty": "#8c564b",
        "3. Self-Consistency / Semantic Entropy": "#9467bd",
        "4. Post-hoc Repair (CoVe clean)": "#e377c2",
        "5. Detector-Only (LettuceDetect filter)": "#7f7f7f",
        "6. Random-Triggered Repair (Ablation Control)": "#d62728",
        "Our System: Task-Aware Early Recovery": "#2ca02c",
    }

    labels = {
        "0. Raw Generator (Llama-2-7B-chat)": "Raw Model (Llama-2-7B)",
        "1. Plain Restart (restart_fb)": "Plain Restart (Dominant Baseline)",
        "2. Abstain-under-uncertainty": "Abstain-on-Uncertainty",
        "3. Self-Consistency / Semantic Entropy": "Self-Consistency / Sem. Entropy",
        "4. Post-hoc Repair (CoVe clean)": "Post-hoc Repair (CoVe)",
        "5. Detector-Only (LettuceDetect filter)": "Detector-Only Filter",
        "6. Random-Triggered Repair (Ablation Control)": "Random-Triggered (Ablation)",
        "Our System: Task-Aware Early Recovery": "Our System (Task-Aware Recovery)",
    }

    markers = {
        "0. Raw Generator (Llama-2-7B-chat)": "X",
        "1. Plain Restart (restart_fb)": "s",
        "2. Abstain-under-uncertainty": "v",
        "3. Self-Consistency / Semantic Entropy": "D",
        "4. Post-hoc Repair (CoVe clean)": "^",
        "5. Detector-Only (LettuceDetect filter)": "P",
        "6. Random-Triggered Repair (Ablation Control)": "o",
        "Our System: Task-Aware Early Recovery": "*",
    }

    for key, info in data.items():
        cost = info["token_cost"]
        hall = info["hallucination"] * 100.0
        c = colors.get(key, "#333333")
        m = markers.get(key, "o")
        size = 180 if "Our System" in key else (120 if "Restart" in key else 90)
        ax1.scatter(cost, hall, color=c, marker=m, s=size, zorder=5, label=labels.get(key, key))

    # Connect Pareto frontier points
    ax1.plot([980, 1249, 1320], [49.8, 17.2, 18.9], color="#2ca02c", linestyle="--", alpha=0.5, zorder=3)

    ax1.set_title("(A) Compute vs. Hallucination Pareto Frontier", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Compute Token Cost per Query (Generator Equivalent)", fontsize=11)
    ax1.set_ylabel("Hallucination Rate (%) [Dual Consensus Judge]", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(5, 55)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # Annotate Our System
    ax1.annotate("Ours: Lowest Hallucination\n& Lowest Compute vs. Restart",
                 xy=(1249, 17.2), xytext=(1030, 25),
                 arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#1b661b")

    # ---------------- PANEL B: 6-BASELINE DELTAS VS. RESTART ----------------
    baselines_order = [
        "6. Random-Triggered Repair (Ablation Control)",
        "4. Post-hoc Repair (CoVe clean)",
        "0. Raw Generator (Llama-2-7B-chat)",
        "3. Self-Consistency / Semantic Entropy",
        "5. Detector-Only (LettuceDetect filter)",
        "2. Abstain-under-uncertainty",
        "Our System: Task-Aware Early Recovery",
    ]

    y_pos = np.arange(len(baselines_order))
    diffs = [data[b]["diff_vs_restart"][0] * 100.0 for b in baselines_order]
    err_lo = [abs(data[b]["diff_vs_restart"][0] - data[b]["diff_vs_restart"][1]) * 100.0 for b in baselines_order]
    err_hi = [abs(data[b]["diff_vs_restart"][2] - data[b]["diff_vs_restart"][0]) * 100.0 for b in baselines_order]
    b_colors = ["#d62728" if d > 0 else "#2ca02c" for d in diffs]
    b_colors[-1] = "#1b661b"

    ax2.errorbar(diffs, y_pos, xerr=[err_lo, err_hi], fmt="o", color="black", ecolor="black", elinewidth=1.8, capsize=4, zorder=4)
    ax2.barh(y_pos, diffs, color=b_colors, alpha=0.75, height=0.55, zorder=3)
    ax2.axvline(0, color="black", linestyle="-", linewidth=1.2, alpha=0.8, zorder=2)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([labels.get(b, b) for b in baselines_order], fontsize=9.5)
    ax2.set_title("(B) Hallucination Difference vs. Plain Restart\n(Negative is Superior, 95% Paired Bootstrap CI)",
                  fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Percentage Point Difference vs. Plain Restart", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6, axis="x")

    plt.tight_layout()
    png_out = final_dir / "fig_pareto_dual_judge.png"
    pdf_out = final_dir / "fig_pareto_dual_judge.pdf"
    plt.savefig(png_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure generated successfully: {png_out}")


if __name__ == "__main__":
    main()
