"""Stage 46 (CPU): Publication Figure - Real Policy Trade-offs on RAGTruth TEST.

Generates:
  results/recovery/final/fig_pareto_dual_judge.png
  results/recovery/final/fig_pareto_dual_judge.pdf

Two-panel publication figure:
  Panel A: Compute Cost vs. Hallucination Rate
  Panel B: Answer Coverage vs. Faithful Answers Delivered (Safety-Utility Trade-off)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

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
        print(f"Error: {data_path} not found.")
        return

    data = json.load(open(data_path))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)

    colors = {
        "Raw Generator (Llama-2-7B)": "#2b2b2b",
        "Plain Restart (b=2)": "#1f77b4",
        "Plain Restart (b=4)": "#aec7e8",
        "CARE Controller (b=2)": "#ff7f0e",
        "Abstain on Flag (Pure Filter)": "#8c564b",
        "Task-Aware (Abstain QA / Restart Other)": "#2ca02c",
        "Chain-of-Verification (Clean)": "#9467bd",
    }

    markers = {
        "Raw Generator (Llama-2-7B)": "X",
        "Plain Restart (b=2)": "s",
        "Plain Restart (b=4)": "D",
        "CARE Controller (b=2)": "o",
        "Abstain on Flag (Pure Filter)": "v",
        "Task-Aware (Abstain QA / Restart Other)": "*",
        "Chain-of-Verification (Clean)": "^",
    }

    # PANEL A: Cost vs Hallucination
    for pol, info in data.items():
        if info["token_cost"] is None:
            continue
        cost = info["token_cost"]
        hall = info["hallucination"] * 100.0
        c = colors.get(pol, "#333333")
        m = markers.get(pol, "o")
        size = 180 if "Task-Aware" in pol or "Restart (b=2)" in pol else 110
        ax1.scatter(cost, hall, color=c, marker=m, s=size, zorder=5, label=pol)

    ax1.set_title("(A) Compute Cost vs. Hallucination Rate", fontsize=12, fontweight="bold", pad=10)
    ax1.set_xlabel("Compute Token Cost per Query", fontsize=11)
    ax1.set_ylabel("Hallucination Rate (%)", fontsize=11)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.set_ylim(10, 55)
    ax1.legend(loc="upper right", fontsize=8.5, framealpha=0.9)

    # PANEL B: Coverage vs Faithful Answers Delivered
    for pol, info in data.items():
        cov = info["coverage"] * 100.0
        faith = info["faithful_delivered"] * 100.0
        c = colors.get(pol, "#333333")
        m = markers.get(pol, "o")
        size = 180 if "Task-Aware" in pol or "Restart (b=2)" in pol else 110
        ax2.scatter(cov, faith, color=c, marker=m, s=size, zorder=5, label=pol)

    ax2.set_title("(B) The Safety-Utility Frontier (Coverage vs. Faithful Answers)", fontsize=12, fontweight="bold", pad=10)
    ax2.set_xlabel("Answer Coverage (%)", fontsize=11)
    ax2.set_ylabel("Faithful Answers Delivered (%)", fontsize=11)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.set_xlim(80, 103)
    ax2.set_ylim(45, 85)

    ax2.annotate("Plain Restart: Highest volume\nof faithful answers (80.8%)",
                 xy=(99.6, 80.8), xytext=(85, 78),
                 arrowprops=dict(facecolor="#1f77b4", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#1f77b4")

    ax2.annotate("Task-Aware: Lower hallucination\nbut lower coverage (71.2%)",
                 xy=(88.4, 71.2), xytext=(82, 60),
                 arrowprops=dict(facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=6),
                 fontsize=9, fontweight="bold", color="#2ca02c")

    plt.tight_layout()
    png_out = final_dir / "fig_pareto_dual_judge.png"
    pdf_out = final_dir / "fig_pareto_dual_judge.pdf"
    plt.savefig(png_out, bbox_inches="tight")
    plt.savefig(pdf_out, bbox_inches="tight")
    plt.close()

    print(f"[OK] Figure generated successfully: {png_out}")


if __name__ == "__main__":
    main()
