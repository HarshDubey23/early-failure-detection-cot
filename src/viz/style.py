"""
WHAT: Single source of truth for figure styling (palette, sizes, saving).

WHY:  Brief §6: consistent colourblind-safe palette, labelled axes, no default
      matplotlib titles, readable at single-column width, PNG (300 dpi) AND
      vector PDF for every figure. Defining it once prevents fourteen slightly
      different styles.

HOW:  Figures are drawn at the physical size they occupy in the IEEE paper
      (3.5 in for one column, 7.16 in for both columns) with 8 pt text, so
      when LaTeX places them at \\columnwidth or \\textwidth no scaling happens
      and the text matches the paper's footnote size. `save` writes both
      formats; `PALETTE` fixes the semantic colours used across all figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # file output only; no display needed (CI, servers)
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito colourblind-safe palette. Semantic roles are fixed repo-wide:
# green = correct/success, vermillion = error/failure, blue = main detector.
PALETTE = {
    "correct": "#009E73",
    "error": "#D55E00",
    "accent": "#0072B2",
    "neutral": "#7F7F7F",
    "purple": "#CC79A7",
    "orange": "#E69F00",
    "sky": "#56B4E9",
    "black": "#000000",
}

SINGLE_COLUMN_IN = 3.5   # IEEE \columnwidth
DOUBLE_COLUMN_IN = 7.16  # IEEE \textwidth
SINGLE = (SINGLE_COLUMN_IN, 2.5)
SINGLE_TALL = (SINGLE_COLUMN_IN, 3.2)
DOUBLE = (DOUBLE_COLUMN_IN, 2.5)


def setup() -> None:
    """Apply publication-quality matplotlib defaults (8 pt, embedded TrueType fonts)."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.linewidth": 0.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "pdf.fonttype": 42,  # TrueType in PDF: IEEE PDF eXpress rejects Type 3 fonts
            "ps.fonttype": 42,
        }
    )


def save(fig, name: str, outdir: str | Path) -> list:
    """Save `fig` as `<name>.png` (300 dpi) and `<name>.pdf` (vector).

    Returns:
        List of written file paths.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    png = outdir / f"{name}.png"
    pdf = outdir / f"{name}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return [str(png), str(pdf)]
