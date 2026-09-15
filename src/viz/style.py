"""
WHAT: Single source of truth for figure styling (palette, sizes, saving).

WHY:  Brief §6: consistent colourblind-safe palette, labelled axes, no default
      matplotlib titles left in, readable at single-column width, PNG (300dpi)
      AND vector PDF for every figure. Defining it once prevents fourteen
      slightly different styles.

HOW:  `setup()` applies rcParams; `save(fig, name, outdir)` writes both
      formats; `PALETTE` fixes the semantic colours used across all figures.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

# Colourblind-safe palette (Okabe-Ito inspired): teal = correct/success,
# vermillion = error/failure, blue = accent/model curve.
PALETTE = {
    "correct": "#009E73",
    "error": "#D55E00",
    "accent": "#0072B2",
    "neutral": "#999999",
    "purple": "#CC79A7",
}

COLUMN_WIDTH_IN = 3.5  # IEEE single column; double-width figures use 2x.


def setup() -> None:
    """Apply publication-quality matplotlib defaults."""
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.3,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
        }
    )


def save(fig, name: str, outdir: str | Path) -> list:
    """Save `fig` as `<name>.png` (300dpi) and `<name>.pdf` (vector).

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
