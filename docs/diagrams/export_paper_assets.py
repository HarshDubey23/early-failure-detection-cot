"""
WHAT: Copies the paper's figures (vector PDFs) and the architecture TikZ into
      the Overleaf package, then compiles main.tex with Tectonic.

WHY:  The paper must use exactly the figures this pipeline produced - never a
      hand-edited or stale copy. One command (`make paper`) makes the copy and
      proves the paper still compiles.

HOW:  Plain file copies from results/figures/ and docs/diagrams/ into
      <paper-dir>/figures/, then `tectonic main.tex` inside <paper-dir>. The
      compiled log is scanned for undefined references and overfull boxes.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER_FIGURES = [  # results/figures -> paper/figures
    "fig4_relative_error_position_HEADLINE.pdf",
    "fig7_auroc_vs_prefix_KEY.pdf",
    "fig9_lead_time.pdf",
    "fig10_reliability.pdf",
    "fig15_online_auroc_by_step.pdf",
]
PAPER_DIAGRAMS = [  # docs/diagrams/rendered -> paper/figures (vector PDFs from TikZ)
    "pipeline.pdf",
    "worked_example_processbench.pdf",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export figures into the paper and compile it")
    parser.add_argument("--paper-dir", required=True)
    parser.add_argument("--tectonic", default="tectonic")
    args = parser.parse_args()
    paper_dir = Path(args.paper_dir).resolve()
    figures_dir = paper_dir / "figures"
    if not (paper_dir / "main.tex").exists():
        sys.exit(f"main.tex not found in {paper_dir}")
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name in PAPER_FIGURES:
        source = ROOT / "results" / "figures" / name
        if not source.exists():
            sys.exit(f"{source} missing - run `make all` first.")
        shutil.copy2(source, figures_dir / name)
        print(f"copied {source.relative_to(ROOT)} -> {figures_dir / name}")
    for name in PAPER_DIAGRAMS:
        source = ROOT / "docs" / "diagrams" / "rendered" / name
        if not source.exists():
            sys.exit(f"{source} missing - run `make diagrams` first.")
        shutil.copy2(source, figures_dir / name)
        print(f"copied {source.relative_to(ROOT)} -> {figures_dir / name}")
    shutil.copy2(ROOT / "docs" / "diagrams" / "architecture.tex", figures_dir / "architecture.tex")
    print(f"copied docs/diagrams/architecture.tex -> {figures_dir / 'architecture.tex'}")

    # Refuse to build a paper containing a number the pipeline did not produce.
    check = subprocess.run([sys.executable, str(Path(__file__).with_name("check_paper_values.py")),
                            str(paper_dir / "main.tex")], capture_output=True, text=True)
    print(check.stdout.strip())
    if check.returncode != 0:
        sys.exit("Paper values do not match results/paper_values.json - fix main.tex first.")

    result = subprocess.run([args.tectonic, "--keep-logs", "main.tex"], cwd=paper_dir,
                            capture_output=True, text=True)
    log_text = (paper_dir / "main.log").read_text(encoding="utf-8", errors="replace") \
        if (paper_dir / "main.log").exists() else result.stdout + result.stderr
    if result.returncode != 0:
        print(result.stdout[-3000:], result.stderr[-3000:])
        sys.exit("LaTeX compilation FAILED")
    problems = [line for line in log_text.splitlines()
                if "undefined" in line.lower() or "Overfull \\hbox" in line]
    print(f"compiled {paper_dir / 'main.pdf'}")
    print("log check: " + ("clean" if not problems else f"{len(problems)} warnings"))
    for line in problems[:20]:
        print("  " + line)


if __name__ == "__main__":
    main()
