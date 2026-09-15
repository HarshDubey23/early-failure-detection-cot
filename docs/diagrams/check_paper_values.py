"""
WHAT: Verifies that every \\RES{...} value in the paper exists in the pipeline's
      results/paper_values.json (or is a configuration setting).

WHY:  "A number in the paper that does not match the notebook" is the error an
      examiner checks first. Filling the paper is manual; this check makes a
      typo or a stale number fail loudly before the PDF is built.

HOW:  Extract the argument of every \\RES{...}; split comma lists; compare each
      piece with the set of values in paper_values.json plus the config values
      the paper quotes (model name, temperature, samples, layer, split share,
      bootstrap resamples, prefix fractions). Author names are not wrapped in
      \\RES and are not checked.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def res_arguments(tex: str) -> list:
    """Arguments of every \\RES{...}, allowing one level of nested braces."""
    # Drop LaTeX comments first: the file header mentions "\RES{...}" as prose.
    without_comments = "\n".join(re.sub(r"(?<!\\)%.*", "", line) for line in tex.splitlines())
    return re.findall(r"\\RES\{((?:[^{}]|\{[^{}]*\})*)\}", without_comments)


def allowed_values(values_path: Path, config_path: Path) -> set:
    from src.config import load_config

    placeholders = json.loads(values_path.read_text(encoding="utf-8"))["placeholders"]
    allowed = set()
    for entry in placeholders.values():
        value = str(entry["value"])
        allowed.add(value)
        allowed.update(piece.strip() for piece in re.split(r"[,&]", value))  # lists and table rows
        allowed.update(piece.strip("[] ") for piece in value.split(","))
    cfg = load_config(config_path)
    allowed.update({
        cfg.probes.model.split("/")[-1], f"{cfg.probes.generation_temperature:g}",
        f"{cfg.probes.samples_per_question}", f"{cfg.probes.default_layer}",
        f"{100 * (1 - cfg.split.test_size):g}", f"{cfg.metrics.bootstrap_resamples}",
        *(f"{100 * f:g}" for f in cfg.prefix_grid),
    })
    if placeholders.get("KS_rejects_uniform", {}).get("value") == "yes":
        allowed.add("not uniformly distributed")
    return allowed


def table_values(tables_dir: Path) -> set:
    """Every number in results/tables (CSV + JSON), formatted as it may be quoted.

    Used for the study guide, which cites more tables than the paper. Each
    value is added at 0-3 decimals, as a signed difference, and x100 (percent)
    at 0-1 decimals.
    """
    import pandas as pd

    numbers = []
    for csv_path in tables_dir.glob("*.csv"):
        frame = pd.read_csv(csv_path)
        for column in frame.select_dtypes("number").columns:
            numbers.extend(frame[column].dropna().tolist())

    def walk(node):
        if isinstance(node, dict):
            for item in node.values():
                walk(item)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, (int, float)) and not isinstance(node, bool):
            numbers.append(float(node))

    for json_path in tables_dir.glob("*.json"):
        walk(json.loads(json_path.read_text(encoding="utf-8")))
    allowed = set()
    for number in numbers:
        for decimals in (0, 1, 2, 3):
            allowed.add(f"{number:.{decimals}f}")
            allowed.add(f"{number:+.{decimals}f}")
        for decimals in (0, 1):
            allowed.add(f"{100 * number:.{decimals}f}")
        if float(number).is_integer():
            allowed.add(f"{int(number):,}")
    return allowed


def check(paper_tex: Path, values_path: Path, config_path: Path, tables_dir: Path | None = None) -> list:
    """Return the \\RES pieces that are not backed by a pipeline value."""
    allowed = allowed_values(values_path, config_path)
    if tables_dir is not None:
        allowed |= table_values(tables_dir)
    problems = []
    for argument in res_arguments(paper_tex.read_text(encoding="utf-8")):
        text = argument.replace(r"\_", "_").strip()
        pieces = [text] if text in allowed else [p.strip() for p in text.split(",")]
        for piece in pieces:
            if piece not in allowed:
                problems.append(argument)
    return problems


def main() -> None:
    args = [a for a in sys.argv[1:] if a != "--tables"]
    paper = Path(args[0]) if args else ROOT.parent / "04_Paper_Overleaf_Package/paper/main.tex"
    tables_dir = ROOT / "results/tables" if "--tables" in sys.argv else None
    arguments = res_arguments(paper.read_text(encoding="utf-8"))
    problems = check(paper, ROOT / "results/paper_values.json", ROOT / "configs/default.yaml", tables_dir)
    print(f"{len(arguments)} \\RES values checked; {len(problems)} not found in results/paper_values.json")
    for problem in problems:
        print(f"  UNBACKED: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
