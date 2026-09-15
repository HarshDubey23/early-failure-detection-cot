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


def check(paper_tex: Path, values_path: Path, config_path: Path) -> list:
    """Return the \\RES pieces that are not backed by a pipeline value."""
    allowed = allowed_values(values_path, config_path)
    problems = []
    for argument in res_arguments(paper_tex.read_text(encoding="utf-8")):
        text = argument.replace(r"\_", "_").strip()
        pieces = [text] if text in allowed else [p.strip() for p in text.split(",")]
        for piece in pieces:
            if piece not in allowed:
                problems.append(argument)
    return problems


def main() -> None:
    paper = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "04_Paper_Overleaf_Package/paper/main.tex"
    arguments = res_arguments(paper.read_text(encoding="utf-8"))
    problems = check(paper, ROOT / "results/paper_values.json", ROOT / "configs/default.yaml")
    print(f"{len(arguments)} \\RES values checked; {len(problems)} not found in results/paper_values.json")
    for problem in problems:
        print(f"  UNBACKED: {problem}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
