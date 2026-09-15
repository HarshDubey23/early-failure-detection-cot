# Detecting and Mitigating Choke Points in Large Language Model Reasoning

![Python 3.10](https://img.shields.io/badge/python-3.10-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

When an LLM answers wrongly, the error usually enters earlier in the
chain-of-thought and propagates: every later step inherits it. This repository
detects that during generation rather than after — a prefix-conditioned
detector predicts eventual failure from a partially generated trajectory, and a
cost-aware policy intervenes only when the risk exceeds a calibrated threshold.

Headline result: first errors concentrate rather than distributing uniformly
(KS test in `docs/RESULTS.md`), with failure predictable from a fraction of the
chain — exact numbers in `docs/RESULTS.md` (every value computed at runtime;
unrun stages read `NOT_RUN`, never a guess).

System architecture (vector TikZ, `docs/diagrams/architecture.tex`): offline, a
detector is fitted on annotated traces split by question and calibrated against
a corruption budget; online, a linear probe scores each step boundary and the
default action is to continue — intervention triggers only on alarm, escalating
from cheap verification to costly regeneration with a revert safeguard.

## Installation

```bash
pip install -r requirements.txt        # or: conda env create -f environment.yml
```

Requires Python 3.10+. GPU (CUDA) only needed for stage 06 (activation
extraction); everything else runs on CPU and the pipeline skips E4 gracefully
without one.

## Quickstart — 5-minute smoke test

```bash
make smoke        # Windows: use git-bash/WSL make, or run the scripts in order
make test         # pytest incl. the leakage test
```

`make smoke` uses `configs/quick.yaml` (labelled SYNTHETIC data, no network, no
GPU) and exercises every stage end to end, writing `docs/RESULTS_SMOKE.md`.

## Full reproduction

```bash
make data          # download + cache datasets
make preprocess    # P1–P6 -> data/processed/processed_traces.csv
make eda           # E1, E2 -> tables + F1–F5
make baselines     # E3 -> F6, F7
make activations   # E4 extraction (GPU; skips gracefully if absent)
make probes        # E4 -> F8, F14
make analysis      # E5, E6, E7 -> F9, F10, F13
make intervention  # E8 -> F11, F12
make report        # docs/RESULTS.md + LaTeX fragments
make all           # everything in order
```

## Repository map

| Path | Contents |
|---|---|
| `configs/` | `default.yaml` (full runs), `quick.yaml` (smoke) — all hyperparameters |
| `src/config.py`, `src/utils/` | validated config, seeds + grouped splits, atomic I/O + manifest, logging |
| `src/data/` | loaders (6 benchmarks, uniform `Trace` schema), P1–P6, segmentation |
| `src/features/` | surface, lexical (TF-IDF), uncertainty, activations |
| `src/models/` | linear / MLP / GRU probes + majority, length-only, entropy-only baselines |
| `src/analysis/` | onset, statistics, predictability, lead time, calibration |
| `src/intervention/` | Algorithm 2 policies + offline simulator |
| `src/viz/` | one style source + one function per figure F1–F14 |
| `scripts/01–10` | thin CLI wrappers (no logic); one stage each |
| `tests/` | leakage (critical), preprocessing, metrics, reproducibility + CI workflow |
| `results/` | tables (CSV + `.tex`), figures (PNG 300dpi + PDF vector), manifest |
| `docs/` | METHODOLOGY, DESIGN_DECISIONS, RESULTS, LEARNING_NOTES, BENCHMARKS, diagrams |
| `notebooks/` | `00_walkthrough.ipynb` (narrated tour), `01_original_prototype.ipynb` |

## Results summary

See `docs/RESULTS.md` (auto-generated): headline onset numbers, KS /
Mann–Whitney / Kruskal–Wallis with effect sizes, prefix AUROC curve, lead-time
and ECE tables, intervention recovery-vs-corruption, and the paper-placeholder
mapping (`main.tex` `\RES{…}` → source table). Paper figures: `fig4_*HEADLINE*`
(Fig. 2) and `fig7_*KEY*` (Fig. 3) in `results/figures/`.

## Datasets and licences

ProcessBench (`Qwen/ProcessBench`, cite Zheng et al. 2024) — onset analysis.
PRM800K (GitHub `openai/prm800k`, MIT, Lightman et al. 2023) — step labels.
GSM8K (`openai/gsm8k` MIT, Cobbe et al. 2021) and MATH/MATH-500 (Hendrycks et
al. 2021) — trajectory generation. BIG-Bench Mistake (Apache-2.0, Tyen et al.
2024) and StrategyQA (Geva et al. 2021) — cross-domain transfer. Details and
identifier verification in `docs/BENCHMARKS.md`. Raw datasets are never
committed (`data/raw/` is git-ignored); activation tensors are never committed
(extraction script + manifest checksums instead).

## Citation

```bibtex
@misc{chokepoints2026,
  title  = {Detecting and Mitigating Choke Points in Large Language Model Reasoning},
  author = {Your Name},
  year   = {2026},
  note   = {Undergraduate research project; code and results in this repository}
}
```

Prior-work citations live in the paper (`main.tex` bibliography): Lightman et
al. 2023, Zheng et al. 2024, Tyen et al. 2024, Huang et al. 2024, Orgad et al.
2024, Wang et al. 2023/2024, Farquhar et al. 2024, Kossen et al. 2024, Wei et
al. 2022.

## Acknowledgements

Supervisor guidance and feedback (see paper Acknowledgment). Built from the
MSE-1 Colab prototype (`notebooks/01_original_prototype.ipynb`); methodology
follows the IEEE paper skeleton (`04_Paper_Overleaf_Package/` in the working
folder, not committed here).
