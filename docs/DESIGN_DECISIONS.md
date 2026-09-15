# Design Decisions

> Every non-trivial choice in this repository, with its justification and the
> alternatives rejected. If a specification in the agent brief is believed to be
> suboptimal, it is implemented as specified and the objection is recorded under
> "Open disagreements" — never silently substituted.

## Implementation plan (written before any code)

The working folder was read in full before coding:

- `MSE1_ChokePoints_Experiments.ipynb` — the Colab prototype (Parts 0–8). This is
  the behavioural specification: dataset inspection, P1–P6, Table 1, Figures 1–8,
  KS / Mann–Whitney / Kruskal–Wallis tests, surface + TF-IDF baselines,
  prefix-conditioned predictability curve, optional GPU hidden-state probe.
- `04_Paper_Overleaf_Package/paper/main.tex` — the IEEE paper skeleton. Its
  notation (`𝒯`, `ρ = e/n`, `f_θ`, `Δ = e − k*`, recovery/corruption) is the
  vocabulary every module uses. No variable is renamed without mapping to it.
- `04_Paper_Overleaf_Package/paper/figures/architecture.tex` — the system
  architecture diagram, copied verbatim into `docs/diagrams/`.
- `05_MASTER_GUIDE.pdf` — the execution guide and viva script. `LEARNING_NOTES.md`
  and the walkthrough notebook answer exactly the viva questions it lists.
- `03_Paper_Preview.pdf` — the compiled target output.

Build order follows §12 of the brief: tooling → data → preprocessing →
statistics/EDA → baselines → activations/probes → lead time/calibration/transfer →
intervention → report. Each stage prints a summary and exits non-zero on failure.

## D1 — Uniform trace schema

All loaders return `Trace` dataclasses (`src/data/loaders.py`). Downstream code
never touches dataset-native columns. This is what makes E1–E8 dataset-agnostic
and what makes the grouped-split assertion enforceable in one place.

Alternative rejected: per-dataset dataframes with adapter functions at each call
site. That duplicates the schema-mapping logic and is where column-name
assumptions hide.

## D2 — Preprocessing rules P1–P6 as named functions

Each rule is a pure function `apply_p1_...` … `apply_p6_...` in
`src/data/preprocessing.py`, applied in order by `apply_all()`. Pure functions
are unit-testable on synthetic fixtures with known answers, which is exactly
what `test_preprocessing.py` does.

- P4 (`relative_onset = e / n`) uses 1-indexed error positions, matching
  ProcessBench annotation convention. A trace with `e = -1` is error-free and
  yields `relative_onset = None`, never `0.0` (which would falsely place an error
  at the start of the chain).
- P6 is enforced, not just documented: `grouped_split()` asserts train/test
  question-ID disjointness and raises on violation.

## D3 — Grouped splitting everywhere

Row-wise splitting leaks because steps within a trajectory are correlated. Every
split in this repo (baselines, prefix curve, probes, transfer) goes through
`src/utils/seeds.py::grouped_split` or sklearn `GroupShuffleSplit` with
`groups=question_id`, followed by `assert_no_leakage()`. `test_no_leakage.py`
is the critical test and runs in CI.

## D4 — Metrics with bootstrap CIs, seeded

`src/analysis/statistics.py` implements AUROC with 1000 seeded bootstrap
resamples, ECE with 10 bins, and KS / Mann–Whitney / Kruskal–Wallis wrappers
that always return effect sizes beside p-values. Seeds come only from the config
file (`configs/*.yaml` → `src/config.py`), never from module-level constants.

## D5 — Representation ladder: surface → lexical → uncertainty → hidden-state

`E3` runs every representation through the same prefix-grid harness
(`src/analysis/predictability.py`) so the figure overlays are comparable.
If the hidden-state probe does not beat the lexical baseline, that null result
is reported, not hidden (ground rule §1.4).

## D6 — GPU-absent behaviour

`make activations` probes `torch.cuda.is_available()`. If absent, it prints a
clear skip message, writes a manifest row with `status: SKIPPED_GPU_ABSENT`,
and every downstream stage runs on non-activation representations. Skipped
results appear as `NOT_RUN` in `RESULTS.md`, never as zeros or guesses.

## D7 — Dataset identifiers are verified, not assumed

Each loader prints and logs the live schema before mapping columns, and raises
`DatasetSchemaError` (naming the dataset and observed columns) when a column is
missing. `docs/BENCHMARKS.md` records whether each identifier resolved at build
time. Unresolvable identifiers are recorded as unverified; no guess is
substituted (brief §13).

Verified at plan time from the prototype + paper text (re-verified at runtime by
`01_download_data.py`):

| Dataset | Identifier used | Status at plan time |
|---|---|---|
| ProcessBench | `Qwen/ProcessBench` (HF) | used in prototype; verify configs at runtime |
| GSM8K | `openai/gsm8k` + config `main` (HF) | used in prototype Part 7 |
| PRM800K | `openai/prm800k` (GitHub, not HF) | loader reads local checkout; network fetchopt-in only |
| MATH / MATH-500 | `hendrycks/competition_math` and `HuggingFaceH4/MATH-500` (HF) | verify at runtime |
| BIG-Bench Mistake | Google Research GitHub `bigbench` mistake-detection task | local-checkout loader; verify at runtime |
| StrategyQA | `wics/strategy-qa` (HF mirror of BigBio original) | verify at runtime |

## D8 — Smoke test uses labelled synthetic data

`configs/quick.yaml` sets `synthetic: true`. The pipeline then generates a tiny
seeded synthetic corpus (known AUROC ≈ 1.0 by construction on one feature, 0.5
on noise) instead of downloading datasets. Synthetic outputs are written under
`results/` with a `synthetic_` prefix and a manifest flag, so they can never be
mistaken for real results. This is what keeps `make smoke` under 5 minutes with
no network and no GPU.

Alternative rejected: subsampling the real datasets for smoke. That still needs
network + HF credentials and is not guaranteed fast.

## D9 — Figures: one function per figure, style in one place

`src/viz/style.py` owns the palette (colourblind-safe), grid, font, and
save-both-formats helper. `src/viz/figures.py` exposes `plot_f1 … plot_f14`,
each taking a dataframe and an output directory. Scripts never contain plotting
logic. TikZ diagrams live in `docs/diagrams/` as vector source.

## D10 — Intervention is offline simulation only

`src/intervention/simulate.py` replays logged trajectories against Algorithm 2
from the paper; it never calls a live model inside the evaluation loop. Recovery
and corruption are always reported together, plus net benefit at matched token
budget, against the comparators in E8 (no-op, always-intervene,
self-consistency N ∈ {2,4,8} approximated by resampling, oracle upper bound).

## Open disagreements

1. **Breadth vs depth of dataset coverage.** The brief asks for six loaders with
   equal weight, but the paper's claims rest on ProcessBench (+ GSM8K-generated
   trajectories for activations). Implemented as specified (all six loaders
   exist), but analysis defaults and the report prioritise ProcessBench. Running
   E7 transfer on all six without calibrated step labels risks comparing
   incommensurable targets.
2. **Probe capacity.** The brief asks for linear, 2-layer MLP, and GRU sequence
   probes; the paper deliberately argues for a linear probe first ("linearly
   decodable is a stronger claim"). Implemented all three as specified, but the
   headline curve uses the linear probe and MLP/GRU are reported as extensions.
3. **Fixed layer 18 vs layer sweep.** The paper text names layer 18 while the
   brief (E4) requires a multi-layer sweep. The sweep is implemented
   (`layers:` list in config); layer 18 remains the default single-layer value
   so the paper placeholder stays fillable.
4. **Prefix-grid endpoints.** The paper lists {10,25,40,50,75,100}% while the
   brief §4 lists {0.1,…,1.0} in tenths. Config `prefix_grid` defaults to the
   paper's six points (cheaper, matches placeholders); full tenths available by
   config change.
