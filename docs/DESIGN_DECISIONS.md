# Design Decisions

> Every non-trivial choice in this repository, with its justification and the
> alternatives rejected. When a specification in the agent brief is believed to be
> suboptimal, it is implemented as specified and the objection is recorded under
> "Open disagreements" - never silently substituted.

## Implementation plan (written before any code)

The working folder was read in full before coding:

- `MSE1_ChokePoints_Experiments.ipynb` - the Colab prototype (Parts 0-8): dataset
  inspection, P1-P6, Table 1, Figures 1-8, KS / Mann-Whitney / Kruskal-Wallis,
  surface + TF-IDF baselines, prefix-conditioned predictability curve, optional GPU
  hidden-state probe. Copied to `notebooks/01_original_prototype.ipynb`.
- `04_Paper_Overleaf_Package/paper/main.tex` - the IEEE paper skeleton. Its notation
  (T, rho = e/n, f_theta, Delta = e - k*, recovery/corruption) is the vocabulary every
  module uses.
- `04_Paper_Overleaf_Package/paper/figures/architecture.tex` - copied verbatim into
  `docs/diagrams/`.
- `05_MASTER_GUIDE.pdf` - execution guide and viva script; `LEARNING_NOTES.md` answers
  the questions it lists.
- `03_Paper_Preview.pdf` - the compiled target layout.

Build order follows brief §12: tooling -> data -> preprocessing -> EDA/statistics ->
baselines -> activations/probes -> lead time/calibration/transfer -> intervention ->
report. Each stage prints a summary and exits non-zero on failure.

## Revision log: bugs found when the pipeline first met the live data (2026-09-15)

The first build was verified only on synthetic smoke data. Inspecting the live datasets
before `make all` exposed defects that would have put wrong numbers into the paper. All
are fixed and each has a regression test (`tests/test_loaders.py`, `tests/test_metrics.py`).

| # | Defect | Effect had it shipped | Fix |
|---|---|---|---|
| R1 | ProcessBench is one config (`default`) with four *splits* (gsm8k, math, olympiadbench, omnimath); the loader read only the first split | 400 of 3,400 traces analysed, a single tier named `default`, Kruskal-Wallis not computable | `load_processbench` iterates every split; the split name is the tier |
| R2 | ProcessBench `label` and BIG-Bench Mistake `mistake_index` are **0-indexed**; the code used them as the paper's 1-indexed `e` | every rho shifted early by 1/n, rho = 0 possible (272 ProcessBench traces have label 0) | `zero_indexed_to_first_error` adds 1; P4 now raises if e is outside 1..n |
| R3 | `question_id` came from the row id, but ProcessBench has 3,400 ids and only 2,842 distinct problems (OlympiadBench: 561 of 1,000) | the same problem in train and test = leakage through the "grouped" split | `question_id` = hash of whitespace-normalised problem text |
| R4 | detector target was `has_error` (a process error exists), but the paper defines y as final-answer incorrectness; ProcessBench provides `final_answer_correct` and 521 traces have a process error but a correct answer | the headline AUROC would answer a different question than the paper states | primary target `answer_incorrect`, `has_error` reported as a secondary target (D11) |
| R5 | rank-biserial effect size had the wrong sign (`1 - 2U/(n1 n2)`) | "erroneous traces are longer" reported as negative | `2U1/(n1 n2) - 1`, sign tested |
| R6 | intervention simulator counted a trace as *corrupted* when the revert safeguard fired, which is the opposite of what reverting does; the "oracle" intervened on every trace; self-consistency rows mixed units (trajectories labelled as tokens) and reported corruption 0 | E8 table internally inconsistent | rewritten (D10) |
| R7 | lead-time and intervention risk series were scored on training traces too | optimistic timing numbers | series produced for held-out test questions only |
| R8 | smoke and real runs wrote figures to the same folder with identical names | a smoke run overwrote real figures | smoke outputs live under `results/smoke/` |
| R9 | the GPU path never graded generated answers, so stage 07 could never train a probe even on a GPU | E4 silently empty forever | stage 06 grades each trajectory against the GSM8K gold answer; stage 07 logic moved to a tested function |
| R10 | CI pinned Python 3.10 while the tested stack (pandas 3, numpy 2.4) needs >= 3.11 | CI red on first push | CI and environment use Python 3.13, the tested version |

## D1 - Uniform trace schema

All loaders return `Trace` dataclasses (`src/data/loaders.py`). Downstream code never
touches dataset-native columns, which makes E1-E8 dataset-agnostic and the grouped-split
assertion enforceable in one place. `first_error_idx` is always the paper's 1-indexed e.
Rejected: per-dataset dataframes with adapters at every call site (where column-name
assumptions hide).

## D2 - Preprocessing rules P1-P6 as named functions

Each rule is a pure function in `src/data/preprocessing.py`, tested on fixtures with
known answers. P3 (sentinel -1 -> error-free) is applied at load time by
`zero_indexed_to_first_error` and re-applied by `apply_all` as a guard. P4 raises on an
out-of-range e. P6 is enforced, not documented: every split goes through
`grouped_split`, which asserts disjointness. On the live data P5 dropped 0 traces
(ProcessBench minimum n is 2); that is reported, not hidden.

## D3 - Grouped splitting everywhere, verified from outside

`make_split` computes ONE grouped split for E3 and it is reused by E5, E6, E7 and E8, so
every experiment scores the same held-out questions. `tests/test_no_leakage.py` checks
each split-consuming function from the outside and scans the source so that no module
can call a row-wise splitter or build its own `GroupShuffleSplit`.

## D4 - Metrics with bootstrap CIs, seeded

AUROC with 1,000 seeded bootstrap resamples, ECE with 10 equal-width bins, and every
hypothesis test returns an effect size beside its p-value (KS: D; Mann-Whitney:
rank-biserial r; Kruskal-Wallis: eta squared). Seeds come only from the config file.

## D5 - Representation ladder, with the confound floor on the same axes

E3 runs every CPU-available representation through the same split: surface (logistic
regression on [n_steps, mean_step_chars, total_chars]), **length-only** (rank by prefix
step count), lexical (TF-IDF fitted on train prefixes only), plus surface-MLP and
surface-GRU as nonlinear / sequence extensions. Uncertainty features and hidden states
are `NOT_RUN` with reasons (`table2_representation_status.csv`): ProcessBench stores no
token log-probabilities, and activations need a GPU.

## D6 - GPU-absent behaviour

`make activations` checks `torch.cuda.is_available()`. Without CUDA it logs
`SKIPPED_GPU_ABSENT`, stage 07 writes empty tables and explicit "NOT RUN" panels for F8
and F14, and every other stage runs. The report reads `NOT_RUN`, never a number.

## D7 - Dataset identifiers are verified, not assumed

`src/data/verification.py` queries the HuggingFace Hub / GitHub API for every identifier
in the brief and records RESOLVES / DISABLED_ON_HUB / NOT_FOUND in
`results/tables/dataset_identifier_verification.json`. A repo that exists but is
*disabled* (taken down) counts as unverified. No substitute dataset is ever used.

## D8 - Smoke test uses labelled synthetic data, in its own folder

`configs/quick.yaml` generates a seeded synthetic corpus (`src/data/synthetic.py`) with a
known construction and writes everything under `results/smoke/`. `make smoke` needs no
network and no GPU and ran in about 1.5 minutes on the build machine.

## D9 - Figures: one function per figure, sized for the paper

`src/viz/style.py` fixes the Okabe-Ito palette and draws figures at their printed size
(3.5 in single column, 7.16 in double) with 8 pt text and TrueType fonts, so LaTeX places
them without scaling. `src/viz/figures.py` has one function per figure; missing inputs
produce a labelled NOT RUN panel.

## D10 - Intervention is an offline simulation with declared assumptions

No live model runs in the evaluation loop, so the *effect* of an intervention is not
observable here. `src/intervention/simulate.py` separates what is measured (which test
traces alarm and when, which would fail, how many characters a regeneration discards)
from what is assumed (`intervention.assumed` in the config: repair / corruption
probabilities, the probability the revert safeguard catches a corruption, verification
cost, characters per token). Comparators: no intervention, always intervene (tau = 0),
an oracle that regenerates exactly the failing traces from the true e, and
self-consistency N in {2, 4, 8} under an independent-samples model (optimistic, stated).
Every E8 number is labelled SIMULATED in `RESULTS.md`, and no E8 number enters the paper.

## D11 - Detector target: answer incorrectness first, process error second

The paper (Problem Formulation) defines y = 1 when the final answer disagrees with gold,
and the abstract promises prediction of "eventual answer incorrectness". ProcessBench
supplies this directly (`final_answer_correct`). The brief's E3 wording ("eventual
incorrectness") agrees. The prototype used "trace contains an error" instead; that target
is kept as the secondary curve (F7b) because it is what onset analysis (E1, E5) is about.

## D12 - Robustness check for the KS onset test

rho = e/n is discrete and bounded below by 1/n, so errors spread *evenly over each trace's
steps* do not give a continuous Uniform(0,1) rho. `ks_discrete_uniform_null` therefore
tests rho against the correct discrete null (e ~ Uniform{1..n}, keeping each trace's n)
by Monte Carlo, and reports how large the KS distance to Uniform(0,1) is expected to be
under that null. The brief's KS test is still reported exactly as specified; this is an
addition so the two can be read together.

## D13 - Stepwise risk series for lead time and intervention

For each absolute step count k, one detector (config: lexical) is fitted on train traces
with at least k steps and scores the first k steps of test traces; r_k therefore uses
s_1..s_k only. Detectors are fitted while at least 50 train traces reach step k and
both classes occur; beyond that the last detector is reused (the cut-off k is recorded).
Rejected: evaluating one prefix-fraction detector at k/n, which requires knowing n in
advance - impossible online.

## D14 - Processed data stored as JSON lines

Processed traces are written as JSONL so step lists and booleans survive a round trip.
The earlier CSV + `ast.literal_eval` approach re-parsed Python list reprs of free text.

## D15 - Paper figures and diagrams are produced by script

`make diagrams` renders TikZ to PDF/PNG with Tectonic; `make paper` copies the vector
figure PDFs and `architecture.tex` into the Overleaf package and compiles it. The real
worked example is selected from ProcessBench by a fixed rule written in
`docs/diagrams/render.py`, so it is data rather than an invented illustration.

## Open disagreements

1. **KS against Uniform(0,1) is the wrong null for "errors concentrate".** Implemented as
   specified (brief E2, paper Sec. VI-A), but a rejection can be produced entirely by the
   lattice of rho = e/n on short traces. The discrete-null test (D12) is reported beside
   it, and the paper text states both. On the live data this matters: the observed KS
   distance to Uniform(0,1) is *smaller* than the distance expected if errors were spread
   evenly over steps, so the KS rejection by itself is not evidence of concentration. The
   paper's claim rests on the discrete-null test, which does reject and shows errors
   occurring earlier than even placement (see `RESULTS.md` E2).
2. **Prefix features leak total length.** Algorithm 1 sets k = ceil(f n), so the step
   count of a prefix at fraction f reveals n. Every "25 %" prediction therefore knows the
   final length, which an online detector cannot. Implemented as specified; the
   length-only baseline is plotted on the same axes so the size of this confound is
   visible, and the stepwise detector used for lead time (D13) does not have it.
3. **Breadth vs depth of dataset coverage.** Six loaders exist as specified, but only
   ProcessBench and BIG-Bench Mistake carry what the experiments need (step text +
   first-error labels + graded answers without a GPU). GSM8K is used only for GPU
   generation; MATH-500, StrategyQA and PRM800K loaders are implemented but feed no
   experiment in this build, and StrategyQA transfer is `NOT_RUN` (it has decompositions,
   not model traces with graded answers).
4. **Probe capacity.** The brief asks for linear, MLP and GRU probes; the paper argues for
   a linear probe first. All three exist; the headline uses logistic regression and
   MLP/GRU are extensions (run on surface features, since activations are unavailable).
5. **Fixed layer 18 vs a layer sweep.** The paper names layer 18; the brief requires a
   sweep. The sweep (`probes.layer_sweep`) is implemented; 18 remains the default layer
   for F14 so the paper's setup sentence stays true when E4 is run.
6. **Prefix grid.** The paper lists {10, 25, 40, 50, 75, 100} %, the brief tenths. Resolved
   by using the union (12 points) so neither is overridden.
7. **Intervention parameters are not evidence.** The brief asks for recovery / corruption
   rates in E8; without a live model those rates are driven by assumed parameters. They
   are produced, labelled SIMULATED, and kept out of the paper.
