# Learning Notes

A plain-language walkthrough of the whole repository, written for the person who will
be questioned on it. Read it with the code open. Every number quoted here is from
`docs/RESULTS.md` (build of 2026-09-15); if you re-run the pipeline and a number changes,
`RESULTS.md` is the authority, not this file.

---

## Part 1 - The project in five minutes

**The question.** When a language model reasons step by step and gets the answer wrong,
where does it go wrong, and can we tell *before the end* that it is going to fail?

**The data.** ProcessBench: 3,400 step-by-step maths solutions written by 12 open models.
Humans marked the first wrong step in each (or "no error"), and each solution records
whether its final answer is right.

**What we measured, and what we found.**

| Question | Experiment | Finding (honest version) |
|---|---|---|
| Where does the first error happen? | E1, E2 | Mean relative position 0.500. Errors sit **earlier** than if they were spread evenly over each solution's steps (expected 0.578); the clearest gap is the last step: 4.1% of first errors vs 15.5% expected. |
| Can we predict a wrong answer from part of the solution? | E3 | Weakly. Text detector (TF-IDF) AUROC 0.588 at 25% of the chain, 0.626 at 100%; significantly above length alone (0.539) but not above surface features (0.618). Mean 0.577 at 25% over 5 splits. Without knowing trace length: 0.566 after one step. |
| Do alarms come before the error? | E5 | Often, but the alarm fires on correct solutions almost as often, so the "warning" is mostly not specific. |
| Are the risk scores trustworthy as probabilities? | E6 | The logistic detectors are well calibrated (ECE 0.021-0.046); the GRU is not (0.164). |
| Does it work on other kinds of reasoning? | E7 | No. On BIG-Bench Mistake the text detector is at or below chance on every task, and inverted on Dyck languages (0.172). |
| Does the internal state of a model carry more signal? | E4 | **Not run** - needs a GPU. |
| Would intervening help? | E8 | Only simulated with assumed repair rates; not evidence. |

**The one-sentence summary for the viva:** "Errors enter earlier than chance placement
would put them, which supports a detection window, but a detector built from the text of
the partial solution is only weakly predictive, is not specific in its timing, and does
not transfer to other reasoning tasks; whether hidden states do better is the open next
step."

---

## Part 2 - The research ideas you must be able to explain

### Trajectory, onset and relative onset

A solution is a list of steps s_1 ... s_n. If step e is the first wrong one, the
**relative onset** is rho = e / n. Dividing by n makes "step 4 of 5" (0.8) and "step 4 of
40" (0.1) comparable.

*Trap you avoided:* ProcessBench stores the first wrong step **0-indexed** (0 means the
first step). Using that number directly as e would make every rho one step too early and
allow rho = 0. The loader adds 1 (`zero_indexed_to_first_error`), and rule P4 refuses any
e outside 1..n.

### Two different "wrong" labels

- `has_error`: a human found a wrong step somewhere (65.3% of traces).
- `answer_incorrect` (the paper's y): the final answer is wrong (50.0%).

They differ: 521 solutions contain a wrong step but still reach the right answer. The
paper promises to predict **wrong answers**, so that is the main target; the process-error
target is reported alongside (Fig. F7b), and it is easier to predict (lexical AUROC 0.735
at 100%) because errors leave textual traces even when the answer recovers.

### Why AUROC

AUROC = the probability that a randomly chosen failing solution gets a higher risk score
than a randomly chosen successful one. 0.5 is chance, 1.0 is perfect. It does not depend
on a threshold or on the class balance.

*Be precise in the viva:* on the paper's own target ProcessBench is **exactly balanced**
(50% wrong answers in every tier), so "classes are imbalanced" is not the reason here.
The reasons are: AUROC is threshold-free (the threshold is chosen later, for the
intervention policy), and the other labels and datasets are imbalanced (65.3% process
errors; 85% wrong answers in four BIG-Bench Mistake tasks), so one metric works everywhere.

### Leakage and grouped splitting (rule P6)

If the same problem appears in training and testing, the detector can memorise the
problem instead of learning about failure, and AUROC is inflated. ProcessBench has 3,400
rows but only 2,842 distinct problems (several models solved the same problem). So the
question id is a hash of the problem text, and every split puts all copies of a problem on
one side. `grouped_split` asserts that the train and test question sets do not overlap,
and `tests/test_no_leakage.py` checks every place a split is made.

### The KS test, and why a second test was needed

The Kolmogorov-Smirnov (KS) test compares the observed rho values with a Uniform(0,1)
distribution; D is the largest gap between the two cumulative curves. It gives D = 0.119,
p = 1.2e-27, so "not uniform".

*The subtlety an examiner may probe:* rho = e/n can only take values 1/n, 2/n, ..., 1. For
a 2-step solution rho is 0.5 or 1.0, never 0.2. So even if errors were placed completely
evenly over steps, rho would not look Uniform(0,1). A simulation shows that evenly placed
errors would give an even larger D (about 0.155) than we observed. So the KS rejection
alone does not prove that errors "concentrate".

The fair test (`ks_discrete_uniform_null`) simulates errors placed evenly over each
solution's *own* steps 1,000 times and compares. The observed distribution is far from
that (D to the null = 0.149 vs 0.026 at the null's 95th percentile, p <= 0.001), and errors
come **earlier** (mean rho 0.500 vs 0.578). Fig. F4 shows both reference curves.

### Kruskal-Wallis and effect size

Kruskal-Wallis asks whether rho differs between the four difficulty tiers: H = 14.50,
p = 0.002, so yes, statistically. But eta squared = 0.005 means tier explains about half a
percent of the variation. With thousands of traces, tiny differences become significant;
the effect size tells you it is small. Always quote both.

### The predictability curve

For each fraction f of the solution (10% ... 100%), keep the first k = ceil(f n) steps,
train a classifier on training problems, and measure AUROC on test problems. The curve
shows how much knowing more of the solution helps.

*The confound you must mention:* k = ceil(f n) means a 25% prefix of a 20-step solution
has 5 steps while a 25% prefix of a 4-step solution has 1 step, so the prefix length
reveals the final length n, which a live detector would not know. The **length-only**
baseline measures exactly how much of the curve that explains (0.515 at 25%, 0.539 at
100%). A **paired bootstrap** on the same test traces shows the lexical detector beats
length-only at every prefix from 25% onward (e.g. +0.073, p = 0.004 at 25%) but not at 10%
or 20%, and it never significantly beats the surface detector (smallest p = 0.068). Two
further checks: over 5 independent grouped splits, lexical AUROC at 25% averages 0.577
(SD 0.015; the main split's 0.588 is on the high side), and a **length-blind** step-wise
detector, which never uses n, still reaches AUROC 0.566 after one step and 0.613 after six.

*Why a paired test and not "the intervals overlap":* both detectors are scored on the same
traces, so their errors are correlated; overlapping intervals do not mean "no difference".
*Why question-level bootstrap:* traces of the same problem are correlated, so resampling
single traces makes intervals too narrow; the code resamples whole questions.

### Calibration and ECE

A detector is calibrated if, among solutions it scores at 0.7, about 70% really fail. ECE
(expected calibration error) bins scores into 10 bins and averages |observed failure rate
- mean score|, weighted by bin size. Accurate-but-uncalibrated detectors cannot be given a
meaningful threshold like "intervene at 80% risk".

### Lead time

For a threshold tau, the first alarm k* is the first step where risk >= tau. Lead time
Delta = e - k*: positive means the alarm came before the error. *Report it with the false
alarm rate:* at tau = 0.5 the alarm fires on 90.8% of erroneous solutions but also on
72.3% of error-free ones. Because most first errors come after step 1, an alarm that fires
early on almost everything will show Delta > 0 for free.

### Recovery, corruption and why the simulation is not a result

Recovery = P(fixed | would have failed); corruption = P(broken | would have succeeded).
A policy that fixes as many as it breaks is worthless, so both are always shown. E8 has no
model to actually re-run, so what an intervention does is *assumed* (for example
"regeneration fixes 60% of failing traces"). Only the alarms and costs are measured. That
is why no E8 number is in the paper.

---

## Part 3 - Module by module

For each: the problem it solves, the key idea, which research question it serves, and
what an examiner may ask.

### `src/config.py` - validated configuration
- **Problem:** hidden defaults and magic numbers make results irreproducible.
- **Idea:** every setting lives in `configs/default.yaml`; every key is required, so a typo
  fails loudly instead of silently using a default.
- **Serves:** reproducibility of everything.
- **Examiner:** "How would I reproduce your table?" -> "`make all` with that config; the
  manifest records the config hash and git commit for each stage."

### `src/data/loaders.py` - one schema for every dataset
- **Problem:** each dataset has different column names and conventions.
- **Idea:** every loader returns `Trace` objects (question id, steps, 1-indexed first error,
  final correctness). Loaders print the live schema and raise `DatasetSchemaError` listing
  the real columns if one is missing.
- **Serves:** all experiments.
- **Examiner:** "How do you know the columns are right?" -> "The loader checks the required
  columns on the live data, and `tests/test_loaders.py` pins the index conversion and the
  answer-vs-process label on fixtures." "Why ProcessBench?" -> "The question is *where*
  reasoning fails, so it needs step-position labels; question-answer datasets cannot
  support that."

### `src/data/verification.py` - does each dataset identifier exist?
- **Idea:** asks the HuggingFace and GitHub APIs about every dataset in the brief. A page
  can exist but be disabled; `hendrycks/competition_math` is, so it is recorded as unusable
  rather than replaced with a mirror.

### `src/data/preprocessing.py` - rules P1-P6
- P1 merge the tiers but keep the tier column (so difficulty can be tested).
- P2 count steps (the denominator of rho).
- P3 the sentinel -1 means error-free.
- P4 rho = e/n, refusing e outside 1..n.
- P5 drop solutions with fewer than 2 steps (none were dropped: the minimum is 2).
- P6 split by question, never by row (enforced in `src/utils/seeds.py`).
- `quality_report` produces the "no unexpected missing values" statement: missing
  first-error values occur exactly for error-free traces and nowhere else.

### `src/data/segmentation.py` - step boundaries
- **Idea:** a step is a non-empty line. Used only for generated text (GPU stage 06);
  ProcessBench already provides steps.
- **Examiner:** "Are lines real reasoning steps?" -> "Approximately; stated as a
  limitation."

### `src/data/synthetic.py` - smoke-test data
- **Idea:** a tiny fake corpus with known structure so `make smoke` runs everything in about
  1.5 minutes with no internet or GPU. Outputs go to `results/smoke/` so they can never be
  confused with real results. It also demonstrated the KS subtlety: errors placed evenly
  by construction still "failed" the plain KS test.

### `src/features/surface.py`, `lexical.py`, `uncertainty.py`, `activations.py`
- **Surface:** number of steps, mean and total characters. The cheapest floor.
- **Lexical:** TF-IDF - which words and word pairs appear, weighted by rarity. The
  vocabulary is learned from **training prefixes only** (learning it from test text would
  leak).
- **Uncertainty:** token entropy / logit margin. Implemented, **not run**: ProcessBench has
  no token probabilities.
- **Activations:** hidden state at the last token of each step, for several layers, from
  one forward pass. Implemented, **not run** (GPU).
- **Examiner:** "If hidden states do not beat TF-IDF, what does that mean?" -> "That the
  model's internals add nothing beyond what is visible in the text; that would be reported
  as a null result."

### `src/models/probe.py`, `mlp_probe.py`, `sequence_probe.py`, `baselines.py`
- **Linear probe:** logistic regression on standardised features (statistics from the
  training split only). A linear success shows the signal is *linearly decodable*.
  `layer_depth_auroc` runs it for every layer and depth bin with one grouped split (E4).
- **MLP (64-32):** tests for non-linear signal. On surface features it matched logistic
  regression (0.626 vs 0.618 at 100%), so non-linearity adds little there.
- **GRU:** reads the per-step features in order. It did worse (0.558) and was poorly
  calibrated (ECE 0.164): more capacity on three weak features mostly adds noise.
- **Baselines:** majority class (AUROC 0.5 by definition, computed not assumed) and
  length-only (rank by number of steps).
- **Examiner:** "Why only a linear probe as the main model?" -> "A linear success is a
  stronger, more interpretable claim; complex probes are extensions, and here they did not
  help."

### `src/analysis/onset.py` (E1)
- Per-tier and pooled table with bootstrap confidence intervals: resample the traces 1,000
  times with replacement and take the 2.5th and 97.5th percentiles of the mean.
- Adds two reference values: the mean rho and the share of final-step errors expected if
  errors were spread evenly over steps.

### `src/analysis/statistics.py` (E2 + metric primitives)
- KS, the discrete-null Monte Carlo test, Mann-Whitney U with rank-biserial r (r > 0:
  failing traces are longer; observed r = +0.175 for process errors), Kruskal-Wallis with
  eta squared, AUROC with bootstrap CI, ECE.
- **Examiner:** "What is a Monte Carlo p-value?" -> "The fraction of 1,000 simulated
  datasets under the null that look at least as extreme as the real one (plus one in
  numerator and denominator). With 1,000 simulations the smallest possible value is
  1/1001, which is why it is reported as <= 0.001."

### `src/analysis/predictability.py` (E3)
- One shared split, then a curve per representation. Returns test scores at 100% for the
  ROC figure (F6).

### `src/analysis/lead_time.py` (E5)
- **Idea:** to simulate a live monitor, fit one detector per step count k on training
  traces that have at least k steps; score step k of each test trace using only steps
  1..k. Detectors are fitted up to k = 19 (after that fewer than 50 training traces reach
  that step, so the last detector is reused, and this is recorded).
- Reports alarm rates on erroneous *and* error-free traces.

### `src/analysis/calibration.py` (E6)
- Reliability tables and a precision / recall / intervention-rate table per threshold.

### `src/analysis/transfer.py` (E7)
- Train on ProcessBench, test on each BIG-Bench Mistake task. Also leave-one-tier-out
  within ProcessBench (the lexical detector falls to chance on GSM8K, 0.487, when trained
  only on the harder tiers).
- **Examiner:** "Surface features transfer to word sorting at 0.832 - better than
  in-domain. Is that transfer?" -> "Unlikely; within that task wrong answers differ in
  length, so length-like features separate them for a task-specific reason. The lexical
  detector, which reads content, is below chance there."

### `src/intervention/policies.py` and `simulate.py` (E8)
- `decide` is Algorithm 2: below tau do nothing; above tau verify, or regenerate if above
  tau_hi, backtracking to the step with the highest risk so far.
- The simulator separates measured quantities from assumed ones (Part 2). The revert
  safeguard is modelled as catching a harmful revision with an assumed probability.
- Self-consistency uses an independent-samples formula; with per-sample accuracy about
  0.51 it predicts almost no gain, and real samples are correlated, so that row is a model,
  not a measurement.
- Two things the table does show honestly: no threshold at or above 0.8 ever alarms (the
  detector's scores never get that high), and tau = 0.3 alarms on every trace, so it equals
  "always intervene".

### `src/viz/style.py`, `src/viz/figures.py`
- One palette (colour-blind safe), figures drawn at their printed size (3.5 in or 7.16 in)
  with 8 pt text, saved as PNG and vector PDF. Missing inputs give a clearly labelled
  NOT RUN panel (F8, F14).

### `src/reporting/report.py` and `scripts/10_build_report.py`
- Builds `docs/RESULTS.md` and `results/paper_values.json` from the saved tables. Every
  paper number has a row saying where it came from. Interpretive sentences come from
  written rules (for example "p < 0.05").

### `scripts/01`-`10`, `Makefile`, `tests/`, manifest
- Each script: read config -> run one stage -> write tables and figures -> add a manifest
  row -> print a summary. `make all` runs them in order and stops at the first failure.
- Tests: leakage (critical), loaders, preprocessing, metrics against hand-computed
  answers, reproducibility (same seed, identical tables).

### `docs/diagrams/`
- TikZ sources rendered to PDF/PNG by `render.py`. `worked_example_processbench.tex` is a
  real ProcessBench trace chosen by a fixed rule, with the first error highlighted;
  `worked_example.tex` is the paper's illustrative train example and is labelled as such.

---

## Part 4 - Questions you are likely to be asked, with answers the data support

- **"Do choke points exist?"** -> "The data show first errors sit earlier than even
  placement over steps would predict, which is consistent with a detection window. They do
  not show that those states *cause* failure; that needs counterfactual resampling."
- **"How early can you predict failure?"** -> "From 25% of the chain, AUROC 0.588 (95% CI
  0.552-0.626) with a text detector - above chance but weak. It is significantly better than
  trace length (+0.073, p = 0.004) but not than simple surface statistics, and it averages
  0.577 over five resampled splits."
- **"Isn't this just hallucination detection?"** -> "Hallucination detection judges a
  finished answer; this asks whether a partial trajectory predicts the outcome. Same target,
  different input and timing."
- **"Why haven't you done mitigation?"** -> "Unconditional self-correction can reduce
  accuracy (Huang et al.). With a detector this weak and this unspecific, intervening would
  touch most correct solutions; detection has to improve first. The simulation exists only
  to show how that trade-off would be measured."
- **"What is the most important limitation?"** -> "Hidden-state probing, the actual
  hypothesis, did not run for lack of a GPU; everything reported uses text features."
- **"Why does the discrete-null test matter?"** -> "Because the obvious KS test would reject
  uniformity even if errors were spread evenly, purely because short solutions only allow a
  few rho values. The claim survives only when tested against the right null."
- **"Your transfer AUROC on Dyck languages is 0.17. Is that a bug?"** -> "No; below 0.5
  means the ranking is inverted - features that signal failure in maths signal success
  there. It is evidence that the detector learned maths-specific surface cues."

---

## Glossary

- **AUROC** - probability a random failing trace scores above a random succeeding one; 0.5
  chance, 1.0 perfect; threshold- and base-rate-independent.
- **Bootstrap CI** - resample the test set with replacement many times; the middle 95% of
  the recomputed statistic is the interval.
- **Calibration / ECE** - whether predicted risks match observed failure rates; ECE is the
  weighted average gap over 10 score bins.
- **Corruption rate** - P(an intervention breaks a solution that would have been right).
- **Discrete null** - errors placed uniformly over each trace's own steps 1..n.
- **Effect size** - how big a difference is (D, r, eta squared), as opposed to how sure we
  are that it exists (p).
- **Grouped split** - train/test split where all rows of one question stay together.
- **Lead time (Delta)** - first erroneous step minus first alarm step; positive = warned
  early.
- **Leakage** - information from the test set reaching training, inflating scores.
- **Length confound** - a detector succeeding because failing solutions are longer, not
  because it understands failure.
- **Monte Carlo test** - p-value from simulations of the null hypothesis.
- **Probe** - a small classifier trained on frozen features to test what information they
  contain; a linear probe tests linear decodability.
- **Process error vs answer error** - a wrong step somewhere vs a wrong final answer.
- **Recovery rate** - P(an intervention fixes a solution that would have been wrong).
- **Relative onset (rho)** - first erroneous step divided by number of steps.
- **TF-IDF** - word (and word-pair) counts weighted down when a word is common everywhere.
