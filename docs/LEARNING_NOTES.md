# Learning Notes

Plain-language walkthrough for the researcher. For each module: the problem it
solves, the key idea, which research question it serves, and what an examiner
might ask. Read this with the code open.

## Map: research question -> code

- RQ1 "Where do failures originate?" -> `src/data/*`, `src/analysis/onset.py`,
  `src/analysis/statistics.py`, scripts 01–04, figures F1–F5.
- RQ2 "How early is failure predictable?" -> `src/features/*`,
  `src/models/*`, `src/analysis/predictability.py`, script 05, figures F6–F8.
- RQ3 "How much warning does detection give?" -> `src/analysis/lead_time.py`,
  `src/analysis/calibration.py`, script 08, figures F9–F10.
- RQ4 "Does it transfer? Does mitigation pay?" ->
  `src/intervention/*`, script 09, figures F11–F13.

## Module-by-module

### src/config.py — typed config loader

Problem: magic numbers scattered through scripts make runs irreproducible.
Idea: every numeric choice lives in `configs/*.yaml`; the loader validates it.
Serves: ground rule 5 (reproduce any run from the config alone).
Examiner: "How do I reproduce your numbers?" → "Fix the config hash in
manifest.json; every seed and hyperparameter is in that file."

### src/data/loaders.py — six loaders, one schema

Problem: six benchmarks, six different column names.
Idea: map everything to the `Trace` dataclass once; downstream code never sees
native columns. Each loader prints the live schema first and raises
`DatasetSchemaError` naming observed columns when mapping fails.
Serves: RQ1–RQ4 (everything downstream).
Examiner: "Why this dataset?" → "The question is about *where* reasoning
fails, so I need step-position labels. QA corpora cannot support that.
ProcessBench annotates the first erroneous step."

### src/data/preprocessing.py — rules P1–P6

Problem: raw traces are not comparable (tiers mixed, lengths differ).
Idea: six small pure functions. P4 (`ρ = e/n`) is the load-bearing one: step 4
of 5 and step 4 of 40 are different events. P6 (grouped splitting) is enforced
in `seeds.py`, not just documented.
Examiner: "Why split by question?" → "Steps within a trace are correlated;
row-wise splitting leaks across the boundary and inflates AUROC."

### src/data/segmentation.py — step boundaries

Problem: activations must be read at the moment each step completes.
Idea: line-break heuristic, centralised so the approximation is stated once
(paper Limitations admits it is approximate, not semantic).
Examiner: "Are steps semantic units?" → "No — heuristic line breaks, stated
as a limitation. A better segmenter is future work."

### src/features/* — the representation ladder

Surface (counts) → lexical (TF-IDF) → uncertainty (entropy/margin) →
hidden-state (activations). Ordered by cost. The honest comparison: if the
hidden-state probe does not beat TF-IDF, internal states add nothing visible —
report it, don't hide it.

### src/models/* — probes and baselines

Linear probe first: success means the signal is *linearly decodable*, a
stronger claim than a black-box win. MLP/GRU are extensions. Baselines
(majority, length-only, entropy-only) are reported even when they win.

### src/analysis/* — statistics, curves, timing, calibration

`statistics.py` always pairs p-values with effect sizes. `predictability.py`
runs every representation through one harness (same splits, same seeds).
`lead_time.py` turns risks into Δ distributions. `calibration.py` answers
whether τ can be trusted.

### src/intervention/* — Algorithm 2 + offline replay

Default action is *do nothing* (Huang et al.: unconditional revision degrades
accuracy). The simulator replays logged trajectories with stated repair
assumptions — a sensitivity analysis, not a claim about a live model.

### src/viz/* — style once, figures F1–F14

Palette and sizes in `style.py`; one function per figure; PNG+PDF always.

### scripts/01–10, tests/, manifest

Scripts are thin CLIs: read config → run → write tables/figures/manifest row →
print summary. Tests guard leakage (critical), P1–P6, metrics, seeds. The
manifest (`results/manifest.json`) records what ran with which config hash.

## Glossary

- **AUROC**: P(a random failing trace scores above a random succeeding one).
  0.5 random, 0.7 useful, 0.8 strong. Used because classes are imbalanced.
- **Calibration / ECE**: whether "80% risk" fails 80% of the time. Accurate but
  uncalibrated detectors cannot drive thresholds.
- **Leakage**: train/test contamination via correlated rows. Prevented by
  grouped (by-question) splits with disjointness assertions.
- **Probe**: small classifier trained on frozen representations to test what
  information they contain. Linear probes test linear decodability.
- **Relative onset (ρ)**: first-error index ÷ trace length. Makes positions
  comparable across traces.
- **Lead time (Δ)**: `e − k*`: error index minus first-alarm index. Positive =
  warning before the error.
- **Recovery / corruption**: P(fixed | would fail) and P(broken | would
  succeed). Always reported together; net benefit = recovery − corruption.
