# Errors are not choke points: decomposition and difficulty controls (stage 11)

Run: 2026-10-24, Linux CPU, ProcessBench snapshot `3bdcd53` (3,400 traces, 2,842 problems),
`make decomposition`. Every number below is read from `results/tables/decomp_*` and was
produced by `scripts/11_choke_point_decomposition.py`. Detectors are cross-fitted with
5-fold grouped (by question) cross-validation, repeated over 5 seeds (42, 1, 2, 3, 4);
intervals are question-level cluster-bootstrap 95% intervals (1,000 resamples) on seed 42;
"seeds" gives mean ± SD of the AUROC over the 5 seeds.

Outcome groups: **clean** (no annotated error) 1,179 · **recovered** (annotated first error,
correct final answer) 521 · **fatal** (first error, wrong final answer) 1,700. No trace is
error-free with a wrong answer.

## A. One first error in four is not fatal, and fatality does not depend on where it occurs

| Stratum | n errors | P(answer wrong \| first error) | 95% CI |
|---|---:|---:|---|
| All first errors | 2,221 | **0.765** | 0.748–0.782 |
| GSM8K | 207 | 0.966 | 0.942–0.990 |
| MATH | 594 | 0.842 | 0.814–0.869 |
| OlympiadBench | 661 | 0.756 | 0.725–0.788 |
| Omni-MATH | 759 | **0.659** | 0.624–0.692 |
| Onset ρ in (0, .25] | 445 | 0.760 | 0.722–0.798 |
| ρ in (.25, .5] | 857 | 0.760 | 0.732–0.790 |
| ρ in (.5, .75] | 561 | 0.766 | 0.735–0.800 |
| ρ in (.75, 1) | 267 | 0.749 | 0.699–0.800 |
| Error on the final step | 91 | 0.890 | 0.824–0.945 |

- Fatality falls with difficulty: on Omni-MATH a third of first errors still end in a
  correct answer (F16).
- Fatality is flat in the position of the error (0.749–0.766 over the four onset bins);
  only an error in the final step is more often fatal. Fatal and recovered traces have the
  same onset (Mann–Whitney p = 0.21, rank-biserial r = 0.04). In a logistic regression with
  tier and generator controls, neither onset ρ (OR 1.70, coefficient CI −0.41 to 1.36) nor
  steps remaining (OR 1.25 per log-step, CI −0.10 to 0.57) is significant.
- Caveat: ProcessBench balances correct and incorrect answers within each tier, so tier-level
  fatality reflects the benchmark's construction as well as the models.

**What the 521 "recovered" traces are.** Human coding of a seeded, tier-stratified sample of 80
recovered traces (two independent coders, 40 items overlapping) is in progress; the protocol and
blinded sheets are in [`CODING_PROTOCOL.md`](CODING_PROTOCOL.md) and `results/coding/`. No
coding result is reported until both sheets are complete. A coder-independent measurement (the
forced-answer probes of Run C) addresses the same question.

Explicit re-checking language after the first error is only slightly more common in recovered
than in fatal traces (18.6% vs 15.5%; Fisher OR 1.25, p = 0.09), so explicit self-correction does
not explain most recoveries. Zhang et al. (2025) report flawed-reasoning-with-correct-answer
rates for the same four sources (5.1% GSM8K to 43.4% Omni-MATH).

## B. Early failure prediction is a difficulty signal

AUROC for "final answer incorrect", out-of-fold over all 3,400 traces (F17):

| Detector | AUROC (seed 42) | 95% CI | 5 seeds |
|---|---:|---|---|
| Question text only (TF-IDF) | 0.561 | 0.543–0.578 | 0.562 ± 0.005 |
| Generator + tier only | 0.539 | 0.521–0.557 | 0.536 ± 0.002 |
| **Question + generator** | **0.588** | 0.570–0.607 | 0.587 ± 0.003 |
| Prefix 25% only | 0.564 | 0.545–0.582 | 0.568 ± 0.006 |
| Prefix 100% only | 0.600 | 0.582–0.619 | 0.601 ± 0.004 |
| Question + generator + prefix 25% | 0.593 | 0.575–0.611 | 0.595 ± 0.004 |
| Question + generator + prefix 50% | 0.610 | 0.592–0.628 | 0.609 ± 0.004 |
| Question + generator + prefix 100% | 0.619 | 0.601–0.637 | 0.620 ± 0.003 |
| Trace length (100%) | 0.542 | 0.524–0.561 | 0.542 ± 0.002 |

Paired increments (same traces, cluster bootstrap):

| Comparison | Δ AUROC | 95% CI | p | seeds with Δ > 0 |
|---|---:|---|---:|---:|
| Q+G+prefix 25% − Q+G | +0.005 | −0.004 to +0.014 | 0.30 | 5/5 |
| Q+G+prefix 50% − Q+G | +0.022 | +0.011 to +0.031 | 0.001 | 5/5 |
| Q+G+prefix 100% − Q+G | +0.031 | +0.020 to +0.041 | 0.001 | 5/5 |
| prefix 25% − Q+G | −0.025 | −0.046 to −0.004 | 0.012 | 0/5 |

**Within-problem test.** 326 problems have both a failing and a succeeding trace (466 pairs,
always from different generators). Concordance (share of pairs where the failing trace scores
higher): generator + tier 0.548 [0.504, 0.591]; prefix 25% 0.485 [0.440, 0.536]; prefix 100%
0.539 [0.487, 0.589]; Q+G+prefix 100% 0.586 [0.535, 0.634]. No text detector beats generator
identity within a problem (largest gain +0.038, CI −0.017 to +0.090).

Reading: the 25%-prefix result in the paper (0.588) is fully explained by what the question
and the generator already reveal. Reasoning text adds a real but small +0.02–0.03 AUROC only
from half-way through the chain. This replicates, for text detectors on ProcessBench, the
difficulty confound Bulut (2026) reports for activation probes.

## C. Failure detectors detect errors and difficulty, not commitment

AUROC between outcome groups, using out-of-fold scores:

| Detector | recovered vs clean | fatal vs recovered | fatal vs clean |
|---|---:|---:|---:|
| Prefix 100%, trained on answer incorrect | 0.703 | **0.456** | 0.664 |
| Prefix 25%, trained on answer incorrect | 0.667 | 0.443 | 0.617 |
| Q+G+prefix 100%, trained on answer incorrect | 0.679 | 0.501 | 0.672 |
| Prefix 100%, trained on "has an error" | 0.819 | **0.376** | 0.709 |

A detector trained to predict a wrong answer ranks recovered traces *above* fatal ones (0.456,
CI 0.428–0.484). Once question and generator are included the ranking is at chance (0.501).
The text separates "has an error / is a hard problem" from "clean", but carries no information
about whether an error will be fatal.

## D. At the moment of the first error, fatality is not predictable from the reasoning

Among the 2,221 erroneous traces, predicting fatal vs recovered:

| Features | AUROC | 95% CI |
|---|---:|---|
| Error position only (online) | 0.484 | 0.455–0.512 |
| Question + generator | 0.671 | 0.646–0.695 |
| + error position | 0.670 | 0.645–0.695 |
| + text of the error step | 0.670 | 0.644–0.694 |
| + prefix up to the error | 0.668 | 0.643–0.693 |
| + one step after the error | 0.672 | 0.646–0.696 |
| + offline length and all post-error text | 0.675 | 0.648–0.701 |

Every reasoning-text increment over question + generator + position is within ±0.003 AUROC
(all p ≥ 0.08). Whether an error is fatal is predicted by the problem and the model, not by
the error.

## Bugs found and fixed while reproducing

1. `length_only_scores` broke ties in step count with an argsort-of-argsort, giving tied
   traces arbitrary distinct ranks. The baseline was therefore platform-dependent (0.539 on
   Windows, 0.535 on Linux at 100% for the same split). Fixed with average ranks. Corrected
   values on the paper's split (`table2d_length_only_tie_fix.csv`): 0.521 / 0.535 / 0.541 /
   0.534 / 0.533 at 10 / 25 / 50 / 75 / 100%. The paper's "lexical − length-only = +0.073,
   p = 0.004 at 25%" becomes **+0.052 [0.005, 0.100], p = 0.032**.
2. Stage 05 crashed when torch is missing instead of marking the GRU row NOT_RUN. Fixed.

Reproduced exactly from the cached data: every onset statistic (KS D = 0.1185, discrete-null
p = 0.001, mean ρ 0.500 vs 0.578, Kruskal–Wallis H = 14.50) and the lexical (0.626) and surface
(0.618) AUROCs at 100%.
