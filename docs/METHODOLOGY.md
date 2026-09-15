# Methodology

Notation matches `04_Paper_Overleaf_Package/paper/main.tex`. Every symbol maps to one
name in code; the right-hand column says where.

| Paper | Meaning | Code |
|---|---|---|
| q | question | `Trace.question`, grouped by `question_id` |
| T = (s_1, ..., s_n, a) | reasoning trajectory | `Trace.steps`, `n_steps` |
| y in {0,1} | 1 when the final answer disagrees with gold | `answer_incorrect` (from ProcessBench `final_answer_correct`) |
| e in {1..n} or empty | first erroneous step, 1-indexed | `first_error_idx` (loader adds 1 to the 0-indexed source label) |
| rho = e / n | relative onset, in (0, 1] | `relative_onset` (P4) |
| n - e | steps remaining after the first error | `steps_remaining` |
| T_{1:k} | prefix of the first k steps | `prefix_steps(steps, f)` with k = max(1, ceil(f n)) |
| f_theta : T_{1:k} -> r_k | detector, r_k ~ P(y = 1 given T_{1:k}) | `src/analysis/predictability.py`, `lead_time.stepwise_risk_series` |
| k* = min{k : r_k >= tau} | first alarm | `lead_time.first_alarm` |
| Delta = e - k* | lead time (> 0: warned before the error) | `lead_time.lead_times` |
| Recovery = P(corrected given y = 1) | | `simulate.simulate_policy` (`recovery_rate`) |
| Corruption = P(broken given y = 0) | | `simulate.simulate_policy` (`corruption_rate`) |

A second outcome, `has_error` (a process error exists anywhere, i.e. e is defined), is
reported alongside y. The two differ: in ProcessBench 521 traces contain a process error
yet reach the correct final answer.

## Data

- **ProcessBench** (primary): 3,400 traces from 12 generator models across four tiers
  (gsm8k 400, math 1,000, olympiadbench 1,000, omnimath 1,000); human-annotated earliest
  erroneous step; `final_answer_correct`.
- **BIG-Bench Mistake** (transfer): 2,186 PaLM 2-L traces on five symbolic tasks with a
  0-indexed first-mistake index; correctness = exact match of the extracted answer and
  target after whitespace normalisation.

## Preprocessing (P1-P6)

P1 merge tiers, keep tier as `subset`; P2 count steps; P3 sentinel -1 -> error-free;
P4 rho = e/n with e checked to lie in 1..n; P5 drop n < 2; P6 split by question id,
never by row. Question ids are hashes of whitespace-normalised question text, because
the same problem appears in several rows (2,842 distinct problems in 3,400 rows).

## E1 - Onset characterisation

Per tier and pooled: counts, % process error, % answer incorrect, median/max n, mean and
median e, mean rho with bootstrap 95% CI (1,000 resamples), median rho, % of errors with
rho <= 0.5, mean n - e with CI. Reference value: the mean rho expected if e were uniform
over each trace's own steps, mean((n + 1) / (2n)).

## E2 - Statistical tests

- KS of rho against Uniform(0,1) (as specified by the brief and paper); effect size D.
- Discrete-null robustness check: Monte Carlo (1,000 simulations) with e ~ Uniform{1..n}
  per trace; goodness of fit measured as the sup-distance between the observed ECDF of rho
  and the null CDF; two-sided test on mean rho; and the KS-to-Uniform(0,1) distance
  expected under this null.
- Mann-Whitney U on n, erroneous vs error-free (and incorrect vs correct answer);
  rank-biserial r = 2U1 / (n1 n2) - 1, positive when the first group is longer.
- Kruskal-Wallis on rho across tiers; eta^2 = (H - k + 1) / (N - k).

## E3 - Predictability curve

One grouped split (25% of questions held out, seed 42) shared by every representation.
For each f in {0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 1.00}:
build T_{1:k} with k = max(1, ceil(f n)); fit on train prefixes; score test prefixes;
AUROC with bootstrap 95% CI. Representations: surface (standardised with train statistics,
L2 logistic regression C = 1), length-only (rank by k, direction from train), lexical
(TF-IDF uni+bigrams, max 20,000 features, min_df 2, fitted on train prefixes only),
surface-MLP (64-32), surface-GRU (hidden 32, 200 full-batch epochs, per-step features
[step chars, cumulative chars, position]). Majority class is the constant score.

Caveat (open disagreement 2): because k = ceil(f n), a prefix reveals n.

All AUROC intervals use a question-level (cluster) bootstrap with 1,000 resamples. Detectors
are compared with a paired bootstrap on the same test traces (lexical - length-only,
lexical - surface, surface - length-only at every prefix fraction). E3 is repeated on 5
independent grouped splits (seeds 42, 1, 2, 3, 4) at 25%, 50% and 100%.

## E4 - Activation probing (GPU only; not run in this build)

Generate 4 trajectories per GSM8K test question (first 120) with
Qwen2.5-Math-1.5B-Instruct (temperature 0.7, top-p 0.95, max 400 new tokens), split steps
on line breaks, grade the last number against the gold answer, run one forward pass and
take the hidden state at each step's final token for layers {4, 8, 12, 16, 18, 20, 24, 28}.
Linear probes (C = 0.1) per layer x depth bin {0-25, 25-50, 50-75, 75-100 %} on one
grouped split.

## E5 - Lead time

Stepwise detectors (lexical; one per absolute step count k, trained on train traces with
>= k steps) give r_1..r_n for each test trace using s_1..s_k only. For
tau in {0.3, ..., 0.9}: alarm rate, P(Delta > 0) with never-alarming traces counted as not
warned, fraction alarming exactly at e, median and quartiles of Delta, and the alarm rate
on error-free test traces. The same series gives a length-blind AUROC of r_k after k = 1..10
observed steps, over test traces with n > k.

## E6 - Calibration

ECE with 10 equal-width bins and reliability tables for every probabilistic detector at
the full prefix on the test split; precision / recall / intervention rate per threshold.

## E7 - Transfer

Full-trace lexical and surface detectors fitted on ProcessBench train questions;
in-domain AUROC on ProcessBench test questions; transfer AUROC on each BIG-Bench Mistake
task and all tasks pooled; drop = in-domain - transfer. Within maths: leave one tier out.
StrategyQA: NOT_RUN (no model traces with graded answers).

## E8 - Intervention (simulation)

Algorithm 2 replayed once per test trace at the first alarm of the stepwise risk series:
verify when tau <= r_k < tau_hi (0.8), regenerate from b = argmax_{i<=k} r_i when
r_k >= tau_hi, budget B = 256 tokens. Measured: alarms, outcomes, discarded-suffix length
(characters / 4 as tokens). Assumed (config): P(repair) verify 0.4 / regenerate 0.6,
P(break) verify 0.05 / regenerate 0.10, P(revert catches a break) 0.5, verification 32
tokens. Comparators: none, always (tau = 0), oracle (regenerate failing traces from e,
always repaired), self-consistency N in {2, 4, 8} under independence.
Net benefit = recovery - corruption; net accuracy change = (repaired - broken) / N.
