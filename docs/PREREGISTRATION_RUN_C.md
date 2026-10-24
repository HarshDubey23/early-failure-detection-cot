# Preregistration: Run C (confirmatory and robustness run)

Written on 2026-10-24, **before any Run C data exist**. The commit that adds this file is the
timestamp. Run C is launched from a later commit, and any deviation from this plan will be
reported as a deviation.

## 1. What is already known, and how it was obtained

**Preregistered before any rollout.** H1-H4 in `docs/COMMITMENT_EXPERIMENT.md` were committed in
`a6fb04c` (2026-10-24 06:30 UTC), before Run A (results recorded in `5316604`, 07:44 UTC). They
fixed δ = 1/8 (with 1/16 and 1/4 as sensitivity), the commitment categories, the decisive step
(largest drop in v) and the sample-split comparison of decisive-step and first-error
backtracking.

**Exploratory: analyses added after seeing Run A/B data.** None of these can be called
confirmed from Runs A/B:

| Exploratory finding (Runs A/B; Qwen2.5-1.5B, Qwen2-1.5B, Qwen2.5-Math-7B) | Estimate |
|---|---|
| restart − first-error backtracking | +0.106, +0.215, +0.068 (all 95% CIs > 0) |
| decisive-step − restart | +0.011, −0.020, +0.040 (all CIs include 0; equivalence not tested) |
| first error = decisive step vs a position-matched null | 0.529 vs 0.171, 0.344 vs 0.214, 0.423 vs 0.159 |
| repeated sample splitting (50 splits) instead of one split | analysis refinement |

## 2. Data collected in Run C (`scripts/kaggle_run_c.sh`)

1. **New model family and new questions.** Llama-3.1-8B-Instruct (weights:
   `unsloth/Meta-Llama-3.1-8B-Instruct`, an ungated copy of the Meta release) resampled at every
   prefix of its own ProcessBench traces. The traces are all fatal and recovered traces with a
   gold answer, plus 60 clean traces (selection seed 42). m = 16 continuations per prefix,
   temperature 0.7, top-p 0.95, at most 1,024 new tokens, float16, tensor parallel 2.
   Llama's 526 traces cover 512 questions, most of which are not among the Qwen models' traces.
2. **Independent sampling batches.** Three new batches (seeds 1001, 2002, 3003; per-prompt seeds
   hashed from (trace, step, seed)), each with m = 16, for Qwen2.5-1.5B-Instruct and
   Qwen2-1.5B-Instruct on the same traces as Run B. This raises precision to 64 continuations per
   prefix and allows test-retest comparisons between batches.
3. **Step signals.**
   - Qwen2.5-Math-PRM-7B step scores for every trace of the four generators, then all of
     ProcessBench (implementation check: ProcessBench F1 should be close to the published value).
   - Per-step negative log-likelihood and entropy under each trace's own generator.
   - Exact token counts.
4. **Forced-answer probes.** At every prefix, m = 8 immediate answers ("Final answer:
   $\boxed{"), giving the answer curve a_k.

If the 11.5-hour limit cuts the Llama run, the analysis uses complete traces only. Traces are
processed fatal first, then recovered, then clean, and the number analysed will be reported.

## 3. Confirmatory hypotheses (Llama-3.1-8B-Instruct)

The point estimates come from `scripts/13_commitment_analysis.py` as committed at launch. The
quantities stage 13 does not yet print are the C3 difference interval, the 90% TOST interval for
C4 and the Holm p-values. They are computed from its per-trace outputs exactly as defined here,
by a script written before Run C data are downloaded. The settings are δ = 1/8, 50 A/B splits,
and 95% trace-bootstrap intervals with 2,000 resamples.

| ID | Hypothesis | Estimator | Confirmed if |
|---|---|---|---|
| C1 | Restarting beats resampling from just before the annotated first error | restart − first-error, sample-split pool (v_0^A > δ) | 95% CI lower bound > 0 |
| C2 | Resampling from just before the decisive step beats first-error backtracking | decisive − first-error, same pool | 95% CI lower bound > 0 |
| C3 | The annotated first error is the decisive step more often than a position-matched null | share(d = e) − mean null probability, not-capability-limited fatal traces | 95% CI lower bound > 0 |
| C4 | Decisive-step backtracking and restart are equivalent within ±0.05 | decisive − restart, TOST | 90% CI inside (−0.05, +0.05) → equivalent; 90% CI outside → different; else **inconclusive** |

C1-C3 are also reported with Holm-adjusted bootstrap p-values. A "not confirmed" result will be
reported as such, and no claim from Runs A/B that fails here will be stated as general.

## 4. Measurement robustness (Qwen 1.5B models; new batches only)

- **R1.** Restart − first-error re-estimated **only on the 48 new continuations per prefix**
  (Run B samples excluded). It is reported per model and pooled (trace bootstrap stratified by
  model). Confirmed if the pooled 95% CI lower bound > 0.
- **R2.** Test-retest reliability of the value curve: correlation of v_k between independent
  batches over all prefixes. Also the agreement of the decisive step between batches, against
  the agreement expected under the position-matched null.
- **R3.** Decisive-step uncertainty: for every not-capability-limited fatal trace, the posterior
  probability of each candidate step being the decisive step. This uses 2,000 draws of each v_k
  from Beta(c + ½, m − c + ½). Reported: the share of traces whose modal decisive step has
  posterior probability ≥ 0.5, and the posterior-averaged share of d = e.

## 5. Practical policies at a matched end-to-end budget

The policies are fixed here. They are evaluated on fatal traces (all of them, since a deployed
system cannot know v_0; the not-capability-limited subset is secondary). Each policy picks a
restart point k, and "resample from k" means sampling continuations of the first k steps.

| ID | Policy | Extra cost counted |
|---|---|---|
| P0 | restart (k = 0) | none |
| P1 | PRM first error: k = (first step with PRM score < 0.5) − 1; restart if none | PRM forward pass over the trace |
| P2 | PRM largest drop: k = argmax_j (p_{j−1} − p_j) − 1 with p_0 = 1 | PRM forward pass |
| P3 | probe search: 2 probe continuations at each of k = ⌊n/4⌋, ⌊n/2⌋, ⌊3n/4⌋ (from half A); take the latest probed k with ≥ 1 success, else restart | all probe tokens, including their prefills |
| P4 | learned: logistic regression on step features, trained **only on the three Qwen generators** (labels: decisive step from half A); target = argmax predicted probability | PRM and generator forward passes |
| P5 | human first error (reference, not deployable) | none |
| P6 | oracle decisive step from half A | all half-A rollouts at every prefix |

The step features are PRM score, PRM drop, generator NLL mean/max, entropy mean/max, relative
position and step tokens.

**Cost (end to end).**
- Cost is counted in tokens processed, weighted by model size relative to the generator
  (FLOPs ∝ parameters × tokens).
- Every prefill (question + prefix) counts at full cost once per restart point (prefix caching).
- Every generated token counts.
- Restart pays the question prefill plus its generated tokens.
- Decode-only accounting is reported as a secondary view only.

**Budget and score.**
- Per trace, the budget is B = b × (cost of one restart attempt), with b ∈ {1, 2, 4, 8}; b = 4
  is primary.
- A policy gets N = ⌊(B − overhead − prefill) / mean generated tokens from k⌋ attempts.
- Success is P(at least one of N attempts is correct), estimated on half B as the
  Jeffreys-posterior mean of 1 − (1 − p)^N. The unbiased pass@N estimator is shown alongside
  whenever N ≤ the half-B sample size.

**Confirmatory (Llama, b = 4).**
- P2 − P0 and P4 − P0, each with a 95% trace-bootstrap CI.
- The outcome is reported as better (CI > 0), worse (CI < 0) or inconclusive.
- On the Qwen models the same numbers are exploratory, because the policy code is written
  after seeing their data.

## 6. Exploratory (descriptive) analyses planned

- Answer curves a_k by outcome group.
- For recovered traces:
  - *answer already correct at the error*: a_e ≥ 0.5 (the forced answer is correct right
    after the error step);
  - *repair*: a_e < 0.5 and the written trace still ends correct.
- Agreement of these labels with human coding when it exists.
- Threshold sensitivity of C1-C4 and of the categories at δ ∈ {1/16, 1/8, 1/4}.
- Alternative decisive-step definitions:
  - first k with v_k ≤ δ thereafter;
  - first drop larger than the clean-trace noise floor.
- PRM implementation check against the published ProcessBench F1.
