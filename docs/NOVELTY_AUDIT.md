# Novelty audit and revision plan (2026-10-24)

## Verdict on the current paper

The engineering is strong: grouped splits, cluster bootstrap, paired tests, a discrete null for
onset, reported NOT_RUN stages. A reviewer in this area would still reject it, for four reasons:

1. **The headline is a difficulty signal.** A detector that never reads the reasoning (question
   text + generator) reaches AUROC 0.588, the paper's "25% of the chain" number. Adding the 25%
   prefix adds +0.005 (n.s.); within a problem the 25% prefix is at chance (0.485). See
   `docs/RESULTS_DECOMPOSITION.md` §B. Bulut (arXiv 2609.03436, Sep 2026) makes exactly this
   critique of early-failure probes, so reviewers will look for it.
2. **The titular concept is never measured.** The introduction separates error, choke point and
   intervention point, then every experiment uses the first error. Choke points stay undefined.
3. **The planned representation experiment (E4) is no longer new.** Afzal et al. (arXiv
   2505.24362, 2025) and Lugoloobi et al. (arXiv 2602.09924, 2026; probe AUROC about 0.84 on
   Qwen2.5-Math-1.5B) already predict CoT success from activations, before or during
   generation.
4. **"Mitigating" is in the title but no mitigation is run;** E8 replays assumed repair rates.

## Closest prior work (checked October 2026)

| Work | What it does | Gap left |
|---|---|---|
| ProcessBench (Zheng et al., ACL 2025) | human first-error labels for 3,400 traces | no notion of which errors decide the outcome |
| Zhang et al. 2025, *Lessons of developing PRMs* (arXiv 2501.07301) | MC rollout labels disagree with human labels; flawed-reasoning-correct-answer rates by source | treats disagreement as noise; no step-level comparison |
| Thought Anchors (Bogdan et al. 2025, arXiv 2506.19143) | resampling importance of CoT sentences | no human error labels; not about failure |
| Critical Tokens (Lin et al., ICML 2025) | tokens whose replacement flips outcomes, via rollouts | token level, for training; no error labels |
| Lost at the Beginning (Liao et al. 2025, arXiv 2506.22058) | first step strongly shapes the outcome | not tied to annotated errors or commitment |
| The Point of No Return (Merrill & Srivastava 2026, arXiv 2605.17113) | counterfactual localisation of commitment, 50 continuations per prefix | deception only; names incorrect solutions as future work |
| Counterfactual Recoverability in OPD (arXiv 2608.04408) | branch from error states: continue vs roll back | distillation; not mapped against human labels or position |
| Knowing Before Saying (2505.24362), LLMs Encode Their Failures (2602.09924) | hidden-state prediction of success | pooled AUROC; difficulty not separated |
| It's the Problem, Not the Path (Bulut 2026, arXiv 2609.03436) | early signals add little beyond difficulty | probes on R1-style data; not text detectors, not ProcessBench |
| When Reasoning Traces Become Performative (arXiv 2605.11746); Reasoning Theater (arXiv 2603.05488) | answers often fixed before or apart from the written CoT | no human first-error labels |
| CROP (arXiv 2605.30085) | conformal certification of error-free prefixes | controls annotated-error risk, not success probability |

No paper found compares **counterfactual commitment (on-policy resampling) with human
first-error annotations**, nor quantifies what failure detectors do on non-fatal errors.

## What is genuinely new now

Done on real data (CPU):

1. On ProcessBench, early-prefix failure detection is explained by question + generator; the
   reasoning adds +0.02 to +0.03 AUROC only from the second half of the chain, and nothing
   within a problem.
2. 23.5% of first errors are not fatal; fatality ranges from 0.966 (GSM8K) to 0.659
   (Omni-MATH) and is flat in error position. In a 40-trace sample only 4 involve an actual
   repair.
3. Failure detectors cannot distinguish fatal from non-fatal errors (0.456; 0.376 for an error
   detector), and fatality at the moment of the error is not predictable from the reasoning text
   beyond question + generator (Δ ≤ 0.003).

Ready to run (GPU, `docs/COMMITMENT_EXPERIMENT.md`):

4. On-policy counterfactual value curves v_k for ProcessBench traces written by open Qwen
   models: is the annotated first error the decisive step, how often does commitment precede
   or follow it, and does backtracking to the decisive step beat backtracking to the first error?

## Proposed reframing

**Title:** *Errors Are Not Choke Points: Difficulty, Error Onset and Commitment in LLM Reasoning
Failures*

**Abstract (current evidence only; extend with E10 once it has run):**

> Process supervision assumes that the first incorrect step of a reasoning trace is where it
> fails, and early-warning detectors assume that failure is visible in a partial trace. We test
> both assumptions on 3,400 ProcessBench traces with expert first-error labels. First, a
> detector that reads only the question and the generator's identity predicts an incorrect
> answer with AUROC 0.588; a TF-IDF detector reading the first quarter of the reasoning adds
> nothing beyond it (+0.005, n.s.), and within a problem it is at chance. Reasoning text adds
> 0.02–0.03 AUROC only after half the chain. Second, 23.5% of annotated first errors are
> followed by a correct answer, rising to 34% on Omni-MATH, and fatality does not depend on where
> the error occurs. A coded sample shows these are rarely repairs: the written reasoning mostly
> does not determine the answer. Third, failure detectors rank non-fatal errors as riskier than
> fatal ones (AUROC 0.456), and once an error has occurred its fatality is unpredictable from the
> text. Early failure detection on this benchmark therefore measures problem difficulty and error
> presence, not commitment. We release a difficulty-controlled evaluation protocol and an
> on-policy counterfactual procedure that measures where trajectories actually become
> unrecoverable.

**Contributions:** (1) a difficulty-controlled protocol for early failure detection
(question-only and generator baselines, incremental AUROC, within-problem concordance) that any
detector paper can adopt; (2) the error-fatality decomposition and coded taxonomy of non-fatal
errors; (3) evidence that text detectors track error presence and difficulty, not commitment;
(4) the counterfactual commitment measurement and its backtracking implication (after E10).

**Drop or demote:** the hidden-state probe as a contribution (cite it as prior work, keep only if
compared under the within-problem protocol); "Mitigating" in the title until E10's backtracking
result exists; Algorithm 2 replay with assumed rates (appendix at most).

## Corrections needed in the current PDF

- Ref [17]: first author is **S.** (Saurav) Kadavath, not "J. Kadavath".
- Ref [8]: ProcessBench was published at ACL 2025 (long paper 50); cite the venue, not only arXiv.
- Ref [4] (Orgad et al.): check for a published version and cite it if one exists.
- Abstract/Results: "lexical − length-only +0.073, p = 0.004" becomes +0.052 [0.005, 0.100],
  p = 0.032 after the tie fix; the length-only column of Table III changes (see
  `results/tables/table2d_length_only_tie_fix.csv`).
- Table I lists "Corrects: Phase II" for this work and the title says "Mitigating": nothing is
  corrected yet.
- Section VI-B reports 0.588 at 25% without a question-only baseline; add §B of the new results.
- Cite Bulut 2026, Zhang et al. 2025, Thought Anchors, Point of No Return, Afzal et al. 2025,
  Lugoloobi et al. 2026, and the performative-CoT papers above.

## What would make it stand out

The CPU results are a solid, honest workshop paper on their own. What turns it into a paper a
strong group would notice is E10 with a clear answer to one question: *is the human-annotated
first error the point of no return?* Either answer matters for process reward models. Run the
1.5B pilot, check the noise floor, then the 7B-Math model, then a second 7B model for
replication. Have a second person code the 40 recovered traces (or 100) and report agreement.
