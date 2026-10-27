# Related work and positioning (checked 2026-10-27)

Supersedes the October 24 table in [`NOVELTY_AUDIT.md`](NOVELTY_AUDIT.md). Every paper below was
looked up on its arXiv or venue page on the date above; author lists in `paper/refs.tex` were
corrected against those pages.

## The claim this repository makes

On failed generations, (i) human first-error labels locate the counterfactual choke point only
about half the time, (ii) process reward models track the label rather than the choke point, and
(iii) no deployable choice of resampling point beats restart at matched compute, with a free oracle
gaining about three points at a 4x budget; (iv) the same holds for hallucination recovery in RAG,
where the measured gain of a detector-gated loop depends on whether the grader shares training data
with the detector.

## Closest work and what is different

| Work | What it does | Difference |
|---|---|---|
| **Bulut 2026**, *It's the Problem, Not the Path* (arXiv 2609.03436) | Gemma-4 E4B and Ministral-3 3B on 89 MATH problems; truncated prefixes vs matched-token restarts; preregistered difficulty-controlled early-signal test; within-problem reanalysis of a published probe | Closest in spirit and earlier (Sep 2026). It studies *all* reasoning trajectories of two reasoning models and finds that long prefixes mostly buy the same successes more cheaply. We study *failed* traces with expert error labels, four instruction models, 125,832 continuations, verifier costs charged, and a bound on what any localiser can gain. Our difficulty result (Part 1) is the same lesson on human-annotated traces; we cite it as such. |
| **Qin, Alvarez-Melis, Jelassi, Malach 2025**, *To Backtrack or Not to Backtrack* (arXiv 2504.07052) | Trained small models on CountDown and Sudoku; sequential backtracking vs best-of-n at matched FLOPs | Backtracking wins on deep search trees, loses on shallow ones. Ours is on-policy natural-language reasoning and RAG with pretrained models, and asks where to resample a failed trace, not whether to train a backtracking model. |
| **Thought Anchors** (Bogdan, Macar, Nanda, Conmy 2025, arXiv 2506.19143); **Thought Branches** (Macar, Bogdan, Rajamanoharan, Nanda, ICLR 2026) | Resampling to measure which reasoning sentences matter | Same tool; we point it at human error labels, verifiers and recovery policies, and at failed traces. |
| **The Point of No Return** (Merrill & Srivastava 2026, arXiv 2605.17113) | Counterfactual localisation of commitment to deception | Different behaviour (deception); they name incorrect solutions as future work, which is our setting. |
| **Not Every Divergence Should Be Suppressed** (Jiang et al. 2026, arXiv 2608.04408) | Counterfactual recoverability of error states in on-policy distillation | Training-time use; no comparison with human labels or restart at inference. |
| **ProcessBench** (Zheng et al., ACL 2025); **Lessons of developing PRMs** (Zhang et al. 2025) | Human first-error labels; Monte Carlo step labels disagree with them | They treat the disagreement as label noise. ChokeBench measures it and shows which target the PRM tracks. |
| **Noise-aware PRM learning** (Xie et al. 2026, arXiv 2601.12748); **MCNIG** (Royer et al. 2026, arXiv 2603.17815) | Cleaner automatic step labels for PRM training | Aim at the human-label target; our results suggest the counterfactual target matters for recovery. |
| **Beyond the Frontier: Stochastic Backtracking** (Tran et al., ICML 2026, arXiv 2605.25143) | PRM-guided search that revisits a persistent pool of prefixes; fewer tokens than best-of-N | Positive result for PRM search on mostly-correct search trees; our free-oracle ceiling bounds the gain from revisiting prefixes of failed on-policy traces. |
| **Snell et al.** (ICLR 2025) | Compute-optimal test-time scaling; verifier search vs best-of-N depends on difficulty | We add counterfactual ground truth for the restart point and charge verifier costs. |
| **Knowing Before Saying** (Afzal et al. 2025); **LLMs Encode Their Failures** (Lugoloobi et al., ICLR 2026) | Hidden states predict success before or during generation | Pooled AUROC; we show the text-detector analogue is explained by question and generator. |
| **Reasoning Theater** (Boppana et al. 2026); **When Reasoning Traces Become Performative** (Li et al., ICML 2026) | Answers often fixed apart from the written CoT | Consistent with our non-fatal errors and late answer curves. |
| **How LM Hallucinations Can Snowball** (Zhang, Press, Merrill, Liu, Smith, ICML 2024) | Early wrong claims lead to more wrong claims | Tested with RAGTruth human spans: no excess hazard after a hallucinated sentence. |
| **CoVe** (Dhuliawala et al., Findings of ACL 2024); **RARR** (Gao et al., ACL 2023); **LettuceDetect** (Kovács & Recski 2025); **RAGTruth** (Niu et al., ACL 2024) | Detect-then-revise hallucination repair; token-level detectors; human span labels | We test *where* to resample at matched budget, preregistered, and show grader dependence when the grader shares the detector's training data. |
| **Luby, Sinclair, Zuckerman 1993**; **Gomes, Selman, Crato, Kautz 2000** | Optimal restart strategies for Las Vegas algorithms; heavy tails make restarts pay | The classical form of the same question. Appendix A gives a supermartingale bound for LLM decoding. |

## What no paper found does

- Compare on-policy counterfactual choke points with expert first-error labels, across model
  families, with replication and posterior uncertainty.
- Grade verifiers on the choke point and bound what a free, perfect localiser can gain over restart.
- Test, preregistered, whether choosing where to resample helps hallucination recovery, and show
  that the measured benefit of a detector-gated loop depends on the grader's relation to the detector.

## Threats to novelty to watch

- A Bulut-style restart control on failed traces of long-CoT reasoning models would overlap with
  Part 2; running ours on R1-distilled or Qwen3 models first is the natural next step.
- PRM papers that train on Monte Carlo value targets may report choke-point localisation; ChokeBench
  gives them a common yardstick.
