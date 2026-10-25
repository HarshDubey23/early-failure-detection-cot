# Preregistration: Run R, recovering from hallucinations

Written on 2026-10-25, **before any Run R data exist**. The commit that adds this file is the
timestamp. Run R is launched from this commit or a later one; any change to what is written here
after data exist is reported as a deviation, with its reason.

## 1. Question

The math study (`docs/COMMITMENT_EXPERIMENT.md`, Runs A-C) found that the human-annotated first
error is often not where a reasoning trace becomes doomed, that resampling from just before it can
lose to restarting, and that at a matched budget no practical backtracking policy beat restart.

Run R asks the same questions about hallucinations in retrieval-augmented generation, where the
success criterion (every claim supported by the given source) can be checked at inference time,
and then tests a system built on the answer:

> Does choosing between **claim repair, backtracking, restarting and abstaining** according to how
> recoverable the response still is reduce hallucinations at a matched compute budget, compared
> with the base model, verification-and-revision (Chain-of-Verification), and each fixed recovery
> strategy?

## 2. What exists before this run

- RAGTruth's human labels, analysed without any model (`scripts/30_ragtruth_label_analysis.py`,
  `results/recovery/ragtruth_labels/`). These are descriptive. They motivated the design (most
  hallucinated responses are mostly faithful; hallucinated sentences cluster; QA sources whose
  evidence is insufficient carry more hallucination) and are not tested again here.
- The pipeline, tested on CPU with fake models (`MOCK=1 bash scripts/kaggle_recovery.sh`) and by
  `tests/test_recovery.py`. That includes planted-effect simulations: when some responses are
  committed at the backtracking point, the controller recovers most of what backtracking alone
  loses; with no such effect it does not beat both fixed strategies. These are properties of the
  code, not results.
- No generator, detector or judge output for Run R exists.

## 3. Data, models and settings

**Benchmarks.**

| Benchmark | Items | Role |
|---|---|---|
| RAGTruth test (Niu et al., ACL 2024) | 450 sources: 150 QA (MS MARCO), 150 Summary (CNN/DM, news), 150 Data2txt (Yelp) | primary |
| HotpotQA, controlled evidence | 150 distractor-dev questions (seed 7, no yes/no answers), each twice: 6 passages with both gold paragraphs (answerable) and 6 distractor passages (insufficient, gold answer string absent) | generalization, abstention, factual accuracy |

**Generators.**

| Generator | Weights | Drafts |
|---|---|---|
| Llama-2-7B-chat (**primary**) | `NousResearch/Llama-2-7b-chat-hf` (ungated copy of Meta's release) | RAGTruth: the released responses with their human labels (on-policy). HotpotQA: generated (T = 0.7, top-p 0.95) |
| Llama-3.1-8B-Instruct (secondary) | `unsloth/Meta-Llama-3.1-8B-Instruct` | generated for both benchmarks |

**Detector and evaluators.**

- **D**, the detector the system uses: LettuceDetect-base (`KRLabsOrg/lettucedect-base-modernbert-en-v1`,
  token classifier trained on RAGTruth train). A sentence's score is its highest token probability;
  a sentence is flagged at τ = 0.5. Input is the (task prompt, response) pair.
- **E1**, primary evaluator: Qwen2.5-7B-Instruct as a judge (greedy). It lists the response's
  unsupported sentences (prompt in `src/recovery/prompts.py`). It is from a different family than
  the generators and the detector.
- **E2**, secondary evaluator: LettuceDetect-large. It shares training data with D, so it is
  expected to favour outputs D selected, and is reported as a robustness check only.
- **Human**: RAGTruth labels, available for the released Llama-2 drafts only.

**Sampling.**
- Continuations from a sentence boundary use the generator that wrote the draft. For RAGTruth
  drafts they use the draft's recorded temperature with top-p 1.0; otherwise T = 0.7 with
  top-p 0.95.
- At most 600 new tokens, float16, vLLM on 2x T4.
- m = 8 continuations per boundary.

## 4. The system and the policies compared

A **draft** is scored by D. If D flags no sentence, every D-triggered policy returns the draft.
Otherwise, with j* the first flagged sentence:

**CARE** (the controller under test):
1. **Claim repair.** Each flagged sentence is rewritten by the generator. The prompt shares the
   task, source and draft as a common prefix. If the repaired text passes D, it is returned.
2. **Commitment-aware resampling.** The remaining budget is spent on continuations from
   k = 0 (restart) and k = j* − 1 (backtrack). Each boundary's value v_k, the probability that a
   continuation from k passes D, has a Beta(1, 1) posterior updated after every paid continuation.
   Each step draws two continuations from the boundary maximising −log(1 − ṽ_k) / (marginal
   cost of k), with ṽ_k sampled from the posterior. The first D-clean continuation is returned.
3. **Fallback.** If nothing passes D within the budget, QA abstains ("Unable to answer based on
   given passages."). Other tasks return the repaired text with the sentences D still flags
   removed.

**Compared policies** (all share the draft and D's flags; `src/recovery/replay.py`):

| Name | What it does |
|---|---|
| base | the draft |
| cove_all | factored Chain-of-Verification on every draft (no detector) |
| repairdel | claim repair, then delete what D still flags |
| restart_fb | resample from k = 0 until D-clean; else abstain (QA) / delete flagged sentences |
| backtrack_fb | the same from k = j* − 1 |
| care@restart | CARE with step 2 restricted to k = 0 (**the mechanism ablation**) |

Reported as secondary and exploratory:
- `delete` and `repair` without deletion.
- `abstain_flag`, which abstains whenever a QA draft is flagged.
- `cove_flag`.
- `cascade_fb`, a fixed rule: half the budget backtracking, then restarts.
- `restart` and `backtrack`, which have no fallback.
- `care@backtrack`, `care@norepair`, `care@noabstain`, `care@probe3` (one probe per system point,
  then commit), and `care@oracle` (boundary chosen with E1's v_k, not deployable).

## 5. Cost and budget

- Cost is counted in generator-equivalent tokens, with FLOPs taken as parameters × tokens.
  Detector passes are weighted by params(D) / params(generator).
- Primary view, prefix caching within a request:
  - a restart pays only its generated tokens and detector pass, since the prompt is cached from the
    draft;
  - a backtrack pays the kept sentences once per boundary;
  - repair and CoVe pay their shared prefix once, plus each call's own tail;
  - every generated token counts.
- The no-cache view, where every call pays its full prompt, is reported alongside.
- The budget is B = b × U, where U is the cost of producing and checking the draft:
  - b = 2 is primary, a deployed system at most triples the cost of answering;
  - b = 1 and b = 4 are secondary;
  - repair cost counts against CARE's budget.
- A continuation may start only while the money spent plus its expected cost fits within B.
  The expected cost is the length of the draft's remainder, plus the prefill if not yet paid, plus
  the detector pass.
- Replay: every policy draws from the same bank of m continuations per boundary in R = 20 random
  orders. The outcome of an item is its mean over the orders.

## 6. Metrics

- **Hallucination rate.** Share of outputs with at least one unsupported sentence, judged by E1
  (primary), E2 and D. An abstention or an empty output counts as not hallucinated.
- **Coverage.** Share of outputs that are not abstentions.
- **Factual accuracy** (HotpotQA, answerable items). The output covers the question and contains
  the gold answer after SQuAD normalisation.
- **Correct abstention** (HotpotQA, insufficient items). The output abstains.
- **Correct behaviour.** Accuracy on answerable items and correct abstention on insufficient ones.
- **Cost.** Total and recovery cost in both views, plus generated tokens.
- **Latency.**
  - Measured live for 12 flagged and 4 unflagged RAGTruth items, for CARE, restart_fb,
    backtrack_fb, repairdel and cove (`scripts/36_live.py`).
  - Modelled for every item from single-request prefill and decode rates measured in the same
    session.

## 7. Confirmatory tests (RAGTruth, Llama-2-7B-chat, b = 2, E1)

Differences are paired by item. 95% intervals use a source-clustered bootstrap with 2,000
resamples. p-values use a two-sided cluster sign-flip test with 10,000 permutations.

| ID | Comparison | Supported if |
|---|---|---|
| H0 | CARE − base (sanity check, not in the Holm family) | 95% CI upper bound < 0 |
| H1 | CARE − restart_fb | hallucination: CI upper bound < 0 and Holm p < 0.05; coverage: CI lower bound > −0.02 |
| H2 | CARE − backtrack_fb | same |
| H3 | CARE − repairdel | same |
| H4 | CARE − cove_all | same |
| H5 | CARE − care@restart (does commitment-aware allocation add anything?) | same |

- H1-H5 form one Holm family.
- Each is reported as **CARE better**, **CARE worse** (CI lower bound > 0, Holm p < 0.05) or
  **not shown**.
- Equivalence of CARE and restart_fb is reported with TOST at ±0.03 (90% CI).
- Cost differences are reported for every comparison. A hallucination gain bought with more
  compute is described as such. Every policy is capped by the same budget, but repairdel and
  cove_all are not budgeted.

**Evaluator check, fixed now.**
- E1's agreement with the human labels on the 450 released drafts is computed before the policy
  comparison is read.
- If E1's response-level F1 is below 0.50 or Cohen's κ is below 0.20, E1 is unreliable. E2
  becomes the primary evaluator for H0-H5, and this is reported.

## 8. Mechanism (RAGTruth, Llama-2-7B-chat, human labels)

The boundaries are k ∈ {0, e−2, e−1, e, e+1} for each human-hallucinated draft, where e is the
first hallucinated sentence (`src/recovery/ragtruth.mechanism_points`). Position-matched placebo
anchors are used for 39 human-clean drafts.

- **M1, recovery.** u_{e−1} − u_0, where u is the share of continuations with no unsupported
  sentence. Evaluated with E1, at k = 0 and e−1 for every hallucinated draft, and with E2.
  Two-sided. In the math study the analogue favoured restart for three of four models.
- **M2, snowballing.** Does keeping the hallucinated sentence raise the hazard of the rest?
  - The jump is h_e − h_{e−1}, with h the share of unsupported continuation sentences (E2).
  - It is compared with the same jump at placebo anchors.
  - Excess jump CI excludes 0 → supported.
- **M3, doomed responses.** The share of hallucinated drafts whose restarts are rarely clean,
  u_0 ≤ 1/8. Descriptive.

## 9. Generalization (HotpotQA controlled evidence)

- Primary generator: Llama-2-7B-chat; Llama-3.1-8B is secondary; b = 2.
- Outcomes are correct behaviour, factual accuracy, coverage and E1 hallucination.
- Comparisons are CARE vs base, restart_fb and cove_all, in one Holm family per generator.
- The same RAGTruth tests (H0-H5) on Llama-3.1-8B are secondary.

## 10. Incomplete runs and exclusions

- Parts run in the priority order of `scripts/kaggle_recovery.sh`. If the time limit cuts a part,
  the analysis uses items whose scoring is complete, and the numbers analysed and dropped are
  reported.
- Prompts longer than the context window are recorded as skipped. Their items are excluded from
  every policy, so comparisons stay paired.
- If E1 fails to parse an output, E2's verdict is used for that output and the count is reported.

## 11. What will not be claimed

- No result from replay is described as a live deployment result; latency is labelled measured or
  modelled.
- A "not shown" result is not described as "no difference" unless TOST shows equivalence.
- Nothing from E2 or D alone is described as a reduction in hallucinations.
- No human evaluation of Run R outputs exists. A blinded comparison sheet will be prepared, and
  any human result will be reported only after people have actually completed it.
