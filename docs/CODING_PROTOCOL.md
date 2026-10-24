# Coding protocol: recovered traces

A *recovered* trace has an expert-annotated first error (ProcessBench) and still ends with the
correct final answer. 521 of the 3,400 ProcessBench traces are recovered. This protocol asks
*why* the answer survives the error.

## Materials

`python scripts/18_coding.py make` creates the coding materials in `results/coding/`:

- `coding_items.html`: 80 items, sampled with seed 2026 and stratified by tier, with all 7 GSM8K
  items kept. Each item shows the question and the steps, with the annotated first error marked.
- `coder_A_blank.csv`: all 80 items, for the first coder.
- `coder_B_blank.csv`: the first 40 items in a different order, for the second coder, who
  provides the agreement estimate.
- `sample_key.csv`: maps items to trace ids and generators. **Coders do not open this file.**

Coders work independently, do not discuss items, and never see the generator, the other
coder's sheet or any counterfactual (rollout) result. Save your sheet as `coder_A.csv` or
`coder_B.csv` in the same folder.

## Fields

| Field | Values | Meaning |
|---|---|---|
| `label_ok` | `yes` / `no` / `unsure` | Does the marked step contain a mathematical or logical error? Judge only that step. |
| `category` | one value from the list below | Why the final answer is still correct. |
| `confidence` | `1` (guess), `2` (fairly sure), `3` (certain) | Confidence in `category`. |
| `note` | free text, one line | What the error is and why you chose the category. Required when `category` is `other` or `label_ok` is not `yes`. |

## Categories (choose the first that applies, in this order)

1. **`label_questionable`**: you answered `label_ok = no`. The marked step looks correct, so the
   trace is not evidence that errors can be survived.
2. **`explicit_repair`**: a later step *says* that something earlier was wrong ("wait",
   "let me re-check", "this is incorrect", recomputes a value) and the correction removes the
   error's effect.
3. **`silent_repair`**: a later step uses the *correct* value or argument again without
   acknowledging the error. The written chain is repaired, but nobody says so.
4. **`error_not_load_bearing`**: the error is real, but nothing that leads to the final answer
   depends on it. Examples: a wrong side remark, or an error in a branch that is later
   abandoned.
5. **`answer_not_determined_by_reasoning`**: the error propagates, and the reasoning after it
   stays wrong or incomplete, yet the stated final answer is correct. Typical cases:
   - the answer is a guess among few options (yes/no, a small integer, a famous constant);
   - it is copied from a pattern;
   - it does not follow from the preceding steps.
   The written reasoning does not determine the answer.
6. **`other`**: none of the above (explain in `note`).

## Rules for ambiguous cases

- If two categories seem to apply, take the first one in the list, so the order above decides.
- If you cannot tell whether a later step repairs the error, choose `answer_not_determined_by_reasoning`
  with confidence 1 and explain in `note`.
- If the problem is outside your competence (for example, a hard olympiad item), still code it.
  Set `confidence = 1` and state this in `note`. Do not skip items.

## Analysis

`python scripts/18_coding.py agree --a results/coding/coder_A.csv --b results/coding/coder_B.csv`
reports:

- per-field percent agreement and Cohen's kappa on the 40 overlapping items;
- coder A's category distribution, with bootstrap intervals.

The paper reports these numbers once both sheets are complete. Until then, the paper states
that the human coding of recovered traces is pending.

The forced-answer probes of Run C (`scripts/15_answer_probe.py`) give a coder-independent
measurement of the same question. If the model's forced answer is already correct right after
the error step (a_e ≥ 0.5), the written reasoning did not determine the answer. Agreement
between that measurement and the human categories is reported as an additional check.
