# Blinded human rating kit: base model vs detect-restart-abstain (DRA)

This kit is the decisive test of the RAG claim: do people find fewer unsupported sentences in DRA's
outputs than in the base model's? Every number so far comes from LLM graders.

- **Design** (`design.json`). DRA returns the base draft unchanged whenever the detector passes it,
  so the two policies differ on 208 of 450 RAGTruth items. 80 of these, drawn at random, are shown
  as A/B pairs in random order. On the other items the difference is 0 by construction, so
  `score` rescales the paired difference by the share of items that differ (0.462) to give the
  change over all items.
- **Raters.** Two people, working independently. Each copies `sheet_blank.csv` to
  `sheet_annotator_<name>.csv`. For every pair, they read the source and task (`sheet.html` is
  easier to read) and write, for output A and output B, the numbers of the sentences the source
  does not support (comma-separated, empty if none), then which output they would rather receive
  (`A`, `B` or `tie`). This takes about 2 to 3 hours per rater.
- **Blinding.** The key that maps A/B to policies is not in the repository. Its SHA-256 is in
  `KEY.sha256`, committed before any rating, so anyone can check that the published key was not
  changed afterwards.
- **Scoring.** After both sheets are filled, put the key back as `KEY_do_not_open.csv` and run

      python scripts/38_human_eval_kit.py score --out results/recovery/human_eval_base_vs_dra

  This reports each policy's human hallucination rate on the rated pairs, the paired difference with
  a bootstrap CI, the same rescaled to all items, the preference shares, and Cohen's kappa between
  the two raters.

**Planned analysis** (fixed before rating). The primary result is base minus DRA on all items, from
the first rater's sheet. The second rater gives agreement (kappa) and a replication. The reduction
counts as human-verified if the 95% interval of the rescaled difference excludes 0 for both raters.
