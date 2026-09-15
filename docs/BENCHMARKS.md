# Benchmarks

Each dataset: source, size, licence, schema, role here, identifier verification.

## ProcessBench (primary — error-onset analysis)

- Source: HuggingFace `Qwen/ProcessBench` (Zheng et al., 2024, arXiv:2412.06559).
- Size: ~3,400 traces across four difficulty tiers (GSM8K, MATH, OlympiadBench,
  OmniMATH) — exact N printed by `01_download_data.py` at runtime.
- Licence: research use; see dataset card. Cite Zheng et al. (2024).
- Schema: step-segmented traces + first-error index (`-1` = correct). The loader
  prints live column names before mapping; missing columns raise
  `DatasetSchemaError` with the observed list.
- Used for: E1, E2, E3, E5, E6 (all onset/timing analysis).
- Identifier status: used in the Colab prototype (`REPO = "Qwen/ProcessBench"`);
  re-verified at runtime by stage 01. If it fails to resolve, the error is
  recorded in `RESULTS.md` and no substitute is used.

## PRM800K (secondary — step-level correct/incorrect/neutral)

- Source: GitHub `openai/prm800k` (Lightman et al., 2023). NOT a HuggingFace
  repo — the loader reads a local checkout (`phase1_train.jsonl`, …).
- Licence: MIT (code/data release). Cite Lightman et al. (2023).
- Schema: per-step human judgements; first `incorrect` step → `first_error_idx`.
- Used for: secondary onset analysis, probe training data.
- Identifier status: local-checkout loader; network fetch is opt-in only.

## GSM8K (trajectory generation for activations)

- Source: HuggingFace `openai/gsm8k`, config `main`.
- Licence: MIT. Cite Cobbe et al. (2021).
- Schema: question–answer pairs, outcome labels only (`first_error_idx = None`).
- Used for: generating trajectories with `Qwen/Qwen2.5-Math-1.5B-Instruct` (E4).
- Identifier status: used in prototype Part 7; verified at runtime.

## MATH / MATH-500 (harder trajectory generation)

- Source: HuggingFace `HuggingFaceH4/MATH-500`, fallback
  `hendrycks/competition_math`. The loader records which resolved.
- Licence: MIT. Cite Hendrycks et al. (2021).
- Schema: competition problems, outcome labels only.
- Used for: harder-regime trajectory generation (E4 extension).
- Identifier status: VERIFY AT RUNTIME — if neither resolves, recorded here as
  unverified and the stage marks E4-extension rows NOT_RUN.

## BIG-Bench Mistake (cross-domain: symbolic reasoning)

- Source: Google Research `bigbench` repo, mistake-detection tasks; loader
  reads local JSONL (`steps` + `first_mistake`).
- Licence: Apache-2.0 (BigBench). Cite Tyen et al. (2024) for the mistake
 -detection framing.
- Schema: traces with first-mistake index → `first_error_idx`.
- Used for: E7 cross-domain transfer.
- Identifier status: VERIFY AT RUNTIME (local-checkout loader).

## StrategyQA (cross-domain: implicit multi-hop)

- Source: HuggingFace `wics/strategy-qa` (fallback `bigbio/strategy_qa`).
- Licence: MIT. Cite Geva et al. (2021).
- Schema: questions with decompositions (no error annotations;
  `first_error_idx = None`). Outcome-detector transfer only.
- Used for: E7.
- Identifier status: VERIFY AT RUNTIME.

## Verification log

`01_download_data.py` prints resolution status for every identifier at runtime.
Any identifier that fails to resolve is appended here (and in `RESULTS.md`) as
UNVERIFIED with the observed error — a guess is never substituted.
