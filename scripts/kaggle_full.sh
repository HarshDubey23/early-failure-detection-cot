#!/usr/bin/env bash
# Full counterfactual-commitment run for one Kaggle "Save Version" (background) session, 2x T4.
# Small models first (minutes each), then Qwen2.5-Math-7B under a time limit so the version
# always finishes and saves its outputs. Every stage-12 run resumes from its JSONL.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=results/commitment/full_run.log
mkdir -p results/commitment
say() { echo "$(date -u '+%H:%M:%S') $*" | tee -a "$LOG"; }

say "setup";  bash scripts/kaggle_commitment.sh setup 2>&1 | tail -n 6 | tee -a "$LOG"
say "vllm";   bash scripts/kaggle_commitment.sh vllm  2>&1 | tail -n 3 | tee -a "$LOG"

# 1) Qwen2.5-1.5B-Instruct, on-policy, all 116 traces, m = 16 (independent replicate of the pilot).
say "run Qwen2.5-1.5B-Instruct"
BACKEND=vllm bash scripts/kaggle_commitment.sh run Qwen2.5-1.5B-Instruct 16 2>&1 | grep "^\[" | tee -a "$LOG"

# 2) Qwen2-1.5B-Instruct, on-policy, all 88 traces, m = 16 (second model family member).
say "run Qwen2-1.5B-Instruct"
BACKEND=vllm bash scripts/kaggle_commitment.sh run Qwen2-1.5B-Instruct 16 2>&1 | grep "^\[" | tee -a "$LOG"

# 3) Qwen2.5-Math-7B-Instruct over both GPUs, m = 12, clean traces capped at 60, at most 8.5 hours.
#    Called directly (not through the wrapper) so `timeout` stops the generator itself; the
#    analysis then uses every trace whose prefixes all finished.
say "run Qwen2.5-Math-7B-Instruct"
timeout 30600 python scripts/12_counterfactual_rollouts.py --generator Qwen2.5-Math-7B-Instruct \
  --backend vllm --tp 2 --dtype half --m 12 --max-tokens 1024 --max-clean 60 \
  --out results/commitment/Qwen2.5-Math-7B-Instruct_s0.jsonl > results/commitment/Qwen2.5-Math-7B-Instruct_s0.stdout 2>&1
say "7B generation exit code $? (124 = time limit reached)"
python scripts/13_commitment_analysis.py --rollouts results/commitment/Qwen2.5-Math-7B-Instruct_s0.jsonl \
  --out results/commitment/tables_Qwen2.5-Math-7B-Instruct 2>&1 | grep "^\[\|traces" | tee -a "$LOG"

tar czf /kaggle/working/commitment_results.tar.gz results/commitment
say "done; archive: /kaggle/working/commitment_results.tar.gz"
