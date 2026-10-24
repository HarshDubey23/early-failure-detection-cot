#!/usr/bin/env bash
# Counterfactual commitment experiment on a Kaggle / Colab GPU notebook.
#
#   bash scripts/kaggle_commitment.sh setup                       # deps + ProcessBench (minutes)
#   bash scripts/kaggle_commitment.sh smoke Qwen2.5-1.5B-Instruct  # 3 traces, m=4: checks the GPU path
#   bash scripts/kaggle_commitment.sh run   Qwen2.5-1.5B-Instruct 16 [extra stage-12 flags]
#
# `run` starts one stage-12 process per visible GPU (shards by trace id), waits for all of
# them, then runs the stage-13 analysis. Stage 12 resumes from its JSONL, so re-running the
# same command after a timeout continues where it stopped.
set -euo pipefail
MODE=${1:-run}
GEN=${2:-Qwen2.5-1.5B-Instruct}
M=${3:-16}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")
OUT=results/commitment
mkdir -p "$OUT"

case "$MODE" in
  setup)
    pip install -q "math-verify[antlr4_13_2]" || pip install -q math-verify
    python scripts/12a_prepare_processbench.py
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available())"
    ;;
  smoke)
    python scripts/12_counterfactual_rollouts.py --generator "$GEN" --backend hf --dtype half \
      --m 4 --max-tokens 768 --max-per-group 1 --out "$OUT/smoke_${GEN}.jsonl"
    python - "$OUT/smoke_${GEN}.jsonl" <<'PY'
import json, sys
for line in open(sys.argv[1]):
    r = json.loads(line)
    print(f"{r['trace_id'][:40]:<40} k={r['k']:<2} gold={str(r['gold'])[:15]:<15} "
          f"correct={r['n_correct']}/{r['m']} boxed={r['n_boxed']} answers={[str(a)[:12] for a in r['answers']]} "
          f"tokens={r['new_tokens']}")
PY
    ;;
  run)
    NG=$(nvidia-smi -L | wc -l)
    echo "starting $NG shard(s) for $GEN, m=$M ${EXTRA[*]:-}"
    pids=()
    for i in $(seq 0 $((NG - 1))); do
      CUDA_VISIBLE_DEVICES=$i python scripts/12_counterfactual_rollouts.py --generator "$GEN" \
        --backend hf --dtype half --m "$M" --shard "$i" --num-shards "$NG" \
        --out "$OUT/${GEN}_s${i}.jsonl" "${EXTRA[@]}" > "$OUT/${GEN}_s${i}.stdout" 2>&1 &
      pids+=($!)
    done
    for p in "${pids[@]}"; do wait "$p"; done
    tail -n 3 "$OUT/${GEN}"_s*.stdout
    python scripts/13_commitment_analysis.py --rollouts "$OUT/${GEN}"_s*.jsonl --out "$OUT/tables_${GEN}"
    ;;
  *)
    echo "unknown mode $MODE"; exit 1;;
esac
