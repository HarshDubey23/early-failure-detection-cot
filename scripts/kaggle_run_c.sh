#!/usr/bin/env bash
# Run C: confirmatory and robustness run on a Kaggle notebook with 2x T4 (see
# docs/PREREGISTRATION_RUN_C.md). Every part has its own time budget; the results archive
# /kaggle/working/run_c_results.tar.gz is rewritten after each part, so a run that is cut
# short still leaves everything finished so far.
#
#   bash scripts/kaggle_run_c.sh            # full run (about 10-11 h)
#   SMOKE=1 bash scripts/kaggle_run_c.sh    # every part on a few traces (about 25 min)
set -uo pipefail
OUT=results/run_c
mkdir -p "$OUT"
START=$(date +%s)
LIMIT=${LIMIT:-41400}            # 11.5 h for the whole run; Kaggle stops sessions at 12 h
SMOKE=${SMOKE:-0}
ARCHIVE=/kaggle/working/run_c_results.tar.gz
export TOKENIZERS_PARALLELISM=false

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$OUT/run_c.log"; }
left() { echo $(( LIMIT - ($(date +%s) - START) )); }
pack() { tar -czf "$ARCHIVE" "$OUT" 2>/dev/null; log "packed $(du -h "$ARCHIVE" | cut -f1)"; }
part() {  # part NAME MAX_SECONDS COMMAND...
  local name=$1 budget=$2; shift 2
  local b=$(( $(left) - 600 )); [ "$b" -gt "$budget" ] && b=$budget
  if [ "$b" -lt 300 ]; then log "SKIP $name: no time left"; return; fi
  log "START $name (budget ${b}s)"
  timeout --kill-after=120 "$b" bash -c "$*" >> "$OUT/$name.out" 2>&1
  log "END $name exit=$? ($(( ($(date +%s) - START) / 60 )) min elapsed)"
  pack
}

if [ "$SMOKE" = 1 ]; then
  LIM="--limit 2"; LIM12="--max-per-group 1"; M=4; MREP=4
else
  LIM=""; LIM12=""; M=16; MREP=16
fi

# 0. Dependencies and data
part setup 1500 '
  pip install -q "math-verify[antlr4_13_2]" || pip install -q math-verify
  pip install -q vllm 2>&1 | tail -n 2
  pip uninstall -y -q torchaudio || true
  python scripts/12a_prepare_processbench.py
  python -c "import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.device_count())"
  nvidia-smi --query-gpu=index,name,memory.total --format=csv'

# 1. Generator step signals (NLL, entropy, exact token counts): one forward pass per trace
part signals_gen 3000 "
  CUDA_VISIBLE_DEVICES=0 python scripts/14_step_signals.py --mode generator --generators Qwen2.5-1.5B-Instruct $LIM --out $OUT/signals_gen.Qwen2.5-1.5B-Instruct.jsonl &
  CUDA_VISIBLE_DEVICES=1 python scripts/14_step_signals.py --mode generator --generators Qwen2-1.5B-Instruct $LIM --out $OUT/signals_gen.Qwen2-1.5B-Instruct.jsonl &
  wait
  python scripts/14_step_signals.py --mode generator --generators Qwen2.5-Math-7B-Instruct $LIM --out $OUT/signals_gen.Qwen2.5-Math-7B-Instruct.jsonl
  python scripts/14_step_signals.py --mode generator --generators Llama-3.1-8B-Instruct $LIM --out $OUT/signals_gen.Llama-3.1-8B-Instruct.jsonl"

# 2. Process reward model step scores: the four rolled-out generators first, then all of ProcessBench
part signals_prm 3000 "
  python scripts/14_step_signals.py --mode prm --generators Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct Qwen2.5-Math-7B-Instruct Llama-3.1-8B-Instruct $LIM --out $OUT/signals_prm.jsonl
  python scripts/14_step_signals.py --mode prm --generators all $LIM --out $OUT/signals_prm.jsonl"

# 3. Forced-answer probes at every prefix (answer curve a_k)
part probes 4800 "
  CUDA_VISIBLE_DEVICES=0 python scripts/15_answer_probe.py --generator Qwen2.5-1.5B-Instruct $LIM --out $OUT/probe.Qwen2.5-1.5B-Instruct.jsonl &
  CUDA_VISIBLE_DEVICES=1 python scripts/15_answer_probe.py --generator Qwen2-1.5B-Instruct $LIM --out $OUT/probe.Qwen2-1.5B-Instruct.jsonl &
  wait
  python scripts/15_answer_probe.py --generator Qwen2.5-Math-7B-Instruct --tp 2 --max-clean 60 $LIM --out $OUT/probe.Qwen2.5-Math-7B-Instruct.jsonl
  python scripts/15_answer_probe.py --generator Llama-3.1-8B-Instruct --tp 2 --max-clean 60 $LIM --out $OUT/probe.Llama-3.1-8B-Instruct.jsonl"

# 4. Three new independent sampling batches for both 1.5B models (same traces as Run B)
for SEED in 1001 2002 3003; do
  part "rep_b$SEED" 2700 "
    for G in Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct; do
      for S in 0 1; do
        CUDA_VISIBLE_DEVICES=\$S python scripts/12_counterfactual_rollouts.py --generator \$G --backend vllm --dtype half \
          --m $MREP --max-tokens 1536 --seed $SEED --seed-mode hash --select-seed 42 --batch-id b$SEED \
          --shard \$S --num-shards 2 $LIM12 --out $OUT/rollouts.\$G.b$SEED.s\$S.jsonl > $OUT/rollouts.\$G.b$SEED.s\$S.stdout 2>&1 &
      done
      wait
    done"
done

# 5. New model family: Llama-3.1-8B-Instruct on its own ProcessBench traces (fatal, recovered, 60 clean)
part llama $(( $(left) - 900 )) "
  python scripts/12_counterfactual_rollouts.py --generator Llama-3.1-8B-Instruct --backend vllm --dtype half --tp 2 \
    --m $M --max-tokens 1024 --max-clean 60 --seed 4004 --seed-mode hash --select-seed 42 --batch-id b4004 $LIM12 \
    --out $OUT/rollouts.Llama-3.1-8B-Instruct.b4004.jsonl"

log "ALL DONE"
pack
