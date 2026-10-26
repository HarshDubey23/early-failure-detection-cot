#!/usr/bin/env bash
# Hallucination-recovery run (Run R) on a Kaggle notebook with 2x T4. Plan and hypotheses:
# docs/PREREGISTRATION_RECOVERY.md. Every part has its own time budget, writes JSONL that resumes
# after an interruption, and the archive /kaggle/working/recovery_results.tar.gz is rewritten
# after each part, so a run cut short keeps everything finished so far.
#
# Priority order: Llama-2-7B-chat on RAGTruth (released, human-labelled drafts) and on the
# controlled HotpotQA set, scoring, the LLM judge, the live latency check; then Llama-3.1-8B.
#
#   bash scripts/kaggle_recovery.sh                 # mode from scripts/recovery_mode
#   SMOKE=1 bash scripts/kaggle_recovery.sh         # a few items per benchmark (~30 min)
#   MOCK=1 bash scripts/kaggle_recovery.sh          # CPU, fake models: tests the sequencing only
set -uo pipefail
RUN=results/recovery/run
ITEMS=data/recovery
mkdir -p "$RUN" "$ITEMS"
START=$(date +%s)
LIMIT=${LIMIT:-23400}            # 6.5 h for the whole run
BANK_BUDGET=${BANK_BUDGET:-9000} # cap on the sampling part, so scoring and judging keep their time
SMOKE=${SMOKE:-0}
MOCK=${MOCK:-0}
[ -f scripts/recovery_mode ] && [ "$MOCK" = 0 ] && . scripts/recovery_mode
ARCHIVE=${ARCHIVE:-/kaggle/working/recovery_results.tar.gz}
export TOKENIZERS_PARALLELISM=false

log()  { echo "[$(date '+%F %T')] $*" | tee -a "$RUN/run.log"; }
left() { echo $(( LIMIT - ($(date +%s) - START) )); }
pack() { mkdir -p "$(dirname "$ARCHIVE")"; tar -czf "$ARCHIVE" "$RUN" "$ITEMS" 2>/dev/null; log "packed $(du -h "$ARCHIVE" | cut -f1)"; }
gpu_clean() {
  [ "$MOCK" = 1 ] && return
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do kill -9 "$p" 2>/dev/null || true; done
  sleep 5
}
want() { [ -z "${PARTS:-}" ] && return 0; for w in $PARTS; do case "$1" in "$w"*) return 0;; esac; done; return 1; }
part() {  # part NAME MAX_SECONDS COMMAND...
  local name=$1 budget=$2; shift 2
  if ! want "$name"; then log "SKIP $name (not in PARTS)"; return; fi
  gpu_clean
  local b=$(( $(left) - 300 )); [ "$b" -gt "$budget" ] && b=$budget
  if [ "$b" -lt 120 ]; then log "SKIP $name: no time left"; return; fi
  log "START $name (budget ${b}s)"
  timeout --kill-after=120 "$b" bash -c "$*" >> "$RUN/$name.out" 2>&1
  log "END $name exit=$? ($(( ($(date +%s) - START) / 60 )) min elapsed)"
  tail -n "${TAIL:-8}" "$RUN/$name.out" | sed "s/^/    [$name] /" | tee -a "$RUN/run.log"
  pack
}

# Resume: when a previous run's output is attached as notebook input, continue from its files
# (every stage skips work it finds already done).
PREV=$(find /kaggle/input -path '*results/recovery/run/run.log' 2>/dev/null | head -1)
if [ -n "$PREV" ] && [ ! -s "$RUN/scores_D.jsonl" ]; then
  P=$(dirname "$PREV")
  cp -rn "$P"/. "$RUN"/ && cp -rn "$P/../../../data/recovery"/. "$ITEMS"/
  log "resumed from $P: $(cat "$RUN"/bank.*.jsonl 2>/dev/null | wc -l) bank rows"
fi

if [ "$MOCK" = 1 ]; then
  BK="--backend mock"; SBK="--backend mock"; LIM="--limit 12"; M="--m 4 --m-curve 4 --max-clean-curves 3"
  LIVE="--n-flagged 2 --n-unflagged 1"
elif [ "$SMOKE" = 1 ]; then
  BK=""; SBK=""; LIM="--limit 9"; M="--m 2 --m-curve 2 --max-clean-curves 3"; LIVE="--n-flagged 2 --n-unflagged 1"
else
  BK=""; SBK=""; LIM=""; M="--m 8 --m-curve 8 --max-clean-curves 40"; LIVE="--n-flagged 12 --n-unflagged 4"
fi
L2="$ITEMS/items.ragtruth.llama-2-7b-chat.jsonl $ITEMS/items.hotpot.llama-2-7b-chat.jsonl"
L3="$ITEMS/items.ragtruth.llama-3.1-8b-instruct.jsonl $ITEMS/items.hotpot.llama-3.1-8b-instruct.jsonl"
G="python scripts/32_generate.py --out-dir $RUN $BK"
SC="python scripts/33_score.py --run-dir $RUN $SBK"

# 0. dependencies, data, items, detector check
part setup 1800 "
  set -e
  if [ '$MOCK' = 0 ]; then
    pip install -q vllm 2>&1 | tail -n 2
    pip uninstall -y -q torchaudio || true
    pip install -q --no-deps lettucedetect || true
    python -c 'import vllm, torch, transformers; print(vllm.__version__, torch.__version__, transformers.__version__, torch.cuda.device_count())'
  fi
  [ -d data/external/RAGTruth ] || git clone --depth 1 https://github.com/ParticleMedia/RAGTruth data/external/RAGTruth
  for GEN in llama-2-7b-chat llama-3.1-8b-instruct; do
    python scripts/31_build_items.py --benchmark ragtruth --generator \$GEN $LIM --out-dir $ITEMS
    python scripts/31_build_items.py --benchmark hotpot --generator \$GEN --n-questions 150 $LIM --out-dir $ITEMS $( [ "$MOCK" = 1 ] && echo --synthetic )
  done
  if [ '$MOCK' = 0 ]; then CUDA_VISIBLE_DEVICES=0 python scripts/37_check_detector.py --n 60 --out $RUN/detector_check.json; fi"

# 1. Llama-2-7B-chat: drafts for HotpotQA (RAGTruth drafts are the released, human-labelled ones)
part l2_drafts 1800 "$G --items $L2 --generator llama-2-7b-chat --stage drafts"
part l2_flags 1200 "CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L2 --kinds draft"
part l2_bank $BANK_BUDGET "$G --items $L2 --generator llama-2-7b-chat --stage bank repair cove --flags $RUN/scores_D.jsonl $M"
part l2_score 3600 "
  CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L2 --stage 1 &
  CUDA_VISIBLE_DEVICES=1 $SC --scorer E2 --items $L2 --stage 1 &
  wait
  CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L2 --stage 2
  CUDA_VISIBLE_DEVICES=1 $SC --scorer E2 --items $L2 --stage 2"
part l2_judge 5400 "$SC --scorer E1 --items $L2 --stage 1 2"
part live 1800 "python scripts/36_live.py --items $ITEMS/items.ragtruth.llama-2-7b-chat.jsonl --generator llama-2-7b-chat --run-dir $RUN $LIVE $( [ "$MOCK" = 1 ] && echo --backend mock )"

# 2. Llama-3.1-8B-Instruct (secondary: modern model, drafts generated here, judged only)
part l3_drafts 2400 "$G --items $L3 --generator llama-3.1-8b-instruct --stage drafts"
part l3_flags 1200 "CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L3 --kinds draft"
part l3_bank 7200 "$G --items $L3 --generator llama-3.1-8b-instruct --stage bank repair cove --flags $RUN/scores_D.jsonl $M"
part l3_score 2400 "
  CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L3 --stage 1 &
  CUDA_VISIBLE_DEVICES=1 $SC --scorer E2 --items $L3 --stage 1 &
  wait
  CUDA_VISIBLE_DEVICES=0 $SC --scorer D --items $L3 --stage 2
  CUDA_VISIBLE_DEVICES=1 $SC --scorer E2 --items $L3 --stage 2"
part l3_judge 3600 "$SC --scorer E1 --items $L3 --stage 1 2"

log "ALL DONE"
pack
