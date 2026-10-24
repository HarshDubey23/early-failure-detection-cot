#!/usr/bin/env bash
# Analyse Run C after extracting run_c_results.tar.gz into the repository root (results/run_c/).
# Implements the analyses fixed in docs/PREREGISTRATION_RUN_C.md. CPU only.
set -euo pipefail
R=results/run_c
C=results/commitment
PY=${PY:-python3}
QWEN_SMALL="Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct"
ALL="Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct Qwen2.5-Math-7B-Instruct Llama-3.1-8B-Instruct"

# Confirmatory C1-C4: Llama-3.1-8B-Instruct (new family, new questions)
if ls $R/rollouts.Llama-3.1-8B-Instruct.*.jsonl >/dev/null 2>&1; then
  $PY scripts/13_commitment_analysis.py --rollouts $R/rollouts.Llama-3.1-8B-Instruct.*.jsonl \
      --out $C/tables_Llama-3.1-8B-Instruct
  for d in 0.0625 0.25; do
    $PY scripts/13_commitment_analysis.py --rollouts $R/rollouts.Llama-3.1-8B-Instruct.*.jsonl \
        --primary-delta $d --out $C/tables_Llama-3.1-8B-Instruct_delta$d > /dev/null
  done
  $PY scripts/16_robustness.py --gen Llama-3.1-8B-Instruct --rollouts $R/rollouts.Llama-3.1-8B-Instruct.*.jsonl \
      --out $C/robustness_Llama-3.1-8B-Instruct
fi

# R1: the exploratory restart-vs-first-error effect on the NEW batches only (Run B samples excluded)
NEW=()
for g in $QWEN_SMALL; do
  files=$(ls $R/rollouts.$g.b*.s*.jsonl 2>/dev/null || true)
  [ -z "$files" ] && continue
  NEW+=($files)
  $PY scripts/13_commitment_analysis.py --rollouts $files --out $C/tables_${g}_newbatches
  # all batches together (Run B + new): precision m = 64
  $PY scripts/13_commitment_analysis.py --rollouts $C/${g}_s*.jsonl $files --out $C/tables_${g}_allbatches
  # R2/R3: reliability between independent batches and decisive-step posterior at m = 64
  $PY scripts/16_robustness.py --gen $g --rollouts $C/${g}_s*.jsonl $files --out $C/robustness_${g}_allbatches
done
if [ ${#NEW[@]} -gt 0 ]; then
  $PY scripts/13_commitment_analysis.py --rollouts "${NEW[@]}" --out $C/tables_pooled_1p5b_newbatches
fi
for d in 0.0625 0.25; do
  for g in Qwen2.5-1.5B-Instruct Qwen2-1.5B-Instruct Qwen2.5-Math-7B-Instruct; do
    $PY scripts/13_commitment_analysis.py --rollouts $C/${g}_s*.jsonl --primary-delta $d \
        --out $C/tables_${g}_delta$d > /dev/null
  done
done

# Practical policies at a matched budget (confirmatory on Llama, exploratory on Qwen)
ROLL=$(ls $C/*_s*.jsonl $R/rollouts.*.jsonl 2>/dev/null)
$PY scripts/17_policies.py --gens $ALL --rollouts $ROLL \
    --signals-gen $(ls $R/signals_gen.*.jsonl 2>/dev/null) --signals-prm $(ls $R/signals_prm.jsonl 2>/dev/null) \
    --out $C/policies

# PRM implementation check and answer curves
[ -f $R/signals_prm.jsonl ] && $PY scripts/19_prm_check.py --prm $R/signals_prm.jsonl --out $C/prm_check.json
ls $R/probe.*.jsonl >/dev/null 2>&1 && $PY scripts/20_answer_curves.py --probes $R/probe.*.jsonl --out $C/answer_curves
echo "Run C analysis done"
