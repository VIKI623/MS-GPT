#!/bin/bash
# Canonical test evaluation over a density-calibrated posterior-query band,
# optionally with a posterior-aligned adapter. Reference hardware: one A6000 Ada.
#
# Args: BENCHMARK  ADAPTER(dir or NONE)  NSPEC  TAG  THR_LOW  THR_HIGH
# Env:  GPUS  (space-separated GPU ids; default "0")
#
# Example (NPLIB1 shared MLM, full test = 672 spectra, canonical density band):
#   GPUS="0" bash scripts/launch/evaluate_ms_gpt.sh \
#       nplib1 NONE 672 mlm 0.062 0.292
# Example (NPLIB1 canonical posterior-aligned adapter, step3000):
#   GPUS="0" bash scripts/launch/evaluate_ms_gpt.sh \
#       nplib1 checkpoints/posterior_aligned_adapters/nplib1 672 adapted 0.062 0.292
set -euo pipefail
BENCHMARK="${1:?benchmark (massspecgym|nplib1)}"; ADAPTER="${2:?adapter directory or NONE}"; NSPEC="${3:?n spectra}"
TAG="${4:?tag}"; TLO="${5:?thr_low}"; THI="${6:?thr_high}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # reduce fragmentation OOM
GPUS="${GPUS:-0}"; GPU_ARR=($GPUS); SHARDS=${#GPU_ARR[@]}
PER=$(( (NSPEC+SHARDS-1)/SHARDS ))
OUT="eval_results/${BENCHMARK}_${TAG}"; mkdir -p "$OUT"
ADAPTER_ARG=""; [ "$ADAPTER" != "NONE" ] && ADAPTER_ARG="--adapter_checkpoint $ADAPTER"
j=0
for g in "${GPU_ARR[@]}"; do
  START=$((j*PER)); END=$((START+PER)); [ $END -gt $NSPEC ] && END=$NSPEC
  if [ $START -lt $NSPEC ]; then
    CUDA_VISIBLE_DEVICES=$g python scripts/query_mlm.py \
      --cache_dir data/posterior_cache --benchmark "$BENCHMARK" --split test \
      --mlm_checkpoint checkpoints/mlm/ms_gpt_mlm.pt \
      --mlm_config configs/mlm_pretraining.yaml $ADAPTER_ARG \
      --query_method density_calibrated_band --num_queries 20 --samples_per_query 5 --batch_size 16 --threshold_low $TLO --threshold_high $THI \
      --start_idx "$START" --end_idx "$END" \
      --output "$OUT/post_shard${j}.jsonl" > "$OUT/shard${j}.log" 2>&1 &
  fi
  j=$((j+1))
done
wait
cat "$OUT"/post_shard*.jsonl > "$OUT/raw.jsonl"
python scripts/repair_jsonl.py "$OUT/raw.jsonl" "$OUT/post.jsonl"
python - "$OUT/post.jsonl" <<'PY'
import json,sys
p=sys.argv[1];c=[]
for l in open(p):
  try:
    o=json.loads(l)
    if isinstance(o,dict) and "cache_row" in o: c.append(l)
  except: pass
open(p,"w").writelines(c)
PY
python scripts/rank_candidates.py --candidates_jsonl "$OUT/post.jsonl" --cache_dir data/posterior_cache --benchmark "$BENCHMARK" --split test --num_workers 48 --output_json "$OUT/metrics_frequency_posterior_similarity.json" >/dev/null 2>&1
python - "$OUT/metrics_frequency_posterior_similarity.json" "$TAG" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]));f=d["frequency_posterior_similarity"];n=d["n_spec"]
print("RESULT %s n=%d T1=%.2f T5=%.2f T10=%.2f T20=%.2f"%(sys.argv[2],n,f["top1"]/n*100,f["top5"]/n*100,f["top10"]/n*100,f["top20"]/n*100))
PY
