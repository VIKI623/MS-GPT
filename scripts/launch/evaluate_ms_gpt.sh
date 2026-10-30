#!/bin/bash
# Generate, rank and evaluate candidates, optionally with a posterior-aligned adapter.
#
# Args: BENCHMARK  ADAPTER(dir or NONE)  NSPEC  TAG  THR_LOW  THR_HIGH  [query options...]
# Env:  GPUS  (space-separated GPU ids; default "0")
#
# The reproduction entry points read thresholds, sampling settings and split
# sizes from the selected inference profile and posterior cache:
#   bash scripts/launch/reproduce_main_results.sh nplib1 ms_gpt_mlm frigid_legacy
#   bash scripts/launch/reproduce_ablation_results.sh nplib1 ms_gpt_mlm frigid_legacy
set -euo pipefail
BENCHMARK="${1:?benchmark (massspecgym|nplib1)}"
ADAPTER="${2:?adapter directory or NONE}"
NSPEC="${3:?n spectra}"
TAG="${4:?tag}"
TLO="${5:?thr_low}"
THI="${6:?thr_high}"
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"  # reduce fragmentation OOM
read -r -a GPU_ARR <<< "${GPUS:-0}"
SHARDS=${#GPU_ARR[@]}
CACHE_DIR="${MS_GPT_CACHE_DIR:?Set MS_GPT_CACHE_DIR to the selected posterior-cache profile}"
WORKERS="${WORKERS:-48}"
REFERENCE_QUERIES="${NUM_REFERENCE_QUERIES:-20}"
MODEL_ARGS=()
if [[ -n "${MS_GPT_INFERENCE_CONFIG:-}" ]]; then
  MODEL_ARGS+=(--inference_config "$MS_GPT_INFERENCE_CONFIG")
  REFERENCE_QUERIES=$(python - "$MS_GPT_INFERENCE_CONFIG" <<'PY'
import json, os, sys
config = json.load(open(sys.argv[1]))
print(os.environ.get("NUM_REFERENCE_QUERIES", config["num_queries"]))
PY
  )
else
  MODEL_ARGS+=(--mlm_checkpoint "${MS_GPT_MLM_CHECKPOINT:-checkpoints/mlm/ms_gpt_mlm.pt}"
               --mlm_config "${MS_GPT_MLM_CONFIG:-configs/mlm_pretraining.yaml}")
fi
[[ -z "${MS_GPT_MLM_CHECKPOINT:-}" ]] || MODEL_ARGS+=(--mlm_checkpoint "$MS_GPT_MLM_CHECKPOINT")
[[ -z "${MS_GPT_MLM_CONFIG:-}" ]] || MODEL_ARGS+=(--mlm_config "$MS_GPT_MLM_CONFIG")
[[ -z "${BATCH_SIZE:-}" ]] || MODEL_ARGS+=(--batch_size "$BATCH_SIZE")
PER=$(( (NSPEC+SHARDS-1)/SHARDS ))
OUT="eval_results/${BENCHMARK}_${TAG}"
mkdir -p "$OUT"
ADAPTER_ARGS=()
if [[ "$ADAPTER" != "NONE" ]]; then
  ADAPTER_ARGS=(--adapter_checkpoint "$ADAPTER")
fi
PIDS=()
SHARD_FILES=()
j=0
for g in "${GPU_ARR[@]}"; do
  START=$((j*PER))
  END=$((START+PER))
  [[ $END -le $NSPEC ]] || END=$NSPEC
  if [ $START -lt $NSPEC ]; then
    CUDA_VISIBLE_DEVICES=$g python scripts/query_mlm.py \
      --cache_dir "$CACHE_DIR" --benchmark "$BENCHMARK" --split test \
      "${MODEL_ARGS[@]}" "${ADAPTER_ARGS[@]}" \
      --threshold_low "$TLO" --threshold_high "$THI" \
      "${@:7}" \
      --start_idx "$START" --end_idx "$END" \
      --output "$OUT/post_shard${j}.jsonl" > "$OUT/shard${j}.log" 2>&1 &
    PIDS+=("$!")
    SHARD_FILES+=("$OUT/post_shard${j}.jsonl")
  fi
  j=$((j+1))
done
for pid in "${PIDS[@]}"; do
  wait "$pid" || exit 1
done
cat "${SHARD_FILES[@]}" > "$OUT/post.jsonl"
python scripts/rank_candidates.py \
  --candidates_jsonl "$OUT/post.jsonl" --cache_dir "$CACHE_DIR" \
  --benchmark "$BENCHMARK" --split test \
  --threshold_low "$TLO" --threshold_high "$THI" --num_workers "$WORKERS" \
  --num_reference_queries "$REFERENCE_QUERIES" \
  --output_jsonl "$OUT/ranked.jsonl"
python scripts/evaluate_structural_metrics.py \
  --ranked_jsonl "$OUT/ranked.jsonl" \
  --metadata "$CACHE_DIR/$BENCHMARK/meta_real_test.parquet" \
  --num_workers "$WORKERS" --no_mces \
  --output_json "$OUT/structural_metrics.json"
