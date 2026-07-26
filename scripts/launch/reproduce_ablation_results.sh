#!/bin/bash
# Reproduce the component ablations on one A6000 Ada GPU.
# Usage: GPU=0 bash scripts/launch/reproduce_ablation_results.sh [all|nplib1|massspecgym]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

TARGET="${1:-all}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-48}"

evaluate_without_adapter() {
  local benchmark="$1" test_size="$2" threshold_low="$3" threshold_high="$4"
  GPUS="$GPU" bash scripts/launch/evaluate_ms_gpt.sh \
    "$benchmark" NONE "$test_size" no_adapter "$threshold_low" "$threshold_high"
  python scripts/evaluate_structural_metrics.py \
    --candidates_jsonl "eval_results/${benchmark}_no_adapter/post.jsonl" \
    --cache_dir data/posterior_cache --benchmark "$benchmark" --split test \
    --num_workers "$WORKERS" --no_mces \
    --output_json "eval_results/${benchmark}_no_adapter/structural_metrics.json"
}

evaluate_point_query() {
  local benchmark="$1" test_size="$2" threshold="$3" tag="$4"
  local output_dir="eval_results/${benchmark}_${tag}"
  mkdir -p "$output_dir"

  CUDA_VISIBLE_DEVICES="$GPU" python scripts/query_mlm.py \
    --cache_dir data/posterior_cache --benchmark "$benchmark" --split test \
    --mlm_checkpoint checkpoints/mlm/ms_gpt_mlm.pt \
    --mlm_config configs/mlm_pretraining.yaml \
    --query_method point --threshold "$threshold" \
    --num_queries 1 --samples_per_query 100 --batch_size 16 \
    --start_idx 0 --end_idx "$test_size" \
    --output "$output_dir/post.jsonl"

  python scripts/rank_candidates.py \
    --candidates_jsonl "$output_dir/post.jsonl" \
    --cache_dir data/posterior_cache --benchmark "$benchmark" --split test \
    --num_workers "$WORKERS" \
    --output_json "$output_dir/exact_match.json"
  python scripts/evaluate_structural_metrics.py \
    --candidates_jsonl "$output_dir/post.jsonl" \
    --cache_dir data/posterior_cache --benchmark "$benchmark" --split test \
    --num_workers "$WORKERS" --no_mces \
    --output_json "$output_dir/structural_metrics.json"
}

run_nplib1() {
  evaluate_without_adapter nplib1 672 0.062 0.292
  evaluate_point_query nplib1 672 0.222 density_match
  evaluate_point_query nplib1 672 0.340 validation_f1
  evaluate_point_query nplib1 672 0.377 validation_cosine
}

run_massspecgym() {
  evaluate_without_adapter massspecgym 17556 0.069 0.274
  evaluate_point_query massspecgym 17556 0.215 density_match
  evaluate_point_query massspecgym 17556 0.257 validation_f1
  evaluate_point_query massspecgym 17556 0.269 validation_cosine
}

case "$TARGET" in
  all) run_nplib1; run_massspecgym ;;
  nplib1) run_nplib1 ;;
  massspecgym) run_massspecgym ;;
  *)
    echo "Usage: $0 [all|nplib1|massspecgym]" >&2
    exit 2
    ;;
esac
