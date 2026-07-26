#!/bin/bash
# Reproduce the released MS-GPT main results on one A6000 Ada GPU.
# Usage: GPU=0 bash scripts/launch/reproduce_main_results.sh [all|nplib1|massspecgym]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

TARGET="${1:-all}"
GPU="${GPU:-0}"
WORKERS="${WORKERS:-48}"

evaluate_benchmark() {
  local benchmark="$1"
  local adapter="$2"
  local test_size="$3"
  local threshold_low="$4"
  local threshold_high="$5"
  local output_dir="eval_results/${benchmark}_main"

  GPUS="$GPU" bash scripts/launch/evaluate_ms_gpt.sh \
    "$benchmark" "$adapter" "$test_size" main \
    "$threshold_low" "$threshold_high"

  python scripts/evaluate_structural_metrics.py \
    --candidates_jsonl "$output_dir/post.jsonl" \
    --cache_dir data/posterior_cache \
    --benchmark "$benchmark" --split test \
    --num_workers "$WORKERS" --no_mces \
    --output_json "$output_dir/structural_metrics.json"
}

case "$TARGET" in
  all)
    evaluate_benchmark nplib1 checkpoints/posterior_aligned_adapters/nplib1 \
      672 0.062 0.292
    evaluate_benchmark massspecgym checkpoints/posterior_aligned_adapters/massspecgym \
      17556 0.069 0.274
    ;;
  nplib1)
    evaluate_benchmark nplib1 checkpoints/posterior_aligned_adapters/nplib1 \
      672 0.062 0.292
    ;;
  massspecgym)
    evaluate_benchmark massspecgym checkpoints/posterior_aligned_adapters/massspecgym \
      17556 0.069 0.274
    ;;
  *)
    echo "Usage: $0 [all|nplib1|massspecgym]" >&2
    exit 2
    ;;
esac
