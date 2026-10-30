#!/bin/bash
# Train a released posterior-aligned adapter recipe.
# Usage: train_posterior_aligned_adapter.sh BENCHMARK [decoder] [mist_variant] [training options...]
set -euo pipefail

BENCHMARK="${1:?benchmark (nplib1|massspecgym)}"
DECODER="${2:-ms_gpt_mlm}"
MIST_VARIANT="${3:-mist_official}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC:-1}"
PROFILE="checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT/$BENCHMARK/training_config.json"
CACHE_ARGS=()
[[ -z "${MS_GPT_CACHE_DIR:-}" ]] || CACHE_ARGS+=(--cache_dir "$MS_GPT_CACHE_DIR")
LABELS="${MS_GPT_LABELS:-eval_results/${BENCHMARK}_${DECODER}_${MIST_VARIANT}_frozen_train/recoverability_labels.parquet}"
torchrun --nproc_per_node="$NPROC" scripts/train_posterior_aligned_adapter.py \
  --config "$PROFILE" \
  "${CACHE_ARGS[@]}" \
  --recoverability_labels "$LABELS" \
  --output_dir "${MS_GPT_ADAPTER_OUTPUT:-checkpoints/adapter_training/$DECODER/$MIST_VARIANT/$BENCHMARK}" \
  "${@:4}"
