#!/bin/bash
# Train the paper's posterior-aligned adapter.
# Usage: train_posterior_aligned_adapter.sh BENCHMARK MAX_STEPS THRESHOLD_LOW THRESHOLD_HIGH
set -euo pipefail

BENCHMARK="${1:?benchmark (nplib1|massspecgym)}"
MAX_STEPS="${2:?maximum optimizer steps}"
THRESHOLD_LOW="${3:?density-band lower threshold}"
THRESHOLD_HIGH="${4:?density-band upper threshold}"

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

NPROC="${NPROC:-8}"
torchrun --nproc_per_node="$NPROC" scripts/train_posterior_aligned_adapter.py \
  --benchmark "$BENCHMARK" \
  --mlm_checkpoint checkpoints/mlm/ms_gpt_mlm.pt \
  --mlm_config configs/mlm_pretraining.yaml \
  --cache_dir data/posterior_cache \
  --recoverability_labels "data/posterior_cache/$BENCHMARK/recoverability_labels.parquet" \
  --threshold_low "$THRESHOLD_LOW" \
  --threshold_high "$THRESHOLD_HIGH" \
  --rank 4 --alpha 8 --dropout 0.05 \
  --batch_size 8 --max_steps "$MAX_STEPS" --learning_rate 1e-4 \
  --warmup_steps 150 --num_workers 4 --save_interval 500 \
  --output_dir "checkpoints/posterior_aligned_adapters/$BENCHMARK" --seed 42
