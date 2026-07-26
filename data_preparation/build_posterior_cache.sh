#!/bin/bash
# Build the MIST posterior cache: for each (benchmark, split), run the benchmark-paired
# MIST encoder over the benchmark spectra to produce pi_raw in [0,1]^4096 (the
# spectrum-induced fingerprint posterior) plus aligned ground-truth labels.
# Output: data/posterior_cache/{benchmark}/{pi_raw,z_star,meta}_real_{split}.{npy,parquet}
#
# Env: MS_GPT_MIST_MASSSPECGYM_CKPT, MS_GPT_MIST_NPLIB1_CKPT,
#      MS_GPT_MASSSPECGYM_DATA, MS_GPT_NPLIB1_DATA, GPUS (default "0")
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
GPUS="${GPUS:-0}"; GPU_ARR=($GPUS); NSH=${#GPU_ARR[@]}

build() {  # benchmark  split  spectrum_parquet_dir
  local benchmark="$1" split="$2" pq="$3"
  echo ">>> posterior cache: $benchmark/$split  ($pq)"
  j=0
  for g in "${GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python scripts/build_posterior_cache.py \
      --benchmark "$benchmark" --split "$split" \
      --spectrum_parquet "$pq" --cache_dir data/posterior_cache \
      --shard_idx "$j" --n_shards "$NSH" --gpu 0 --batch_size 32 &
    j=$((j+1))
  done
  wait
  python scripts/merge_posterior_cache.py \
    --cache_dir data/posterior_cache --benchmark "$benchmark" \
    --split "$split" --cleanup
}

# MassSpecGym: train (for density band + LoRA labels), val, test
for split in train val test; do
  build massspecgym "$split" "${MS_GPT_MASSSPECGYM_DATA:?set MS_GPT_MASSSPECGYM_DATA}/$split"
done
# NPLIB1
for split in train val test; do
  build nplib1 "$split" "${MS_GPT_NPLIB1_DATA:?set MS_GPT_NPLIB1_DATA}/$split"
done

echo "Done. Caches under data/posterior_cache/{massspecgym,nplib1}/"
