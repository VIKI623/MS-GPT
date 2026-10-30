#!/bin/bash
# Build the MIST posterior cache: for each (benchmark, split), run the benchmark-paired
# MIST encoder over the benchmark spectra to produce pi_raw in [0,1]^4096 (the
# spectrum-induced fingerprint posterior) plus aligned ground-truth labels.
# Usage: build_posterior_cache.sh [benchmark] [mist_variant]
# Benchmarks: massspecgym, nplib1, all.
# Peak defaults: mist_official=10, frigid_legacy=40; MAX_PEAKS overrides the default.
#
# Env: MS_GPT_MIST_MASSSPECGYM_CKPT, MS_GPT_MIST_NPLIB1_CKPT,
#      MS_GPT_MIST_OFFICIAL_MASSSPECGYM_CKPT, MS_GPT_MIST_OFFICIAL_NPLIB1_CKPT,
#      MS_GPT_MASSSPECGYM_DATA, MS_GPT_NPLIB1_DATA,
#      MAX_PEAKS, GPUS (default "0")
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
read -r -a GPU_ARR <<< "${GPUS:-0}"
NSH=${#GPU_ARR[@]}
TARGET="${1:-all}"
case "$TARGET" in
  massspecgym|nplib1|all) ;;
  *) echo "Unknown benchmark: $TARGET" >&2; exit 2 ;;
esac
MIST_VARIANT="${2:-mist_official}"
case "$MIST_VARIANT" in
  mist_official) MIST_IMPLEMENTATION=mist_official; MAX_PEAKS="${MAX_PEAKS:-10}" ;;
  frigid_legacy) MIST_IMPLEMENTATION=mist_frigid_legacy; MAX_PEAKS="${MAX_PEAKS:-40}" ;;
  *) echo "Unknown MIST checkpoint family: $MIST_VARIANT" >&2; exit 2 ;;
esac
build() {  # benchmark  split  spectrum_parquet_dir
  local benchmark="$1" split="$2" pq="$3"
  local j=0 g pid
  local pids=()
  echo ">>> posterior cache: $benchmark/$split  ($pq)"
  for g in "${GPU_ARR[@]}"; do
    CUDA_VISIBLE_DEVICES=$g python scripts/build_posterior_cache.py \
      --benchmark "$benchmark" --split "$split" \
      --spectrum_parquet "$pq" --cache_dir "$CACHE_DIR" \
      --mist_variant "$MIST_VARIANT" --mist_checkpoint "$checkpoint" \
      --max_peaks "$MAX_PEAKS" \
      --shard_idx "$j" --n_shards "$NSH" --gpu 0 --batch_size "${BATCH_SIZE:-32}" &
    pids+=("$!")
    j=$((j+1))
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || return 1
  done
  python scripts/merge_posterior_cache.py \
    --cache_dir "$CACHE_DIR" --benchmark "$benchmark" \
    --split "$split" --n_shards "$NSH" --cleanup
}

for benchmark in massspecgym nplib1; do
  [[ "$TARGET" == "all" || "$TARGET" == "$benchmark" ]] || continue
  input_profile="annotated_top$MAX_PEAKS"
  CACHE_DIR="${MS_GPT_CACHE_DIR:-data/posterior_cache/$MIST_IMPLEMENTATION/$input_profile}"
  if [[ "$TARGET" == "all" && -n "${MS_GPT_CACHE_DIR:-}" ]]; then
    CACHE_DIR="$MS_GPT_CACHE_DIR/$input_profile"
  fi
  checkpoint="checkpoints/mist/$MIST_VARIANT/mist_${benchmark}.pt"
  checkpoint_env="MS_GPT_MIST_${benchmark^^}_CKPT"
  if [[ "$MIST_VARIANT" == "mist_official" ]]; then
    checkpoint_env="MS_GPT_MIST_OFFICIAL_${benchmark^^}_CKPT"
  fi
  checkpoint="${!checkpoint_env:-$checkpoint}"
  if [[ ! -f "$checkpoint" ]]; then
    if [[ "$TARGET" == "all" ]]; then continue; fi
    echo "Checkpoint is not included: $checkpoint" >&2
    exit 1
  fi
  data_env="MS_GPT_${benchmark^^}_DATA"
  data_dir="${!data_env:-data/benchmarks/$benchmark}"
  for split in train val test; do
    build "$benchmark" "$split" "$data_dir/$split"
  done
  echo "Done. $benchmark caches under $CACHE_DIR/"
done
