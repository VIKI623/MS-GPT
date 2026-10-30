#!/bin/bash
# Evaluate Base and train/val-calibrated point controls for a released profile.
# Usage: reproduce_ablation_results.sh [all|nplib1|massspecgym] [decoder] [mist_variant]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
TARGET="${1:-all}"
DECODER="${2:-ms_gpt_mlm}"
MIST_VARIANT="${3:-mist_official}"
GPU="${GPUS:-${GPU:-0}}"
WORKERS="${WORKERS:-48}"
[[ -d "checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT" ]] || {
  echo "Asset combination is not included: $DECODER/$MIST_VARIANT" >&2; exit 1;
}
case "$TARGET" in
  all|nplib1|massspecgym) ;;
  *) echo "Unknown benchmark: $TARGET" >&2; exit 2 ;;
esac

for benchmark in nplib1 massspecgym; do
  [[ "$TARGET" == "all" || "$TARGET" == "$benchmark" ]] || continue
  profile="checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT/$benchmark/inference_config.json"
  if [[ ! -f "$profile" ]]; then
    [[ "$TARGET" == "all" ]] && continue
    echo "Asset combination is not included: $profile" >&2
    exit 1
  fi
  profile_values=$(python - "$profile" "$TARGET" <<'PY'
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq
config = json.loads(Path(sys.argv[1]).read_text())
calibration = json.loads(Path(config["calibration_file"]).read_text())["benchmarks"][config["benchmark"]]
point_keys = ("density_match_threshold", "validation_f1_threshold", "validation_cosine_threshold")
if any(key not in calibration for key in point_keys):
    raise SystemExit("Run data_preparation/compute_density_calibration.sh for this profile before point-query ablations.")
cache = Path(os.environ.get("MS_GPT_CACHE_DIR", calibration["cache_dir"]))
if sys.argv[2] == "all" and os.environ.get("MS_GPT_CACHE_DIR"):
    cache /= Path(calibration["cache_dir"]).name
for key in ("threshold_low", "threshold_high"):
    print(calibration[key])
print(cache)
print(pq.read_metadata(cache / config["benchmark"] / "meta_real_test.parquet").num_rows)
print(config.get("batch_size", 16))
print(config["num_queries"] * config["samples_per_query"])
print(config["num_queries"])
for key in point_keys:
    print(calibration[key])
PY
  )
  mapfile -t values <<< "$profile_values"
  cache="${values[2]}"
  points=("${values[7]}" "${values[8]}" "${values[9]}")

  names=(base density_match validation_f1 validation_cosine)
  for i in "${!names[@]}"; do
    query_args=()
    if (( i > 0 )); then
      query_args=(--query_method point --threshold "${points[$((i-1))]}"
                  --num_queries 1 --samples_per_query "${values[5]}")
    fi
    MS_GPT_INFERENCE_CONFIG="$profile" NUM_REFERENCE_QUERIES="${values[6]}" \
    MS_GPT_CACHE_DIR="$cache" BATCH_SIZE="${values[4]}" GPUS="$GPU" WORKERS="$WORKERS" \
      bash scripts/launch/evaluate_ms_gpt.sh \
        "$benchmark" NONE "${values[3]}" "${DECODER}_${MIST_VARIANT}_${names[$i]}" \
        "${values[0]}" "${values[1]}" "${query_args[@]}"
  done
done
