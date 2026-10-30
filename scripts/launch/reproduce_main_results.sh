#!/bin/bash
# Evaluate a released decoder/MIST/adapter combination on the complete test split.
# Usage: reproduce_main_results.sh PROFILE [decoder] [mist_variant]
set -euo pipefail

cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

TARGET="${1:-massspecgym}"
DECODER="${2:-ms_gpt_mlm}"
MIST_VARIANT="${3:-mist_official}"
GPU="${GPUS:-${GPU:-0}}"
WORKERS="${WORKERS:-48}"
[[ -d "checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT" ]] || {
  echo "Asset combination is not included: $DECODER/$MIST_VARIANT" >&2; exit 1;
}

evaluate_profile() {
  local profile_name="$1"
  local adapter="checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT/$profile_name"
  local profile="$adapter/inference_config.json"
  [[ -f "$profile" ]] || { echo "Asset combination is not included: $profile" >&2; return 1; }
  local profile_values
  profile_values=$(python - "$profile" "$TARGET" <<'PY'
import json, os, sys
from pathlib import Path
import pyarrow.parquet as pq

config = json.loads(Path(sys.argv[1]).read_text())
calibration = json.loads(Path(config["calibration_file"]).read_text())["benchmarks"][config["benchmark"]]
cache_dir = Path(os.environ.get("MS_GPT_CACHE_DIR", calibration["cache_dir"]))
if sys.argv[2] == "all" and os.environ.get("MS_GPT_CACHE_DIR"):
    cache_dir /= Path(calibration["cache_dir"]).name
for key in ("threshold_low", "threshold_high"):
    print(calibration[key])
print(pq.read_metadata(cache_dir / config["benchmark"] / "meta_real_test.parquet").num_rows)
print(cache_dir)
print(config.get("batch_size", 16))
print(config["benchmark"])
print(config["num_queries"])
PY
  )
  local values
  mapfile -t values <<< "$profile_values"
  local threshold_low="${values[0]}" threshold_high="${values[1]}" test_size="${values[2]}"
  local cache_dir="${values[3]}"
  local benchmark="${values[5]}"
  local tag="${DECODER}_${MIST_VARIANT}_main"

  MS_GPT_INFERENCE_CONFIG="$profile" \
  NUM_REFERENCE_QUERIES="${values[6]}" \
  MS_GPT_CACHE_DIR="$cache_dir" BATCH_SIZE="${values[4]}" GPUS="$GPU" WORKERS="$WORKERS" bash scripts/launch/evaluate_ms_gpt.sh \
    "$benchmark" "$adapter" "$test_size" "$tag" \
    "$threshold_low" "$threshold_high"
}

case "$TARGET" in
  all)
    for profile_name in nplib1 massspecgym; do
      if [[ -f "checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT/$profile_name/inference_config.json" ]]; then
        evaluate_profile "$profile_name"
      fi
    done
    ;;
  nplib1|massspecgym)
    evaluate_profile "$TARGET"
    ;;
  *)
    echo "Usage: $0 [massspecgym|nplib1|all] [ms_gpt_mlm|ms_gpt_mlm_msg_50m] [mist_official|frigid_legacy]" >&2
    exit 2
    ;;
esac
