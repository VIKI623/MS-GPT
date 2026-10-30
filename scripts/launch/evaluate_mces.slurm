#!/bin/bash
#SBATCH --job-name=ms-gpt-mces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.log

# Usage:
#   sbatch scripts/launch/evaluate_mces.slurm BENCHMARK RANKED_JSONL [OUTPUT_JSON]
# Cluster-specific account and partition settings are intentionally omitted.
set -euo pipefail

BENCHMARK="${1:?benchmark (nplib1|massspecgym)}"
RANKED="${2:?ranked JSONL produced by reproduce_main_results.sh}"
OUTPUT="${3:-$(dirname "$RANKED")/mces_metrics.json}"
WORKERS="${SLURM_CPUS_PER_TASK:-48}"
DECODER="${MS_GPT_DECODER:-ms_gpt_mlm}"
MIST_VARIANT="${MS_GPT_MIST_VARIANT:-mist_official}"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"
PROFILE="checkpoints/posterior_aligned_adapters/$DECODER/$MIST_VARIANT/$BENCHMARK/inference_config.json"
CACHE_DIR="${MS_GPT_CACHE_DIR:-$(python - "$PROFILE" <<'PY'
import json, sys
config = json.load(open(sys.argv[1]))
calibration = json.load(open(config["calibration_file"]))["benchmarks"][config["benchmark"]]
print(calibration["cache_dir"])
PY
)}"

python scripts/evaluate_structural_metrics.py \
  --ranked_jsonl "$RANKED" \
  --metadata "$CACHE_DIR/$BENCHMARK/meta_real_test.parquet" \
  --num_workers "$WORKERS" --mces \
  --output_json "$OUTPUT"
