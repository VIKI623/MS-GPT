#!/bin/bash
#SBATCH --job-name=ms-gpt-mces
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.log

# Usage:
#   sbatch scripts/launch/evaluate_mces.slurm BENCHMARK CANDIDATES_JSONL [OUTPUT_JSON]
# Cluster-specific account and partition settings are intentionally omitted.
set -euo pipefail

BENCHMARK="${1:?benchmark (nplib1|massspecgym)}"
CANDIDATES="${2:?candidate JSONL produced by reproduce_main_results.sh}"
OUTPUT="${3:-eval_results/${BENCHMARK}_main/mces_metrics.json}"
WORKERS="${SLURM_CPUS_PER_TASK:-48}"

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

python scripts/evaluate_structural_metrics.py \
  --candidates_jsonl "$CANDIDATES" \
  --cache_dir data/posterior_cache \
  --benchmark "$BENCHMARK" --split test \
  --num_workers "$WORKERS" --mces \
  --output_json "$OUTPUT"
