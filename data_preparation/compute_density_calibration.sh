#!/bin/bash
# Calibrate a posterior cache on train, and its point-query controls on val.
# Usage: compute_density_calibration.sh [benchmark] [mist_variant] [kappa_low] [kappa_high]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

TARGET="${1:-all}"
MIST_VARIANT="${2:-mist_official}"
KAPPA_LOW="${3:-0.85}"
KAPPA_HIGH="${4:-1.25}"
CONFIG="density_calibration/$MIST_VARIANT/canonical_band.json"
if (( $# > 4 )); then
  echo "Usage: $0 [benchmark] [mist_variant] [kappa_low] [kappa_high]" >&2
  exit 2
fi
case "$TARGET" in
  all|nplib1|massspecgym) ;;
  *) echo "Unknown benchmark: $TARGET" >&2; exit 2 ;;
esac
case "$MIST_VARIANT" in
  mist_official) MIST_IMPLEMENTATION=mist_official; MAX_PEAKS="${MAX_PEAKS:-10}" ;;
  frigid_legacy) MIST_IMPLEMENTATION=mist_frigid_legacy; MAX_PEAKS="${MAX_PEAKS:-40}" ;;
  *) echo "Unknown MIST checkpoint family: $MIST_VARIANT" >&2; exit 2 ;;
esac
input_profile="annotated_top$MAX_PEAKS"
CACHE_DIR="${MS_GPT_CACHE_DIR:-data/posterior_cache/$MIST_IMPLEMENTATION/$input_profile}"
if [[ "$TARGET" == "all" && -n "${MS_GPT_CACHE_DIR:-}" ]]; then
  CACHE_DIR="$MS_GPT_CACHE_DIR/$input_profile"
fi
for benchmark in nplib1 massspecgym; do
  [[ "$TARGET" == "all" || "$TARGET" == "$benchmark" ]] || continue
  python scripts/compute_density_calibration.py \
    --config "$CONFIG" --benchmark "$benchmark" --cache_dir "$CACHE_DIR" \
    --max_peaks "$MAX_PEAKS" --validation_points \
    --density_ratio_band "$KAPPA_LOW" "$KAPPA_HIGH"
done
