#!/bin/bash
# Derive the density-calibrated threshold band from the training split's
# ground-truth fingerprint density. The band endpoints are theta(kappa)=A^-1(kappa*D),
# where D is the mean on-bit count of true fingerprints and A(.) is the posterior
# survival function. The canonical band uses kappa in [0.95, 1.5] and contains
# the density-match point kappa=1.
#
# Reproduces the density calibrations and resolved intervals under
# analysis/density_calibration/.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -n "${MS_GPT_ENV_BIN:-}" ] && export PATH="$MS_GPT_ENV_BIN:$PATH"

for benchmark in massspecgym nplib1; do
  echo ">>> density calibration: $benchmark (train)"
  python scripts/compute_density_calibration.py --benchmark "$benchmark" --split train
done

echo "Done. See analysis/density_calibration/ for calibration files."
