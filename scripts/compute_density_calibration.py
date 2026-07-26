"""Calibrate the posterior-query threshold band from training density.

Operating point in DENSITY-RATIO space rather than raw threshold:
  D     = mean true on-bit count of training-set Morgan FPs (z*)   [no test leakage]
  A(t)  = mean active-bit count of the binarized posterior (pi > t), train set
  theta(kappa) = A^{-1}(kappa * D)  -- the threshold whose mean density is kappa*D

The sweep interval is [theta(kappa_low), theta(kappa_high)] with kappa_low > kappa_high
(denser -> sparser). The kappa band is shared across datasets; the raw thresholds it
produces adapt to each dataset's posterior calibration.

Usage:
  python scripts/compute_density_calibration.py --benchmark massspecgym --split train
"""
import argparse
import json
import numpy as np
from pathlib import Path

GRID = np.round(np.arange(0.0025, 0.6001, 0.0025), 4).astype(np.float32)
DENSITY_RATIOS = [2.0, 1.75, 1.5, 1.25, 1.0, 0.95, 0.94, 0.9, 0.8]


def threshold_for_density_ratio(A, D, kappa):
    """Largest t with A(t) >= kappa*D, linearly interpolated (A is decreasing)."""
    target = kappa * D
    # A decreasing; find crossing
    if A[0] < target:
        return float(GRID[0])          # even densest threshold is below target
    if A[-1] >= target:
        return float(GRID[-1])         # never drops below target on the grid
    k = int(np.searchsorted(-A, -target))  # first index where A[k] < target
    t0, t1, a0, a1 = GRID[k - 1], GRID[k], A[k - 1], A[k]
    frac = (a0 - target) / (a0 - a1) if a0 != a1 else 0.0
    return float(t0 + frac * (t1 - t0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache_dir", default="data/posterior_cache")
    ap.add_argument("--benchmark", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()

    d = Path(args.cache_dir) / args.benchmark
    tag = f"real_{args.split}"
    pi = np.load(d / f"pi_raw_{tag}.npy", mmap_mode="r")
    zsp = np.load(d / f"z_star_{tag}.npy", mmap_mode="r")
    N = pi.shape[0]
    CH = 4096

    A = np.zeros(len(GRID))
    dens = 0.0
    for s in range(0, N, 2000):
        e = min(s + 2000, N)
        p = np.asarray(pi[s:e], dtype=np.float32)
        zs = np.unpackbits(np.asarray(zsp[s:e]), axis=1)[:, :CH]
        dens += zs.sum()
        for k, t in enumerate(GRID):
            A[k] += (p > t).sum()
    A /= N
    D = dens / N

    thetas = {
        f"k{ratio}": round(threshold_for_density_ratio(A, D, ratio), 4)
        for ratio in DENSITY_RATIOS
    }
    out = {
        "benchmark": args.benchmark, "split": args.split, "n": int(N),
        "true_density_D": round(float(D), 3),
        "grid_min": float(GRID[0]), "grid_max": float(GRID[-1]), "grid_step": 0.0025,
        "theta_by_kappa": thetas,
        # store a coarse A-curve for plotting / reproducibility
        "A_curve": {str(round(float(t), 3)): round(float(a), 2)
                    for t, a in zip(GRID[::8], A[::8])},
    }
    output_dir = Path("analysis/density_calibration")
    output_dir.mkdir(parents=True, exist_ok=True)
    op = output_dir / f"density_calibration_{args.benchmark}_{args.split}.json"
    json.dump(out, open(op, "w"), indent=2)
    print(f"{args.benchmark}/{tag}: N={N}  D={D:.2f} bits")
    for kk in DENSITY_RATIOS:
        print(f"  kappa={kk:<5} -> theta={thetas[f'k{kk}']:.4f}  (mean {kk*D:.1f} bits)")
    print(f"saved {op}")


if __name__ == "__main__":
    main()
