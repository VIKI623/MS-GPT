"""Calibrate density-band and point-query thresholds from train/val posteriors."""
import argparse
import json
from pathlib import Path

import numpy as np


DENSITY_GRID = np.unique(np.r_[
    0., np.geomspace(1e-5, .01, 120), np.arange(.01, 1., .001), 1.,
]).astype(np.float32)
POINT_GRID = np.round(np.arange(.001, 1., .001), 3).astype(np.float32)


def threshold_counts(posterior, truth, thresholds):
    """Count active bits and true positives under the strict posterior > t rule."""
    order = np.argsort(posterior, axis=1)
    sorted_pi = np.take_along_axis(posterior, order, axis=1)
    below = np.stack([np.searchsorted(row, thresholds, side="right") for row in sorted_pi])
    predicted = posterior.shape[1] - below
    ordered_truth = np.take_along_axis(truth, order, axis=1)
    prefix = np.pad(ordered_truth.cumsum(axis=1, dtype=np.int64), ((0, 0), (1, 0)))
    true_count = truth.sum(axis=1, dtype=np.int64)[:, None]
    intersection = true_count - np.take_along_axis(prefix, below, axis=1)
    return predicted, intersection, true_count


def iterate_counts(directory, split, thresholds, chunk_size):
    posterior = np.load(directory / f"pi_raw_real_{split}.npy", mmap_mode="r")
    packed = np.load(directory / f"z_star_real_{split}.npy", mmap_mode="r")
    if len(posterior) != len(packed) or not len(posterior):
        raise ValueError("Posterior and truth caches must contain the same nonzero number of rows")
    for start in range(0, len(posterior), chunk_size):
        values = np.asarray(posterior[start:start + chunk_size], dtype=np.float32)
        truth = np.unpackbits(packed[start:start + chunk_size], axis=1)[:, :values.shape[1]]
        yield threshold_counts(values, truth, thresholds)


def calibrate(directory, density_ratio_band, *, validation_points=False, chunk_size=512):
    """Calibrate density-band endpoints on train and optional point queries on val."""
    low, high = map(float, density_ratio_band)
    if not 0 < low <= high:
        raise ValueError("Density ratios must satisfy 0 < low <= high")
    active = np.zeros(len(DENSITY_GRID), dtype=np.float64)
    true_bits, n_train = 0, 0
    for predicted, _, truth in iterate_counts(directory, "train", DENSITY_GRID, chunk_size):
        active += predicted.sum(axis=0)
        true_bits += int(truth.sum())
        n_train += len(truth)
    active /= n_train
    density = true_bits / n_train
    ratios = sorted(set([low, 1.0, high]))
    theta = {str(ratio): float(np.interp(ratio * density, active[::-1], DENSITY_GRID[::-1]))
             for ratio in ratios}
    report = {
        "calibration_split": "train", "training_rows": n_train,
        "true_density_D": density,
        "threshold_low": theta[str(high)],
        "threshold_high": theta[str(low)],
        "density_match_threshold": theta["1.0"], "theta_by_kappa": theta,
    }
    if validation_points:
        f1 = np.zeros(len(POINT_GRID), dtype=np.float64)
        cosine = np.zeros_like(f1)
        n_val = 0
        for predicted, intersection, truth in iterate_counts(directory, "val", POINT_GRID, chunk_size):
            f1 += (2 * intersection / np.maximum(predicted + truth, 1)).sum(axis=0)
            cosine += (intersection / np.sqrt(np.maximum(predicted * truth, 1))).sum(axis=0)
            n_val += len(truth)
        f1 /= n_val
        cosine /= n_val
        best_f1, best_cosine = int(f1.argmax()), int(cosine.argmax())
        report.update(
            validation_rows=n_val,
            validation_f1_threshold=float(POINT_GRID[best_f1]),
            validation_cosine_threshold=float(POINT_GRID[best_cosine]),
            validation_f1=float(f1[best_f1]), validation_cosine=float(cosine[best_cosine]),
            point_selection={"split": "val", "aggregation": "mean_per_spectrum",
                             "grid_step": .001, "tie_break": "smallest_threshold"},
        )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("density_calibration/mist_official/canonical_band.json"),
                        help="Canonical JSON containing the density-ratio band and benchmark endpoints")
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--density_ratio_band", type=float, nargs=2,
                        help="Override the configured density-ratio band")
    parser.add_argument("--max_peaks", type=int,
                        help="Fragment count used to build the cache; defaults to the benchmark entry")
    parser.add_argument("--validation_points", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--output", type=Path,
                        help="Canonical JSON to update; defaults to --config")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    band = args.density_ratio_band or config["density_ratio_band"]
    max_peaks = args.max_peaks
    if max_peaks is None:
        max_peaks = config["benchmarks"][args.benchmark]["max_fragment_peaks"]
    directory = args.cache_dir / args.benchmark
    report = calibrate(directory, band,
                       validation_points=args.validation_points, chunk_size=args.chunk_size)
    report.update(max_fragment_peaks=max_peaks, cache_dir=str(args.cache_dir))
    output = args.output or args.config
    canonical = json.loads(output.read_text()) if output.is_file() else {"benchmarks": {}}
    canonical["density_ratio_band"] = band
    canonical["benchmarks"][args.benchmark] = report
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(canonical, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
