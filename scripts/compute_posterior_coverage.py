"""Compute posterior-coverage diagnostics used to construct posterior-aligned adapter labels."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from ms_gpt.data.posterior_queries import BAND_POINTS, resolve_threshold_band


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", default="train", choices=("train", "val", "test"))
    parser.add_argument("--threshold_low", type=float, required=True)
    parser.add_argument("--threshold_high", type=float, required=True)
    parser.add_argument("--num_queries", type=int, default=BAND_POINTS)
    args = parser.parse_args()

    directory = Path(args.cache_dir) / args.benchmark
    tag = f"real_{args.split}"
    meta = pd.read_parquet(directory / f"meta_{tag}.parquet")
    pi = np.load(directory / f"pi_raw_{tag}.npy", mmap_mode="r")
    z_star = np.load(directory / f"z_star_{tag}.npy", mmap_mode="r")
    band = resolve_threshold_band(args.threshold_low, args.threshold_high)
    grid = np.linspace(*band, args.num_queries, dtype=np.float32)

    coverage = np.zeros(len(meta), np.float32)
    best_threshold = np.zeros(len(meta), np.float32)
    for start in range(0, len(meta), 2000):
        end = min(start + 2000, len(meta))
        posterior = np.asarray(pi[start:end], dtype=np.float32)
        truth = np.unpackbits(np.asarray(z_star[start:end]), axis=1)[:, :pi.shape[1]]
        best = np.zeros(end - start, np.float32)
        best_t = np.zeros(end - start, np.float32)
        for threshold in grid:
            query = posterior > threshold
            intersection = (query & truth).sum(axis=1)
            union = (query | truth).sum(axis=1)
            score = intersection / np.maximum(union, 1)
            better = score > best
            best = np.where(better, score, best)
            best_t = np.where(better, threshold, best_t)
        coverage[start:end] = best
        best_threshold[start:end] = best_t

    output = pd.DataFrame({
        "cache_row": np.arange(len(meta), dtype=np.int64),
        "identifier": meta["identifier"].astype(str),
        "posterior_coverage": coverage,
        "best_threshold": best_threshold,
    })
    path = directory / f"posterior_coverage_{tag}.parquet"
    output.to_parquet(path, index=False)
    print(f"Saved {len(output)} rows to {path}")


if __name__ == "__main__":
    main()
