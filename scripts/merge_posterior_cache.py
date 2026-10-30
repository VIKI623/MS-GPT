"""Merge per-GPU posterior-cache shards for one benchmark and split."""
import argparse
import os
import re

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--n_shards", type=int,
                        help="Expected shard count for this run; excludes older shard files.")
    parser.add_argument("--cleanup", action="store_true")
    args = parser.parse_args()

    dom_dir = os.path.join(args.cache_dir, args.benchmark)
    tag = f"real_{args.split}"
    pattern = re.compile(rf"^meta_{tag}_shard(\d+)\.parquet$")
    if args.n_shards is None:
        meta_files = sorted(
            (f for f in os.listdir(dom_dir) if pattern.match(f)),
            key=lambda f: int(pattern.match(f).group(1)),
        )
    else:
        meta_files = [f"meta_{tag}_shard{index:02d}.parquet" for index in range(args.n_shards)]
    if not meta_files:
        raise SystemExit(f"No shards found at {dom_dir}/meta_{tag}_shard*.parquet")

    metas, arrays = [], {name: [] for name in ("pi_raw", "z_star")}
    for meta_file in meta_files:
        shard = pattern.match(meta_file).group(1)
        meta = pd.read_parquet(os.path.join(dom_dir, meta_file))
        shard_arrays = {
            name: np.load(os.path.join(dom_dir, f"{name}_{tag}_shard{shard}.npy"))
            for name in arrays
        }
        if any(value.shape[0] != len(meta) for value in shard_arrays.values()):
            raise SystemExit(f"Row-count mismatch in {args.benchmark}/{tag} shard {shard}")
        metas.append(meta)
        for name, value in shard_arrays.items():
            arrays[name].append(value)

    merged_meta = pd.concat(metas, ignore_index=True)
    if not np.array_equal(merged_meta.global_row.to_numpy(), np.arange(len(merged_meta))):
        raise ValueError("Posterior shards must cover consecutive global rows without gaps or duplicates")
    merged_meta["cache_row"] = np.arange(len(merged_meta), dtype=np.int64)
    merged_meta.to_parquet(os.path.join(dom_dir, f"meta_{tag}.parquet"), index=False)
    for name, parts in arrays.items():
        np.save(os.path.join(dom_dir, f"{name}_{tag}.npy"), np.concatenate(parts, axis=0))

    if args.cleanup:
        for meta_file in meta_files:
            shard = pattern.match(meta_file).group(1)
            os.remove(os.path.join(dom_dir, meta_file))
            for name in arrays:
                os.remove(os.path.join(dom_dir, f"{name}_{tag}_shard{shard}.npy"))
    print(f"Merged {len(merged_meta)} rows for {args.benchmark}/{tag}")


if __name__ == "__main__":
    main()
