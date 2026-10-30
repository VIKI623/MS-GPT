"""Build one shard of the spectrum-induced posterior cache.

For one benchmark split, this script:
  1. Reads the spectrum records.
  2. Runs the benchmark-paired MIST encoder to produce π_raw ∈ [0,1]^4096.
  3. Computes z* = MorganFP(smiles) ∈ {0,1}^4096 via RDKit.
  4. Writes aligned posterior, oracle-fingerprint and metadata shards:
        {cache_dir}/{benchmark}/meta_real_{split}_shard{shard_idx:02d}.parquet
        {cache_dir}/{benchmark}/pi_raw_real_{split}_shard{shard_idx:02d}.npy
        {cache_dir}/{benchmark}/z_star_real_{split}_shard{shard_idx:02d}.npy  (packed bits)

Run multiple instances in parallel across GPUs by setting different
--shard_idx values. Then run merge_posterior_cache.py to consolidate.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/build_posterior_cache.py \\
        --benchmark massspecgym --split test \\
        --spectrum_parquet $MS_GPT_MASSSPECGYM_DATA/test \\
        --mist_variant mist_official \\
        --shard_idx 0 --n_shards 8
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from ms_gpt.data.mlm_pretraining_dataset import smiles_to_fingerprint
from ms_gpt.data.benchmark_spectra import load_benchmark_spectra
from ms_gpt.data.mist_inputs import prepare_mist_batch


FP_BITS = 4096


def load_mist(benchmark, device, variant, checkpoint=None):
    if variant == "mist_official":
        from ms_gpt.models.mist_official_encoder import MIST
        ckpt = checkpoint or os.environ.get(
            f"MS_GPT_MIST_OFFICIAL_{benchmark.upper()}_CKPT",
            f"checkpoints/mist/mist_official/mist_{benchmark}.pt",
        )
        return MIST(ckpt).to(device)
    ckpt = checkpoint or os.environ.get(
        f"MS_GPT_MIST_{benchmark.upper()}_CKPT",
        f"checkpoints/mist/frigid_legacy/mist_{benchmark}.pt",
    )
    from ms_gpt.models.mist_frigid_encoder import MIST
    return MIST(ckpt, benchmark, device=str(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--spectrum_parquet", required=True,
                        help="Benchmark Parquet directory with one spectrum per row.")
    parser.add_argument("--cache_dir", help="Output cache root; defaults to the selected MIST input profile.")
    parser.add_argument("--mist_variant", choices=("frigid_legacy", "mist_official"), default="mist_official",
                        help="Checkpoint family, independent of spectrum preprocessing.")
    parser.add_argument("--mist_checkpoint", help="Override the benchmark-paired MIST checkpoint.")
    parser.add_argument("--max_peaks", type=int,
                        help="Number of annotated peaks; defaults to mist_official=10, frigid_legacy=40.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    args.mist_implementation = (
        "mist_official" if args.mist_variant == "mist_official" else "mist_frigid_legacy"
    )
    input_protocol = "annotated_topn"
    if args.max_peaks is None:
        args.max_peaks = 10 if args.mist_variant == "mist_official" else 40
    if args.max_peaks < 1:
        parser.error("--max_peaks must be positive")
    if args.cache_dir is None:
        args.cache_dir = f"data/posterior_cache/{args.mist_implementation}/annotated_top{args.max_peaks}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.cache_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    # Load spectrum records and shard
    print(f"Loading spectra: {args.spectrum_parquet}")
    recs = load_benchmark_spectra(args.spectrum_parquet)
    n_total = len(recs)
    per_shard = (n_total + args.n_shards - 1) // args.n_shards
    s = args.shard_idx * per_shard
    e = min(s + per_shard, n_total)
    shard_recs = recs[s:e]
    print(f"  total {n_total}, shard {args.shard_idx}/{args.n_shards} → [{s}:{e}] = {len(shard_recs)}")

    if not shard_recs:
        print("Empty shard, nothing to do.")
        return

    print("Loading MIST encoder...")
    mist = load_mist(args.benchmark, device, args.mist_variant, args.mist_checkpoint)
    print(f"  implementation: {mist.mist_implementation}; input profile: {input_protocol}")

    N = len(shard_recs)
    pi_raw = np.zeros((N, FP_BITS), dtype=np.float32)
    z_star = np.empty((N, FP_BITS), dtype=np.uint8)
    failed_idx = []
    fragment_counts = np.zeros(N, dtype=np.int32)

    # 1. Compute z* (cheap, CPU)
    print("Computing z* from SMILES...")
    for i, r in enumerate(tqdm(shard_recs, desc="z*")):
        fp = smiles_to_fingerprint(r["smiles"], n_bits=FP_BITS)
        if fp is None:
            failed_idx.append(i)
            z_star[i] = 0
        else:
            z_star[i] = fp.astype(np.uint8)

    # 2. Encode π_raw in batches
    print("Encoding π_raw with MIST...")
    BS = args.batch_size
    with torch.no_grad():
        for start in tqdm(range(0, N, BS), desc="MIST"):
            batch = shard_recs[start:start + BS]
            sb = prepare_mist_batch(batch, max_peaks=args.max_peaks, device=device)
            B = sb["batch_peak_num"].shape[0]
            pi_raw[start:start + B] = mist.predict(sb).cpu().numpy()
            fragment_counts[start:start + B] = sb["batch_peak_num"].cpu().numpy() - 1

    # 3. Pack oracle fingerprints.
    z_star_packed = np.packbits(z_star, axis=1)                       # (N, 512)

    # 4. Build metadata.
    n_active_zstar = z_star.sum(axis=1).astype(np.int32)
    meta = pd.DataFrame({
        "shard_row": np.arange(N, dtype=np.int64),
        "global_row": np.arange(s, e, dtype=np.int64),
        "identifier": [r["identifier"] for r in shard_recs],
        "smiles": [r["smiles"] for r in shard_recs],
        "inchikey_14": [r["inchikey"][:14] for r in shard_recs],
        "formula": [r["formula"] for r in shard_recs],
        "adduct": [r["adduct"] for r in shard_recs],
        "collision_energy": [r["collision_energy"] for r in shard_recs],
        "source": "benchmark",
        "split": args.split,
        "benchmark": args.benchmark,
        "n_active_z_star": n_active_zstar,
        "fp_failed": np.isin(np.arange(N), failed_idx),
        "mist_implementation": args.mist_implementation,
        "input_protocol": input_protocol,
        "n_annotated_peaks": [len(r["peak_formula"]["inten"]) for r in shard_recs],
        "n_fragment_peaks": fragment_counts,
    })
    meta["normalization_denominator"] = [
        max(r["peak_formula"]["inten"], default=0.0) for r in shard_recs
    ]

    # 5. Save shard outputs.
    tag = f"real_{args.split}_shard{args.shard_idx:02d}"
    meta_path = os.path.join(out_dir, f"meta_{tag}.parquet")
    pi_path = os.path.join(out_dir, f"pi_raw_{tag}.npy")
    zs_path = os.path.join(out_dir, f"z_star_{tag}.npy")

    meta.to_parquet(meta_path, index=False)
    np.save(pi_path, pi_raw)
    np.save(zs_path, z_star_packed)
    input_config = {
        "mist_implementation": args.mist_implementation,
        "input_protocol": input_protocol,
        "max_peaks": args.max_peaks,
        "model_config": mist.model_config,
        "normalization": "selected_fragment_max",
        "intensity_transform": "sqrt",
        "cls_position": "last",
        "cls_intensity": 1.0,
    }
    (Path(out_dir) / f"input_config_{tag}.json").write_text(json.dumps(input_config, indent=2) + "\n")

    n_failed = int(meta["fp_failed"].sum())
    print(f"\nDone. {N} rows ({n_failed} failed).")
    print(f"  meta:   {meta_path}")
    print(f"  pi_raw: {pi_path}  ({pi_raw.nbytes/1024**2:.1f} MB)")
    print(f"  z_star: {zs_path}  ({z_star_packed.nbytes/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
