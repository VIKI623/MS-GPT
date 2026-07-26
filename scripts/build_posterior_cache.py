"""Build one shard of the spectrum-induced posterior cache.

For one benchmark split, this script:
  1. Reads the spectrum records.
  2. Runs the benchmark-paired MIST encoder to produce π_raw ∈ [0,1]^4096.
  3. Computes z* = MorganFP(smiles) ∈ {0,1}^4096 via RDKit.
  4. Writes aligned posterior, oracle-fingerprint and metadata shards:
        {cache_dir}/{benchmark}/meta_{shard_idx}.parquet
        {cache_dir}/{benchmark}/pi_raw_{shard_idx}.npy
        {cache_dir}/{benchmark}/z_star_{shard_idx}.npy      (packed bits)

Run multiple instances in parallel across GPUs by setting different
--shard_idx values. Then run merge_posterior_cache.py to consolidate.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/build_posterior_cache.py \\
        --benchmark massspecgym --split test \\
        --spectrum_parquet $MS_GPT_MASSSPECGYM_DATA/test \\
        --cache_dir data/posterior_cache \\
        --shard_idx 0 --n_shards 8
"""
import argparse
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from ms_gpt.data.mlm_pretraining_dataset import smiles_to_fingerprint
from ms_gpt.data.benchmark_spectrum_dataset import BenchmarkSpectrumDataset
from ms_gpt.data.spectrum_utils import prepare_spectrum_batch
from ms_gpt.models.mist_massspecgym_encoder import MassSpecGymMISTEncoder
from ms_gpt.models.mist_nplib1_encoder import NPLIB1MISTEncoder


FP_BITS = 4096


def load_mist(benchmark, device):
    if benchmark == "massspecgym":
        ckpt = os.environ.get("MS_GPT_MIST_MASSSPECGYM_CKPT")
        if not ckpt:
            raise SystemExit("Set MS_GPT_MIST_MASSSPECGYM_CKPT")
        return MassSpecGymMISTEncoder(checkpoint_path=ckpt, device=str(device))
    if benchmark == "nplib1":
        ckpt = os.environ.get("MS_GPT_MIST_NPLIB1_CKPT")
        if not ckpt:
            raise SystemExit("Set MS_GPT_MIST_NPLIB1_CKPT")
        return NPLIB1MISTEncoder(checkpoint_path=ckpt, device=str(device))
    raise ValueError(f"unknown benchmark: {benchmark}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--spectrum_parquet", required=True,
                        help="Parquet dir of spectrum records (must match BenchmarkSpectrumDataset schema).")
    parser.add_argument("--cache_dir", default="data/posterior_cache")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--n_shards", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    out_dir = os.path.join(args.cache_dir, args.benchmark)
    os.makedirs(out_dir, exist_ok=True)

    # Load spectrum records and shard
    print(f"Loading spectra: {args.spectrum_parquet}")
    recs = BenchmarkSpectrumDataset(args.spectrum_parquet).records
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
    mist = load_mist(args.benchmark, device)
    cls_type = mist.cls_type_value

    N = len(shard_recs)
    pi_raw = np.empty((N, FP_BITS), dtype=np.float32)
    z_star = np.empty((N, FP_BITS), dtype=np.uint8)
    failed_idx = []

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
            try:
                sb = prepare_spectrum_batch(
                    [r["peak_formula"] for r in batch],
                    [r["precursor_mz"] for r in batch],
                    [r["formula"] for r in batch],
                    [r["adduct"] for r in batch],
                    [r["collision_energy"] for r in batch],
                    device=str(device), cls_type_value=cls_type,
                )
                B = sb["batch_peak_num"].shape[0]
                if "batch_spec_instrument" not in sb:
                    sb["batch_spec_instrument"] = torch.zeros(B, dtype=torch.long, device=device)
                peak_emb = mist.formula_transformer(sb)
                cls_mask = sb["batch_peak_type"] == cls_type
                cls_emb = peak_emb[cls_mask]
                pi = mist.fp_grower(cls_emb)[-1].cpu().numpy()
                pi_raw[start:start + B] = pi
            except Exception as exc:
                # Mark this batch's rows as failed; keep zero placeholders
                print(f"  batch [{start}:{start+BS}] failed: {exc}")
                for j in range(start, min(start + BS, N)):
                    failed_idx.append(j)

    # 3. Pack oracle fingerprints.
    z_star_packed = np.packbits(z_star, axis=1)                       # (N, 512)

    # 4. Build metadata.
    n_active_zstar = z_star.sum(axis=1).astype(np.int32)
    meta = pd.DataFrame({
        "shard_row": np.arange(N, dtype=np.int64),
        "global_row": np.arange(s, e, dtype=np.int64),
        "identifier": [r.get("identifier", "") for r in shard_recs],
        "smiles": [r["smiles"] for r in shard_recs],
        "inchikey_14": [r["inchikey"][:14] for r in shard_recs],
        "formula": [r["formula"] for r in shard_recs],
        "adduct": [r.get("adduct", "") for r in shard_recs],
        "collision_energy": [float(r.get("collision_energy", float("nan"))) for r in shard_recs],
        "source": "benchmark",
        "split": args.split,
        "benchmark": args.benchmark,
        "n_active_z_star": n_active_zstar,
        "fp_failed": np.isin(np.arange(N), failed_idx),
    })

    # 5. Save shard outputs.
    tag = f"real_{args.split}_shard{args.shard_idx:02d}"
    meta_path = os.path.join(out_dir, f"meta_{tag}.parquet")
    pi_path = os.path.join(out_dir, f"pi_raw_{tag}.npy")
    zs_path = os.path.join(out_dir, f"z_star_{tag}.npy")

    meta.to_parquet(meta_path, index=False)
    np.save(pi_path, pi_raw)
    np.save(zs_path, z_star_packed)

    n_failed = int(meta["fp_failed"].sum())
    print(f"\nDone. {N} rows ({n_failed} failed).")
    print(f"  meta:   {meta_path}")
    print(f"  pi_raw: {pi_path}  ({pi_raw.nbytes/1024**2:.1f} MB)")
    print(f"  z_star: {zs_path}  ({z_star_packed.nbytes/1024**2:.1f} MB)")


if __name__ == "__main__":
    main()
