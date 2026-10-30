"""
Offline preprocessing for pretraining data.

Reads local parquet files (safe-gpt dataset), decodes SAFE→SMILES,
then runs RDKit (stereo removal, InChI key filtering, FP, formula) in parallel.
Writes train/shard_*.parquet and metadata/preparation.json under output_dir.
SAFE tokenization is deferred to the DataLoader.
Validation and test structures from both benchmarks are excluded by default.

Usage:
    python scripts/prepare_mlm_corpus.py \\
        --input_dir $MS_GPT_SAFE_GPT_RAW \\
        --output_dir data/pretrain_100m \\
        --num_workers 64
"""

import argparse
import json
import time
from glob import glob
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey

RDLogger.DisableLog("rdApp.*")

from ms_gpt.data.mlm_pretraining_dataset import ELEMENTS, formula_str_to_vec
from ms_gpt.utils.paths import env_path_or


def load_exclusion_keys(path=None):
    """Read an override file or collect both benchmarks' held-out structures."""
    if path is not None:
        with open(path) as handle:
            return {line.strip() for line in handle if line.strip()}
    keys = set()
    for benchmark in ("nplib1", "massspecgym"):
        root = env_path_or(f"MS_GPT_{benchmark.upper()}_DATA", f"data/benchmarks/{benchmark}")
        for split in ("val", "test"):
            files = sorted((root / split).rglob("*.parquet"))
            if not files:
                raise FileNotFoundError(f"No benchmark Parquet files in {root / split}")
            for parquet_file in files:
                inchikeys = pd.read_parquet(parquet_file, columns=["inchikey"])["inchikey"]
                keys.update(inchikeys.str[:14])
    return keys


def process_structure_record(safe_str, fp_bits=4096):
    """Decode SAFE→SMILES, then compute RDKit features."""
    try:
        import safe as sf
        smiles = sf.decode(safe_str)
        if not smiles:
            return None

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        Chem.RemoveStereochemistry(mol)
        canon = Chem.MolToSmiles(mol)

        mol2 = Chem.MolFromSmiles(canon)
        if mol2 is None:
            return None

        inchi_str = MolToInchi(mol2)
        if inchi_str is None:
            return None
        ik = InchiToInchiKey(inchi_str)
        if ik is None:
            return None
        ik_2d = ik.split("-")[0]

        fp = AllChem.GetMorganFingerprintAsBitVect(mol2, radius=2, nBits=fp_bits)
        fp_indices = list(fp.GetOnBits())

        mol_h = Chem.AddHs(mol2)
        formula = Chem.rdMolDescriptors.CalcMolFormula(mol_h)

        fvec = formula_str_to_vec(formula).astype(int).tolist()

        return {
            "smiles": canon,
            "inchikey_2d": ik_2d,
            "formula": formula,
            "formula_vec": fvec,
            "fp_indices": fp_indices,
        }
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory with raw safe-gpt parquet files")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--exclude_inchikeys", type=str, default=None,
                        help="Override benchmark val/test exclusions with an InChIKey-14 file")
    parser.add_argument("--num_workers", type=int, default=64)
    parser.add_argument("--shard_size", type=int, default=500000)
    parser.add_argument("--max_molecules", type=int, default=None)
    parser.add_argument("--fp_bits", type=int, default=4096)
    parser.add_argument("--chunk_size", type=int, default=50000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    train_dir = output_dir / "train"
    metadata_dir = output_dir / "metadata"
    train_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    exclude_keys = load_exclusion_keys(args.exclude_inchikeys)
    print(f"Loaded {len(exclude_keys)} InChI keys to exclude")

    input_files = sorted(glob(str(Path(args.input_dir) / "*.parquet")))
    print(f"Found {len(input_files)} input parquet files")

    shard_id = 0
    buffer = []
    total_raw = 0
    total_kept = 0
    total_excluded = 0
    total_failed = 0
    t0 = time.time()
    done = False

    pool = Pool(args.num_workers)

    try:
        for file_idx, parquet_file in enumerate(input_files):
            if done:
                break
            df = pd.read_parquet(parquet_file, columns=["input"])
            safe_strings = df["input"].tolist()
            print(f"File {file_idx}/{len(input_files)}: {Path(parquet_file).name} "
                  f"({len(safe_strings):,} rows)", flush=True)

            for start in range(0, len(safe_strings), args.chunk_size):
                if done:
                    break
                chunk = safe_strings[start:start + args.chunk_size]
                results = pool.map(partial(process_structure_record, fp_bits=args.fp_bits), chunk)
                total_raw += len(chunk)

                for r in results:
                    if r is None:
                        total_failed += 1
                        continue
                    if r["inchikey_2d"] in exclude_keys:
                        total_excluded += 1
                        continue
                    buffer.append(r)
                    total_kept += 1
                    if args.max_molecules is not None and total_kept >= args.max_molecules:
                        done = True
                        break

                if len(buffer) >= args.shard_size:
                    df_out = pd.DataFrame(buffer)
                    df_out.to_parquet(train_dir / f"shard_{shard_id:05d}.parquet", index=False)
                    elapsed = time.time() - t0
                    rate = total_raw / elapsed
                    print(f"  Shard {shard_id}: {total_kept:,} kept / {total_raw:,} raw "
                          f"({total_excluded} excluded, {total_failed} failed) | "
                          f"{rate:.0f} mol/s | {elapsed/60:.1f} min")
                    shard_id += 1
                    buffer = []

    finally:
        pool.close()
        pool.join()

    if buffer:
        df_out = pd.DataFrame(buffer)
        df_out.to_parquet(train_dir / f"shard_{shard_id:05d}.parquet", index=False)
        shard_id += 1

    elapsed = time.time() - t0
    print(f"\nDone! {total_kept:,} molecules in {shard_id} shards")
    print(f"Raw: {total_raw:,} | Failed: {total_failed:,} | Excluded: {total_excluded:,}")
    print(f"Time: {elapsed/3600:.2f} hours ({total_raw/elapsed:.0f} mol/s)")

    meta = {
        "total_molecules": total_kept,
        "total_raw": total_raw,
        "total_excluded": total_excluded,
        "total_failed": total_failed,
        "num_shards": shard_id,
        "shard_size": args.shard_size,
        "fp_bits": args.fp_bits,
        "elements": ELEMENTS,
        "exclude_inchikeys_count": len(exclude_keys),
    }
    with open(metadata_dir / "preparation.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
