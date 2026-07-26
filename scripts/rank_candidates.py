"""Rank conditional-MLM candidates by frequency and posterior similarity.
Strategy: frequency_posterior_similarity (frequency + Tanimoto).

Usage:
    python scripts/rank_candidates.py \\
        --candidates_jsonl eval_results/gen_pretrain_shard*.jsonl \\
        --num_workers 32 \\
        --output_json eval_results/metrics_pretrain.json
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
from collections import Counter
from glob import glob
from multiprocessing import Pool
from tqdm import tqdm

from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray

RDLogger.DisableLog("rdApp.*")


def featurize_candidate(smi, fp_bits=4096):
    """SMILES → (ik14, formula, morgan_fp, canonical_smiles) or None."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        mol2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if mol2 is None:
            return None
        inchi = MolToInchi(mol2)
        if not inchi:
            return None
        ik = InchiToInchiKey(inchi).split("-")[0]
        formula = rdMolDescriptors.CalcMolFormula(mol2)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=fp_bits)
        arr = np.zeros(fp_bits, dtype=np.float32)
        ConvertToNumpyArray(fp, arr)
        return ik, formula, arr, Chem.MolToSmiles(mol2)
    except Exception:
        return None


def tanimoto_similarity(a, b):
    inter = (a * b).sum()
    union = ((a + b) > 0).sum()
    return float(inter / max(union, 1))


# Density-calibrated band for the band-marginalized tie-break similarity: a
# frequency tie is broken by max_t Tanimoto((posterior > t), candidate) over the
# density band -- the same posterior/fingerprint similarity used by the
# fidelity-binning analysis, replacing the single thresholded posterior reference.
BAND = {"nplib1": (0.062, 0.292), "massspecgym": (0.069, 0.274)}
BAND_POINTS = 20


def band_reference_queries(posterior, band_low, band_high, num=BAND_POINTS):
    thresholds = np.linspace(band_low, band_high, num, dtype=np.float32)
    return posterior[None, :] > thresholds[:, None]


def band_max_similarity(queries, query_popcounts, candidate_fp):
    candidate_bits = candidate_fp > 0
    intersection = (queries & candidate_bits).sum(1)
    union = query_popcounts + candidate_bits.sum() - intersection
    return float(np.max(intersection / np.maximum(union, 1)))


def rank_spectrum_candidates(args_tuple):
    """Parse + frequency_posterior_similarity rerank for one spectrum. Returns intermediate result."""
    record, formula_filter, posterior, band = args_tuple
    target_inchikey14 = record["target_inchikey14"]
    target_formula = record["target_formula"]

    band_queries = band_reference_queries(posterior, band[0], band[1])
    band_popcounts = band_queries.sum(1)

    total_valid = 0
    keys = Counter()
    ik_to_fps = {}
    ik_to_smi = {}

    for smi in record["candidate_smiles"]:
        r = featurize_candidate(smi)
        if r is None:
            continue
        ik, formula, fp_arr, canon = r
        total_valid += 1
        if formula_filter and target_formula and formula != target_formula:
            continue
        keys[ik] += 1
        ik_to_fps.setdefault(ik, []).append(fp_arr)
        if ik not in ik_to_smi:
            ik_to_smi[ik] = canon

    cands = []
    for k, freq in keys.items():
        sim = np.mean([
            band_max_similarity(band_queries, band_popcounts, fp)
            for fp in ik_to_fps[k]
        ])
        cands.append((k, freq, sim))
    cands.sort(key=lambda x: (-x[1], -x[2]))
    ranked = [c[0] for c in cands[:20]]

    hits = {}
    for k in (1, 5, 10, 20):
        hits[k] = 1 if target_inchikey14 in ranked[:k] else 0

    return {
        "cache_row": record["cache_row"],
        "target_inchikey14": target_inchikey14,
        "target_formula": target_formula,
        "total_generated": len(record["candidate_smiles"]),
        "total_valid": total_valid,
        "n_unique": len(keys),
        "hits_frequency_posterior_similarity": hits,
        "ranked_frequency_posterior_similarity": ranked,
        "ik_to_smi": ik_to_smi,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_jsonl", nargs="+", required=True,
                        help="JSONL files produced by query_mlm.py")
    parser.add_argument("--cache_dir", default="data/posterior_cache")
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test", "sim"), default="test")
    parser.add_argument("--formula_filter", action="store_true", default=True)
    parser.add_argument("--no_formula_filter", dest="formula_filter", action="store_false")
    parser.add_argument("--num_workers", type=int, default=32)
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    # Expand globs
    paths = []
    for p in args.candidates_jsonl:
        paths.extend(sorted(glob(p)))
    print(f"Loading {len(paths)} JSONL file(s)...")

    records = []
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    records.sort(key=lambda r: r["cache_row"])
    print(f"  {len(records)} spectra loaded")

    pi_path = Path(args.cache_dir) / args.benchmark / f"pi_raw_real_{args.split}.npy"
    posterior = np.load(pi_path, mmap_mode="r")
    band = BAND[args.benchmark]
    print(f"Band-marginalized tie-break: {args.benchmark} band {band} x{BAND_POINTS}")

    print(f"\nRanking candidates with {args.num_workers} workers...")
    tasks = [
        (r, args.formula_filter,
         np.asarray(posterior[int(r["cache_row"])], dtype=np.float32), band)
        for r in records
    ]
    with Pool(args.num_workers) as pool:
        results = list(tqdm(
            pool.imap(rank_spectrum_candidates, tasks, chunksize=16),
            total=len(tasks), desc="frequency_posterior_similarity",
        ))

    n = len(results)
    total_valid = sum(r["total_valid"] for r in results)
    total_gen = sum(r["total_generated"] for r in results)
    agg_fs = {1: 0, 5: 0, 10: 0, 20: 0}
    for r in results:
        for k in (1, 5, 10, 20):
            agg_fs[k] += r["hits_frequency_posterior_similarity"][k]

    print(f"\n{'='*60}")
    print(f"frequency_posterior_similarity ({n} spectra, formula_filter={args.formula_filter})")
    print(f"  Validity: {total_valid}/{total_gen} = {total_valid/max(total_gen,1)*100:.1f}%")
    for k in (1, 5, 10, 20):
        print(f"  Top-{k:<2}: {agg_fs[k]:5d}/{n} = {agg_fs[k]/n*100:.2f}%")

    # Save metrics
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    output = {
        "n_spec": n,
        "total_valid": total_valid,
        "total_generated": total_gen,
        "frequency_posterior_similarity": {f"top{k}": agg_fs[k] for k in (1, 5, 10, 20)},
    }
    with open(args.output_json, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved → {args.output_json}")


if __name__ == "__main__":
    main()
