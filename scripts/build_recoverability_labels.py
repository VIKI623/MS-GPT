"""Build recoverability groups for posterior-aligned adaptation.

Input: frozen-MLM candidate JSONL and the aligned posterior-coverage table.

For each spectrum, parse its frozen candidate pool (100 draws in released profiles), compute:
  frozen_hit_top100 = target InChIKey-14 in the formula-filtered candidate pool
  frozen_hit_top10  = target in the top 10
  frozen_hit_top1   = target ranked first
  target_rank_frozen = target rank (-1 if absent)
  target_frequency_frozen = target generation frequency

Join with posterior_coverage (by cache_row), assign group:
  frozen_hit   (frozen_hit_top100=True)
  residual     (not hit, posterior_coverage >= cov_good)
  low_coverage (not hit, posterior_coverage < cov_good)

Output: target ranks, posterior coverage and recoverability group for each cache row.

Usage:
  python scripts/build_recoverability_labels.py \
    --frozen_jsonl eval_results/massspecgym_frozen_train/post.jsonl \
    --cache_dir data/posterior_cache/mist_official/annotated_top10 --benchmark massspecgym \
    --posterior_coverage data/posterior_cache/mist_official/annotated_top10/massspecgym/posterior_coverage_real_train.parquet \
    --threshold_low 0.03112947114434112 --threshold_high 0.7947460954389134 \
    --coverage_threshold 0.40 --workers 48 \
    --output eval_results/massspecgym_frozen_train/recoverability_labels.parquet
"""
import argparse
import json
from collections import Counter
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
from ms_gpt.pipeline.postprocessing import decode_safe
from ms_gpt.data.posterior_queries import (
    BAND_POINTS,
    band_max_similarity,
    band_reference_queries,
    resolve_threshold_band,
)

RDLogger.DisableLog("rdApp.*")


def featurize_candidate(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None: return None
        Chem.RemoveStereochemistry(mol)
        m2 = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        if m2 is None: return None
        inchi = MolToInchi(m2)
        if not inchi: return None
        ik = InchiToInchiKey(inchi).split("-")[0]
        formula = rdMolDescriptors.CalcMolFormula(m2)
        fp = AllChem.GetMorganFingerprintAsBitVect(m2, 2, nBits=4096)
        arr = np.zeros(4096, dtype=np.float32); ConvertToNumpyArray(fp, arr)
        return ik, formula, arr
    except Exception:
        return None


def label_spectrum(task, num_reference_queries=BAND_POINTS):
    rec, posterior, band = task
    target_key = rec["target_inchikey14"]
    target_formula = rec.get("target_formula", "")
    band_queries = band_reference_queries(posterior, band[0], band[1], num=num_reference_queries)
    band_popcounts = band_queries.sum(1)
    keys = Counter()
    similarities, candidate_scores = {}, {}
    smiles = (filter(None, map(decode_safe, rec["candidate_safe"]))
              if "candidate_safe" in rec else rec["candidate_smiles"])
    for smi in smiles:
        if smi not in candidate_scores:
            features = featurize_candidate(smi)
            score = None
            if features is not None:
                key, formula, fp = features
                if not target_formula or formula == target_formula:
                    score = (key, band_max_similarity(band_queries, band_popcounts, fp))
            candidate_scores[smi] = score
        if candidate_scores[smi] is None:
            continue
        key, score = candidate_scores[smi]
        keys[key] += 1
        similarities.setdefault(key, []).append(score)
    ranked = sorted(keys, key=lambda key: (-keys[key], -np.mean(similarities[key])))
    hit100 = target_key in keys
    target_rank = ranked.index(target_key) if target_key in ranked else -1
    hit10 = 0 <= target_rank < 10
    hit1 = target_rank == 0
    target_frequency = keys.get(target_key, 0)
    return {"cache_row":int(rec["cache_row"]), "target_inchikey14":target_key,
            "frozen_hit_top1":int(hit1), "frozen_hit_top10":int(hit10),
            "frozen_hit_top100":int(hit100), "target_rank_frozen":target_rank,
            "target_frequency_frozen":int(target_frequency)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--frozen_jsonl", required=True)
    ap.add_argument("--posterior_coverage", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    ap.add_argument("--split", choices=("train","val","test"), default="train")
    ap.add_argument("--coverage_threshold", type=float, default=0.40)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--output", required=True)
    ap.add_argument("--threshold_low", type=float)
    ap.add_argument("--threshold_high", type=float)
    ap.add_argument("--num_reference_queries", type=int, default=BAND_POINTS)
    args=ap.parse_args()
    try:
        band = resolve_threshold_band(args.threshold_low, args.threshold_high)
    except ValueError as exc:
        ap.error(str(exc))

    with open(args.frozen_jsonl) as handle:
        recs = [json.loads(line) for line in handle if line.strip()]
    print(f"{len(recs)} frozen records")
    pi_path = Path(args.cache_dir)/args.benchmark/f"pi_raw_real_{args.split}.npy"
    posterior = np.load(pi_path, mmap_mode="r")
    print(f"Band-marginalized tie-break: {args.benchmark} band {band} x{args.num_reference_queries}")
    tasks = ((r, np.asarray(posterior[int(r["cache_row"])], dtype=np.float32), band) for r in recs)
    with Pool(args.workers) as p:
        rows = list(p.imap(partial(label_spectrum, num_reference_queries=args.num_reference_queries),
                           tasks, chunksize=32))
    df=pd.DataFrame(rows)

    pdg = pd.read_parquet(args.posterior_coverage)[
        ["cache_row", "posterior_coverage"]
    ]
    df=df.merge(pdg, on="cache_row", how="left", validate="one_to_one")
    if df["posterior_coverage"].isna().any():
        raise ValueError("Every candidate row requires coverage from the corresponding posterior cache")

    df["group"] = np.select(
        [df["frozen_hit_top100"] == 1, df["posterior_coverage"] >= args.coverage_threshold],
        ["frozen_hit", "residual"], default="low_coverage",
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.output, index=False)
    print(f"saved {args.output}  n={len(df)}")
    print("group sizes:\n", df["group"].value_counts())
    print(f"frozen recall@100={df.frozen_hit_top100.mean()*100:.1f}% "
          f"@10={df.frozen_hit_top10.mean()*100:.1f}% @1={df.frozen_hit_top1.mean()*100:.1f}%")


if __name__=="__main__":
    main()
