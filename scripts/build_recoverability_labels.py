"""Build recoverability groups for posterior-aligned adaptation.

Input: frozen-MLM candidate JSONL and the aligned posterior-coverage table.

For each spectrum, parse the 100 frozen candidates, compute:
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
    --posterior_coverage data/posterior_cache/massspecgym/posterior_coverage_real_train.parquet \
    --coverage_threshold 0.40 --workers 48 \
    --output data/posterior_cache/massspecgym/recoverability_labels.parquet
"""
import argparse, json
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.DataStructs import ConvertToNumpyArray
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


def tanimoto_similarity(a, b):
    inter=(a*b).sum(); union=((a+b)>0).sum(); return float(inter/max(union,1))


# Density-calibrated band for the band-marginalized tie-break similarity, matching
# the fidelity-binning analysis: a frequency tie is broken by
# max_t Tanimoto((posterior > t), candidate) over the density band.
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


def label_spectrum(task):
    rec, posterior, band = task
    target_key = rec["target_inchikey14"]
    target_formula = rec.get("target_formula", "")
    band_queries = band_reference_queries(posterior, band[0], band[1])
    band_popcounts = band_queries.sum(1)
    keys=Counter(); ikfp={}
    for smi in rec["candidate_smiles"]:
        r=featurize_candidate(smi)
        if r is None: continue
        ik,f,fp=r
        if target_formula and f != target_formula:
            continue
        keys[ik]+=1; ikfp.setdefault(ik,[]).append(fp)
    cands=[]
    for k,fr in keys.items():
        sim=np.mean([band_max_similarity(band_queries,band_popcounts,fp) for fp in ikfp[k]]); cands.append((k,fr,sim))
    cands.sort(key=lambda x:(-x[1],-x[2]))
    ranked=[c[0] for c in cands]
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
    ap.add_argument("--cache_dir", default="data/posterior_cache")
    ap.add_argument("--benchmark", choices=("massspecgym","nplib1"), required=True)
    ap.add_argument("--split", choices=("train","val","test","sim"), default="train")
    ap.add_argument("--coverage_threshold", type=float, default=0.40)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--output", required=True)
    args=ap.parse_args()

    recs=[]
    for l in open(args.frozen_jsonl):
        l=l.strip()
        if not l: continue
        try:
            o=json.loads(l)
            if isinstance(o,dict) and "cache_row" in o: recs.append(o)
        except: pass
    print(f"{len(recs)} frozen records")
    pi_path = Path(args.cache_dir)/args.benchmark/f"pi_raw_real_{args.split}.npy"
    posterior = np.load(pi_path, mmap_mode="r")
    band = BAND[args.benchmark]
    print(f"Band-marginalized tie-break: {args.benchmark} band {band} x{BAND_POINTS}")
    tasks=[(r, np.asarray(posterior[int(r["cache_row"])], dtype=np.float32), band) for r in recs]
    with Pool(args.workers) as p:
        rows=p.map(label_spectrum, tasks, chunksize=32)
    df=pd.DataFrame(rows)

    pdg = pd.read_parquet(args.posterior_coverage)[
        ["cache_row", "posterior_coverage"]
    ]
    df=df.merge(pdg, on="cache_row", how="left")
    df["posterior_coverage"]=df["posterior_coverage"].fillna(0.0)

    def assign_recoverability_group(r):
        if r["frozen_hit_top100"] == 1:
            return "frozen_hit"
        return "residual" if r["posterior_coverage"] >= args.coverage_threshold else "low_coverage"
    df["group"]=df.apply(assign_recoverability_group, axis=1)

    df.to_parquet(args.output, index=False)
    print(f"saved {args.output}  n={len(df)}")
    print("group sizes:\n", df["group"].value_counts())
    print(f"frozen recall@100={df.frozen_hit_top100.mean()*100:.1f}% "
          f"@10={df.frozen_hit_top10.mean()*100:.1f}% @1={df.frozen_hit_top1.mean()*100:.1f}%")


if __name__=="__main__":
    main()
