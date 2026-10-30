"""Rank SAFE generation outputs and save ordered, formula-matched candidates."""

import argparse
from glob import glob
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from rdkit import RDLogger
from tqdm import tqdm

from ms_gpt.data.posterior_queries import BAND_POINTS, resolve_threshold_band
from ms_gpt.pipeline.postprocessing import CandidateRanker, candidate_record


def _init_worker(posterior_path, band, num_queries, top_k):
    global POSTERIORS, BAND, NUM_QUERIES, TOP_K
    POSTERIORS = np.load(posterior_path, mmap_mode="r")
    BAND, NUM_QUERIES, TOP_K = band, num_queries, top_k
    RDLogger.DisableLog("rdApp.*")


def rank_record(record):
    ranker = CandidateRanker(POSTERIORS[int(record["cache_row"])], BAND,
                             num_reference_queries=NUM_QUERIES)
    if "candidate_safe" in record:
        molecules = ranker.rank(record["candidate_safe"], record["target_formula"], TOP_K)
    else:
        molecules = ranker.rank_smiles(record["candidate_smiles"], record["target_formula"], TOP_K)
    output = {key: record[key] for key in ("cache_row", "identifier", "target_formula") if key in record}
    output["candidates"] = [candidate_record(mol) for mol in molecules]
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates_jsonl", nargs="+", required=True)
    parser.add_argument("--cache_dir", type=Path, required=True)
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    parser.add_argument("--threshold_low", type=float, required=True)
    parser.add_argument("--threshold_high", type=float, required=True)
    parser.add_argument("--num_reference_queries", type=int, default=BAND_POINTS)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_jsonl", type=Path, required=True)
    args = parser.parse_args()
    band = resolve_threshold_band(args.threshold_low, args.threshold_high)
    paths = [path for pattern in args.candidates_jsonl for path in sorted(glob(pattern))]
    if not paths:
        parser.error("No candidate JSONL files matched")
    records = []
    for path in paths:
        with open(path) as source:
            records.extend(json.loads(line) for line in source if line.strip())
    records.sort(key=lambda r: r["cache_row"])
    posterior_path = args.cache_dir / args.benchmark / f"pi_raw_real_{args.split}.npy"
    initargs = (posterior_path, band, args.num_reference_queries, args.top_k)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with args.output_jsonl.open("w") as output:
        if args.num_workers == 1:
            _init_worker(*initargs)
            for result in tqdm(map(rank_record, records), total=len(records), desc="Rank"):
                output.write(json.dumps(result) + "\n")
        else:
            with Pool(args.num_workers, initializer=_init_worker, initargs=initargs) as pool:
                for result in tqdm(pool.imap(rank_record, records, chunksize=1), total=len(records), desc="Rank"):
                    output.write(json.dumps(result) + "\n")
    print(f"Saved {len(records)} ordered candidate lists to {args.output_jsonl}")


if __name__ == "__main__":
    main()
