"""Batch Acc/Tanimoto/close-match/MCES evaluation of rank_candidates.py outputs.

Usage:
    python scripts/evaluate_structural_metrics.py \
        --ranked_jsonl path/to/ranked.jsonl \
        --metadata path/to/meta_real_test.parquet \
        --num_workers 16 --no_mces --output_json path/to/structural_metrics.json

The shared evaluator scores the requested ordered prefixes (default: k=1 and k=10).
This script handles I/O, parallel execution and aggregation only.
"""

import argparse
from glob import glob
import json
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from tqdm import tqdm

from ms_gpt.pipeline.evaluation import MCES_THRESHOLD, MCES_TIME_LIMIT, METRIC_KS, evaluate_molecules


def process_spectrum(task):
    record, target_smiles, solver, ks = task
    molecules = [Chem.MolFromSmiles(c["smiles"]) for c in record["candidates"]]
    if any(mol is None for mol in molecules):
        raise ValueError(f"Invalid ranked structure at cache row {record['cache_row']}")
    return {"cache_row": record["cache_row"], "identifier": record.get("identifier", ""),
            "n_candidates": len(molecules),
            **evaluate_molecules(molecules, target_smiles, ks=ks, mces_solver=solver)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranked_jsonl", nargs="+", required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=METRIC_KS,
                        help="Top-k values to evaluate (default: 1 10)")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--mces_solver", default="PULP_CBC_CMD")
    parser.add_argument("--mces", dest="mces", action="store_true", default=True)
    parser.add_argument("--no_mces", dest="mces", action="store_false")
    parser.add_argument("--output_json", type=Path, required=True)
    args = parser.parse_args()
    if any(k < 1 for k in args.ks):
        parser.error("--ks values must be positive integers")
    RDLogger.DisableLog("rdApp.*")
    solver = None
    if args.mces:
        import pulp
        import myopic_mces

        solvers = pulp.listSolvers(onlyAvailable=True)
        solver = args.mces_solver
        if solver not in solvers:
            raise RuntimeError(f"MCES solver {solver} is unavailable; available solvers: {solvers}")
        print(f"MCES: solver={solver}, threshold={MCES_THRESHOLD}, stronger_bound=True, time_limit={MCES_TIME_LIMIT}")
    meta = pd.read_parquet(args.metadata)
    id_to_smiles = dict(zip(meta["identifier"].astype(str), meta["smiles"]))
    records = []
    for pattern in args.ranked_jsonl:
        for path in sorted(glob(pattern)):
            with open(path) as source:
                records.extend(json.loads(line) for line in source if line.strip())
    if not records:
        parser.error("No ranked candidate records found")
    tasks = [(r, id_to_smiles[str(r["identifier"])] if r.get("identifier")
              else meta.iloc[int(r["cache_row"])]["smiles"], solver, args.ks) for r in records]
    if args.num_workers == 1:
        results = list(tqdm(map(process_spectrum, tasks), total=len(tasks), desc="Score"))
    else:
        with Pool(args.num_workers) as pool:
            results = list(tqdm(pool.imap(process_spectrum, tasks, chunksize=1), total=len(tasks), desc="Score"))
    per_spectrum = pd.DataFrame(results)
    metric_columns = [column for column in per_spectrum if "@" in column]
    output = {"n_spec": len(results), "mces_enabled": args.mces,
              **per_spectrum[metric_columns].mean().to_dict()}
    for name in metric_columns:
        if name.startswith(("mces@", "tanimoto@")):
            output[f"median_{name}"] = float(per_spectrum[name].median())
    if args.mces:
        output["mces_fallback_pairs"] = int(per_spectrum["mces_fallback_pairs"].sum())
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    per_spectrum.to_parquet(args.output_json.with_suffix(".parquet"), index=False)
    for k in args.ks:
        print(f"@{k}: acc={output[f'acc@{k}']:.4%}, tanimoto={output[f'tanimoto@{k}']:.4f}"
              + f", close_match={output[f'close_match@{k}']:.4%}"
              + (f", mces={output[f'mces@{k}']:.4f}" if args.mces else ""))
    print(f"Saved {len(results)} per-spectrum results and means to {args.output_json.parent}")


if __name__ == "__main__":
    main()
