"""Evaluate structural metrics for ``query_mlm.py`` JSONL outputs.

Candidates are canonicalized, grouped by InChIKey-14, and ranked by generation
frequency. Posterior-fingerprint similarity breaks frequency ties for the
known-formula metrics. Accuracy, fingerprint similarities, and match rates use
the candidates compatible with the target formula. MCES uses the complete
ranked candidate pool and its paper-reported solver configuration.

Metrics are reported as the mean across all spectra for k in {1, 5, 10}:

* ``acc@k``: an exact InChIKey-14 match occurs in the first k candidates.
* ``mces@k``: minimum myopic MCES distance in the first k candidates.
* ``tanimoto@k``: maximum Morgan radius-2/2048 Tanimoto similarity.
* ``rdkit_tanimoto@k``: maximum RDKit topological similarity.
* ``cosine@k``: maximum Morgan radius-2/2048 cosine similarity.
* ``close_match@k``: RDKit topological similarity is at least 0.675.
* ``meaningful_match@k``: RDKit topological similarity is at least 0.4.

Usage:
    python scripts/evaluate_structural_metrics.py \\
        --candidates_jsonl path/to/post.jsonl \\
        --cache_dir data/posterior_cache --benchmark massspecgym \\
        --split test --num_workers 48 --no_mces \\
        --output_json path/to/metrics_structural.json

For the full MCES run, submit ``scripts/launch/evaluate_mces.slurm`` on a CPU
cluster instead of passing ``--no_mces``.
"""

import argparse
import json
import os
from collections import Counter, defaultdict
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.inchi import InchiToInchiKey

RDLogger.DisableLog("rdApp.*")

try:
    import pulp
    from myopic_mces import MCES

    _MCES_AVAIL = True
except ModuleNotFoundError:
    _MCES_AVAIL = False


METRIC_KS = (1, 5, 10)
MCES_THRESHOLD = 100
MCES_EMPTY_VALUE = 100.0


from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator


def canonical_mol_from_smiles(smi):
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
        return TautomerEnumerator().Canonicalize(mol)
    except Exception:
        return None


def remove_stereo(mol):
    if mol is None:
        return None
    rw_mol = Chem.RWMol(mol)
    Chem.RemoveStereochemistry(rw_mol)
    return rw_mol.GetMol()


def mol_to_inchi_no_stereo(mol):
    mol = remove_stereo(mol)
    if mol is None:
        return None
    try:
        return Chem.MolToInchi(mol)
    except Exception:
        return None


def tanimoto_np(a, b):
    intersection = (a * b).sum()
    union = ((a + b) > 0).sum()
    return float(intersection / max(union, 1))


# Density-calibrated band for the band-marginalized tie-break similarity. This is
# the same band the query generation sweeps (density_calibrated_query_band) and
# the same posterior/fingerprint similarity used by the fidelity-binning analysis:
# a frequency tie is broken by max_t Tanimoto((posterior > t), candidate).
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


def _rank_candidates(rec, band_queries, band_popcounts):
    """Return candidate metadata and the rankings used by the metrics.

    Frequency ties are broken by band-marginalized posterior similarity --
    max_t Tanimoto((posterior > t), candidate) over the density band -- rather
    than a single thresholded posterior reference.
    """
    counts = Counter()
    key_to_mol = {}
    key_to_inchi = {}
    key_to_formula = {}
    key_to_fps = defaultdict(list)

    for smi in rec["candidate_smiles"]:
        mol = remove_stereo(canonical_mol_from_smiles(smi))
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            inchi = Chem.MolToInchi(mol)
            if not inchi:
                continue
            key = InchiToInchiKey(inchi).split("-")[0]
            formula = rdMolDescriptors.CalcMolFormula(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=4096)
            fp_array = np.zeros(4096, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, fp_array)
        except Exception:
            continue

        counts[key] += 1
        key_to_fps[key].append(fp_array)
        if key not in key_to_mol:
            key_to_mol[key] = mol
            key_to_inchi[key] = inchi
            key_to_formula[key] = formula

    scored = []
    for key, frequency in counts.items():
        similarity = float(
            np.mean([
                band_max_similarity(band_queries, band_popcounts, fp)
                for fp in key_to_fps[key]
            ])
        )
        scored.append((key, frequency, similarity))
    scored.sort(key=lambda item: (-item[1], -item[2]))

    target_formula = rec.get("target_formula")
    if target_formula:
        formula_ranking = [
            key for key, _, _ in scored if key_to_formula[key] == target_formula
        ]
    else:
        formula_ranking = [key for key, _, _ in scored]

    # Counter preserves first-observed order for equal-frequency candidates.
    frequency_ranking = [key for key, _ in counts.most_common()]
    limit = max(METRIC_KS)
    return (
        key_to_mol,
        key_to_inchi,
        formula_ranking[:limit],
        frequency_ranking[:limit],
    )


def process_spectrum(task):
    """Compute every requested metric for one spectrum."""
    rec, gt_smiles, do_mces, solver, posterior, band = task
    band_queries = band_reference_queries(posterior, band[0], band[1])
    band_popcounts = band_queries.sum(1)

    true_mol = remove_stereo(canonical_mol_from_smiles(gt_smiles))
    if true_mol is None:
        return None
    true_inchi = mol_to_inchi_no_stereo(true_mol)
    if true_inchi is None:
        return None

    true_smi = Chem.MolToSmiles(true_mol)
    true_num_bonds = true_mol.GetNumBonds()
    true_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
        true_mol, 2, nBits=2048
    )
    rdkit_fp_generator = AllChem.GetRDKitFPGenerator()
    true_rdkit_fp = rdkit_fp_generator.GetFingerprint(true_mol)

    key_to_mol, key_to_inchi, formula_ranking, mces_ranking = _rank_candidates(
        rec, band_queries, band_popcounts
    )

    tanimoto_values = []
    rdkit_tanimoto_values = []
    cosine_values = []
    for key in formula_ranking:
        predicted_mol = key_to_mol[key]
        try:
            predicted_morgan_fp = AllChem.GetMorganFingerprintAsBitVect(
                predicted_mol, 2, nBits=2048
            )
            tanimoto_values.append(
                DataStructs.TanimotoSimilarity(true_morgan_fp, predicted_morgan_fp)
            )
            cosine_values.append(
                DataStructs.CosineSimilarity(true_morgan_fp, predicted_morgan_fp)
            )
        except Exception:
            tanimoto_values.append(0.0)
            cosine_values.append(0.0)
        try:
            predicted_rdkit_fp = rdkit_fp_generator.GetFingerprint(predicted_mol)
            rdkit_tanimoto_values.append(
                DataStructs.TanimotoSimilarity(true_rdkit_fp, predicted_rdkit_fp)
            )
        except Exception:
            rdkit_tanimoto_values.append(0.0)

    mces_values = []
    if do_mces:
        for key in mces_ranking:
            predicted_mol = key_to_mol[key]
            try:
                distance = MCES(
                    true_smi,
                    Chem.MolToSmiles(predicted_mol),
                    solver=solver,
                    threshold=MCES_THRESHOLD,
                    always_stronger_bound=False,
                    solver_options={"msg": 0},
                )[1]
                mces_values.append(float(distance))
            except Exception:
                mces_values.append(
                    float(true_num_bonds + predicted_mol.GetNumBonds())
                )

    prefix_max_tanimoto = [0.0]
    prefix_max_rdkit = [0.0]
    prefix_max_cosine = [0.0]
    for tanimoto, rdkit_tanimoto, cosine in zip(
        tanimoto_values, rdkit_tanimoto_values, cosine_values
    ):
        prefix_max_tanimoto.append(max(prefix_max_tanimoto[-1], tanimoto))
        prefix_max_rdkit.append(max(prefix_max_rdkit[-1], rdkit_tanimoto))
        prefix_max_cosine.append(max(prefix_max_cosine[-1], cosine))

    prefix_min_mces = [MCES_EMPTY_VALUE]
    for distance in mces_values:
        prefix_min_mces.append(min(prefix_min_mces[-1], distance))

    earliest_match = next(
        (
            rank
            for rank, key in enumerate(formula_ranking)
            if key_to_inchi[key] == true_inchi
        ),
        -1,
    )

    out = {}
    for k in METRIC_KS:
        formula_idx = min(k, len(formula_ranking))
        out[f"acc@{k}"] = 1.0 if 0 <= earliest_match < k else 0.0
        out[f"tanimoto@{k}"] = prefix_max_tanimoto[formula_idx]
        out[f"rdkit_tanimoto@{k}"] = prefix_max_rdkit[formula_idx]
        out[f"cosine@{k}"] = prefix_max_cosine[formula_idx]
        out[f"close_match@{k}"] = (
            1.0 if prefix_max_rdkit[formula_idx] >= 0.675 else 0.0
        )
        out[f"meaningful_match@{k}"] = (
            1.0 if prefix_max_rdkit[formula_idx] >= 0.4 else 0.0
        )
        if do_mces:
            mces_idx = min(k, len(mces_ranking))
            out[f"mces@{k}"] = prefix_min_mces[mces_idx]
    return out


def _load_records(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(sorted(glob(pattern)))
    if not paths:
        raise FileNotFoundError("no candidate JSONL files matched the supplied paths")

    print(f"[jsonl] loading {len(paths)} file(s)")
    records = []
    for path in paths:
        with open(path) as handle:
            records.extend(json.loads(line) for line in handle if line.strip())
    records.sort(key=lambda rec: rec["cache_row"])
    print(f"[jsonl] {len(records)} spectra")
    return records


def _ground_truth_for_record(rec, id_to_smiles, row_to_smiles):
    identifier = str(rec.get("identifier", ""))
    if identifier in id_to_smiles:
        return id_to_smiles[identifier]
    cache_row = rec.get("cache_row")
    if isinstance(cache_row, int) and 0 <= cache_row < len(row_to_smiles):
        return row_to_smiles[cache_row]
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates_jsonl", nargs="+", required=True)
    parser.add_argument("--cache_dir", default="data/posterior_cache")
    parser.add_argument(
        "--benchmark", choices=("massspecgym", "nplib1"), required=True
    )
    parser.add_argument(
        "--split", choices=("train", "val", "test", "sim"), default="test"
    )
    parser.add_argument("--num_workers", type=int, default=48)
    parser.add_argument("--mces", dest="mces", action="store_true", default=True)
    parser.add_argument("--no_mces", dest="mces", action="store_false")
    parser.add_argument("--output_json", required=True)
    args = parser.parse_args()

    if args.mces and not _MCES_AVAIL:
        raise RuntimeError(
            "MCES evaluation requires the optional 'metrics' dependencies; "
            "install them or pass --no_mces"
        )

    solver = None
    if args.mces:
        available_solvers = pulp.listSolvers(onlyAvailable=True)
        if not available_solvers:
            raise RuntimeError("MCES evaluation requires an available PuLP solver")
        solver = available_solvers[0]
        print(
            f"[mces] using solver={solver}, threshold={MCES_THRESHOLD}, "
            "stronger_bound=False, pair_time_limit=none"
        )

    meta_path = (
        Path(args.cache_dir)
        / args.benchmark
        / f"meta_real_{args.split}.parquet"
    )
    print(f"[meta] {meta_path}")
    meta = pd.read_parquet(meta_path)
    row_to_smiles = meta["smiles"].astype(str).tolist()
    id_to_smiles = dict(
        zip(meta["identifier"].astype(str), meta["smiles"].astype(str))
    )
    print(f"[meta] {len(id_to_smiles)} identifiers")

    pi_path = (
        Path(args.cache_dir) / args.benchmark / f"pi_raw_real_{args.split}.npy"
    )
    posterior = np.load(pi_path, mmap_mode="r")
    band = BAND[args.benchmark]
    print(f"[band] {args.benchmark} tie-break band {band} x{BAND_POINTS} points")

    records = _load_records(args.candidates_jsonl)
    tasks = []
    missing = []
    for rec in records:
        gt_smiles = _ground_truth_for_record(rec, id_to_smiles, row_to_smiles)
        if gt_smiles is None:
            missing.append(rec.get("cache_row"))
            continue
        pi_vec = np.asarray(posterior[int(rec["cache_row"])], dtype=np.float32)
        tasks.append((rec, gt_smiles, args.mces, solver, pi_vec, band))
    if missing:
        preview = ", ".join(map(str, missing[:10]))
        raise ValueError(
            f"ground-truth SMILES missing for {len(missing)} records "
            f"(cache rows: {preview})"
        )

    print(f"[run] {args.num_workers} workers, mces={args.mces}")
    results = Parallel(n_jobs=args.num_workers, verbose=5)(
        delayed(process_spectrum)(task) for task in tasks
    )
    invalid_targets = sum(result is None for result in results)
    if invalid_targets:
        raise ValueError(
            f"failed to parse ground-truth structures for {invalid_targets} spectra"
        )
    n_spectra = len(results)
    if n_spectra == 0:
        raise ValueError("no spectra were available for evaluation")
    print(f"[run] {n_spectra} valid results")

    totals = defaultdict(float)
    per_sample = defaultdict(list)
    for result in results:
        for metric, value in result.items():
            totals[metric] += value
            if metric.startswith(
                ("mces@", "tanimoto@", "rdkit_tanimoto@", "cosine@")
            ):
                per_sample[metric].append(value)
    means = {metric: total / n_spectra for metric, total in totals.items()}
    medians = {
        f"median_{metric}": float(np.median(values))
        for metric, values in per_sample.items()
    }

    print("=" * 64)
    print(f"structural metrics ({n_spectra} spectra, mces={args.mces})")
    for k in METRIC_KS:
        summary = (
            f"  @ k={k:<2d}: acc={means[f'acc@{k}'] * 100:5.2f}%  "
            f"tanimoto={means[f'tanimoto@{k}']:.3f}  "
            f"rdkit_tan={means[f'rdkit_tanimoto@{k}']:.3f}  "
            f"cm={means[f'close_match@{k}'] * 100:5.2f}%  "
            f"mm={means[f'meaningful_match@{k}'] * 100:5.2f}%"
        )
        if args.mces:
            summary += f"  mces={means[f'mces@{k}']:6.3f}"
        print(summary)

    output = {
        "n_spec": n_spectra,
        "mces_enabled": args.mces,
        **{metric: float(value) for metric, value in means.items()},
        **medians,
    }
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w") as handle:
        json.dump(output, handle, indent=2)
    print(f"saved -> {args.output_json}")


if __name__ == "__main__":
    main()
