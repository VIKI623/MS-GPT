"""Score an ordered molecule list without filtering, grouping or reranking."""

from functools import lru_cache

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ..utils.mol_repair import molecule_data


METRIC_KS = (1, 10)
MCES_THRESHOLD = 15
MCES_TIME_LIMIT = 600
MCES_EMPTY_VALUE = 100.0
CLOSE_MATCH_THRESHOLD = 0.675
MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
RDKIT = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048)


@lru_cache(maxsize=32768)
def _mces_distance(target_smiles, candidate_smiles, solver):
    if target_smiles == candidate_smiles:
        return 0.0, False
    from myopic_mces import MCES

    try:
        distance = MCES(target_smiles, candidate_smiles, solver=solver,
                        threshold=MCES_THRESHOLD, always_stronger_bound=True,
                        solver_options={"msg": 0, "threads": 1, "timeLimit": MCES_TIME_LIMIT})[1]
        return float(distance), False
    except Exception:
        bonds = sum(Chem.MolFromSmiles(s).GetNumBonds() for s in (target_smiles, candidate_smiles))
        return float(bonds), True


def evaluate_molecules(molecules, target_smiles, *, ks=METRIC_KS, mces_solver=None):
    """Evaluate normalized RDKit Mols in their supplied order.

    For each prefix of k candidates, accuracy tests for an InChIKey-14 match,
    Tanimoto takes the maximum radius-2 Morgan similarity (2048 bits), and MCES
    takes the minimum distance. Close match tests whether the maximum RDKit
    topological-fingerprint similarity is at least 0.675. Every metric uses
    the same ordered prefix.
    MCES follows FRIGID: threshold 15, stronger bound, and a 600-second limit.
    Empty pools score zero on accuracy/Tanimoto/close match and 100 on MCES.
    """
    target = molecule_data(target_smiles)
    if target is None:
        raise ValueError(f"Invalid target structure: {target_smiles}")
    true_morgan = MORGAN.GetFingerprint(target["mol"])
    true_rdkit = RDKIT.GetFingerprint(target["mol"])
    values = {name: [] for name in ("acc", "tanimoto", "close_match")}
    if mces_solver is not None:
        values["mces"] = []
    fallbacks = 0
    for mol in molecules[:max(ks)]:
        fp = MORGAN.GetFingerprint(mol)
        values["acc"].append(float(Chem.MolToInchiKey(mol)[:14] == target["key"]))
        values["tanimoto"].append(DataStructs.TanimotoSimilarity(true_morgan, fp))
        values["close_match"].append(float(DataStructs.TanimotoSimilarity(
            true_rdkit, RDKIT.GetFingerprint(mol)) >= CLOSE_MATCH_THRESHOLD))
        if mces_solver is not None:
            distance, fallback = _mces_distance(target["smiles"], Chem.MolToSmiles(mol), mces_solver)
            values["mces"].append(distance)
            fallbacks += fallback

    metrics = {}
    for k in ks:
        for name, scores in values.items():
            metrics[f"{name}@{k}"] = (min([MCES_EMPTY_VALUE, *scores[:k]]) if name == "mces"
                                      else max(scores[:k], default=0.0))
    if mces_solver is not None:
        metrics["mces_fallback_pairs"] = fallbacks
    return metrics
