"""Decode SAFE sequences, rank unique structures and complete short formula pools."""

from functools import lru_cache

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ..data.posterior_queries import (
    BAND_POINTS,
    band_reference_queries,
    resolve_threshold_band,
)
from ..utils.mol_repair import formula_counts, molecule_data, repair_candidates


FINGERPRINT = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


@lru_cache(maxsize=8192)
def decode_safe(sequence):
    """Return decoded SMILES, or None for an invalid SAFE sequence."""
    import safe

    return safe.decode(sequence, ignore_errors=True)


@lru_cache(maxsize=2048)
def _fingerprinted(smiles):
    candidate = molecule_data(smiles)
    if candidate is None:
        return None
    return dict(candidate, fingerprint=FINGERPRINT.GetFingerprint(candidate["mol"]))


class CandidateRanker:
    """Rank one spectrum's candidates against its posterior and calibrated band."""

    def __init__(self, posterior, probability_band, *, num_reference_queries=BAND_POINTS):
        posterior = np.asarray(posterior)
        if posterior.shape != (4096,):
            raise ValueError("Expected one 4096-bit posterior vector")
        band = resolve_threshold_band(*probability_band)
        queries = band_reference_queries(posterior, *band, num=num_reference_queries)
        self.queries = [DataStructs.CreateFromBinaryText(row.tobytes())
                        for row in np.packbits(queries, axis=1, bitorder="little")]

    def _similarity(self, fingerprint):
        return max(DataStructs.BulkTanimotoSimilarity(fingerprint, self.queries))

    def rank(self, safe_sequences, target_formula, top_k=10):
        """Return at most top_k ordered RDKit Mols from raw SAFE sequences.

        Existing formula matches retain their order; unique repairs fill the
        remaining slots. Each parent contributes at most one repair, chosen
        by posterior similarity after deduplication. Repairs inherit the parent
        frequency and are sorted within the appended suffix by frequency and
        their own posterior similarity, both descending.
        """
        return self.rank_smiles(
            (smiles for sequence in safe_sequences if (smiles := decode_safe(sequence))),
            target_formula,
            top_k,
        )

    def rank_smiles(self, generated_smiles, target_formula, top_k=10):
        """Rank already decoded SMILES, including saved generation outputs."""
        target = formula_counts(target_formula)
        if target is None:
            raise ValueError(f"Unsupported molecular formula: {target_formula}")
        if top_k < 1:
            raise ValueError("top_k must be positive")

        groups, scores = {}, {}
        for smiles in generated_smiles:
            candidate = _fingerprinted(smiles)
            if candidate is None:
                continue
            if smiles not in scores:
                scores[smiles] = self._similarity(candidate["fingerprint"])
            key = candidate["key"]
            if key not in groups:
                groups[key] = dict(candidate, frequency=0, similarities=[])
            group = groups[key]
            if (formula_counts(group["formula"]) != target
                    and formula_counts(candidate["formula"]) == target):
                group.update(candidate)
            group["frequency"] += 1
            group["similarities"].append(scores[smiles])
        for group in groups.values():
            group["matched"] = formula_counts(group["formula"]) == target
            group["posterior_similarity"] = float(np.mean(group.pop("similarities")))
        ranked = sorted(groups.values(), key=lambda c: (
            not c["matched"], -c["frequency"], -c["posterior_similarity"]))
        selected = [c for c in ranked if c["matched"]][:top_k]
        n_matched = len(selected)
        seen = {c["key"] for c in selected}
        parents = [c for c in ranked if not c["matched"]]

        for parent in parents:
            if len(selected) >= top_k:
                break
            products, _ = repair_candidates(parent, target_formula)
            candidate = max(
                (dict(product, posterior_similarity=self._similarity(
                    FINGERPRINT.GetFingerprint(product["mol"])))
                 for product in products if product["key"] not in seen),
                key=lambda c: c["posterior_similarity"], default=None,
            )
            if candidate is None:
                continue
            seen.add(candidate["key"])
            selected.append(dict(candidate, frequency=parent["frequency"],
                                 parent_key=parent["key"]))
        selected[n_matched:] = sorted(
            selected[n_matched:], key=lambda c: (-c["frequency"], -c["posterior_similarity"]))

        molecules = []
        for candidate in selected:
            mol = Chem.Mol(candidate["mol"])
            mol.SetIntProp("frequency", candidate["frequency"])
            mol.SetDoubleProp("posterior_similarity", candidate["posterior_similarity"])
            if "parent_key" in candidate:
                mol.SetProp("parent_key", candidate["parent_key"])
            molecules.append(mol)
        return molecules


def candidate_record(mol):
    """Serialize a ranked Mol and its generation or repair provenance."""
    record = {"smiles": Chem.MolToSmiles(mol),
              "frequency": mol.GetIntProp("frequency"),
              "posterior_similarity": mol.GetDoubleProp("posterior_similarity")}
    if mol.HasProp("parent_key"):
        record["parent_key"] = mol.GetProp("parent_key")
    return record
