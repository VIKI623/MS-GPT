"""Bounded local edits and formula-matched candidate-pool completion."""

from collections import Counter
from functools import lru_cache
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator


BONDS = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
TAUTOMERS = TautomerEnumerator()


def formula_counts(formula):
    # RDKit writes charge after the elemental formula, e.g. C4H12N+ or C6H18N2+2.
    formula = re.sub(r"[+-]\d*$", "", formula)
    if not re.fullmatch(r"(?:[A-Z][a-z]?\d*)+", formula):
        return None
    counts = Counter()
    for symbol, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        counts[symbol] += int(count or 1)
    return counts


@lru_cache(maxsize=8192)
def molecule_data(smiles):
    """Use the MS-GPT tautomer and stereochemistry convention."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = TAUTOMERS.Canonicalize(mol)
        Chem.RemoveStereochemistry(mol)
        Chem.SanitizeMol(mol)
        inchi = Chem.MolToInchi(mol)
        if not inchi:
            return None
        return dict(mol=mol, inchi=inchi, key=Chem.InchiToInchiKey(inchi).split("-")[0],
                    formula=rdMolDescriptors.CalcMolFormula(mol), smiles=Chem.MolToSmiles(mol))
    except Exception:
        return None


def _heavy_delta(mol, target):
    current = Counter(a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() != 1)
    return {e: target[e] - current[e] for e in current.keys() | target.keys()
            if e != "H" and target[e] != current[e]}


def _lower_bound(delta):
    return max(sum(max(v, 0) for v in delta.values()), sum(max(-v, 0) for v in delta.values()))


def _finish(mol, smiles, target):
    """Normalize a sanitized edit with matching elemental counts."""
    try:
        if len(Chem.GetMolFrags(mol)) != 1:
            return None
        if any(a.GetNumRadicalElectrons() for a in mol.GetAtoms()):
            return None
        candidate = molecule_data(smiles)
        if candidate and formula_counts(candidate["formula"]) == target:
            return candidate
    except Exception:
        pass
    return None


def _edit_steps(mol, target):
    """Propose formula-directed atom, bond, proton and component-joining edits."""
    base = Chem.Mol(mol)
    try:
        Chem.Kekulize(base, clearAromaticFlags=True)
    except Chem.KekulizeException:
        return
    delta = _heavy_delta(base, target)
    organic = {"B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I"}

    def reset(atom):
        if not atom.GetFormalCharge() and not atom.GetNumRadicalElectrons():
            atom.SetNumExplicitHs(0)
            atom.SetNoImplicit(False)

    if delta:
        missing = sorted(e for e, n in delta.items() if n > 0 and e in organic)
        classes = Chem.CanonicalRankAtoms(mol, breakTies=False)
        seen_classes = set()
        for atom in base.GetAtoms():
            i = atom.GetIdx()
            if classes[i] in seen_classes:
                continue
            seen_classes.add(classes[i])
            if atom.GetIsotope():
                continue
            if delta.get(atom.GetSymbol(), 0) < 0:
                if atom.GetDegree() <= 1:
                    product = Chem.RWMol(base)
                    for neighbor in product.GetAtomWithIdx(i).GetNeighbors():
                        reset(neighbor)
                    product.RemoveAtom(i)
                    yield product.GetMol(), "terminal_delete"
                for symbol in missing:
                    product = Chem.RWMol(base)
                    new = Chem.Atom(symbol)
                    product.ReplaceAtom(i, new)
                    yield product.GetMol(), "atom_replace"
            if atom.GetSymbol() not in {"B", "C", "N", "O", "Si", "P", "S"} or atom.GetFormalCharge():
                continue
            for symbol in missing:
                max_order = min(3, max(Chem.GetPeriodicTable().GetValenceList(symbol)))
                for order in range(1, max_order + 1):
                    if atom.GetTotalNumHs() < order and atom.GetSymbol() not in {"P", "S"}:
                        continue
                    product = Chem.RWMol(base)
                    new_index = product.AddAtom(Chem.Atom(symbol))
                    product.AddBond(i, new_index, BONDS[order])
                    reset(product.GetAtomWithIdx(i))
                    yield product.GetMol(), "terminal_add"
        return

    h_delta = target["H"] - formula_counts(rdMolDescriptors.CalcMolFormula(mol))["H"]
    # Bond-order edits preserve the atom connectivity, including ring topology.
    if h_delta and h_delta % 2 == 0:
        change = -1 if h_delta > 0 else 1
        for bond in base.GetBonds():
            order = int(bond.GetBondTypeAsDouble())
            if order + change not in BONDS:
                continue
            atoms = (bond.GetBeginAtom(), bond.GetEndAtom())
            if change > 0 and any(a.GetTotalNumHs() < 1 for a in atoms):
                continue
            if any(a.GetFormalCharge() or a.GetNumRadicalElectrons() for a in atoms):
                continue
            product = Chem.RWMol(base)
            product.GetBondWithIdx(bond.GetIdx()).SetBondType(BONDS[order + change])
            for atom in atoms:
                reset(product.GetAtomWithIdx(atom.GetIdx()))
            yield product.GetMol(), "bond_order"

    if h_delta:
        change = 1 if h_delta > 0 else -1
        for atom in base.GetAtoms():
            if atom.GetSymbol() not in {"N", "O", "P", "S"} or atom.GetNumRadicalElectrons():
                continue
            charge, hydrogens = atom.GetFormalCharge() + change, atom.GetTotalNumHs() + change
            if abs(charge) > 1 or hydrogens < 0:
                continue
            product = Chem.RWMol(base)
            edited = product.GetAtomWithIdx(atom.GetIdx())
            edited.SetFormalCharge(charge)
            edited.SetNumExplicitHs(hydrogens)
            edited.SetNoImplicit(True)
            yield product.GetMol(), "protonation" if change > 0 else "deprotonation"

    for atom in base.GetAtoms():
        if atom.GetNumRadicalElectrons() and atom.GetSymbol() in organic:
            product = Chem.RWMol(base)
            edited = product.GetAtomWithIdx(atom.GetIdx())
            edited.SetNumRadicalElectrons(0)
            reset(edited)
            yield product.GetMol(), "valence_completion"

    fragments = Chem.GetMolFrags(base)
    if len(fragments) > 1 and h_delta <= -2:
        labels = {i: k for k, fragment in enumerate(fragments) for i in fragment}
        ends = [a for a in base.GetAtoms() if a.GetTotalNumHs() and not a.GetFormalCharge()
                and a.GetSymbol() in {"C", "N", "O", "P", "S"}]
        for pos, a in enumerate(ends):
            for b in ends[pos+1:]:
                if labels[a.GetIdx()] == labels[b.GetIdx()]:
                    continue
                product = Chem.RWMol(base)
                product.AddBond(a.GetIdx(), b.GetIdx(), Chem.BondType.SINGLE)
                reset(product.GetAtomWithIdx(a.GetIdx()))
                reset(product.GetAtomWithIdx(b.GetIdx()))
                yield product.GetMol(), "join_components"


def repair_candidates(parent, target_formula, *, max_edits=2, max_plans=128):
    """Bounded breadth-first local edits, with RDKit determining H and valence.

    Retain at most 16 intermediate states per depth. Ring connectivity is
    preserved; disconnected parents may be joined. No target structure is used.
    """
    target = formula_counts(target_formula)
    if (_lower_bound(_heavy_delta(parent["mol"], target)) > max_edits
            or len(Chem.GetMolFrags(parent["mol"])) > max_edits + 1):
        return [], 0
    initial = Chem.MolFromSmiles(parent["smiles"])
    if initial is None:
        initial = Chem.Mol(parent["mol"])
    states, seen, outputs, tried = [(initial, ())], {parent["smiles"]}, {}, 0
    for depth in range(1, max_edits + 1):
        pending = [(iter(_edit_steps(mol, target)), path) for mol, path in states]
        next_states = []
        level_limit = max_plans if depth == max_edits else max(1, max_plans // max_edits)
        level_tried = 0
        while pending and tried < max_plans and level_tried < level_limit:
            following = []
            for iterator, path in pending:
                if tried >= max_plans or level_tried >= level_limit:
                    break
                try:
                    product, kind = next(iterator)
                except StopIteration:
                    continue
                following.append((iterator, path))
                tried += 1
                level_tried += 1
                try:
                    Chem.SanitizeMol(product)
                    counts = formula_counts(rdMolDescriptors.CalcMolFormula(product))
                    if depth == max_edits and counts != target:
                        continue
                    smiles = Chem.MolToSmiles(product)
                    if smiles in seen:
                        continue
                    seen.add(smiles)
                    steps = (*path, kind)
                    if counts == target:
                        candidate = _finish(product, smiles, target)
                        if candidate and candidate["key"] not in outputs:
                            outputs[candidate["key"]] = dict(candidate, edit_cost=depth, edit_operations=steps)
                    elif depth < max_edits:
                        heavy = _lower_bound(_heavy_delta(product, target))
                        fragments = len(Chem.GetMolFrags(product))
                        if heavy <= max_edits - depth and fragments <= max_edits - depth + 1:
                            next_states.append(((heavy, abs(counts["H"] - target["H"]), fragments, smiles), product, steps))
                except (ValueError, RuntimeError):
                    pass
                if tried >= max_plans or level_tried >= level_limit:
                    break
            pending = following
        states = [(mol, path) for _, mol, path in sorted(next_states, key=lambda x: x[0])[:16]]
        if not states:
            break
    return sorted(outputs.values(), key=lambda c: (c["edit_cost"], c["smiles"])), tried
