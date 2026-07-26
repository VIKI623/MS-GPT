"""chem_utils.py"""

import re
import numpy as np
from functools import reduce

import torch
from rdkit import Chem
from rdkit.Chem import Atom
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
import datamol as dm
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdApp.info')


try:
    from rdkit.Chem.MolStandardize.tautomer import TautomerCanonicalizer, TautomerTransform
    _RD_TAUTOMER_CANONICALIZER = 'v1'
    _TAUTOMER_TRANSFORMS = (
        TautomerTransform('1,3 heteroatom H shift',
                          '[#7,S,O,Se,Te;!H0]-[#7X2,#6,#15]=[#7,#16,#8,Se,Te]'),
        TautomerTransform('1,3 (thio)keto/enol r', '[O,S,Se,Te;X2!H0]-[C]=[C]'),
    )
except ModuleNotFoundError:
    from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator  # newer rdkit
    _RD_TAUTOMER_CANONICALIZER = 'v2'

P_TBL = Chem.GetPeriodicTable()

ROUND_FACTOR = 4

ELECTRON_MASS = 0.000548579909
CHEM_FORMULA_SIZE = "([A-Z][a-z]*)([0-9]*)"

VALID_ELEMENTS = [
    "C",
    "H",
    "As",
    "B",
    "Br",
    "Cl",
    "Co",
    "F",
    "Fe",
    "I",
    "K",
    "N",
    "Na",
    "O",
    "P",
    "S",
    "Se",
    "Si",
]
VALID_ATOM_NUM = [Atom(i).GetAtomicNum() for i in VALID_ELEMENTS]

CHEM_ELEMENT_NUM = len(VALID_ELEMENTS)

ATOM_NUM_TO_ONEHOT = torch.zeros((max(VALID_ATOM_NUM) + 1, CHEM_ELEMENT_NUM))

# Convert to onehot
ATOM_NUM_TO_ONEHOT[VALID_ATOM_NUM, torch.arange(CHEM_ELEMENT_NUM)] = 1

VALID_MONO_MASSES = np.array(
    [P_TBL.GetMostCommonIsotopeMass(i) for i in VALID_ELEMENTS]
)
CHEM_MASSES = VALID_MONO_MASSES[:, None]

ELEMENT_VECTORS = np.eye(len(VALID_ELEMENTS), dtype=np.int64)
ELEMENT_VECTORS_MASS = np.hstack([ELEMENT_VECTORS, CHEM_MASSES])
ELEMENT_TO_MASS = dict(zip(VALID_ELEMENTS, CHEM_MASSES.squeeze()))

ELEMENT_DIM_MASS = len(ELEMENT_VECTORS_MASS[0])
ELEMENT_DIM = len(ELEMENT_VECTORS[0])

# Reasonable normalization vector for elements
# Estimated by max counts (+ 1 when zero)
NORM_VEC = np.array([81, 158, 2, 1, 3, 10, 1, 17, 1, 6, 1, 19, 2, 34, 6, 6, 2, 6])

NORM_VEC_MASS = np.array(NORM_VEC.tolist() + [1471])

# Assume 64 is the highest repeat of any 1 atom
MAX_ELEMENT_NUM = 64

element_to_ind = dict(zip(VALID_ELEMENTS, np.arange(len(VALID_ELEMENTS))))
element_to_position = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS))
element_to_position_mass = dict(zip(VALID_ELEMENTS, ELEMENT_VECTORS_MASS))

# Element group (periodic table group) mapping for GGNN
ELEMENT_TO_GROUP = {
    "C": 4, "H": 0, "As": 3, "B": 2, "Br": 6, "Cl": 6, "Co": 7, "F": 6,
    "Fe": 7, "I": 6, "K": 1, "N": 3, "Na": 1, "O": 5, "P": 3, "S": 5,
    "Se": 5, "Si": 4,
}
ELEMENT_GROUP_DIM = len(set(ELEMENT_TO_GROUP.values()))  # 8 unique groups
ELEMENT_GROUP_VECTORS = np.eye(ELEMENT_GROUP_DIM)
element_to_group = {k: ELEMENT_GROUP_VECTORS[v] for k, v in ELEMENT_TO_GROUP.items()}

# GGNN-specific constants (for molecular featurization)
MAX_H = 6  # Maximum hydrogen count (one-hot encoding dimension)
MAX_BONDS = 4  # Number of bond types (single=0, double=1, triple=2, aromatic=3)
DEFAULT_PE_EMBED_K = 14  # Default positional encoding dimension


# Map ion to adduct mass, don't use electron
ion2mass = {
    # positive mode - basic adducts
    "[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+Na]+": ELEMENT_TO_MASS["Na"] - ELECTRON_MASS,
    "[M+K]+": ELEMENT_TO_MASS["K"] - ELECTRON_MASS,
    "[M+NH4]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,
    "[M+H3N+H]+": ELEMENT_TO_MASS["N"] + ELEMENT_TO_MASS["H"] * 4 - ELECTRON_MASS,  # alias for [M+NH4]+
    "[M]+": 0 - ELECTRON_MASS,
    
    # positive mode - in-source losses
    "[M+H-H2O]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M-H2O+H]+": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,  # alias
    "[M+H-2H2O]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,
    "[M-H4O2+H]+": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 3 - ELECTRON_MASS,  # alias
    
    # positive mode - metal substitution
    "[M+2Na-H]+": 2 * ELEMENT_TO_MASS["Na"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    "[M+2K-H]+": 2 * ELEMENT_TO_MASS["K"] - ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
    
    # negative mode - basic adducts
    "[M]-": 0 + ELECTRON_MASS,
    "[M-H]-": -ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M+Cl]-": ELEMENT_TO_MASS["Cl"] + ELECTRON_MASS,
    "[M+FA-H]-": 46.00548 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,  # Formic acid adduct
    "[M+HCOO]-": 46.00548 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,  # alias for [M+FA-H]-
    "[M+Ac-H]-": 60.02113 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,  # Acetic acid adduct
    "[M+CH3COO]-": 60.02113 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,  # alias for [M+Ac-H]-
    
    # negative mode - in-source losses
    "[M-H-H2O]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,
    "[M-H2O-H]-": -ELEMENT_TO_MASS["O"] - ELEMENT_TO_MASS["H"] * 3 + ELECTRON_MASS,  # alias
    "[M-H-2H2O]-": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 5 + ELECTRON_MASS,
    "[M-2H2O-H]-": -ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] * 5 + ELECTRON_MASS,  # alias
    "[M-H-CO2]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,
    "[M-CO2-H]-": -ELEMENT_TO_MASS["C"] - ELEMENT_TO_MASS["O"] * 2 - ELEMENT_TO_MASS["H"] + ELECTRON_MASS,  # alias
    "[M-H-NH3]-": -ELEMENT_TO_MASS["N"] - ELEMENT_TO_MASS["H"] * 4 + ELECTRON_MASS,
    "[M-NH3-H]-": -ELEMENT_TO_MASS["N"] - ELEMENT_TO_MASS["H"] * 4 + ELECTRON_MASS,  # alias
    
    # negative mode - metal substitution
    "[M-2H+Na]-": -2 * ELEMENT_TO_MASS["H"] + ELEMENT_TO_MASS["Na"] + ELECTRON_MASS,
    "[M-2H+K]-": -2 * ELEMENT_TO_MASS["H"] + ELEMENT_TO_MASS["K"] + ELECTRON_MASS,
}


def is_positive_adduct(adduct_str: str) -> bool:
    """Check the adduct string is positive or negative (return True if positive)"""
    return adduct_str[-1] == '+'


# Define rdbe mult
rdbe_mult = np.zeros_like(ELEMENT_VECTORS[0])
els = ["C", "N", "P", "H", "Cl", "Br", "I", "F"]
weights = [2, 1, 1, -1, -1, -1, -1, -1]
for k, v in zip(els, weights):
    rdbe_mult[element_to_ind[k]] = v


def cross_sum(x, y):
    """cross_sum."""
    return (np.expand_dims(x, 0) + np.expand_dims(y, 1)).reshape(-1, y.shape[-1])


def get_all_subsets_dense(
    dense_formula: str, element_vectors
) -> tuple[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        dense_formula (str, element_vectors): _description_
        np (_type_): _description_

    Returns:
        _type_: _description_
    """

    non_zero = np.argwhere(dense_formula > 0).flatten()

    vectorized_formula = []
    for nonzero_ind in non_zero:
        temp = element_vectors[nonzero_ind] * np.arange(
            0, dense_formula[nonzero_ind] + 1
        ).reshape(-1, 1)
        vectorized_formula.append(temp)

    zero_vec = np.zeros((1, element_vectors.shape[-1]))
    cross_prod = reduce(cross_sum, vectorized_formula, zero_vec)

    cross_prod_inds = rdbe_filter(cross_prod)
    cross_prod = cross_prod[cross_prod_inds]
    all_masses = cross_prod.dot(VALID_MONO_MASSES)
    return cross_prod, all_masses


def get_all_subsets(chem_formula: str):
    dense_formula = formula_to_dense(chem_formula)
    return get_all_subsets_dense(dense_formula, element_vectors=ELEMENT_VECTORS)


def rdbe_filter(cross_prod):
    """rdbe_filter.
    Args:
        cross_prod:
    """
    rdbe_total = 1 + 0.5 * cross_prod.dot(rdbe_mult)
    filter_inds = np.argwhere(rdbe_total >= 0).flatten()
    return filter_inds


def formula_to_dense(chem_formula: str) -> np.ndarray:
    """formula_to_dense.

    Args:
        chem_formula (str): Input chemical formal
    Return:
        np.ndarray of vector

    """
    total_onehot = []
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        one_hot = element_to_position[chem_symbol].copy().reshape(1, -1)
        one_hot_repeats = np.repeat(one_hot, repeats=num, axis=0)
        total_onehot.append(one_hot_repeats)

    # Check if null
    if len(total_onehot) == 0:
        dense_vec = np.zeros(len(element_to_position), dtype=np.int64)
    else:
        dense_vec = np.vstack(total_onehot, dtype=np.int64).sum(0)

    return dense_vec


def formula_mass(chem_formula: str) -> float:
    """get formula mass"""
    mass = 0
    for (chem_symbol, num) in re.findall(CHEM_FORMULA_SIZE, chem_formula):
        # Convert num to int
        num = 1 if num == "" else int(num)
        mass += ELEMENT_TO_MASS[chem_symbol] * num
    return mass


def formula_difference(formula_1, formula_2):
    """formula_1 - formula_2"""
    form_1 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_1)
    }
    form_2 = {
        chem_symbol: (int(num) if num != "" else 1)
        for chem_symbol, num in re.findall(CHEM_FORMULA_SIZE, formula_2)
    }

    for k, v in form_2.items():
        form_1[k] = form_1[k] - form_2[k]
    out_formula = "".join([f"{k}{v}" for k, v in form_1.items() if v > 0])
    return out_formula

def vec_to_formula(form_vec) -> str:
    """Convert vector back to formula string using Hill system ordering.
    
    Hill system: C first, then H, then all other elements in alphabetical order.
    """
    elements_with_counts = []
    for i in np.argwhere(form_vec > 0).flatten():
        el = VALID_ELEMENTS[i]
        ct = int(form_vec[i])
        elements_with_counts.append((el, ct))
    
    if not elements_with_counts:
        return ""
    
    # Sort using Hill system: C first, H second, then alphabetical
    def hill_sort_key(item):
        el = item[0]
        if el == "C":
            return (0, el)
        elif el == "H":
            return (1, el)
        else:
            return (2, el)
    
    elements_with_counts.sort(key=hill_sort_key)
    
    # Build formula string
    build_str = ""
    for el, ct in elements_with_counts:
        new_item = f"{el}{ct}" if ct > 1 else f"{el}"
        build_str = build_str + new_item
    return build_str


def uncharged_formula(mol, mol_type="mol") -> str:
    """Compute uncharged formula"""
    if mol_type == "mol":
        chem_formula = CalcMolFormula(mol)
    elif mol_type == "smiles":
        mol = Chem.MolFromSmiles(mol)
        if mol is None:
            return None
        chem_formula = CalcMolFormula(mol)
    else:
        raise ValueError()

    return re.findall(r"^([^\+,^\-]*)", chem_formula)[0]


def canonical_mol_from_inchi(inchi):
    """Canonicalize mol after Chem.MolFromInchi
    Note that this function may be 50 times slower than Chem.MolFromInchi"""
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    if _RD_TAUTOMER_CANONICALIZER == 'v1':
        _molvs_t = TautomerCanonicalizer(transforms=_TAUTOMER_TRANSFORMS)
        mol = _molvs_t.canonicalize(mol)
    else:
        _te = TautomerEnumerator()
        mol = _te.Canonicalize(mol)
    return mol


def form_from_smi(smi: str) -> str:
    """form_from_smi.

    Args:
        smi (str): smi

    Return:
        str
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return CalcMolFormula(mol)


def inchikey_from_smiles(smi: str) -> str:
    """inchikey_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchiKey(mol)


def inchi_from_smiles(smi: str) -> str:
    """inchi_from_smiles.

    Args:
        smi (str): smi

    Returns:
        str:
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ""
    else:
        return Chem.MolToInchi(mol)


def smiles_from_inchi(inchi: str) -> str:
    """smiles_from_inchi.

    Args:
        inchi (str): inchi

    Returns:
        str:
    """
    mol = canonical_mol_from_inchi(inchi)
    if mol is None:
        return ""
    else:
        return Chem.MolToSmiles(mol)


def clipped_ppm(mass_diff: np.ndarray, parentmass: np.ndarray) -> np.ndarray:
    """clipped_ppm.

    Args:
        mass_diff (np.ndarray): mass_diff
        parentmass (np.ndarray): parentmass

    Returns:
        np.ndarray:
    """
    parentmass_copy = parentmass * 1
    parentmass_copy[parentmass < 200] = 200
    ppm = mass_diff / parentmass_copy * 1e6
    return ppm


def normalize_mol(mol: Chem.Mol):
    mol = dm.standardize_mol(mol)
    mol = dm.to_mol(mol, ordered=True, remove_hs=True)
    return mol


def mol_to_planar_inchikey(rdmol: Chem.Mol):
    if rdmol is None:
        return None
    
    planar_inchikey = Chem.MolToInchiKey(rdmol).split("-")[0]
    
    return planar_inchikey