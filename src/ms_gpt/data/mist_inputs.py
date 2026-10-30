"""Shared spectrum inputs for the official and FRIGID MIST encoders."""

import re

import numpy as np
import torch


ELEMENTS = (
    "C", "H", "As", "B", "Br", "Cl", "Co", "F", "Fe",
    "I", "K", "N", "Na", "O", "P", "S", "Se", "Si",
)
ELEMENT_INDEX = {element: index for index, element in enumerate(ELEMENTS)}
ION_INDEX = {
    "[M+H]+": 0,
    "[M+Na]+": 1,
    "[M+K]+": 2,
    "[M-H2O+H]+": 3,
    "[M+H-H2O]+": 3,
    "[M+H3N+H]+": 4,
    "[M+NH4]+": 4,
    "[M]+": 5,
    "[M-H4O2+H]+": 6,
    "[M+H-2H2O]+": 6,
}
FORMULA_PATTERN = re.compile(r"([A-Z][a-z]*)([0-9]*)")
FRAGMENT_TYPE = 0
CLS_TYPE = 3


def formula_to_vector(formula):
    """Encode elemental counts in the shared MIST element order."""
    vector = np.zeros(len(ELEMENTS), dtype=np.float32)
    for element, count in FORMULA_PATTERN.findall(formula):
        vector[ELEMENT_INDEX[element]] += int(count) if count else 1
    return vector


def featurize_spectrum(record, *, max_peaks):
    """Select annotated peaks, normalize by their maximum, then take sqrt.

    ``peak_formula.inten`` contains linear intensities, either absolute or
    source-relative. CLS is appended after fragment intensity processing.
    No checkpoint metadata or unannotated spectrum peaks are consulted.
    """
    if max_peaks < 1:
        raise ValueError("max_peaks must be positive")
    peak_formula = record["peak_formula"]
    intensity = np.asarray(peak_formula["inten"], dtype=np.float64)
    selected = np.argsort(intensity)[::-1][:max_peaks]
    intensity = intensity[selected]
    denominator = float(intensity.max()) if len(intensity) else 0.0
    if denominator > 0:
        intensity = np.sqrt(intensity / denominator)

    formulas = [peak_formula["formula"][index] for index in selected]
    formulas.append(record["formula"])
    count = len(formulas)
    return {
        "form_vec": np.stack([formula_to_vector(formula) for formula in formulas]),
        "ion_vec": np.full(count, ION_INDEX[record["adduct"]], dtype=np.int64),
        "intens": np.append(intensity, 1.0).astype(np.float32),
        "types": np.asarray([FRAGMENT_TYPE] * len(selected) + [CLS_TYPE], dtype=np.int64),
        "num_peaks": count,
        "selected_indices": selected,
        "normalization_denominator": denominator,
    }


def prepare_mist_batch(records, *, max_peaks, device="cpu"):
    """Build the same padded input tensors independently of MIST weights."""
    items = [featurize_spectrum(record, max_peaks=max_peaks) for record in records]
    lengths = np.asarray([item["num_peaks"] for item in items], dtype=np.int64)
    shape = (len(items), int(lengths.max()))
    forms = np.zeros((*shape, len(ELEMENTS)), dtype=np.float32)
    ions = np.zeros(shape, dtype=np.int64)
    intensities = np.zeros(shape, dtype=np.float32)
    types = np.zeros(shape, dtype=np.int64)
    for index, item in enumerate(items):
        count = item["num_peaks"]
        forms[index, :count] = item["form_vec"]
        ions[index, :count] = item["ion_vec"]
        intensities[index, :count] = item["intens"]
        types[index, :count] = item["types"]
    return {
        "batch_peak_formulae_vector": torch.as_tensor(forms, device=device),
        "batch_peak_adduct": torch.as_tensor(ions, device=device),
        "batch_peak_intensity": torch.as_tensor(intensities, device=device),
        "batch_peak_type": torch.as_tensor(types, device=device),
        "batch_peak_num": torch.as_tensor(lengths, device=device),
        "batch_spec_instrument": torch.zeros(len(items), dtype=torch.long, device=device),
    }
