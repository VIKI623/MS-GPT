"""
Spectrum featurization utilities for MIST encoder input.

Converts raw peak_formula dicts + metadata into batched tensors
compatible with MIST FormulaTransformer.
"""

import numpy as np
import torch

from ms_gpt.external.mist.spectrum_featurizers import PeakFormulaFeaturizer
from ms_gpt.external.mist.adduct_classes import get_adduct_class_idx
from ms_gpt.external.mist.collision_energy import get_collision_energy_class_array

COLLISION_ENERGY_REF = dict(
    low_collision_energy_ref=13.0,
    mid_collision_energy_ref=31.0,
    high_collision_energy_ref=68.0,
)

def prepare_spectrum_batch(peak_formula_dicts, precursor_mzs, formulas,
                           adducts, collision_energies, device="cpu",
                           n_highest_peaks=60, collision_energy_ref=None,
                           cls_type_value=3):
    """Convert raw spectrum fields into a batched dict for MIST FormulaTransformer."""
    from functools import partial

    spec_featurizer = PeakFormulaFeaturizer(
        n_highest_peaks=n_highest_peaks, cls_type_value=cls_type_value
    )
    ce_ref = collision_energy_ref or COLLISION_ENERGY_REF
    ce_fn = partial(get_collision_energy_class_array, **ce_ref)

    peak_data_items = []
    for pf, pmz, form, adduct in zip(peak_formula_dicts, precursor_mzs, formulas, adducts):
        item = spec_featurizer(
            peak_formula_dict=pf,
            precursor_mz=pmz,
            formula=form,
            adduct=adduct,
        )
        peak_data_items.append(item)

    batch_size = len(peak_data_items)
    max_peaks = max(item["peak_num"] for item in peak_data_items)

    peak_nums = np.array([item["peak_num"] for item in peak_data_items], dtype=np.int64)
    formula_dim = peak_data_items[0]["peak_formulae_vector"].shape[1]

    peak_type = np.zeros((batch_size, max_peaks), dtype=np.int64)
    peak_adduct = np.zeros((batch_size, max_peaks), dtype=np.int64)
    peak_formulae_vec = np.zeros((batch_size, max_peaks, formula_dim), dtype=np.int64)
    peak_intensity = np.zeros((batch_size, max_peaks), dtype=np.float32)

    for i, item in enumerate(peak_data_items):
        n = item["peak_num"]
        peak_type[i, :n] = item["peak_type"]
        peak_adduct[i, :n] = item["peak_adduct"]
        peak_formulae_vec[i, :n] = item["peak_formulae_vector"]
        peak_intensity[i, :n] = item["peak_intensity"]

    spec_adducts = np.array(
        [get_adduct_class_idx(a) for a in adducts], dtype=np.int64
    )
    ce_array = ce_fn(np.array(collision_energies, dtype=np.float32))
    ce_raw = np.nan_to_num(
        np.array(collision_energies, dtype=np.float32), nan=0.0
    )

    return {
        "batch_peak_num": torch.tensor(peak_nums, device=device),
        "batch_peak_type": torch.tensor(peak_type, device=device),
        "batch_peak_adduct": torch.tensor(peak_adduct, device=device),
        "batch_peak_formulae_vector": torch.tensor(peak_formulae_vec, device=device),
        "batch_peak_intensity": torch.tensor(peak_intensity, dtype=torch.float32, device=device),
        "batch_spec_adduct": torch.tensor(spec_adducts, device=device),
        "batch_spec_collision_energy": torch.tensor(ce_array, device=device),
        "batch_spec_normalized_collision_energy": torch.tensor(
            ce_raw, dtype=torch.long, device=device
        ),
    }
