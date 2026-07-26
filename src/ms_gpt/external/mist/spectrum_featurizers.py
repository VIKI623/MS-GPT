from enum import Enum
import numpy as np
import pandas as pd
from .chem_utils import formula_to_dense, ion2mass
from .adduct_classes import get_adduct_class_idx


class PeakType(Enum):
    Frag = 0
    CLS = 1


class SpectrumFeaturizerMixin:
    
    @classmethod
    def trim_peaks(cls, n_highest_peaks, inten_thresh, peak_intensity, *peak_items):
        # Filter by intensity threshold
        valid_indices = peak_intensity >= inten_thresh
        peak_intensity = peak_intensity[valid_indices]
        peak_items = tuple(item[valid_indices] for item in peak_items)
        
        if len(peak_intensity) <= n_highest_peaks:
            return peak_intensity, *peak_items
        
        # Select top N highest intensity peaks
        sorted_intensity_indices = np.argsort(peak_intensity)[::-1]
        highest_peak_indices = sorted_intensity_indices[:n_highest_peaks]
        
        peak_intensity = peak_intensity[highest_peak_indices]
        peak_items = tuple(item[highest_peak_indices] for item in peak_items)
        
        return peak_intensity, *peak_items
    
    @classmethod
    def normalize_intensities(cls, peak_intensity):
        if len(peak_intensity) == 0:
            return peak_intensity
        
        max_intensity = peak_intensity.max()
        
        if max_intensity == 0:
            return peak_intensity
        
        normalized_intensities = peak_intensity / max_intensity            
        return normalized_intensities
    
    @classmethod
    def preprocess_spectrum(cls, n_highest_peaks, inten_thresh, peak_intensity, *peak_items, intensity_transform="sqrt"):
        peak_intensity = cls.normalize_intensities(peak_intensity)
        peak_intensity, *peak_items = cls.trim_peaks(n_highest_peaks, inten_thresh, peak_intensity, *peak_items)
        if intensity_transform == "sqrt":
            peak_intensity = np.sqrt(peak_intensity)

        return peak_intensity, *peak_items
    
    @classmethod
    def electronic_denoising(cls, peak_intensity, peak_mzs):
        mass, intensity = peak_mzs, peak_intensity
        order = np.argsort(intensity)
        mass = mass[order]
        intensity = intensity[order]
        mass_confirmed = np.array([])
        intensity_confirmed = np.array([])
        while len(intensity) > 0:
            seed_intensity = np.max(intensity)
            idx_left = np.searchsorted(intensity, seed_intensity * 0.999, side='left')
            mass_temp = mass[idx_left:]
            intensity_temp = intensity[idx_left:]
            if len(mass_temp) <= 3:
                mass_confirmed = np.concatenate((mass_confirmed, mass_temp), dtype=np.float32)
                intensity_confirmed = np.concatenate((intensity_confirmed, intensity_temp), dtype=np.float32)
            intensity = intensity[0:idx_left]
            mass = mass[0:idx_left]
            
        if len(mass_confirmed) == 0:
            return peak_intensity, peak_mzs
        
        peak_mzs, peak_intensity = cls.sort_spectrum(mass_confirmed, intensity_confirmed)
        
        return peak_intensity, peak_mzs

    @classmethod
    def sort_spectrum(cls, peak_mzs, *peak_items):
        order = np.argsort(peak_mzs)
        peak_mzs = peak_mzs[order]
        peak_items = tuple(item[order] for item in peak_items)
        
        return peak_mzs, *peak_items


class PeakFormulaFeaturizer(SpectrumFeaturizerMixin):
    def __init__(self, n_highest_peaks=100, inten_thresh=0.0, cls_type_value=None, intensity_transform="sqrt"):
        self.n_highest_peaks = n_highest_peaks
        self.inten_thresh = inten_thresh
        self.cls_type_value = cls_type_value if cls_type_value is not None else PeakType.CLS.value
        self.intensity_transform = intensity_transform

    def __call__(self, peak_formula_dict, precursor_mz, formula, adduct):
        peak_intensity = np.asarray(peak_formula_dict["inten"], dtype=np.float32)
        peak_formulae = np.asarray(peak_formula_dict["formula"], dtype=object)
        peak_mzs = np.asarray(peak_formula_dict["formula_mass_no_adduct"], dtype=np.float32)

        peak_adducts = np.array([adduct] * len(peak_mzs), dtype=object)

        peak_intensity, peak_formulae, peak_mzs, peak_adducts = self.preprocess_spectrum(self.n_highest_peaks, self.inten_thresh, peak_intensity, peak_formulae, peak_mzs, peak_adducts, intensity_transform=self.intensity_transform)

        peak_adducts = [get_adduct_class_idx(adduct)] + [get_adduct_class_idx(peak_adduct) for peak_adduct in peak_adducts]
        peak_type = [self.cls_type_value] + (len(peak_intensity) * [PeakType.Frag.value])
        peak_intensity = np.insert(peak_intensity, 0, 1.0)
        peak_formulae = np.insert(peak_formulae, 0, formula)
        mass_diff = ion2mass[adduct]
        mol_exact_weight = precursor_mz - mass_diff
        peak_mzs = np.insert(peak_mzs, 0, mol_exact_weight)
        
        peak_num = len(peak_intensity)
        peak_formulae_vector = [formula_to_dense(peak_formula) for peak_formula in peak_formulae]
        
        data_item = {
            "peak_num": peak_num,
            "peak_mzs": peak_mzs,
            "peak_formulae_vector": np.vstack(peak_formulae_vector, dtype=np.int64),
            "peak_type": np.array(peak_type, dtype=np.int64),
            "peak_adduct": np.array(peak_adducts, dtype=np.int64),
            "peak_intensity": peak_intensity,
        }
        
        return data_item


class MagmaAugmentedPeakAtomFeaturizer:
    def __init__(self, d_magma_fingerprints=2048):
        self.d_magma_fingerprints = d_magma_fingerprints
    
    def _match_peak_mzs_with_magma_record(self, peak_attr_dict, peak_mzs):
        magma_df = pd.DataFrame(peak_attr_dict)
        magma_df = magma_df.dropna().reset_index(drop=True)
        
        match_record = [None] * len(peak_mzs)
        
        if len(magma_df) > 0:
            magma_masses = magma_df["mz_corrected"].values
            diff_mat = np.abs(peak_mzs[:, None] - magma_masses[None, :])
            min_inds = diff_mat.argmin(1)
            for row_ind, min_ind in enumerate(min_inds):
                diff_val = diff_mat[row_ind, min_ind]
                if diff_val < 1e-4:
                    magma_row = magma_df.loc[min_ind]
                    match_record[row_ind] = magma_row
        
        return match_record    
    
    def _get_peak_atom_mask(self, matched_magma_record, mol_atom_num):
        peak_atom_mask = np.zeros((len(matched_magma_record), mol_atom_num), dtype=bool)
        
        for idx, peak_magma_record in enumerate(matched_magma_record):
            if peak_magma_record is None:
                continue

            frag_inds_str = str(peak_magma_record["frag"])
            if "," in frag_inds_str:
                frag_inds_str_list = frag_inds_str.split(",")
            else:
                frag_inds_str_list = [frag_inds_str]
            
            frag_inds_int_list = []
            for frag_inds_str in frag_inds_str_list:
                try:
                    frag_inds_int = int(frag_inds_str.split('.')[0])
                    frag_inds_int_list.append(frag_inds_int)
                except (ValueError, TypeError):
                    continue
            
            # bin() 返回如 '0b1101' 的字符串，[2:] 去掉 '0b' 前缀。
            # [::-1] 反转字符串，使最低有效位（第0位）位于索引0处。
            for frag_inds_int in frag_inds_int_list:
                binary_str_rev = bin(frag_inds_int)[2:][::-1]
                for i in range(min(len(binary_str_rev), mol_atom_num)):
                    if binary_str_rev[i] == '1':
                        peak_atom_mask[idx, i] = True
                        
        peak_atom_mask = ~peak_atom_mask  # invert mask: True for non-attn atoms
        all_masked_rows = peak_atom_mask.all(axis=-1)
        peak_atom_mask[all_masked_rows] = False
        
        return peak_atom_mask
    
    def _get_peak_fingerprints(self, matched_magma_record):
        peak_magma_fingerprints = np.zeros((len(matched_magma_record), self.d_magma_fingerprints))
        
        for idx, peak_magma_record in enumerate(matched_magma_record):
            if peak_magma_record is None:
                continue
            
            frag_fp = peak_magma_record["frag_fp"]
            
            if pd.isna(frag_fp):
                continue
            
            multi_frag_fp = str(frag_fp).split(";")
            
            frag_fp = multi_frag_fp[0]
            magma_fp_bits = [
                int(float(i)) % self.d_magma_fingerprints
                for i in frag_fp.split(",")
            ]
            
            # update to 1 where active bit
            peak_magma_fingerprints[idx, magma_fp_bits] = 1

        return peak_magma_fingerprints
    
    def __call__(self, peak_attr_dict, peak_mzs, rdmol, attr_names=['peak_atom_mask']):
        matched_magma_record = self._match_peak_mzs_with_magma_record(peak_attr_dict=peak_attr_dict,
                                                                      peak_mzs=peak_mzs)
        
        attr_names = set(attr_names)
        attr_values = []
        
        if 'peak_atom_mask' in attr_names:
            peak_atom_mask = self._get_peak_atom_mask(matched_magma_record=matched_magma_record,
                                                      mol_atom_num=int(rdmol.GetNumAtoms()))
            attr_values.append(peak_atom_mask)
        
        if 'peak_fingerprints' in attr_names:
            peak_fingerprints = self._get_peak_fingerprints(matched_magma_record=matched_magma_record)
            attr_values.append(peak_fingerprints)
        
        
        if len(attr_values) == 0:
            return None
        elif len(attr_values) == 1:
            return attr_values[0]
        else:
            return tuple(attr_values)


preprocess_spectrum = SpectrumFeaturizerMixin.preprocess_spectrum
trim_peaks = SpectrumFeaturizerMixin.trim_peaks
normalize_intensities = SpectrumFeaturizerMixin.normalize_intensities
sort_spectrum = SpectrumFeaturizerMixin.sort_spectrum
electronic_denoising = SpectrumFeaturizerMixin.electronic_denoising


__all__ = [
    "PeakType",
    "PeakFormulaFeaturizer",
    "MagmaAugmentedPeakAtomFeaturizer",
    "preprocess_spectrum",
    "trim_peaks",
    "normalize_intensities",
    "sort_spectrum",
    "electronic_denoising"
]