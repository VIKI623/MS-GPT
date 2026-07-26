"""
Benchmark spectrum dataset for evaluation.

Loads the released benchmark parquet data for structural
per-spectrum posterior inference and evaluation.
"""

from glob import glob

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset


BENCHMARK_COLUMNS = [
    "identifier",
    "formula",
    "smiles",
    "inchikey",
    "precursor_mz",
    "adduct",
    "collision_energy",
    "peak_formula",
]


class BenchmarkSpectrumDataset(Dataset):
    """
    Loads the released benchmark parquet data.

    Each item contains the peak_formula dicts and metadata needed by
    ``prepare_spectrum_batch()`` for MIST posterior inference.

    The parquet groups spectra by formula (group_size spectra per row).
    We flatten to individual spectra for random access.
    """

    def __init__(self, parquet_dir, split_name="train"):
        pattern = f"{parquet_dir}/**/*.parquet"
        parquet_files = sorted(glob(pattern, recursive=True))
        if not parquet_files:
            pattern = f"{parquet_dir}/*.parquet"
            parquet_files = sorted(glob(pattern))

        self.records = []
        for pf in parquet_files:
            table = pq.read_table(pf, columns=BENCHMARK_COLUMNS)
            df = table.to_pandas()
            for _, row in df.iterrows():
                if isinstance(row["identifier"], (list, np.ndarray)):
                    n_spectra = len(row["identifier"])
                    for j in range(n_spectra):
                        self.records.append({
                            "identifier": row["identifier"][j],
                            "formula": row["formula"],
                            "smiles": row["smiles"][j],
                            "inchikey": row["inchikey"][j],
                            "precursor_mz": float(row["precursor_mz"][j]),
                            "adduct": row["adduct"][j],
                            "collision_energy": float(row["collision_energy"][j]),
                            "peak_formula": row["peak_formula"][j],
                        })
                else:
                    self.records.append({
                        "identifier": row["identifier"],
                        "formula": row["formula"],
                        "smiles": row["smiles"],
                        "inchikey": row["inchikey"],
                        "precursor_mz": float(row["precursor_mz"]),
                        "adduct": row["adduct"],
                        "collision_energy": float(row["collision_energy"]),
                        "peak_formula": row["peak_formula"],
                    })

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return self.records[idx]
