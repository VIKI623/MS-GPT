"""Read per-spectrum benchmark annotations and metadata for MIST inference."""

from glob import glob

import pyarrow.parquet as pq


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


def load_benchmark_spectra(parquet_dir):
    """Load one record per spectrum, preserving sorted file order and row order."""
    records = []
    for path in sorted(glob(f"{parquet_dir}/**/*.parquet", recursive=True)):
        records.extend(pq.read_table(path, columns=BENCHMARK_COLUMNS).to_pylist())
    return records
