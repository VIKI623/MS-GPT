"""
Datasets for conditional molecule-language model pretraining.

Two modes:
1. PreprocessedMLMDataset — memory-mapped Arrow IPC + numpy (fast, multi-worker safe)
2. StreamingMLMDataset — on-the-fly from HuggingFace (slow, single-worker fallback)
"""

import pickle
import re
from collections import defaultdict
from pathlib import Path
from glob import glob
from typing import Dict, Optional

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset, IterableDataset

import safe as sf
from rdkit import Chem
from rdkit.Chem import AllChem

ELEMENTS = [
    "C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I",
    "B", "Si", "Se", "As",
]


def smiles_to_fingerprint(smiles: str, n_bits: int = 4096, radius: int = 2) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    return np.array(fp, dtype=np.float32)


def smiles_to_formula_vec(smiles: str) -> Optional[np.ndarray]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    formula_str = Chem.rdMolDescriptors.CalcMolFormula(mol)
    return formula_str_to_vec(formula_str)


def formula_str_to_vec(formula_str: str) -> np.ndarray:
    counts = {}
    for element, count in re.findall(r"([A-Z][a-z]?)(\d*)", formula_str):
        if element:
            counts[element] = int(count) if count else 1
    vec = np.zeros(len(ELEMENTS), dtype=np.float32)
    for i, elem in enumerate(ELEMENTS):
        vec[i] = counts.get(elem, 0)
    return vec


def smiles_to_safe(smiles: str) -> Optional[str]:
    try:
        safe_str = sf.encode(smiles)
        if safe_str:
            return safe_str
    except Exception:
        pass
    return None


class PreprocessedMLMDataset(Dataset):
    """Memory-mapped dataset from preprocessed parquet shards.

    On first load, converts parquet shards to:
      - data.arrow: Arrow IPC (smiles + fp_indices as uint16), memory-mapped
      - formula_vecs.npy: [N, 14] float32, numpy memory-mapped
      - formula_groups.pkl: {formula_str: [indices]}, for FormulaGroupBatchSampler

    All processes share mmap pages — safe with num_workers > 0 and DDP.
    """

    def __init__(self, parquet_dir, tokenizer, max_len=256, fp_bits=4096):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fp_bits = fp_bits

        parquet_dir = Path(parquet_dir)
        arrow_path = parquet_dir / "data.arrow"
        fvec_path = parquet_dir / "formula_vecs.npy"
        groups_path = parquet_dir / "formula_groups.pkl"

        if not arrow_path.exists():
            self.convert(parquet_dir)

        self._mmap = pa.memory_map(str(arrow_path))
        self._table = ipc.open_file(self._mmap).read_all()
        self._smiles = self._table.column("smiles")
        self._fp_idx = self._table.column("fp_indices")
        self._n = len(self._table)

        self.formula_vecs = np.load(str(fvec_path), mmap_mode="r")

        if groups_path.exists():
            with open(groups_path, "rb") as f:
                self.formula_groups = pickle.load(f)
        else:
            self.formula_groups = None

        self.formula_list = None

    @staticmethod
    def convert(parquet_dir):
        """One-time: parquet shards → Arrow IPC + numpy mmap + formula groups."""
        parquet_dir = Path(parquet_dir)
        parquet_files = sorted(glob(str(parquet_dir / "shard_*.parquet")))
        if not parquet_files:
            raise FileNotFoundError(f"No shard_*.parquet in {parquet_dir}")

        arrow_path = parquet_dir / "data.arrow"
        fvec_path = parquet_dir / "formula_vecs.npy"
        groups_path = parquet_dir / "formula_groups.pkl"

        print(f"Converting {len(parquet_files)} shards → Arrow IPC ...")

        total_rows = sum(pq.read_metadata(f).num_rows for f in parquet_files)
        print(f"  {total_rows} molecules total")

        n_elem = len(ELEMENTS)
        formula_vecs = np.zeros((total_rows, n_elem), dtype=np.float32)
        formula_groups = defaultdict(list)

        arrow_schema = pa.schema([
            ("smiles", pa.utf8()),
            ("fp_indices", pa.list_(pa.uint16())),
        ])
        sink = pa.OSFile(str(arrow_path), "wb")
        writer = ipc.new_file(sink, arrow_schema)

        offset = 0
        for i, pf in enumerate(parquet_files):
            table = pq.read_table(pf)
            n = len(table)

            fv = np.array(table.column("formula_vec").to_pylist(), dtype=np.float32)
            formula_vecs[offset:offset + n] = fv

            for j, f in enumerate(table.column("formula").to_pylist()):
                formula_groups[f].append(offset + j)

            fp_arr = table.column("fp_indices").combine_chunks()
            fp_vals = fp_arr.values.cast(pa.uint16())
            fp_uint16 = pa.ListArray.from_arrays(fp_arr.offsets, fp_vals)

            batch = pa.record_batch(
                [table.column("smiles").combine_chunks(), fp_uint16],
                schema=arrow_schema,
            )
            writer.write_batch(batch)

            offset += n
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(parquet_files)} shards")

        writer.close()
        sink.close()

        np.save(str(fvec_path), formula_vecs)

        for k, v in formula_groups.items():
            formula_groups[k] = np.array(v, dtype=np.int64)
        with open(groups_path, "wb") as f:
            pickle.dump(dict(formula_groups), f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"  Arrow:  {arrow_path.stat().st_size / 1e9:.1f} GB")
        print(f"  Numpy:  {fvec_path.stat().st_size / 1e9:.1f} GB")
        print(f"  Groups: {len(formula_groups)} unique formulas")

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        smiles = self._smiles[idx].as_py()
        safe_str = smiles_to_safe(smiles)
        if safe_str is None:
            safe_str = smiles

        tokens = self.tokenizer.encode(safe_str)
        if len(tokens) > self.max_len - 2:
            tokens = tokens[: self.max_len - 2]

        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        pad = self.tokenizer.pad_token_id

        token_seq = [bos] + tokens + [eos]
        seq_len = len(token_seq)
        pad_len = self.max_len - seq_len

        input_ids = torch.tensor(token_seq + [pad] * pad_len, dtype=torch.long)
        labels = input_ids.clone()
        labels[seq_len:] = -100

        attention_mask = torch.zeros(self.max_len, dtype=torch.long)
        attention_mask[:seq_len] = 1

        fp = torch.zeros(self.fp_bits, dtype=torch.float32)
        indices = self._fp_idx[idx].as_py()
        if indices:
            fp[indices] = 1.0

        formula_vec = torch.from_numpy(self.formula_vecs[idx].copy())

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "fingerprint": fp,
            "formula_vec": formula_vec,
        }


class StreamingMLMDataset(IterableDataset):
    """
    Streaming dataset fallback for large-scale data without preprocessing.
    Slow due to on-the-fly RDKit + SAFE computation with num_workers=0.
    """

    def __init__(self, hf_dataset, tokenizer, max_len=256,
                 fp_bits=4096, fp_radius=2):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.fp_bits = fp_bits
        self.fp_radius = fp_radius

    def __iter__(self):
        for item in self.hf_dataset:
            smiles = item.get("smiles") or item.get("SMILES")
            if not smiles:
                continue

            fp = smiles_to_fingerprint(smiles, self.fp_bits, self.fp_radius)
            formula_vec = smiles_to_formula_vec(smiles)
            safe_str = smiles_to_safe(smiles)

            if fp is None or formula_vec is None or safe_str is None:
                continue

            tokens = self.tokenizer.encode(safe_str)
            if len(tokens) > self.max_len - 2:
                continue

            bos = self.tokenizer.bos_token_id
            eos = self.tokenizer.eos_token_id
            pad = self.tokenizer.pad_token_id

            token_seq = [bos] + tokens + [eos]
            seq_len = len(token_seq)
            pad_len = self.max_len - seq_len

            input_ids = torch.tensor(token_seq + [pad] * pad_len, dtype=torch.long)
            labels = input_ids.clone()
            labels[seq_len:] = -100

            attention_mask = torch.zeros(self.max_len, dtype=torch.long)
            attention_mask[:seq_len] = 1

            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
                "fingerprint": torch.tensor(fp),
                "formula_vec": torch.tensor(formula_vec),
            }
