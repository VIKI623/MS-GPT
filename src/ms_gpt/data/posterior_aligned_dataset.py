"""Training data for posterior-aligned adaptation of the conditional MLM."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import safe
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from torch.utils.data import Dataset

from ms_gpt.data.mlm_pretraining_dataset import formula_str_to_vec, smiles_to_safe


RECOVERABILITY_WEIGHTS = {
    "frozen_hit": (0.0, 1.0),
    "residual": (2.0, 0.5),
    "low_coverage": (0.1, 0.25),
}


def encode_posterior_target(smiles, tokenizer, max_len, target_format):
    """Encode a complete structure with the selected decoder's pretraining rules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    Chem.RemoveStereochemistry(mol)
    canonical = Chem.MolToSmiles(mol, isomericSmiles=False)
    if target_format == "arrow":
        representation = smiles_to_safe(canonical) or canonical
    elif target_format == "tokenized_parquet":
        mol = Chem.MolFromSmiles(canonical)
        try:
            representation = safe.encode(canonical, canonical=True, randomize=False, ignore_stereo=True)
        except safe.SAFEFragmentationError:
            representation = canonical
    else:
        raise ValueError(f"Unknown target format: {target_format}")
    ids = tokenizer.encode(representation)
    if target_format == "arrow":
        ids = [tokenizer.bos_token_id] + ids + [tokenizer.eos_token_id]
    if len(ids) > max_len or tokenizer.unk_token_id in ids:
        return None
    decoded = Chem.MolFromSmiles(safe.decode(representation))
    if decoded is None:
        return None
    Chem.RemoveStereochemistry(decoded)
    if Chem.MolToSmiles(decoded, isomericSmiles=False) != canonical:
        return None
    return (np.asarray(ids, dtype=np.int64), Chem.MolToInchiKey(mol).split("-")[0],
            rdMolDescriptors.CalcMolFormula(mol))


class PosteriorAlignedDataset(Dataset):
    """Pair cached spectrum posteriors with SAFE targets and loss weights."""

    def __init__(
        self,
        cache_dir: str | Path,
        benchmark: str,
        tokenizer,
        recoverability_labels: str | Path,
        *,
        split: str = "train",
        max_len: int = 256,
        threshold_low: float,
        threshold_high: float,
        rng_seed: int = 42,
        group_weights: dict | None = None,
        gamma: float = 0.4,
        target_format: str = "arrow",
        token_file: str | Path | None = None,
    ):
        directory = Path(cache_dir) / benchmark
        tag = f"real_{split}"
        self.meta = pd.read_parquet(directory / f"meta_{tag}.parquet")
        self.posterior = np.load(directory / f"pi_raw_{tag}.npy", mmap_mode="r")
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)
        self.rng_seed = int(rng_seed)
        self._worker_rng = None

        labels = pd.read_parquet(recoverability_labels).set_index("cache_row").loc[self.meta.cache_row]
        groups = np.where(labels.frozen_hit_top100.astype(bool), "frozen_hit",
                          np.where(labels.posterior_coverage >= gamma, "residual", "low_coverage"))
        weights = dict(RECOVERABILITY_WEIGHTS)
        if group_weights:
            weights.update(group_weights)
        self.ce_weights = np.array(
            [weights[group][0] for group in groups],
            dtype=np.float32,
        )
        self.kl_weights = np.array(
            [weights[group][1] for group in groups],
            dtype=np.float32,
        )

        if token_file is not None:
            token_data = np.load(token_file, allow_pickle=True)
            tokens = token_data["tokens"]
            valid = token_data["ok_mask"].astype(bool)
            if len(tokens) != len(self.meta) or not np.array_equal(token_data["cache_rows"], self.meta.cache_row):
                raise ValueError("Token rows must match the selected posterior cache")
            self.rows = np.flatnonzero(valid).astype(np.int64)
            self.token_ids = [np.asarray(tokens[i], dtype=np.int64) for i in self.rows]
        else:
            targets = {smiles: encode_posterior_target(smiles, tokenizer, max_len, target_format)
                       for smiles in self.meta.smiles.unique()}
            rows, token_ids = [], []
            for i, row in enumerate(self.meta.itertuples()):
                target = targets[row.smiles]
                if target is None:
                    continue
                ids, key, formula = target
                if key != row.inchikey_14 or re.sub(r"[+-]\d*$", "", formula) != re.sub(r"[+-]\d*$", "", row.formula):
                    continue
                rows.append(i)
                token_ids.append(ids)
            self.rows = np.asarray(rows, dtype=np.int64)
            self.token_ids = token_ids

    def __len__(self):
        return len(self.rows)

    def _rng(self):
        if self._worker_rng is None:
            worker = torch.utils.data.get_worker_info()
            worker_id = worker.id if worker else 0
            self._worker_rng = np.random.default_rng(self.rng_seed + 99991 * worker_id)
        return self._worker_rng

    def __getitem__(self, index):
        cache_row = int(self.rows[index])
        posterior = np.asarray(self.posterior[cache_row], dtype=np.float32)
        threshold = self._rng().uniform(self.threshold_low, self.threshold_high)
        query = (posterior > threshold).astype(np.float32)
        return {
            "input_ids": torch.from_numpy(self.token_ids[index]),
            "fingerprint": torch.from_numpy(query),
            "formula_vec": torch.from_numpy(
                formula_str_to_vec(self.meta.iloc[cache_row]["formula"]).astype(np.float32)
            ),
            "ce_weight": torch.tensor(self.ce_weights[cache_row]),
            "kl_weight": torch.tensor(self.kl_weights[cache_row]),
        }


def collate_posterior_aligned_batch(batch, pad_token_id=0):
    """Pad SAFE targets and stack posterior-derived queries."""
    batch_size = len(batch)
    max_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.full((batch_size, max_length), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_length), dtype=torch.long)
    labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
    for i, item in enumerate(batch):
        length = item["input_ids"].shape[0]
        input_ids[i, :length] = item["input_ids"]
        attention_mask[i, :length] = 1
        labels[i, :length] = item["input_ids"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "fingerprint": torch.stack([item["fingerprint"] for item in batch]),
        "formula_vec": torch.stack([item["formula_vec"] for item in batch]),
        "ce_weight": torch.stack([item["ce_weight"] for item in batch]),
        "kl_weight": torch.stack([item["kl_weight"] for item in batch]),
    }
