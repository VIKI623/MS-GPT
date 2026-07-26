"""Training data for posterior-aligned adaptation of the conditional MLM."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ms_gpt.data.mlm_pretraining_dataset import formula_str_to_vec, smiles_to_safe


RECOVERABILITY_WEIGHTS = {
    "frozen_hit": (0.05, 2.0),
    "residual": (1.0, 1.0),
    "low_coverage": (0.0, 0.5),
}


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
    ):
        directory = Path(cache_dir) / benchmark
        tag = f"real_{split}"
        self.meta = pd.read_parquet(directory / f"meta_{tag}.parquet")
        self.posterior = np.load(directory / f"pi_raw_{tag}.npy", mmap_mode="r")
        self.threshold_low = float(threshold_low)
        self.threshold_high = float(threshold_high)
        self.rng_seed = int(rng_seed)
        self._worker_rng = None

        labels = pd.read_parquet(recoverability_labels).set_index("cache_row")
        weights = dict(RECOVERABILITY_WEIGHTS)
        if group_weights:
            weights.update(group_weights)
        self.ce_weights = np.array(
            [weights[labels.loc[i, "group"]][0] for i in range(len(self.meta))],
            dtype=np.float32,
        )
        self.kl_weights = np.array(
            [weights[labels.loc[i, "group"]][1] for i in range(len(self.meta))],
            dtype=np.float32,
        )

        token_path = directory / f"tokens_{tag}.npz"
        if token_path.exists():
            token_data = np.load(token_path, allow_pickle=True)
            tokens = token_data["tokens"]
            valid = token_data["ok_mask"].astype(bool)
            self.rows = np.flatnonzero(valid).astype(np.int64)
            self.token_ids = [np.asarray(tokens[i], dtype=np.int64) for i in self.rows]
        else:
            bos, eos = tokenizer.bos_token_id, tokenizer.eos_token_id
            rows, token_ids = [], []
            for i, smiles in enumerate(self.meta["smiles"]):
                safe_string = smiles_to_safe(smiles)
                if not safe_string:
                    continue
                ids = tokenizer.encode(safe_string, add_special_tokens=False)
                rows.append(i)
                token_ids.append(np.asarray([bos] + ids[: max_len - 2] + [eos]))
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


def collate_posterior_aligned_batch(batch):
    """Pad SAFE targets and stack posterior-derived queries."""
    batch_size = len(batch)
    max_length = max(item["input_ids"].shape[0] for item in batch)
    input_ids = torch.zeros((batch_size, max_length), dtype=torch.long)
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
