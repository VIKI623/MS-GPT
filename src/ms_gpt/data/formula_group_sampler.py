"""Formula-aware batch sampler for formula-grouped pretraining (C1).

Hosts ``FormulaGroupBatchSampler``, used by ``scripts/pretrain_mlm.py`` to batch
molecules by molecular formula during FP+formula-conditioned pretraining.
"""

from collections import defaultdict

import numpy as np
from torch.utils.data import Sampler


class FormulaGroupBatchSampler(Sampler):
    """Two-pool formula-aware batch sampler.

    - Pure pool: formulas with >= batch_size molecules → each formula shuffled
      internally, chunked into pure same-formula batches.
    - Mixed pool: formulas with < batch_size molecules → all merged, shuffled,
      chunked into mixed batches.

    All batches from both pools are interleaved randomly each epoch.
    No data is discarded.
    """

    def __init__(self, formulas, batch_size, drop_last=True, seed=42,
                 rank=0, world_size=1):
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0
        self.rank = rank
        self.world_size = world_size

        if isinstance(formulas, dict):
            formula_to_idx = formulas
        else:
            formula_to_idx = defaultdict(list)
            for i, f in enumerate(formulas):
                formula_to_idx[f].append(i)

        self.pure_groups = []
        mixed_list = []
        for indices in formula_to_idx.values():
            arr = np.asarray(indices, dtype=np.int64)
            if len(arr) >= batch_size:
                self.pure_groups.append(arr)
            else:
                mixed_list.extend(arr.tolist())
        self.mixed_indices = np.array(mixed_list, dtype=np.int64)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def set_start_batch(self, start):
        self._start_batch = start

    def _build_all_batches(self, rng):
        """Build full batch list (deterministic given rng). All ranks produce the same list."""
        bs = self.batch_size
        all_batches = []

        for group in self.pure_groups:
            indices = group.copy()
            rng.shuffle(indices)
            for start in range(0, len(indices) - bs + 1, bs):
                all_batches.append(indices[start:start + bs])
            remainder = len(indices) % bs
            if remainder and not self.drop_last:
                all_batches.append(indices[len(indices) - remainder:])

        mixed = self.mixed_indices.copy()
        rng.shuffle(mixed)
        for start in range(0, len(mixed) - bs + 1, bs):
            all_batches.append(mixed[start:start + bs])
        remainder = len(mixed) % bs
        if remainder and not self.drop_last:
            all_batches.append(mixed[len(mixed) - remainder:])

        rng.shuffle(all_batches)
        return all_batches

    def __iter__(self):
        rng = np.random.RandomState(self.seed + self.epoch)
        all_batches = self._build_all_batches(rng)
        usable = (len(all_batches) // self.world_size) * self.world_size
        start = getattr(self, '_start_batch', 0)
        rank_indices = list(range(self.rank, usable, self.world_size))
        for idx in rank_indices[start:]:
            yield all_batches[idx].tolist()
        self._start_batch = 0

    def __len__(self):
        bs = self.batch_size
        n = 0
        for g in self.pure_groups:
            n += len(g) // bs
            if len(g) % bs and not self.drop_last:
                n += 1
        n += len(self.mixed_indices) // bs
        if len(self.mixed_indices) % bs and not self.drop_last:
            n += 1
        return n // self.world_size
