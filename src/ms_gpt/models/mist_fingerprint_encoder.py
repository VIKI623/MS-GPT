"""
MIST fingerprint predictor: spectrum → 4096-bit Morgan FP.

The shared components reproduce the checkpoint architecture used by the two
benchmark-paired MIST encoders.
"""

import torch
import torch.nn as nn
import numpy as np

class FingerprintGrowingModule(nn.Module):
    """FPGrowingModule matching the original MIST checkpoint layout."""

    def __init__(self, hidden_input_dim=512, final_target_dim=4096,
                 num_splits=4, reduce_factor=2):
        super().__init__()
        layer_dims = [
            int(np.ceil(final_target_dim / (reduce_factor ** s)))
            for s in range(num_splits + 1)
        ][::-1]
        self.output_dims = layer_dims

        self.initial_predict = nn.Sequential(
            nn.Linear(hidden_input_dim, layer_dims[0]),
            nn.Sigmoid(),
        )
        predict_bricks, gate_bricks = [], []
        for i, dim in enumerate(layer_dims[:-1]):
            out_dim = layer_dims[i + 1]
            predict_bricks.append(nn.Sequential(nn.Linear(dim, out_dim), nn.Sigmoid()))
            gate_bricks.append(nn.Sequential(nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid()))
        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden):
        cur = self.initial_predict(hidden)
        preds = [cur]
        for predict_brick, gate_brick in zip(self.predict_bricks, self.gate_bricks):
            cur = gate_brick(hidden) * predict_brick(cur)
            preds.append(cur)
        return preds
