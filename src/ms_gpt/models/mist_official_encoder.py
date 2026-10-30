"""Inference architecture for MIST official checkpoints."""

import torch
from torch import nn

from ms_gpt.data.mist_batch import as_mist_batch
from ms_gpt.external.mist_official import modules


class FormulaTransformer(modules.FormulaTransformer):
    def __init__(self, config):
        super().__init__(**{
            **config,
            "set_pooling": config.get("set_pooling", "cls"),
            "embed_instrument": config.get("embed_instrument", False),
            "inten_transform": "float",
        })

    def forward(self, batch):
        return super().forward(as_mist_batch(batch))


class MIST(nn.Module):
    cls_type_value = 3
    mist_implementation = "mist_official"

    def __init__(self, checkpoint):
        super().__init__()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.release_config = payload["config"]
        self.checkpoint_step = payload["step"]
        self.best_validation_tanimoto = payload.get("best_validation_tanimoto")
        self.model_config = self.release_config["model"]
        config = self.model_config
        self.encoder = FormulaTransformer(config)
        self.decoder = modules.FPGrowingModule(
            hidden_input_dim=config["hidden_size"],
            final_target_dim=config["output_size"],
            num_splits=config["refine_layers"],
        )
        self.load_state_dict(payload["ema_model"], strict=True)
        self.eval()

    def forward(self, batch):
        return self.decoder(self.encoder(batch))[-1]

    @torch.no_grad()
    def predict(self, batch):
        return self(batch)
