"""FRIGID legacy checkpoint inference with explicit formula-feature settings."""

import torch
from torch import nn

from ms_gpt.data.mist_batch import as_mist_batch
from ms_gpt.external.mist_frigid_legacy.modules import FormulaTransformer, FPGrowingModule


class MIST(nn.Module):
    """Load the encoder and fingerprint head from a FRIGID legacy state dictionary.

    Input tensors are prepared by the shared MS-GPT data layer. The default
    features include precursor losses, the checkpoint's instrument embedding,
    and absolute formula differences for all fragment pairs. Padding keys are
    excluded from attention; peak selection is configured separately.
    """

    cls_type_value = 3
    mist_implementation = "mist_frigid_legacy"

    def __init__(self, checkpoint_path, benchmark, device="cpu", *,
                 no_diffs=False, embed_instrument=True, pairwise_subset_only=False):
        super().__init__()
        hidden = {"nplib1": 512, "massspecgym": 640}[benchmark]
        self.model_config = {
            "hidden_size": hidden,
            "peak_attn_layers": 2,
            "num_heads": 8,
            "spectra_dropout": 0.1,
            "pairwise_featurization": True,
            "no_diffs": no_diffs,
            "pairwise_subset_only": pairwise_subset_only,
            "form_embedder": "pos-cos",
            "set_pooling": "cls",
            "embed_instrument": embed_instrument,
            "inten_transform": "float",
            "output_size": 4096,
        }
        self.encoder = FormulaTransformer(**self.model_config)
        self.decoder = FPGrowingModule(
            hidden_input_dim=hidden, final_target_dim=4096, num_splits=4,
        )
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = state.get("state_dict", state)
        self.encoder.load_state_dict({
            key.removeprefix("spectra_encoder.0."): value
            for key, value in state.items() if key.startswith("spectra_encoder.0.")
        }, strict=True)
        self.decoder.load_state_dict({
            key.removeprefix("spectra_encoder.2."): value
            for key, value in state.items() if key.startswith("spectra_encoder.2.")
        }, strict=True)
        self.to(device)
        self.eval()

    def forward(self, batch):
        return self.decoder(self.encoder(as_mist_batch(batch)))[-1]

    @torch.no_grad()
    def predict(self, batch):
        return self(batch)
