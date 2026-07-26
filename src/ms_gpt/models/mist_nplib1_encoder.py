"""NPLIB1-paired MIST spectrum-to-fingerprint posterior encoder.

Shares the same architecture style as MassSpecGymMISTEncoder (``spectra_encoder.0.*`` prefix,
learned instrument_embedder), but with hidden_size=512 (vs MassSpecGym's 640) and trained on
NPLIB1 data. Use this encoder only with the NPLIB1 benchmark.

The checkpoint layout uses the ``spectra_encoder.0.*`` prefix and a learned
instrument embedding.
"""

import torch
import torch.nn as nn

from ms_gpt.external.mist.nn_module import FormulaTransformer
from ms_gpt.models.mist_fingerprint_encoder import FingerprintGrowingModule


class NPLIB1MISTEncoder(nn.Module):
    def __init__(
        self,
        checkpoint_path,
        device="cpu",
        cls_type_value=3,
    ):
        super().__init__()
        self.cls_type_value = cls_type_value
        self.device = torch.device(device)

        self.formula_transformer = FormulaTransformer(
            hidden_size=512,
            peak_attn_layers=2,
            spectra_dropout=0.1,
            pairwise_featurization=True,
            num_heads=8,
            form_embedder="pos-cos",
            adduct_class_num=8,
            no_diffs=False,
            peak_type_set="full",
            collision_energy_encoder_type="sinusoidal",
            instrument_class_num=6,
        )
        self.fp_grower = FingerprintGrowingModule(
            hidden_input_dim=512, final_target_dim=4096, num_splits=4, reduce_factor=2,
        )

        self._load_checkpoint(checkpoint_path)
        self.to(self.device)
        self.eval()

    def _load_checkpoint(self, path):
        state = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        ft_state = {
            k.replace("spectra_encoder.0.", ""): v
            for k, v in state.items()
            if k.startswith("spectra_encoder.0.")
        }
        if "instrument_embedder" in ft_state:
            embed = ft_state.pop("instrument_embedder")
            self.formula_transformer.register_buffer("instrument_lookup", embed)
        self.formula_transformer.load_state_dict(ft_state, strict=False)

        fg_state = {
            k.replace("spectra_encoder.2.", ""): v
            for k, v in state.items()
            if k.startswith("spectra_encoder.2.")
        }
        self.fp_grower.load_state_dict(fg_state, strict=True)

    @torch.no_grad()
    def predict(self, spectrum_batch):
        peak_embeddings = self.formula_transformer(spectrum_batch)
        peak_types = spectrum_batch["batch_peak_type"]
        cls_mask = peak_types == self.cls_type_value
        cls_embeddings = peak_embeddings[cls_mask]
        growing_fps = self.fp_grower(cls_embeddings)
        return growing_fps[-1]

    @torch.no_grad()
    def predict_from_raw(
        self, peak_formula_dicts, precursor_mzs, formulas, adducts, collision_energies,
        instruments=None,
    ):
        from ms_gpt.data.spectrum_utils import prepare_spectrum_batch

        batch = prepare_spectrum_batch(
            peak_formula_dicts=peak_formula_dicts,
            precursor_mzs=precursor_mzs,
            formulas=formulas,
            adducts=adducts,
            collision_energies=collision_energies,
            device=str(self.device),
            cls_type_value=self.cls_type_value,
        )
        if "batch_spec_instrument" not in batch:
            B = batch["batch_peak_num"].shape[0]
            batch["batch_spec_instrument"] = torch.zeros(B, dtype=torch.long, device=self.device)
        return self.predict(batch)
