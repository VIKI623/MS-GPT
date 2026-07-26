""" modules.py"""
from .spectrum_featurizers import PeakType
import copy
import torch
import numpy as np
from torch import nn
from . import transformer_layer, form_embedders


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLPBlocks(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        dropout: float,
        num_layers: int,
    ):
        super().__init__()
        self.activation = nn.ReLU()
        self.dropout_layer = nn.Dropout(p=dropout)
        self.input_layer = nn.Linear(input_size, hidden_size)
        middle_layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(middle_layer, num_layers - 1)

    def forward(self, x):
        output = x
        output = self.input_layer(x)
        output = self.dropout_layer(output)
        output = self.activation(output)
        for layer_index, layer in enumerate(self.layers):
            output = layer(output)
            output = self.dropout_layer(output)
            output = self.activation(output)
        return output


class FormulaTransformer(nn.Module):
    """FormulaTransformer"""

    def __init__(
        self,
        hidden_size: int,
        peak_attn_layers: int,
        spectra_dropout: float = 0.1,
        pairwise_featurization: bool = True,
        num_heads: int = 8,
        form_embedder: str = "pos-cos",
        adduct_class_num: int = 11,
        no_diffs: bool = True,
        instrument_class_num: int = 0,
        peak_type_set: str = "reduced",
        collision_energy_encoder_type: str = "categorical",
    ):
        """_summary_
        Args:
                    hidden_size (int): _description_
                    peak_attn_layers (int): _description_
                    spectra_dropout (float, optional): _description_. Defaults to 0.1.
                    pairwise_featurization (bool, optional): _description_. Defaults to True.
                    num_heads (int, optional): _description_. Defaults to 8.
                    output_size (int, optional): _description_. Defaults to 2048.
                    form_embedder (str, optional): _description_. Defaults to "pos-cos".

                Raises:
                    ValueError: _description_
        """
        nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.attn_heads = num_heads
        self.dim_feedforward = self.hidden_size * 4
        self.spectra_dropout = spectra_dropout
        self.no_diffs = no_diffs

        self.form_embedder = form_embedder
        self.form_embedder_mod = form_embedders.get_embedder(self.form_embedder)
        
        self.inten_feats = 1

        # peak type set: "full" = {Frag, LOSS, AB_LOSS, CLS}, "reduced" = {Frag, CLS}
        if peak_type_set == "full":
            self.num_types = 4
            self.cls_type = 3  # CLS index in full enum
        else:
            self.num_types = len(PeakType)
            self.cls_type = PeakType.CLS.value

        # collision energy: "categorical" includes 4-dim CE features, "sinusoidal" omits
        self.collision_energy_feats = 0 if collision_energy_encoder_type == "sinusoidal" else 4

        self.adduct_dim = adduct_class_num
        self.pairwise_featurization = pairwise_featurization
        self.instrument_class_num = instrument_class_num

        # Define dense encoders and root formula encoder
        self.formula_dim = self.form_embedder_mod.full_dim
        if self.no_diffs:
            self.input_dim = (
                self.formula_dim
                + self.num_types
                + self.adduct_dim
                + self.inten_feats
                + self.collision_energy_feats
                + self.instrument_class_num
            )
        else:
            self.input_dim = (
                self.formula_dim * 2
                + self.num_types
                + self.adduct_dim
                + self.inten_feats
                + self.collision_energy_feats
                + self.instrument_class_num
            )

        self.intermediate_layer = MLPBlocks(
            input_size=self.input_dim,
            hidden_size=self.hidden_size,
            dropout=self.spectra_dropout,
            num_layers=2,
        )
        
        self.pairwise_featurizer = None
        if self.pairwise_featurization:
            self.pairwise_featurizer = MLPBlocks(
                input_size=self.formula_dim,
                hidden_size=self.hidden_size,
                dropout=self.spectra_dropout,
                num_layers=2,
            )

        # Multihead attention block with residuals
        peak_attn_layer = transformer_layer.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.attn_heads,
            dim_feedforward=self.dim_feedforward,
            dropout=self.spectra_dropout,
            pairwise_featurization=pairwise_featurization,
        )
        self.peak_attn_layers = _get_clones(peak_attn_layer, peak_attn_layers)
        
        types_lookup = nn.functional.one_hot(torch.arange(self.num_types))
        adducts_lookup = nn.functional.one_hot(torch.arange(self.adduct_dim))
        self.register_buffer('types_lookup', types_lookup)
        self.register_buffer('adducts_lookup', adducts_lookup)
        
        if self.instrument_class_num > 0:
            instrument_lookup = nn.functional.one_hot(torch.arange(self.instrument_class_num))
            self.register_buffer('instrument_lookup', instrument_lookup)

    def forward(self, batch):
        """forward."""
        # Step 1: Create embeddings
        num_peaks = batch["batch_peak_num"]
        peak_types = batch["batch_peak_type"]

        peak_dim = peak_types.shape[-1]
        adducts = batch["batch_peak_adduct"]

        # Get form vec and get diffs
        orig_form_vec = batch["batch_peak_formulae_vector"][:, :, :]
        form_diffs = orig_form_vec[:, :, None, :] - orig_form_vec[:, None, :, :]

        # Embed formulae based upon peak type
        form_vec = self.form_embedder_mod(orig_form_vec) # (batch_size, peak_num, embedding_dim)

        inten_tensor = batch["batch_peak_intensity"].unsqueeze(-1)
        one_hot_types = self.types_lookup[peak_types]
        one_hot_adducts = self.adducts_lookup[adducts]

        if self.no_diffs:
            input_vec = [
                form_vec,
                one_hot_types,
                one_hot_adducts,
                inten_tensor,
            ]
        else:
            cls_token_mask = peak_types == self.cls_type
            abs_diffs = form_diffs[cls_token_mask]
            diff_vec = self.form_embedder_mod(abs_diffs)
            input_vec = [
                form_vec,
                diff_vec,
                one_hot_types,
                one_hot_adducts,
                inten_tensor,
            ]

        if self.collision_energy_feats > 0:
            collision_energy_tensor = batch["batch_spec_collision_energy"].unsqueeze(1).expand(-1, peak_dim, -1)
            input_vec.append(collision_energy_tensor)
            
        if self.instrument_class_num > 0:
            instruments = batch["batch_spec_instrument"].unsqueeze(1).expand(-1, peak_dim)
            one_hot_instruments = self.instrument_lookup[instruments]
            input_vec.append(one_hot_instruments)
        
        input_vec = torch.cat(input_vec, dim=-1)
        peak_tensor = self.intermediate_layer(input_vec)

        # Step 3: Run transformer
        # B x Np x d -> Np x B x d
        peak_tensor = peak_tensor.transpose(0, 1)
        # Mask before summing
        peak_dim = peak_tensor.shape[0]
        peaks_aranged = torch.arange(peak_dim, device=peak_tensor.device)

        # batch x num peaks
        attn_mask = ~(peaks_aranged[None, :] < num_peaks[:, None])
        pairwise_features = None
        if self.pairwise_featurization:
            # Make sure to _only_ consider subset fragments, rather than only
            # partial additions/subtractions from across the tree branches
            # Comment out to consider _any_ loss embedding with new parameters
            same_sign = torch.all(form_diffs >= 0, -1) | torch.all(form_diffs <= 0, -1)
            form_diffs[~same_sign].fill_(0)
            form_diffs = torch.abs(form_diffs)
            pairwise_features = self.pairwise_featurizer(
                self.form_embedder_mod(form_diffs)
            )

        # Np x B x d
        for peak_attn_layer in self.peak_attn_layers:
            peak_tensor, pairwise_features = peak_attn_layer(
                peak_tensor,
                pairwise_features=pairwise_features,
                src_key_padding_mask=attn_mask,
            )
        
        return peak_tensor.transpose(0, 1)


class FPGrowingModule(nn.Module):
    """FPGrowingModule.

    Accept an input hidden dim and progressively grow by powers of 2 s.t.

    We eventually get to the final output size...

    """

    def __init__(
        self,
        hidden_input_dim: int = 512,
        final_target_dim: int = 4096,
        num_splits=4,
        reduce_factor=2,
        spectra_dropout=0.1
    ):
        super().__init__()

        self.hidden_input_dim = hidden_input_dim
        self.final_target_dim = final_target_dim
        self.num_splits = num_splits
        self.reduce_factor = reduce_factor

        final_output_size = self.final_target_dim

        # Creates an array where we end with final_size and have num_splits + 1
        # different entries in it (e.g., num_splits = 1 with final dim 4096 has
        # [2048, 4096])
        layer_dims = [
            int(np.ceil(final_output_size / (reduce_factor**num_split)))
            for num_split in range(num_splits + 1)
        ][::-1]

        # Start by predicting into the very first layer dim (e.g., 512  -> 512)
        self.output_dims = layer_dims

        # Define initial predict module
        self.initial_predict = nn.Sequential(
            nn.Linear(hidden_input_dim, hidden_input_dim),
            nn.ReLU(),
            nn.Dropout(spectra_dropout),
            nn.Linear(
                hidden_input_dim,
                layer_dims[0],
            ),
            nn.Sigmoid(),
        )
        predict_bricks = []
        gate_bricks = []
        for layer_dim_ind, layer_dim in enumerate(layer_dims[:-1]):
            out_dim = layer_dims[layer_dim_ind + 1]

            # Need to update nn.Linear layer to be fixed if the right param is
            # called
            lin_predict = nn.Linear(layer_dim, out_dim)
            predict_brick = nn.Sequential(lin_predict, nn.Sigmoid())

            gate_bricks.append(
                nn.Sequential(nn.Linear(hidden_input_dim, out_dim), nn.Sigmoid())
            )
            predict_bricks.append(predict_brick)

        self.predict_bricks = nn.ModuleList(predict_bricks)
        self.gate_bricks = nn.ModuleList(gate_bricks)

    def forward(self, hidden):
        """forward.

        Return dict mapping output dim to the

        """
        cur_pred = self.initial_predict(hidden)
        output_preds = [cur_pred]
        for _out_dim, predict_brick, gate_brick in zip(
            self.output_dims[1:], self.predict_bricks, self.gate_bricks
        ):
            gate_outs = gate_brick(hidden)
            pred_out = predict_brick(cur_pred)
            cur_pred = gate_outs * pred_out
            output_preds.append(cur_pred)
        return output_preds


class MistModel(nn.Module):
    def __init__(self,
        d_magma_fingerprints = 512,
        d_fingerprints = 4096,
        num_fp_growing_predictor_layers = 4,
        hidden_size: int = 512,
        peak_attn_layers: int = 2,
        spectra_dropout: float = 0.1,
        pairwise_featurization: bool = True,
        num_heads: int = 8,
        form_embedder: str = "pos-cos",
        adduct_class_num: int = 11,
        no_diffs: bool = True):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.spec_encoder = FormulaTransformer(
            hidden_size=hidden_size,
            peak_attn_layers=peak_attn_layers,
            spectra_dropout=spectra_dropout,
            pairwise_featurization=pairwise_featurization,
            num_heads=num_heads,
            form_embedder=form_embedder,
            adduct_class_num=adduct_class_num,
            no_diffs=no_diffs)
        
        self.peak_fingerprints_predictor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(spectra_dropout),
            nn.Linear(hidden_size, d_magma_fingerprints),
            nn.Sigmoid()
        )
        
        self.precursor_fingerprints_predictor = FPGrowingModule(
                hidden_input_dim=hidden_size,
                final_target_dim=d_fingerprints,
                num_splits=num_fp_growing_predictor_layers,
                reduce_factor=2,
                spectra_dropout=spectra_dropout)

    def forward(self, batch):
        batch_peak_type = batch["batch_peak_type"]  # (batch_size, max_peak_num)        
        peak_embeddings = self.spec_encoder(batch) # (batch_size, max_peak_num, d_model)
        
        pred_peak_magma_fingerprints = self.peak_fingerprints_predictor(peak_embeddings)
        
        cls_peak_peak_embeddings = peak_embeddings[batch_peak_type == PeakType.CLS.value]
        precursor_fingerprints_multi_layers = self.precursor_fingerprints_predictor(cls_peak_peak_embeddings)
        
        return {
            "pred_growing_fingerprints": precursor_fingerprints_multi_layers,
            "pred_peak_magma_fingerprints": pred_peak_magma_fingerprints
        }