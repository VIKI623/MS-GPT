"""
FP-Conditioned Autoregressive Molecular Language Model.

Architecture: GPT-2 backbone + cross-attention conditioning on Morgan fingerprint + formula.
Based on SAFE-GPT tokenizer/vocabulary, adapted for conditional generation.
"""

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import DynamicCache, GPT2Config, GPT2LMHeadModel
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions


def select_top_k_peaks(peak_embeddings, peak_types, intensities, peak_nums, k, cls_type):
    """Select top-K non-CLS peaks by intensity from MIST encoder output.

    Args:
        peak_embeddings: (B, max_peaks, D) from FormulaTransformer
        peak_types: (B, max_peaks) peak type indices
        intensities: (B, max_peaks) peak intensities
        peak_nums: (B,) valid peak count per sample
        k: number of peaks to select
        cls_type: CLS token type index
    Returns:
        selected: (B, k, D) top-K peak embeddings
        mask: (B, k) bool mask (True = valid)
    """
    B, max_peaks, D = peak_embeddings.shape
    device = peak_embeddings.device

    selected = torch.zeros(B, k, D, device=device, dtype=peak_embeddings.dtype)
    mask = torch.zeros(B, k, dtype=torch.bool, device=device)

    for i in range(B):
        n = peak_nums[i].item()
        valid = torch.zeros(max_peaks, dtype=torch.bool, device=device)
        valid[:n] = True
        frag = valid & (peak_types[i] != cls_type)

        idx = frag.nonzero(as_tuple=True)[0]
        if len(idx) == 0:
            continue

        _, order = intensities[i, idx].sort(descending=True)
        top_idx = idx[order[:k]]
        n_sel = len(top_idx)
        selected[i, :n_sel] = peak_embeddings[i, top_idx]
        mask[i, :n_sel] = True

    return selected, mask


class FingerprintEncoder(nn.Module):
    """Encodes sparse Morgan fingerprint into dense sequence of embeddings."""

    def __init__(self, fp_bits: int = 4096, d_model: int = 768, max_active_bits: int = 256,
                 num_layers: int = 2, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.fp_bits = fp_bits
        self.max_active_bits = max_active_bits
        self.d_model = d_model

        self.bit_embedding = nn.Embedding(fp_bits, d_model)
        self.value_proj = nn.Linear(1, d_model)
        self.combine = nn.Linear(d_model * 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 4,
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.self_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fingerprint: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            fingerprint: (batch, fp_bits) binary or soft fingerprint
        Returns:
            embeddings: (batch, seq_len, d_model) - active bit embeddings
            mask: (batch, seq_len) - padding mask (True = valid)
        """
        batch_size = fingerprint.shape[0]
        device = fingerprint.device

        embeddings_list = []
        masks = []
        for i in range(batch_size):
            active_idx = fingerprint[i].nonzero(as_tuple=True)[0]
            if len(active_idx) > self.max_active_bits:
                active_idx = active_idx[:self.max_active_bits]

            n_active = len(active_idx)
            if n_active == 0:
                active_idx = torch.zeros(1, dtype=torch.long, device=device)
                n_active = 1

            bit_emb = self.bit_embedding(active_idx)
            val_emb = self.value_proj(fingerprint[i, active_idx].unsqueeze(-1))
            combined = self.combine(torch.cat([bit_emb, val_emb], dim=-1))

            pad_len = self.max_active_bits - n_active
            if pad_len > 0:
                padding = torch.zeros(pad_len, self.d_model, device=device)
                combined = torch.cat([combined, padding], dim=0)

            embeddings_list.append(combined)
            mask = torch.cat([
                torch.ones(n_active, device=device, dtype=torch.bool),
                torch.zeros(pad_len, device=device, dtype=torch.bool)
            ]) if pad_len > 0 else torch.ones(n_active, device=device, dtype=torch.bool)
            masks.append(mask)

        embeddings = torch.stack(embeddings_list)
        mask = torch.stack(masks)

        src_key_padding_mask = ~mask
        embeddings = self.self_attn(embeddings, src_key_padding_mask=src_key_padding_mask)
        embeddings = self.norm(embeddings)
        return embeddings, mask


class FormulaEncoder(nn.Module):
    """Encodes molecular formula (element counts) into embeddings."""

    ELEMENTS = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I",
                "B", "Si", "Se", "As"]

    def __init__(self, d_model: int = 768, max_count: int = 200):
        super().__init__()
        self.n_elements = len(self.ELEMENTS)
        self.element_embedding = nn.Embedding(self.n_elements, d_model)
        self.count_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.combine = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, formula_vec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            formula_vec: (batch, n_elements) element count vector
        Returns:
            (batch, n_elements, d_model) formula embeddings
        """
        batch_size = formula_vec.shape[0]
        device = formula_vec.device
        elem_idx = torch.arange(self.n_elements, device=device).unsqueeze(0).expand(batch_size, -1)
        elem_emb = self.element_embedding(elem_idx)
        count_emb = self.count_proj(formula_vec.unsqueeze(-1))
        combined = self.combine(torch.cat([elem_emb, count_emb], dim=-1))
        return self.norm(combined)


class SplitQKVMultiheadAttention(nn.Module):
    """nn.MultiheadAttention with Q/K/V as separate Linear submodules.

    Math is bit-identical to ``nn.MultiheadAttention(batch_first=True)`` so
    that released checkpoints can transfer through ``normalize_mlm_checkpoint_state_dict``
    without retraining. We expose q_proj / k_proj / v_proj as standalone
    nn.Linear modules so peft LoRA can target them directly (the original
    nn.MultiheadAttention bundles them in a single ``in_proj_weight``
    parameter, which peft cannot target).
    """

    def __init__(self, embed_dim: int, num_heads: int,
                 dropout: float = 0.0, bias: bool = True):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} not divisible by num_heads {num_heads}")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = False) -> Tuple[torch.Tensor, None]:
        B, Lq, _ = query.shape
        Lk = key.shape[1]
        q = self.q_proj(query).view(B, Lq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, Lk, self.num_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(q * self.scale, k.transpose(-2, -1))  # (B, H, Lq, Lk)
        if key_padding_mask is not None:
            # key_padding_mask: (B, Lk), True = pad
            attn = attn.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, Lq, self.embed_dim)
        out = self.out_proj(out)
        return out, None


def normalize_mlm_checkpoint_state_dict(sd: dict) -> dict:
    """Convert a checkpoint saved with nn.MultiheadAttention cross_attn into
    one compatible with SplitQKVMultiheadAttention.

    Splits each ``*.cross_attn.in_proj_weight`` (shape 3D × D) into
    ``q_proj.weight``, ``k_proj.weight``, ``v_proj.weight`` (each D × D),
    and similarly for biases. Redundant ``transformer.*`` aliases of
    ``gpt2.transformer.*`` are dropped because the model exposes ``transformer`` as
    a property rather than registering it twice. All other keys pass
    through unchanged.
    """
    out = {}
    for k, v in sd.items():
        if k.startswith("transformer."):
            continue
        if k.endswith("cross_attn.in_proj_weight"):
            base = k[: -len("in_proj_weight")]
            d_model = v.shape[1]
            q, kk, vv = v.split(d_model, dim=0)
            out[base + "q_proj.weight"] = q.contiguous()
            out[base + "k_proj.weight"] = kk.contiguous()
            out[base + "v_proj.weight"] = vv.contiguous()
        elif k.endswith("cross_attn.in_proj_bias"):
            base = k[: -len("in_proj_bias")]
            d_model = v.shape[0] // 3
            q, kk, vv = v.split(d_model, dim=0)
            out[base + "q_proj.bias"] = q.contiguous()
            out[base + "k_proj.bias"] = kk.contiguous()
            out[base + "v_proj.bias"] = vv.contiguous()
        else:
            out[k] = v
    return out


class CrossAttentionConditioner(nn.Module):
    """Cross-attention layer injected into GPT-2 blocks for FP/formula conditioning.

    Uses ``SplitQKVMultiheadAttention`` (math-identical to
    ``nn.MultiheadAttention``) so that LoRA can target Q/K/V via peft.
    ``normalize_mlm_checkpoint_state_dict`` maps released MLM weights to this
    split-QKV representation before loading.
    """

    def __init__(self, d_model: int = 768, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = SplitQKVMultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                context_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model) - MLM hidden states
            context: (batch, ctx_len, d_model) - condition embeddings
            context_mask: (batch, ctx_len) - True=valid, False=padding
        """
        key_padding_mask = ~context_mask if context_mask is not None else None
        residual = x
        x = self.norm(x)
        x, _ = self.cross_attn(x, context, context, key_padding_mask=key_padding_mask)
        x = residual + x
        x = x + self.ffn(self.norm2(x))
        return x


class ConditionalMoleculeLanguageModel(nn.Module):
    """
    GPT-2 with fingerprint + formula cross-attention conditioning.

    Injects cross-attention after every N transformer blocks.
    """

    def __init__(
        self,
        vocab_size: int = 1882,
        d_model: int = 768,
        n_layers: int = 12,
        n_heads: int = 12,
        fp_bits: int = 4096,
        fp_encoder_layers: int = 2,
        cross_attn_every_n: int = 2,
        max_seq_len: int = 256,
        dropout: float = 0.1,
        spectrum_dim: Optional[int] = None,
        n_top_peaks: int = 20,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_top_peaks = n_top_peaks

        gpt2_config = GPT2Config(
            vocab_size=vocab_size,
            n_embd=d_model,
            n_layer=n_layers,
            n_head=n_heads,
            n_positions=max_seq_len,
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            use_cache=False,
            add_cross_attention=False,
        )
        self.gpt2 = GPT2LMHeadModel(gpt2_config)
        # `transformer` is exposed as a read-only alias of `gpt2.transformer` via the
        # property below — intentionally NOT registered as a submodule. Registering it
        # (self.transformer = self.gpt2.transformer) duplicates all 148 GPT-2 keys under
        # a second `transformer.*` prefix in state_dict(); torch.save dedups the shared
        # storage so the file is not bloated, but naive param counts then read ~238M
        # instead of the true ~150M.

        self.fp_encoder = FingerprintEncoder(
            fp_bits=fp_bits, d_model=d_model,
            num_layers=fp_encoder_layers, num_heads=n_heads, dropout=dropout
        )
        self.formula_encoder = FormulaEncoder(d_model=d_model)

        self.cross_attn_layers = nn.ModuleDict()
        for i in range(n_layers):
            if (i + 1) % cross_attn_every_n == 0:
                self.cross_attn_layers[str(i)] = CrossAttentionConditioner(
                    d_model=d_model, num_heads=n_heads, dropout=dropout
                )

        self.context_proj = nn.Linear(d_model, d_model)

        self.spectrum_proj = None
        if spectrum_dim is not None:
            self.spectrum_proj = nn.Sequential(
                nn.Linear(spectrum_dim, d_model),
                nn.LayerNorm(d_model),
            )

    @property
    def transformer(self):
        """Read-only alias of the GPT-2 backbone transformer (see __init__ — not a
        registered submodule, to avoid duplicate ``transformer.*`` keys in state_dict)."""
        return self.gpt2.transformer

    def encode_condition(self, fingerprint: torch.Tensor,
                         formula_vec: torch.Tensor,
                         spectral_tokens: Optional[torch.Tensor] = None,
                         spectral_mask: Optional[torch.Tensor] = None,
                         ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode FP + formula (+ optional spectrum) into conditioning context."""
        fp_emb, fp_mask = self.fp_encoder(fingerprint)
        formula_emb = self.formula_encoder(formula_vec)
        formula_mask = torch.ones(formula_emb.shape[:2], dtype=torch.bool, device=formula_emb.device)

        parts = [fp_emb, formula_emb]
        masks = [fp_mask, formula_mask]

        if spectral_tokens is not None and self.spectrum_proj is not None:
            parts.append(self.spectrum_proj(spectral_tokens))
            masks.append(spectral_mask)

        context = torch.cat(parts, dim=1)
        context_mask = torch.cat(masks, dim=1)
        context = self.context_proj(context)
        return context, context_mask

    def forward(
        self,
        input_ids: torch.Tensor,
        fingerprint: torch.Tensor,
        formula_vec: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        spectral_tokens: Optional[torch.Tensor] = None,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> CausalLMOutputWithCrossAttentions:
        context, context_mask = self.encode_condition(
            fingerprint, formula_vec, spectral_tokens, spectral_mask
        )

        inputs_embeds = self.transformer.wte(input_ids)
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        position_embeds = self.transformer.wpe(position_ids)
        hidden_states = self.transformer.drop(inputs_embeds + position_embeds)

        if attention_mask is not None:
            causal_mask = self._make_causal_mask(input_ids.shape[1], input_ids.device)
            extended_mask = attention_mask.unsqueeze(1).unsqueeze(2) * causal_mask
            extended_mask = (1.0 - extended_mask) * torch.finfo(hidden_states.dtype).min
        else:
            extended_mask = None

        for i, block in enumerate(self.transformer.h):
            outputs = block(hidden_states, attention_mask=extended_mask)
            hidden_states = outputs[0]

            if str(i) in self.cross_attn_layers:
                hidden_states = self.cross_attn_layers[str(i)](
                    hidden_states, context, context_mask
                )

        hidden_states = self.transformer.ln_f(hidden_states)
        logits = self.gpt2.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutputWithCrossAttentions(loss=loss, logits=logits)

    def _make_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        return (1.0 - mask).unsqueeze(0).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        fingerprint: torch.Tensor,
        formula_vec: torch.Tensor,
        max_len: int = 256,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.95,
        bos_token_id: int = 0,
        eos_token_id: int = 1,
        num_samples: int = 1,
        spectral_tokens: Optional[torch.Tensor] = None,
        spectral_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Autoregressive sampling with KV cache — O(n) per sequence."""
        device = fingerprint.device
        batch_size = fingerprint.shape[0] * num_samples

        if num_samples > 1:
            fingerprint = fingerprint.repeat_interleave(num_samples, dim=0)
            formula_vec = formula_vec.repeat_interleave(num_samples, dim=0)
            if spectral_tokens is not None:
                spectral_tokens = spectral_tokens.repeat_interleave(num_samples, dim=0)
                spectral_mask = spectral_mask.repeat_interleave(num_samples, dim=0)

        with torch.autocast(device_type="cuda", enabled=False):
            st = spectral_tokens.float() if spectral_tokens is not None else None
            context, context_mask = self.encode_condition(
                fingerprint.float(), formula_vec.float(), st, spectral_mask
            )
        input_ids = torch.full((batch_size, 1), bos_token_id, dtype=torch.long, device=device)
        unfinished = torch.ones(batch_size, dtype=torch.bool, device=device)
        cache = DynamicCache()

        for step in range(max_len - 1):
            cur_ids = input_ids if step == 0 else input_ids[:, -1:]
            cache_pos = torch.tensor([step], dtype=torch.long, device=device)

            hidden = self.transformer.drop(
                self.transformer.wte(cur_ids)
                + self.transformer.wpe(cache_pos.unsqueeze(0))
            )

            for i, block in enumerate(self.transformer.h):
                out = block(
                    hidden,
                    past_key_value=cache,
                    cache_position=cache_pos,
                    use_cache=True,
                )
                hidden = out[0]
                if str(i) in self.cross_attn_layers:
                    hidden = self.cross_attn_layers[str(i)](
                        hidden, context, context_mask
                    )

            logits = self.gpt2.lm_head(
                self.transformer.ln_f(hidden[:, -1])
            ).float() / temperature

            if top_k > 0:
                thresh = torch.topk(logits, top_k)[0][..., -1, None]
                logits[logits < thresh] = float("-inf")
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                remove = cum_probs > top_p
                remove[..., 1:] = remove[..., :-1].clone()
                remove[..., 0] = False
                logits[remove.scatter(1, sorted_idx, remove)] = float("-inf")

            next_token = torch.multinomial(
                F.softmax(logits, dim=-1), num_samples=1
            )
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            unfinished = unfinished & (next_token.squeeze(1) != eos_token_id)
            if not unfinished.any():
                break

        return input_ids
