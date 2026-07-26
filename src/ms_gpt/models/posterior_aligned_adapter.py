"""LoRA wrapping for ConditionalMoleculeLanguageModel.

Strategy: freeze the entire base model, then attach LoRA adapters to the
modules that read spectrum-induced posterior queries in the conditional MLM:

  - FingerprintEncoder.value_proj         (per-bit value projection)
  - FingerprintEncoder.combine            (combines bit + value embed)
  - FingerprintEncoder.self_attn.layers.*.linear*   (FP encoder FFN)
  - cross_attn_layers.*.cross_attn.in_proj_weight   (Q/K/V proj inside MHA)
  - cross_attn_layers.*.cross_attn.out_proj         (MHA output projection)
  - cross_attn_layers.*.ffn.*                       (cross-attn FFN)
  - context_proj                          (final context projection)

GPT-2 self-attention and the LM head are NOT adapted — those carry the
chemistry-side prior that pretraining captured, which we don't want to
disturb. Formula encoder is also left frozen (formula info is well-learned).

Public API:
    attach_posterior_aligned_adapter(model, rank=16, alpha=16, dropout=0.05) → model
    count_trainable_parameters(model) → (trainable_params, total_params)
    save_adapter(model, path) → writes a small (~few MB) state dict of LoRA params
    load_adapter(model, path)  → loads LoRA weights into a wrapped model
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model


# Modules to target for LoRA. Names are matched as suffixes against
# the qualified module path, so 'value_proj' matches
# 'fp_encoder.value_proj' but not 'gpt2.value_proj' (latter does not exist).
def find_posterior_alignment_targets(
    model: nn.Module,
    *,
    include_cross_attn_ffn: bool = True,
    include_qkv: bool = True,
    include_fp_encoder_attn_ffn: bool = False,
) -> list[str]:
    """Return the qualified module names suitable as peft target_modules.

    peft's LoraConfig matches by suffix of qualified name. We collect
    the full names here so the match is unambiguous in dotted paths.

    Cross-attention's Q/K/V are exposed as separate ``q_proj``/``k_proj``/
    ``v_proj`` Linear modules (see ``SplitQKVMultiheadAttention`` in
    conditional_mlm.py) so peft can target them directly. This is the
    query-reading pathway described in the paper.
    """
    targets: list[str] = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # FingerprintEncoder.value_proj / combine
        if name.endswith("fp_encoder.value_proj"):
            targets.append(name)
        elif name.endswith("fp_encoder.combine"):
            targets.append(name)
        # context_proj
        elif name.endswith("context_proj"):
            targets.append(name)
        # cross-attention Q/K/V projections
        elif (include_qkv and "cross_attn_layers" in name
              and "cross_attn." in name
              and (name.endswith("q_proj") or name.endswith("k_proj")
                   or name.endswith("v_proj"))):
            targets.append(name)
        # cross-attention out_proj (the post-MHA linear)
        elif "cross_attn_layers" in name and name.endswith("out_proj"):
            targets.append(name)
        # cross-attention FFN linears
        elif (include_cross_attn_ffn and "cross_attn_layers" in name
              and (name.endswith("ffn.0") or name.endswith("ffn.3"))):
            targets.append(name)
        # optional: FP encoder self-attn FFN
        elif (include_fp_encoder_attn_ffn
              and "fp_encoder.self_attn.layers" in name
              and (name.endswith("linear1") or name.endswith("linear2"))):
            targets.append(name)
    return targets


def attach_posterior_aligned_adapter(
    model: nn.Module,
    *,
    rank: int = 16,
    alpha: float = 16.0,
    dropout: float = 0.05,
    extra_targets: Iterable[str] | None = None,
):
    """Freeze ``model`` and attach LoRA adapters at the canonical targets.

    Returns the peft-wrapped model. The original module hierarchy is
    preserved; trainable params are only the LoRA A/B matrices.
    """
    # 1. Freeze everything
    for p in model.parameters():
        p.requires_grad = False
    # 2. Decide targets
    targets = find_posterior_alignment_targets(model)
    if extra_targets:
        targets.extend(t for t in extra_targets if t not in targets)
    if not targets:
        raise RuntimeError("No LoRA targets matched — module names changed?")
    # 3. peft LoraConfig
    config = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type=None,
    )
    peft_model = get_peft_model(model, config)
    return peft_model


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


def save_adapter(model, path: str | Path) -> None:
    """Save only the LoRA adapter weights (peft does this natively)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))


def load_adapter(model, path: str | Path):
    """Load LoRA weights from a peft save dir into a (freshly wrapped) model.

    Caller is expected to have already wrapped ``model`` with the matching
    LoraConfig (or to be loading directly via PeftModel.from_pretrained).
    """
    from peft import PeftModel
    return PeftModel.from_pretrained(model, str(path), is_trainable=True)
