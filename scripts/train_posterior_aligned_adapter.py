"""Train the paper's posterior-aligned adapter for one benchmark."""
import argparse
import json
import math
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from safe.tokenizer import SAFETokenizer
from torch.utils.data import DataLoader, DistributedSampler

from ms_gpt.data.posterior_aligned_dataset import (
    PosteriorAlignedDataset,
    collate_posterior_aligned_batch,
)
from ms_gpt.models.conditional_mlm import (
    ConditionalMoleculeLanguageModel,
    normalize_mlm_checkpoint_state_dict,
)
from ms_gpt.models.posterior_aligned_adapter import (
    attach_posterior_aligned_adapter,
    count_trainable_parameters,
    save_adapter,
)


def setup_distributed():
    if "RANK" not in os.environ:
        return 0, 1, 0, False
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank, True


def load_tokenizer():
    path = os.environ.get("MS_GPT_SAFE_GPT_CKPT", "checkpoints/safe_gpt_base")
    tokenizer = SAFETokenizer.from_pretrained(path).get_pretrained()
    tokenizer.add_tokens(["<", ">"])
    return tokenizer


def sequence_cross_entropy(logits, labels):
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:]
    mask = shifted_labels != -100
    safe_labels = shifted_labels.clamp(min=0)
    token_loss = -F.log_softmax(shifted_logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return (token_loss * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def teacher_student_kl(teacher_logits, student_logits, labels):
    teacher_logp = F.log_softmax(teacher_logits.float(), dim=-1)
    student_logp = F.log_softmax(student_logits.float(), dim=-1)
    token_kl = (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)
    mask = labels != -100
    return (token_kl * mask).sum(-1) / mask.sum(-1).clamp(min=1)


def build_conditional_mlm(config, vocab_size):
    return ConditionalMoleculeLanguageModel(
        vocab_size=vocab_size,
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        fp_bits=config.model.fp_bits,
        fp_encoder_layers=config.model.fp_encoder_layers,
        cross_attn_every_n=config.model.cross_attn_every_n,
        max_seq_len=config.model.max_seq_len,
        dropout=0.0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlm_checkpoint", default="checkpoints/mlm/ms_gpt_mlm.pt")
    parser.add_argument("--mlm_config", default="configs/mlm_pretraining.yaml")
    parser.add_argument("--cache_dir", default="data/posterior_cache")
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--recoverability_labels", required=True)
    parser.add_argument("--threshold_low", type=float, required=True)
    parser.add_argument("--threshold_high", type=float, required=True)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=150)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = rank == 0
    device = torch.device(
        f"cuda:{local_rank}" if distributed else f"cuda:{args.gpu}"
    )
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    config = OmegaConf.load(args.mlm_config)
    tokenizer = load_tokenizer()
    base_model = build_conditional_mlm(config, len(tokenizer))
    state_dict = torch.load(args.mlm_checkpoint, map_location="cpu", weights_only=True)
    base_model.load_state_dict(normalize_mlm_checkpoint_state_dict(state_dict))
    model = attach_posterior_aligned_adapter(
        base_model, rank=args.rank, alpha=args.alpha, dropout=args.dropout
    ).to(device)
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )

    dataset = PosteriorAlignedDataset(
        args.cache_dir,
        args.benchmark,
        tokenizer,
        args.recoverability_labels,
        max_len=config.model.max_seq_len,
        threshold_low=args.threshold_low,
        threshold_high=args.threshold_high,
        rng_seed=args.seed,
    )
    sampler = DistributedSampler(dataset, shuffle=True) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_posterior_aligned_batch,
    )

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, step / max(args.warmup_steps, 1))
        * 0.5
        * (
            1.0
            + math.cos(
                math.pi
                * max(0, step - args.warmup_steps)
                / max(1, args.max_steps - args.warmup_steps)
            )
        ),
    )

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "training_config.json"), "w") as handle:
            json.dump(vars(args), handle, indent=2)
        trainable, total = count_trainable_parameters(model)
        print(f"Trainable parameters: {trainable:,}/{total:,}")

    model.train()
    step, start_time = 0, time.time()
    while step < args.max_steps:
        if sampler is not None:
            sampler.set_epoch(step)
        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = model(
                    input_ids=batch["input_ids"],
                    fingerprint=batch["fingerprint"],
                    formula_vec=batch["formula_vec"],
                    attention_mask=batch["attention_mask"],
                )
                per_example_ce = sequence_cross_entropy(output.logits, batch["labels"])
                inner_model = model.module if distributed else model
                with torch.no_grad(), inner_model.disable_adapter():
                    teacher = model(
                        input_ids=batch["input_ids"],
                        fingerprint=batch["fingerprint"],
                        formula_vec=batch["formula_vec"],
                        attention_mask=batch["attention_mask"],
                    )
                per_example_kl = teacher_student_kl(
                    teacher.logits, output.logits, batch["labels"]
                )
                loss = (
                    batch["ce_weight"] * per_example_ce
                    + batch["kl_weight"] * per_example_kl
                ).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            step += 1
            if is_main and step % 50 == 0:
                elapsed = (time.time() - start_time) / 60
                print(f"step={step} loss={loss.item():.4f} elapsed={elapsed:.1f}m")
            if is_main and step % args.save_interval == 0:
                target = os.path.join(args.output_dir, f"step_{step}")
                save_adapter(model.module if distributed else model, target)
            if step >= args.max_steps:
                break

    if is_main:
        save_adapter(model.module if distributed else model, os.path.join(args.output_dir, "final"))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
