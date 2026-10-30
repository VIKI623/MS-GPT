"""Train the paper's posterior-aligned adapter for one benchmark."""
import argparse
from contextlib import nullcontext
from functools import partial
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, DistributedSampler

from ms_gpt.data.mlm_pretraining_dataset import load_molecule_tokenizer
from ms_gpt.data.posterior_aligned_dataset import (
    PosteriorAlignedDataset,
    collate_posterior_aligned_batch,
)
from ms_gpt.data.posterior_queries import resolve_threshold_band
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


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Released adapter training_config.json")
    parser.add_argument("--mlm_checkpoint")
    parser.add_argument("--mlm_config")
    parser.add_argument("--tokenizer_path")
    parser.add_argument("--cache_dir")
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"))
    parser.add_argument("--recoverability_labels")
    parser.add_argument("--token_file", help="Optional targets encoded with the selected decoder's tokenizer")
    parser.add_argument("--threshold_low", type=float)
    parser.add_argument("--threshold_high", type=float)
    parser.add_argument("--gamma", type=float)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--max_steps", type=int, help="Cosine-schedule horizon in optimizer updates")
    parser.add_argument("--selected_step", type=int, help="Adapter step exported at output_dir")
    parser.add_argument("--stop_after_steps", type=int)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--weight_decay", type=float)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--max_grad_norm", type=float)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--save_interval", type=int)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    recipe = json.loads(Path(args.config).read_text())
    for key, value in vars(args).items():
        if value is None and key in recipe and key != "gradient_accumulation_steps":
            setattr(args, key, recipe[key])
    for key, value in {"max_grad_norm": 1.0, "num_workers": 2, "save_interval": 500}.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if args.cache_dir is None:
        profile = json.loads(Path(args.config).with_name("inference_config.json").read_text())
        args.cache_dir = profile["cache_dir"]
    if args.recoverability_labels is None:
        parser.error("Supply --recoverability_labels from the selected decoder's frozen train candidates")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_batch = recipe["global_batch_size"]
    if args.batch_size is None:
        args.batch_size = min(recipe["batch_size_per_gpu"], global_batch // world_size)
    if args.gradient_accumulation_steps is None:
        if world_size == recipe["world_size"] and args.batch_size == recipe["batch_size_per_gpu"]:
            args.gradient_accumulation_steps = recipe["gradient_accumulation_steps"]
        else:
            args.gradient_accumulation_steps = global_batch // max(1, args.batch_size * world_size)
    if (args.batch_size < 1 or args.gradient_accumulation_steps < 1 or
            args.batch_size * args.gradient_accumulation_steps * world_size != global_batch):
        parser.error(f"batch_size * accumulation * world_size must equal {global_batch}")
    if args.stop_after_steps is None:
        args.stop_after_steps = args.selected_step
    if not 0 < args.selected_step <= args.stop_after_steps <= args.max_steps:
        parser.error("Require 0 < selected_step <= stop_after_steps <= max_steps")
    resolve_threshold_band(args.threshold_low, args.threshold_high)
    return args, recipe


def main():
    args, recipe = parse_args()
    rank, world_size, local_rank, distributed = setup_distributed()
    is_main = rank == 0
    device = torch.device(
        f"cuda:{local_rank}" if distributed else f"cuda:{args.gpu}"
    )
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    config = OmegaConf.load(args.mlm_config)
    tokenizer = load_molecule_tokenizer(args.tokenizer_path or config.get("tokenizer_path"))
    base_model = ConditionalMoleculeLanguageModel.from_config(config, len(tokenizer), dropout=0.0)
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
        gamma=args.gamma,
        group_weights={name: (weights["ce_weight"], weights["kl_weight"])
                       for name, weights in recipe["recoverability_groups"].items()},
        target_format=config.data.format,
        token_file=args.token_file,
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                 shuffle=True, seed=args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
        collate_fn=partial(collate_posterior_aligned_batch, pad_token_id=tokenizer.pad_token_id),
    )
    if not len(loader):
        raise ValueError("The training dataset must supply at least one full batch per GPU")

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
            configuration = {**recipe, **vars(args), "world_size": world_size,
                             "batch_size_per_gpu": args.batch_size, "training_rows": len(dataset)}
            json.dump(configuration, handle, indent=2)
            handle.write("\n")
        trainable, total = count_trainable_parameters(model)
        print(f"Trainable parameters: {trainable:,}/{total:,}")

    model.train()
    inner_model = model.module if distributed else model
    step, epoch, start_time = 0, 0, time.time()
    sampler.set_epoch(epoch)
    iterator = iter(loader)
    while step < args.stop_after_steps:
        optimizer.zero_grad(set_to_none=True)
        mean_loss = 0.0
        for micro in range(args.gradient_accumulation_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                epoch += 1
                sampler.set_epoch(epoch)
                iterator = iter(loader)
                batch = next(iterator)
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            sync = model.no_sync() if distributed and micro < args.gradient_accumulation_steps - 1 else nullcontext()
            with sync:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    inputs = {key: batch[key] for key in ("input_ids", "fingerprint", "formula_vec", "attention_mask")}
                    output = model(**inputs)
                    per_example_ce = sequence_cross_entropy(output.logits, batch["labels"])
                    with torch.no_grad(), inner_model.disable_adapter():
                        teacher = inner_model(**inputs)
                    per_example_kl = teacher_student_kl(teacher.logits, output.logits, batch["labels"])
                    loss = (batch["ce_weight"] * per_example_ce + batch["kl_weight"] * per_example_kl).mean()
                (loss / args.gradient_accumulation_steps).backward()
                mean_loss += loss.item() / args.gradient_accumulation_steps
        torch.nn.utils.clip_grad_norm_(parameters, args.max_grad_norm)
        optimizer.step()
        scheduler.step()
        step += 1
        if is_main and (step == 1 or step % 50 == 0):
            elapsed = (time.time() - start_time) / 60
            print(f"step={step} loss={mean_loss:.4f} elapsed={elapsed:.1f}m", flush=True)
        if is_main and (step % args.save_interval == 0 or step == args.selected_step):
            save_adapter(inner_model, os.path.join(args.output_dir, f"step_{step}"))
        if is_main and step == args.selected_step:
            save_adapter(inner_model, args.output_dir)

    if is_main:
        save_adapter(inner_model, os.path.join(args.output_dir, "final"))
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
