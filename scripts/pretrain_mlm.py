"""
Pretrain the fingerprint-formula-conditioned molecule-language model.

Usage (single GPU):
    python scripts/pretrain_mlm.py --config configs/mlm_pretraining.yaml

Usage (multi-GPU DDP):
    torchrun --nproc_per_node=8 scripts/pretrain_mlm.py --config configs/mlm_pretraining.yaml
"""

import argparse
import gc
import math
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.amp import autocast

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from safe.tokenizer import SAFETokenizer
from ms_gpt.models.conditional_mlm import ConditionalMoleculeLanguageModel
from ms_gpt.data.formula_group_sampler import FormulaGroupBatchSampler


def setup_distributed():
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def load_molecule_tokenizer(model_path="checkpoints/safe_gpt_base"):
    tk = SAFETokenizer.from_pretrained(model_path).get_pretrained()
    tk.add_tokens(["<", ">"])
    return tk


def load_safe_gpt_weights(model, model_path="checkpoints/safe_gpt_base", is_main=True):
    """Warm-start from safe-gpt pretrained weights (87M GPT-2 trained on 1.1B molecules)."""
    from transformers import GPT2LMHeadModel
    safe_gpt = GPT2LMHeadModel.from_pretrained(model_path)
    safe_sd = safe_gpt.state_dict()
    our_sd = model.state_dict()

    loaded, skipped = 0, 0
    for safe_key, safe_param in safe_sd.items():
        our_key = f"gpt2.{safe_key}"
        if our_key not in our_sd:
            skipped += 1
            continue

        our_shape = our_sd[our_key].shape
        if safe_param.shape == our_shape:
            our_sd[our_key] = safe_param
            loaded += 1
        elif len(safe_param.shape) == 2 and len(our_shape) == 2:
            # Partial copy (e.g., wte [1880,768] → [1882,768], wpe [1024,768] → [256,768])
            min_rows = min(safe_param.shape[0], our_shape[0])
            min_cols = min(safe_param.shape[1], our_shape[1])
            our_sd[our_key][:min_rows, :min_cols] = safe_param[:min_rows, :min_cols]
            loaded += 1
        else:
            skipped += 1

    model.load_state_dict(our_sd)
    if is_main:
        total = loaded + skipped
        print(f"Warm-start from safe-gpt: {loaded}/{total} params loaded, {skipped} skipped (cross-attn/FP/formula)")
    return model


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
        dropout=config.model.dropout,
    )


def get_lr(step, warmup_steps, base_lr, max_steps):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    config = OmegaConf.load(args.config)
    device = torch.device(f"cuda:{local_rank}")

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)

    safe_gpt_path = config.training.get("safe_gpt_path", "checkpoints/safe_gpt_base")
    tokenizer = load_molecule_tokenizer(safe_gpt_path)
    vocab_size = len(tokenizer)

    model = build_conditional_mlm(config, vocab_size)
    if config.training.get("warm_start_safe_gpt", True):
        model = load_safe_gpt_weights(model, safe_gpt_path, is_main=is_main)
    model = model.to(device)
    if is_main:
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {n_params:.1f}M | World size: {world_size}")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank])

    if is_main and config.training.get("wandb_project"):
        wandb.init(
            project=config.training.wandb_project,
            name=config.training.run_name,
            config=OmegaConf.to_container(config),
        )

    # --- Data loading ---
    use_preprocessed = config.data.get("preprocessed_dir") is not None

    if use_preprocessed:
        from ms_gpt.data.mlm_pretraining_dataset import PreprocessedMLMDataset

        # One-time Arrow conversion (rank 0 only, others wait)
        arrow_path = Path(config.data.preprocessed_dir) / "data.arrow"
        if not arrow_path.exists():
            if is_main:
                PreprocessedMLMDataset.convert(config.data.preprocessed_dir)
            if world_size > 1:
                dist.barrier()

        train_dataset = PreprocessedMLMDataset(
            config.data.preprocessed_dir, tokenizer,
            max_len=config.model.max_seq_len,
            fp_bits=config.model.fp_bits,
        )
        use_formula_group = config.data.get("formula_grouped", False)
        if use_formula_group and train_dataset.formula_groups is not None:
            sampler = None
            batch_sampler = FormulaGroupBatchSampler(
                train_dataset.formula_groups, config.training.batch_size,
                drop_last=True, rank=rank, world_size=world_size,
            )
        elif world_size > 1:
            sampler = DistributedSampler(train_dataset, shuffle=True)
            batch_sampler = None
        else:
            sampler = None
            batch_sampler = None

        loader_kwargs = dict(
            num_workers=config.training.get("num_workers", 8),
            pin_memory=True,
        )
        if batch_sampler is not None:
            train_loader = DataLoader(
                train_dataset, batch_sampler=batch_sampler, **loader_kwargs,
            )
        else:
            train_loader = DataLoader(
                train_dataset, batch_size=config.training.batch_size,
                sampler=sampler, shuffle=(sampler is None),
                drop_last=True, **loader_kwargs,
            )
        if is_main:
            mode = "formula-grouped" if (batch_sampler is not None) else "shuffled"
            print(f"Preprocessed dataset: {len(train_dataset)} molecules ({mode}), "
                  f"num_workers={config.training.get('num_workers', 8)}")
    else:
        from ms_gpt.data.mlm_pretraining_dataset import StreamingMLMDataset
        from datasets import load_dataset
        if config.data.get("hf_dataset"):
            hf_ds = load_dataset(config.data.hf_dataset, split="train", streaming=True)
        else:
            hf_ds = load_dataset("parquet", data_files=config.data.train_path, split="train", streaming=True)
        if world_size > 1:
            hf_ds = hf_ds.shard(num_shards=world_size, index=rank)
        train_dataset = StreamingMLMDataset(
            hf_ds, tokenizer, max_len=config.model.max_seq_len,
            fp_bits=config.model.fp_bits,
        )
        sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=config.training.batch_size,
            num_workers=0, pin_memory=True,
        )
        if is_main:
            print("Streaming dataset (slow mode, consider preprocessing)")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.training.lr,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.95),
    )

    output_dir = Path(config.training.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if world_size > 1 else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        if is_main:
            print(f"Resumed from step {global_step}, epoch {start_epoch}, loss {ckpt['loss']:.4f}")

    save_interval = config.training.get("save_interval", 5000)
    log_interval = config.training.get("log_interval", 100)
    max_steps = config.training.get("max_steps", None)
    warmup_steps = config.training.get("warmup_steps", 0)
    base_lr = config.training.lr
    grad_accum = config.training.gradient_accumulation_steps

    if max_steps is None and use_preprocessed:
        steps_per_epoch = len(train_dataset) // (config.training.batch_size * world_size)
        total_opt_steps = (steps_per_epoch * config.training.max_epochs) // grad_accum
    else:
        total_opt_steps = max_steps or 50000
    if is_main:
        print(f"Cosine schedule over {total_opt_steps} optimizer steps "
              f"(warmup={warmup_steps})")

    total_loss = 0.0
    num_batches = 0
    done = False

    if args.resume and batch_sampler is not None:
        steps_per_epoch = len(batch_sampler)
        skip_batches = global_step % steps_per_epoch if start_epoch * steps_per_epoch < global_step else 0
        if skip_batches > 0:
            batch_sampler.set_start_batch(skip_batches)
            if is_main:
                print(f"Sampler will skip {skip_batches}/{steps_per_epoch} batches in epoch {start_epoch}")

    for epoch in range(start_epoch, config.training.max_epochs):
        if done:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        for batch in pbar:
            opt_step = global_step // grad_accum
            lr = get_lr(opt_step, warmup_steps, base_lr, total_opt_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            fingerprint = batch["fingerprint"].to(device)
            formula_vec = batch["formula_vec"].to(device)

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(
                    input_ids=input_ids,
                    fingerprint=fingerprint,
                    formula_vec=formula_vec,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs.loss / grad_accum

            loss.backward()

            if (num_batches + 1) % grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item() * grad_accum
            num_batches += 1
            global_step += 1

            if is_main and num_batches % log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step, "lr": f"{lr:.2e}"})
                if wandb.run:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/step": global_step,
                        "train/lr": lr,
                    })

            if is_main and global_step % save_interval == 0:
                raw_model = model.module if world_size > 1 else model
                avg = total_loss / num_batches
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg,
                    "best_loss": best_loss,
                }, output_dir / f"mlm_training_step_{global_step}.pt")
                if avg < best_loss:
                    best_loss = avg
                    torch.save(raw_model.state_dict(), output_dir / "best_mlm.pt")
                # Retain only recent training snapshots.
                old_step = global_step - save_interval * 2
                old_ckpt = output_dir / f"mlm_training_step_{old_step}.pt"
                if old_ckpt.exists():
                    old_ckpt.unlink()
                print(f"Step {global_step}: loss={avg:.4f}, lr={lr:.2e}")
                gc.collect()
                torch.cuda.empty_cache()

            if max_steps and global_step >= max_steps:
                done = True
                break

    if is_main:
        raw_model = model.module if world_size > 1 else model
        avg = total_loss / max(num_batches, 1)
        torch.save(raw_model.state_dict(), output_dir / "ms_gpt_mlm.pt")
        print(f"Training complete. Final loss={avg:.4f}, total steps={global_step}")
        if wandb.run:
            wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
