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
from torch.utils.data import DataLoader, DistributedSampler, Subset
from torch.amp import autocast

import wandb
from tqdm import tqdm
from omegaconf import OmegaConf

from ms_gpt.models.conditional_mlm import ConditionalMoleculeLanguageModel
from ms_gpt.data.formula_group_sampler import FormulaGroupBatchSampler
from ms_gpt.data.mlm_pretraining_dataset import load_molecule_tokenizer


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


def load_safe_gpt_weights(model, model_path="checkpoints/safe_gpt_base", is_main=True):
    """Initialize the decoder backbone from its paired SAFE-GPT checkpoint."""
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
        elif (safe_key in {"transformer.wte.weight", "transformer.wpe.weight", "lm_head.weight"}
              and safe_param.shape[1:] == our_shape[1:]):
            # Partial copy (e.g., wte [1880,768] → [1882,768], wpe [1024,768] → [256,768])
            min_rows = min(safe_param.shape[0], our_shape[0])
            our_sd[our_key][:min_rows] = safe_param[:min_rows]
            loaded += 1
        else:
            raise ValueError(f"SAFE-GPT shape mismatch for {safe_key}: {tuple(safe_param.shape)} vs {tuple(our_shape)}")

    model.load_state_dict(our_sd)
    if is_main:
        total = loaded + skipped
        print(f"Warm-start from safe-gpt: {loaded}/{total} params loaded, {skipped} skipped (cross-attn/FP/formula)")
    return model


def get_lr(step, warmup_steps, base_lr, max_steps):
    if step < warmup_steps:
        return base_lr * step / max(warmup_steps, 1)
    decay_ratio = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * decay_ratio))


@torch.no_grad()
def evaluate_loss(model, loader, device):
    """Aggregate validation loss across disjoint per-rank subsets."""
    model.eval()
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        with autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            loss = model(**batch).loss
        count = batch["input_ids"].shape[0]
        totals[0] += loss.double() * count
        totals[1] += count
    if dist.is_initialized():
        dist.all_reduce(totals)
    model.train()
    return (totals[0] / totals[1]).item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, help="Override training.seed")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    rank, world_size, local_rank = setup_distributed()
    is_main = rank == 0
    config = OmegaConf.load(args.config)
    if config.data.get("preprocessed_dir") is None:
        if config.data.get("format", "arrow") != "arrow":
            raise ValueError("Saved token targets require data.preprocessed_dir and format=tokenized_parquet")
        if config.data.get("formula_grouped", False):
            raise ValueError("Formula-grouped sampling requires a preprocessed corpus")
        if config.training.get("max_steps") is None:
            raise ValueError("Streaming pretraining requires training.max_steps for the learning-rate schedule")
    device = torch.device(f"cuda:{local_rank}")

    seed = args.seed if args.seed is not None else config.training.get("seed", 42)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cuda.matmul.allow_tf32 = config.training.get("tf32", False)
    torch.backends.cudnn.allow_tf32 = config.training.get("tf32", False)

    safe_gpt_path = config.training.get("safe_gpt_path", "checkpoints/safe_gpt_base")
    tokenizer = load_molecule_tokenizer(config.get("tokenizer_path", safe_gpt_path))
    vocab_size = len(tokenizer)

    model = ConditionalMoleculeLanguageModel.from_config(config, vocab_size)
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
    batch_sampler = None
    val_loader = None

    if use_preprocessed:
        from ms_gpt.data.mlm_pretraining_dataset import PreprocessedMLMDataset, TokenizedMLMDataset

        data_format = config.data.get("format", "arrow")
        if data_format == "arrow":
            # One-time Arrow conversion (rank 0 only, others wait).
            arrow_path = Path(config.data.preprocessed_dir) / "train" / "data.arrow"
            if not arrow_path.exists():
                if is_main:
                    PreprocessedMLMDataset.convert(config.data.preprocessed_dir)
                if world_size > 1:
                    dist.barrier()
            dataset_class = PreprocessedMLMDataset
        elif data_format == "tokenized_parquet":
            dataset_class = TokenizedMLMDataset
        else:
            raise ValueError(f"Unknown pretraining data format: {data_format}")

        train_dataset = dataset_class(
            config.data.preprocessed_dir, tokenizer,
            max_len=config.model.max_seq_len,
            fp_bits=config.model.fp_bits,
        )
        use_formula_group = config.data.get("formula_grouped", False)
        if use_formula_group and train_dataset.formula_groups is None:
            raise FileNotFoundError("Formula-grouped sampling requires formula_groups.pkl in the pretraining corpus")
        if use_formula_group:
            sampler = None
            batch_sampler = FormulaGroupBatchSampler(
                train_dataset.formula_groups, config.training.batch_size,
                drop_last=True, seed=seed, rank=rank, world_size=world_size,
                mix_remainders=config.data.get("mix_formula_remainders", False),
            )
        elif world_size > 1:
            sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
            batch_sampler = None
        else:
            sampler = None
            batch_sampler = None

        loader_kwargs = dict(
            num_workers=config.training.get("num_workers", 8),
            pin_memory=True,
            collate_fn=getattr(train_dataset, "collate_fn", None),
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
        validation_split = config.data.get("validation_split")
        if validation_split:
            val_dataset = TokenizedMLMDataset(
                config.data.preprocessed_dir, tokenizer,
                max_len=config.model.max_seq_len, fp_bits=config.model.fp_bits,
                split=validation_split,
            )
            val_loader = DataLoader(
                Subset(val_dataset, range(rank, len(val_dataset), world_size)),
                batch_size=config.training.batch_size,
                num_workers=config.training.get("num_workers", 0),
                pin_memory=True, collate_fn=val_dataset.collate_fn,
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
        betas=tuple(config.training.get("betas", [0.9, 0.95])),
    )

    output_dir = Path(config.training.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)

    global_step = 0
    best_loss = float("inf")
    start_epoch = 0
    resume_batch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model = model.module if world_size > 1 else model
        raw_model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        global_step = ckpt.get("global_step", 0)
        best_loss = ckpt.get("best_loss", float("inf"))
        start_epoch = ckpt.get("epoch", 0)
        resume_batch = ckpt.get("batch_in_epoch", 0)
        if is_main:
            print(f"Resumed from step {global_step}, epoch {start_epoch}, loss {ckpt['loss']:.4f}")

    save_interval = config.training.get("save_interval", 5000)
    log_interval = config.training.get("log_interval", 100)
    eval_interval = config.training.get("eval_interval", 0)
    max_steps = config.training.get("max_steps", None)
    warmup_steps = config.training.get("warmup_steps", 0)
    base_lr = config.training.lr
    grad_accum = config.training.gradient_accumulation_steps

    batches_per_epoch = len(train_loader) if use_preprocessed else None
    total_opt_steps = max_steps if max_steps is not None else (
        math.ceil(batches_per_epoch / grad_accum) * config.training.max_epochs
    )
    if is_main:
        print(f"Cosine schedule over {total_opt_steps} optimizer steps "
              f"(warmup={warmup_steps})")

    total_loss = 0.0
    num_batches = 0
    done = max_steps is not None and global_step >= max_steps
    raw_model = model.module if world_size > 1 else model
    optimizer.zero_grad()
    if batches_per_epoch is not None and resume_batch == batches_per_epoch:
        start_epoch += 1
        resume_batch = 0

    for epoch in range(start_epoch, config.training.max_epochs):
        if done:
            break
        if sampler is not None:
            sampler.set_epoch(epoch)
        if batch_sampler is not None:
            batch_sampler.set_epoch(epoch)
            batch_sampler.set_start_batch(resume_batch)

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", disable=not is_main)
        start_batch = resume_batch if batch_sampler is not None else 0
        for batch_index, batch in enumerate(pbar, start=start_batch):
            if batch_index < resume_batch:
                continue
            accumulation = grad_accum
            if batches_per_epoch is not None:
                accumulation = min(
                    grad_accum,
                    batches_per_epoch - (batch_index // grad_accum) * grad_accum,
                )

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
                loss = outputs.loss / accumulation

            loss.backward()
            total_loss += outputs.loss.item()
            num_batches += 1

            end_of_epoch = batch_index + 1 == batches_per_epoch
            if (batch_index + 1) % grad_accum != 0 and not end_of_epoch:
                continue

            lr = get_lr(global_step, warmup_steps, base_lr, total_opt_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if is_main and global_step % log_interval == 0:
                avg_loss = total_loss / num_batches
                pbar.set_postfix({"loss": f"{avg_loss:.4f}", "step": global_step, "lr": f"{lr:.2e}"})
                if wandb.run:
                    wandb.log({
                        "train/loss": avg_loss,
                        "train/step": global_step,
                        "train/lr": lr,
                    })

            if val_loader is not None and eval_interval and global_step % eval_interval == 0:
                val_loss = evaluate_loss(raw_model, val_loader, device)
                if val_loss < best_loss:
                    best_loss = val_loss
                    if is_main:
                        torch.save(raw_model.state_dict(), output_dir / "best_mlm.pt")
                if is_main:
                    print(f"Step {global_step}: val_loss={val_loss:.6f}")
                    if wandb.run:
                        wandb.log({"val/loss": val_loss, "train/step": global_step})

            if is_main and global_step % save_interval == 0:
                avg = total_loss / num_batches
                if val_loader is None and avg < best_loss:
                    best_loss = avg
                    torch.save(raw_model.state_dict(), output_dir / "best_mlm.pt")
                torch.save({
                    "global_step": global_step,
                    "epoch": epoch,
                    "batch_in_epoch": batch_index + 1,
                    "model_state_dict": raw_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg,
                    "best_loss": best_loss,
                }, output_dir / f"mlm_training_step_{global_step}.pt")
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
        resume_batch = 0

    if val_loader is not None:
        if world_size > 1:
            dist.barrier()
        best_path = output_dir / "best_mlm.pt"
        if math.isfinite(best_loss) and best_path.exists():
            raw_model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        val_loss = evaluate_loss(raw_model, val_loader, device)
        if is_main:
            print(f"Export validation loss: {val_loss:.6f}")

    if is_main:
        avg = total_loss / max(num_batches, 1)
        torch.save(raw_model.state_dict(), output_dir / config.training.get("output_name", "ms_gpt_mlm.pt"))
        print(f"Training complete. Final loss={avg:.4f}, total steps={global_step}")
        if wandb.run:
            wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    main()
