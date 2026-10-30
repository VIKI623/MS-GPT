"""Query the conditional molecule-language model with a spectrum posterior.

For each spectrum in the cache:
  1. Read π_raw (and metadata) directly — no online MIST encoding.
  2. Construct the density-calibrated binary query band.
  3. Generate candidates from the conditional MLM for every query.
  4. Save the raw SAFE sequences in one JSONL row per spectrum.

The point-query method is retained for the paper's ablation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_mlm.py \\
        --inference_config checkpoints/posterior_aligned_adapters/ms_gpt_mlm/mist_official/massspecgym/inference_config.json \\
        --split test \\
        --output eval_results/massspecgym_base.jsonl
"""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from ms_gpt.models.conditional_mlm import (
    ConditionalMoleculeLanguageModel, normalize_mlm_checkpoint_state_dict,
)
from ms_gpt.data.mlm_pretraining_dataset import ELEMENTS, formula_str_to_vec, load_molecule_tokenizer
from ms_gpt.data.posterior_queries import BAND_POINTS, construct_queries, resolve_threshold_band


def decode_tokens(tokenizer, token_ids, eos_id):
    tokens = []
    for t in token_ids:
        if t == eos_id:
            break
        tokens.append(int(t))
    return tokenizer.decode(tokens, skip_special_tokens=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference_config", help="Adapter inference_config.json; explicit arguments override it.")
    parser.add_argument("--calibration_file", help="Canonical JSON supplying this benchmark's band endpoints.")
    parser.add_argument("--cache_dir")
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"))
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mlm_checkpoint")
    parser.add_argument("--mlm_config")
    parser.add_argument("--tokenizer_path", help="Tokenizer paired with the decoder; defaults to its model config.")
    parser.add_argument("--query_method", choices=("point", "density_calibrated_band"),
                        default="density_calibrated_band")
    parser.add_argument("--num_queries", type=int, default=BAND_POINTS)
    parser.add_argument("--samples_per_query", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=None,
                        help="Binarization threshold for the point-query ablation "
                             "(z = posterior > threshold); required when "
                             "--query_method point. The density-calibrated band "
                             "ignores it.")
    parser.add_argument("--threshold_low", type=float, default=None)
    parser.add_argument("--threshold_high", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Spectra per MLM batch. The actual MLM batch has "
                             "batch_size * num_queries rows; "
                             "shrink if OOM.")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the conditional MLM (first call slow, then faster).")
    parser.add_argument("--adapter_checkpoint", default=None,
                        help="Optional posterior-aligned adapter directory (created by "
                             "scripts/train_posterior_aligned_adapter.py). When set, the adapter is "
                             "loaded on top of the base pretrain checkpoint.")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--keep_indices_file", default=None,
                        help="Optional .npy of cache-row indices to restrict generation "
                             "to a subset; start_idx/end_idx then index into this array.")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", required=True)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--inference_config")
    config_parser.add_argument("--cache_dir")
    config_args, _ = config_parser.parse_known_args()
    if config_args.inference_config:
        profile = json.loads(Path(config_args.inference_config).read_text())
        # Adapter application is explicit so the same profile also serves Base.
        fields = {action.dest for action in parser._actions} - {"adapter_checkpoint"}
        parser.set_defaults(**{key: value for key, value in profile.items() if key in fields})
    args = parser.parse_args()
    for field in ("benchmark", "mlm_checkpoint", "mlm_config"):
        if getattr(args, field) is None:
            parser.error(f"Supply --{field} or --inference_config")
    if args.calibration_file:
        calibration_path = Path(args.calibration_file)
        if calibration_path.is_file():
            calibration = json.loads(calibration_path.read_text())["benchmarks"][args.benchmark]
            if config_args.cache_dir is None:
                args.cache_dir = calibration["cache_dir"]
            if args.threshold_low is None:
                args.threshold_low = calibration["threshold_low"]
            if args.threshold_high is None:
                args.threshold_high = calibration["threshold_high"]
        elif args.query_method == "density_calibrated_band" and (
            args.threshold_low is None or args.threshold_high is None
        ):
            parser.error(f"Run density calibration first: {calibration_path}")
    if args.cache_dir is None:
        parser.error("Supply --cache_dir, --calibration_file or --inference_config")
    if args.query_method == "point":
        args.num_queries = 1
        if args.threshold is None or not 0 <= args.threshold <= 1:
            parser.error("Point queries require --threshold in [0, 1]")
    else:
        try:
            resolve_threshold_band(args.threshold_low, args.threshold_high)
        except ValueError as exc:
            parser.error(str(exc))
    if min(args.batch_size, args.num_queries, args.samples_per_query) < 1:
        parser.error("Batch size, query count and samples per query must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --- Load cache ---
    dom_dir = os.path.join(args.cache_dir, args.benchmark)
    tag = f"real_{args.split}"
    pi_path = os.path.join(dom_dir, f"pi_raw_{tag}.npy")
    meta_path = os.path.join(dom_dir, f"meta_{tag}.parquet")
    print(f"Loading cache:\n  meta: {meta_path}\n  pi:   {pi_path}")
    meta = pd.read_parquet(meta_path)
    pi = np.load(pi_path, mmap_mode="r")  # (N, 4096) memmap
    n_total = len(meta)
    assert pi.shape[0] == n_total, f"cache mismatch: meta {n_total} vs pi {pi.shape[0]}"

    # Optional: restrict generation to a saved subset of cache rows. start/end
    # then index into that subset rather than the complete cache.
    if args.keep_indices_file:
        base = np.load(args.keep_indices_file).astype(np.int64)
        if base.ndim != 1 or (base < 0).any() or (base >= n_total).any():
            raise ValueError("Subset indices must refer to rows in the selected posterior cache")
        end = args.end_idx if args.end_idx is not None else len(base)
        if not 0 <= args.start_idx <= end <= len(base):
            raise ValueError("Requested range lies outside the subset")
        indices = base[args.start_idx:end]
        print(f"Subset {args.keep_indices_file}: {len(base)} rows; "
              f"this shard [{args.start_idx}:{end}] = {len(indices)} spectra")
    else:
        end = args.end_idx if args.end_idx is not None else n_total
        if not 0 <= args.start_idx <= end <= n_total:
            raise ValueError("Requested range lies outside the posterior cache")
        indices = np.arange(args.start_idx, end)
        print(f"Generating for cache rows [{args.start_idx}:{end}] = {len(indices)} spectra")
    draws_per_spectrum = args.num_queries * args.samples_per_query
    print(f"Query method: {args.query_method}; total candidates/spectrum={draws_per_spectrum}")

    # --- Load conditional MLM ---
    cfg = OmegaConf.load(args.mlm_config)
    tokenizer = load_molecule_tokenizer(args.tokenizer_path or cfg.get("tokenizer_path"))
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"Loading conditional MLM: {args.mlm_checkpoint}")
    model = ConditionalMoleculeLanguageModel.from_config(cfg, len(tokenizer), dropout=0.0)
    sd = torch.load(args.mlm_checkpoint, map_location="cpu", weights_only=True)
    # Pretrain ckpt was saved with nn.MultiheadAttention; convert to split-QKV.
    sd = normalize_mlm_checkpoint_state_dict(sd)
    model.load_state_dict(sd)
    if args.adapter_checkpoint:
        from peft import PeftModel
        print(f"Loading posterior-aligned adapter: {args.adapter_checkpoint}")
        model = PeftModel.from_pretrained(model, args.adapter_checkpoint, is_trainable=False)
        # Merge LoRA into base weights for faster inference (optional but cheap).
        model = model.merge_and_unload()
    model = model.to(device).eval()
    if args.compile:
        # Compile the forward path (KV-cache loop in .generate is opaque to compile,
        # so this only helps the prefill / cross-attn passes). First batch is slow.
        try:
            model = torch.compile(model, mode="reduce-overhead")
            print("torch.compile enabled")
        except Exception as exc:
            print(f"torch.compile failed ({exc}); proceeding without.")

    # --- Generation loop ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    n_written = 0
    with open(args.output, "w") as fout:
        for batch_start in tqdm(range(0, len(indices), args.batch_size)):
            batch_idx = indices[batch_start:batch_start + args.batch_size]

            # Stack all queries for this spectrum batch.
            fp_stack = np.empty(
                (len(batch_idx) * args.num_queries, cfg.model.fp_bits), dtype=np.float32
            )
            fv_stack = np.empty((len(batch_idx) * args.num_queries, len(ELEMENTS)), dtype=np.float32)
            meta_rows = []
            for j, ridx in enumerate(batch_idx):
                pi_i = np.asarray(pi[ridx], dtype=np.float32)
                z = construct_queries(
                    args.query_method,
                    pi_i,
                    num_queries=args.num_queries,
                    threshold=args.threshold,
                    threshold_low=args.threshold_low,
                    threshold_high=args.threshold_high,
                )
                fv = formula_str_to_vec(meta.iloc[ridx]["formula"])
                fp_stack[j * args.num_queries:(j + 1) * args.num_queries] = z
                fv_stack[j * args.num_queries:(j + 1) * args.num_queries] = fv[None, :]
                meta_rows.append(meta.iloc[ridx])

            fp_t = torch.tensor(fp_stack, dtype=torch.float32, device=device)
            fv_t = torch.tensor(fv_stack, dtype=torch.float32, device=device)

            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16,
                                                 enabled=device.type == "cuda"):
                gen_ids = model.generate(
                    fingerprint=fp_t, formula_vec=fv_t,
                    max_len=cfg.model.max_seq_len,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    bos_token_id=bos_id, eos_token_id=eos_id,
                    num_samples=args.samples_per_query,
                )
            # Generated rows are ordered by spectrum, query, then sample.
            generated_tokens = gen_ids.cpu().tolist()
            for j, row in enumerate(meta_rows):
                tokens = generated_tokens[j * draws_per_spectrum:(j + 1) * draws_per_spectrum]
                sequences = [decode_tokens(tokenizer, ids, eos_id) for ids in tokens]

                fout.write(json.dumps({
                    "cache_row": int(batch_idx[j]),
                    "identifier": row.get("identifier", ""),
                    "target_inchikey14": str(row["inchikey_14"]),
                    "target_formula": str(row["formula"]),
                    "query_method": args.query_method,
                    "num_queries": int(args.num_queries),
                    "samples_per_query": int(args.samples_per_query),
                    "candidate_safe": sequences,
                }) + "\n")
                n_written += 1
            if (batch_start + args.batch_size) % 200 == 0:
                fout.flush()
    print(f"\nDone: wrote {n_written} rows → {args.output}")


if __name__ == "__main__":
    main()
