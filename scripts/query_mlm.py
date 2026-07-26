"""Query the conditional molecule-language model with a spectrum posterior.

For each spectrum in the cache:
  1. Read π_raw (and metadata) directly — no online MIST encoding.
  2. Construct the density-calibrated binary query band.
  3. Generate candidates from the conditional MLM for every query.
  4. Write one JSONL row per spectrum.

The point-query method is retained for the paper's ablation.

Usage:
    CUDA_VISIBLE_DEVICES=0 python scripts/query_mlm.py \\
        --cache_dir data/posterior_cache --benchmark massspecgym \\
        --split test \\
        --mlm_checkpoint checkpoints/mlm/ms_gpt_mlm.pt \\
        --mlm_config configs/mlm_pretraining.yaml \\
        --query_method density_calibrated_band \\
        --num_queries 20 --samples_per_query 5 \\
        --threshold_low 0.069 --threshold_high 0.274 \\
        --start_idx 0 --end_idx 2195 \\
        --output eval_results/massspecgym_shard0.jsonl
"""
import argparse
import json
import os
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

from safe.tokenizer import SAFETokenizer
import safe as sf

from ms_gpt.models.conditional_mlm import (
    ConditionalMoleculeLanguageModel, normalize_mlm_checkpoint_state_dict,
)
from ms_gpt.data.mlm_pretraining_dataset import formula_str_to_vec
from ms_gpt.data.posterior_queries import construct_queries


def load_molecule_tokenizer():
    model_path = os.environ.get("MS_GPT_SAFE_GPT_CKPT", "checkpoints/safe_gpt_base")
    tk = SAFETokenizer.from_pretrained(model_path).get_pretrained()
    tk.add_tokens(["<", ">"])
    return tk


def decode_tokens(tokenizer, token_ids, eos_id):
    tokens = []
    for t in token_ids:
        if t == eos_id:
            break
        tokens.append(int(t))
    text = tokenizer.decode(tokens, skip_special_tokens=True)
    try:
        smi = sf.decode(text)
        if smi:
            return smi
    except Exception:
        pass
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--benchmark", choices=("massspecgym", "nplib1"), required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    parser.add_argument("--mlm_checkpoint", required=True)
    parser.add_argument("--mlm_config", required=True)
    parser.add_argument("--query_method", choices=("point", "density_calibrated_band"),
                        default="density_calibrated_band")
    parser.add_argument("--num_queries", type=int, default=20)
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
    args = parser.parse_args()

    if args.query_method == "point":
        args.num_queries = 1

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
    # then index into the subset array (so 8-shard sharding works over the subset).
    if args.keep_indices_file:
        base = np.load(args.keep_indices_file).astype(np.int64)
        end = args.end_idx if args.end_idx else len(base)
        indices = base[args.start_idx:end]
        print(f"Subset {args.keep_indices_file}: {len(base)} rows; "
              f"this shard [{args.start_idx}:{end}] = {len(indices)} spectra")
    else:
        end = args.end_idx if args.end_idx else n_total
        indices = np.arange(args.start_idx, end)
        print(f"Generating for cache rows [{args.start_idx}:{end}] = {len(indices)} spectra")
    print(f"Query method: {args.query_method}; total candidates/spectrum="
          f"{args.num_queries * args.samples_per_query}")

    # --- Load conditional MLM ---
    cfg = OmegaConf.load(args.mlm_config)
    tokenizer = load_molecule_tokenizer()
    vocab_size = len(tokenizer)
    bos_id = tokenizer.bos_token_id
    eos_id = tokenizer.eos_token_id

    print(f"Loading conditional MLM: {args.mlm_checkpoint}")
    model = ConditionalMoleculeLanguageModel(
        vocab_size=vocab_size,
        d_model=cfg.model.d_model,
        n_layers=cfg.model.n_layers,
        n_heads=cfg.model.n_heads,
        fp_bits=cfg.model.fp_bits,
        fp_encoder_layers=cfg.model.fp_encoder_layers,
        cross_attn_every_n=cfg.model.cross_attn_every_n,
        max_seq_len=cfg.model.max_seq_len,
        dropout=0.0,
    )
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
    fout = open(args.output, "w")
    n_written = 0
    try:
        for batch_start in tqdm(range(0, len(indices), args.batch_size)):
            batch_idx = indices[batch_start:batch_start + args.batch_size]
            if len(batch_idx) == 0:
                continue

            # Stack all queries for this spectrum batch.
            fp_stack = np.empty(
                (len(batch_idx) * args.num_queries, cfg.model.fp_bits), dtype=np.float32
            )
            fv_stack = np.empty((len(batch_idx) * args.num_queries, 14), dtype=np.float32)
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

            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
                gen_ids = model.generate(
                    fingerprint=fp_t, formula_vec=fv_t,
                    max_len=cfg.model.max_seq_len,
                    temperature=args.temperature,
                    top_k=args.top_k, top_p=args.top_p,
                    bos_token_id=bos_id, eos_token_id=eos_id,
                    num_samples=args.samples_per_query,
                )
            # Generated rows are ordered by spectrum, query, then sample.
            for j, row in enumerate(meta_rows):
                smis = []
                for q in range(args.num_queries):
                    for c in range(args.samples_per_query):
                        row_idx = (
                            j * args.num_queries + q
                        ) * args.samples_per_query + c
                        ids = gen_ids[row_idx].cpu().tolist()
                        smis.append(decode_tokens(tokenizer, ids, eos_id))

                fout.write(json.dumps({
                    "cache_row": int(batch_idx[j]),
                    "identifier": row.get("identifier", ""),
                    "target_inchikey14": str(row["inchikey_14"]),
                    "target_formula": str(row["formula"]),
                    "query_method": args.query_method,
                    "num_queries": int(args.num_queries),
                    "samples_per_query": int(args.samples_per_query),
                    "candidate_smiles": smis,
                }) + "\n")
                n_written += 1
            if (batch_start + args.batch_size) % 200 == 0:
                fout.flush()
    finally:
        fout.close()

    print(f"\nDone: wrote {n_written} rows → {args.output}")


if __name__ == "__main__":
    main()
