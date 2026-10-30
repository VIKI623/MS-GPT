# MS-GPT

**MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as
Spectrum-Induced Posterior Querying of a Molecule-Language Model**

MS-GPT formulates known-formula MS/MS *de novo* structure elucidation as
spectrum-induced posterior querying of a conditional molecule-language model.
MS-GPT constructs a calibrated band of posterior-derived queries, generates
candidates across the band, and ranks them by generation frequency and posterior
similarity.

![MS-GPT workflow](assets/inference.png)

The 150.1M-parameter conditional molecule-language model is pretrained once on a
molecule-only corpus with oracle fingerprints and formulas, then shared across
NPLIB1 and MassSpecGym. Each benchmark has a dataset-specific MIST spectrum
encoder and a lightweight posterior-aligned LoRA adapter.

## Method

### Spectrum-induced posterior querying

A frozen, dataset-specific MIST encoder maps an MS/MS spectrum and its known
formula to a 4096-bit fingerprint posterior. Active-bit density band calibration
derives dataset-specific threshold intervals from the corresponding encoder
training split.

At the default candidate-pool size `B = 100`, MS-GPT forms `M = 20` thresholded
queries and samples `N = 5` SAFE sequences per query. Generated structures are
grouped by InChIKey-14 and ranked by formula match, generation frequency, and
posterior similarity. Local edits supplement short candidate pools.

### Molecule-only pretraining and posterior-aligned adaptation

Formula-grouped pretraining maps oracle fingerprint-formula queries to target
SAFE sequences without spectra. The main model is trained for two epochs on
100M molecule records after excluding connectivity-equivalent structures
from the validation and test splits of both benchmarks.

Posterior-aligned adaptation samples thresholded queries from the target
spectral domain and trains LoRA modules only in the query-reading pathway. The
formula encoder, SAFE-GPT backbone, language-model head, and all other pretrained
weights remain frozen. A teacher-forced cross-entropy objective is paired with a
KL anchor to the frozen model.


## Results

> [!IMPORTANT]
> **Updated MIST encoders and benchmark results.** 
> 
> This release replaces the earlier FRIGID-released weights with [official MIST checkpoints](https://github.com/coleygroup/MIST), following the encoder update described in [FRIGID](https://github.com/coleygroup/FRIGID). It also includes an attention-mask correction in the MIST implementation.


The paper evaluates exact connectivity recovery under the known-formula protocol
with `B = 100`, using [MIST official checkpoints](https://github.com/coleygroup/MIST#data-and-pretrained-models). All metrics for each benchmark are evaluated on the same final candidate list produced by the unified postprocessing. 

Exact match compares the first 14 InChIKey characters; Tanimoto reports the maximum
radius-2 Morgan-fingerprint similarity among the top-ranked candidates. MCES configurations follow [MassSpecGym in the Wild](https://arxiv.org/abs/2606.19624). 
| Benchmark | Top-1 accuracy (%) | Top-1 Tanimoto | Top-1 MCES ↓ | Top-10 accuracy (%) | Top-10 Tanimoto | Top-10 MCES ↓ |
|---|---:|---:|---:|---:|---:|---:|
| NPLIB1 | **17.09** | **0.5266** | **6.6880** | **27.59** | **0.6081** | **5.1117** |
| MassSpecGym | **10.22** | **0.3889** | **12.0882** | **13.21** | **0.4458** | **10.5485** |

Component ablations retain the same budget of 100 decoder samples per spectrum.
Query-construction controls below are evaluated without posterior-aligned
adaptation. Cells report Top-1/Top-10 exact-match accuracy (%).

| Variant | NPLIB1 accuracy (%) | MassSpecGym accuracy (%) |
|---|---:|---:|
| MS-GPT | **17.09/27.59** | **10.22/13.21** |
| MS-GPT-Base (without posterior-aligned adaptation) | 15.87/23.93 | 9.27/12.75 |
| Density-match point query | 11.60/17.09 | 7.97/10.34 |
| Validation-F1 point query | 13.68/17.83 | 7.42/9.69 |
| Validation-cosine point query | 13.43/17.34 | 6.70/9.06 |

## Reproduction

An NVIDIA RTX 6000 Ada GPU is recommended for reproduction.

### 1. Install the environment

```bash
git clone https://github.com/VIKI623/MS-GPT.git
cd MS-GPT
bash env/setup.sh ms_gpt
conda activate ms_gpt
cp env/.env.example .env
```

Edit `.env` to point to the released checkpoints and benchmark data, then load
the variables:

```bash
set -a
source .env
set +a
```

The setup script installs PyTorch 2.6.0 with the CUDA 12.4 wheel.

### 2. Place the released assets

Download the released asset bundle from the
[SJTU Cloud](https://pan.sjtu.edu.cn/web/share/b1789d4d6ef298565f7f4e8c07603eb9)
and extract it into the repository root:

```text
checkpoints/
├── mlm/{ms_gpt_mlm.pt,ms_gpt_mlm_msg_50m.pt}
├── mist/
│   ├── frigid_legacy/{mist_nplib1.pt,mist_massspecgym.pt}
│   └── mist_official/{mist_nplib1.pt,mist_massspecgym.pt}
├── mist_cf_msg/{mist_cf_msg.ckpt,mist_cf_fast_filter.ckpt}
├── posterior_aligned_adapters/
│   ├── ms_gpt_mlm/
│   │   ├── frigid_legacy/{nplib1,massspecgym}/
│   │   └── mist_official/{nplib1,massspecgym}/
│   └── ms_gpt_mlm_msg_50m/
│       └── mist_official/massspecgym/
├── safe_gpt_base/
└── safe_gpt_base_msg_50m/
data/
├── benchmarks/{nplib1,massspecgym}/
├── sirius_decomp/massspecgym/{train,val,test}.parquet  # optional; formula candidates
├── pretrain_100m/                         # optional; pretraining only
└── msg_pretrain_50m/                      # optional; pretraining only
```

### 3. Build fingerprint-posterior caches

```bash
bash data_preparation/build_posterior_cache.sh
bash data_preparation/compute_density_calibration.sh
```

### 4. Reproduce the reported evaluations

```bash
# Main results
bash scripts/launch/reproduce_main_results.sh all

# Component ablations
bash scripts/launch/reproduce_ablation_results.sh all

# MassSpecGym with the 50M decoder
bash scripts/launch/reproduce_main_results.sh massspecgym ms_gpt_mlm_msg_50m
```

Pass `nplib1` or `massspecgym` instead of `all` to evaluate one benchmark.
Generated candidates and metrics are written to `eval_results/`.

MCES is evaluated on CPU HPC nodes after candidate generation:

```bash
python -m pip install -e '.[metrics]'
sbatch scripts/launch/evaluate_mces.slurm \
  nplib1 eval_results/nplib1_ms_gpt_mlm_mist_official_main/ranked.jsonl
```

## License

The code is released under the [Apache License 2.0](LICENSE).

## Citation

```bibtex
@misc{zhao2026msgptrethinkingmsmsnovo,
      title={MS-GPT: Rethinking MS/MS De Novo Structure Elucidation as Spectrum-Induced Posterior Querying of a Molecule-Language Model},
      author={Xin Zhao and Yumin Liu and Zhuo Li and Weichu Zheng and Feng Zhu and Xiaokang Yang and Yaohui Jin and Yanyan Xu},
      year={2026},
      eprint={2607.23607},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2607.23607},
}
```
