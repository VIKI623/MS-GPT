#!/bin/bash
# MS-GPT environment setup. Creates a conda env, installs deps, and installs the
# `ms_gpt` package in editable mode.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

ENV_NAME="${1:-ms_gpt}"

echo "[1/3] Creating conda env '$ENV_NAME' (python 3.10)"
conda create -y -n "$ENV_NAME" python=3.10
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "[2/3] Installing PyTorch 2.6.0 (CUDA 12.4 wheel)"
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

echo "[3/3] Installing MS-GPT + dependencies"
python -m pip install -r env/requirements.txt
python -m pip install -e .

echo
echo "Done. Activate with: conda activate $ENV_NAME"
echo "Next: copy env/.env.example -> .env, fill in paths, then 'set -a; source .env; set +a'"
