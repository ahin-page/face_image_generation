#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-7}"
export CUDA_HOME="${CUDA_HOME:-/opt/conda/envs/generate_image/lib/python3.11/site-packages/nvidia/cu13}"
export PATH="/opt/conda/envs/generate_image/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib:/opt/conda/envs/generate_image/lib:${LD_LIBRARY_PATH:-}"
export CC="/opt/conda/envs/generate_image/bin/x86_64-conda-linux-gnu-gcc"
export CXX="/opt/conda/envs/generate_image/bin/x86_64-conda-linux-gnu-g++"
export CUDAHOSTCXX="$CXX"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$PROJECT_ROOT/outputs/torch_extensions}"
export PYTHONWARNINGS="${PYTHONWARNINGS:-once}"

/opt/conda/envs/generate_image/bin/python "$SCRIPT_DIR/generate_stylegan2ada_submission.py" "$@"
