#!/usr/bin/env bash
# Start the OpenAI-compatible llama-cpp-python inference server.
#
# Exposes /v1/completions and /health — eval pods in run_eval_shard.py connect
# to these endpoints unchanged; no eval-side code modifications required.
#
# Expected speed on RTX 4050 Laptop (6 GB VRAM) with Q8_0 GGUF: ~10-20 tok/s
# vs 1.3 tok/s with the previous Unsloth + TORCHDYNAMO_DISABLE workaround.
#
# Prerequisites:
#   bash scripts/merge_and_export.py      # merge LoRA into base model
#   bash scripts/convert_to_gguf.sh       # convert to GGUF Q8_0
#   pip install -r serving/requirements-serve.txt  (with CUDA, see that file)
#
# Usage:
#   bash serving/start_server.sh [path/to/model.gguf] [port]

set -euo pipefail

MODEL="${1:-outputs/rinlekha-q8.gguf}"
PORT="${2:-8000}"

# CUDA libs (cudart, cublas, cusparse, etc.) live inside individual nvidia
# sub-packages in the conda env, not on the system library path.
# Collect all their lib dirs and prepend to LD_LIBRARY_PATH.
NVIDIA_LIBS="$(python -c "
import os, sys
base = os.path.join(sys.prefix, 'lib/python3.11/site-packages/nvidia')
dirs = []
for pkg in os.listdir(base):
    lib = os.path.join(base, pkg, 'lib')
    if os.path.isdir(lib):
        dirs.append(lib)
print(':'.join(dirs))
")"
export LD_LIBRARY_PATH="${NVIDIA_LIBS}:${LD_LIBRARY_PATH:-}"

if [[ ! -f "$MODEL" ]]; then
    echo "Model not found: $MODEL"
    echo "Run scripts/merge_and_export.py then scripts/convert_to_gguf.sh first."
    exit 1
fi

echo "Starting llama-cpp-python server"
echo "  model:  $MODEL"
echo "  port:   $PORT"
echo "  layers: all on GPU (-ngl 99)"

python -m llama_cpp.server \
    --model        "$MODEL" \
    --n_gpu_layers 99 \
    --n_ctx        2048 \
    --host         0.0.0.0 \
    --port         "$PORT"
