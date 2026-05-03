#!/usr/bin/env bash
# Step 2 of the GGUF export pipeline: convert merged fp16 HF model to GGUF Q8_0.
#
# Q8_0 chosen over Q4_K_M because:
#   - convert_hf_to_gguf.py converts directly without a separate quantize binary
#   - Q8_0 (4.3 GB) fits in the 6.4 GB VRAM alongside KV cache
#   - Quality difference vs Q4_K_M is negligible for this task
#
# Prerequisites (run once):
#   pip install gguf sentencepiece protobuf
#   CMAKE_ARGS="-DGGML_CUDA=on" pip install "llama-cpp-python[server]>=0.3.0"
#
# Usage:
#   bash scripts/convert_to_gguf.sh [merged_dir] [output_gguf]
#
# Example:
#   bash scripts/convert_to_gguf.sh outputs/merged_gemma3_4b outputs/rinlekha-q8.gguf

set -euo pipefail

MERGED="${1:-outputs/merged_gemma3_4b}"
OUTPUT="${2:-outputs/rinlekha-q8.gguf}"
LLAMA_DIR="$(dirname "$0")/.llama_cpp"

if [[ ! -d "$MERGED" ]]; then
    echo "ERROR: merged model not found at '$MERGED'"
    echo "Run 'python scripts/merge_and_export.py' first."
    exit 1
fi

# Clone llama.cpp (conversion tools only) if not already present
if [[ ! -d "$LLAMA_DIR" ]]; then
    echo "Cloning llama.cpp conversion tools (shallow)..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp "$LLAMA_DIR"
fi

echo "Installing Python conversion dependencies..."
pip install -q gguf sentencepiece protobuf

echo "Converting $MERGED → $OUTPUT (Q8_0)..."
python "$LLAMA_DIR/convert_hf_to_gguf.py" "$MERGED" \
    --outtype q8_0 \
    --outfile "$OUTPUT"

echo ""
echo "GGUF written: $OUTPUT"
du -sh "$OUTPUT"
echo ""
echo "Start inference server:"
echo "  bash serving/start_server.sh $OUTPUT"
