#!/bin/bash
# start_llama_server.sh — Qwen3-Coder-32B (Q5_K_M) on Kaggle 2x Tesla T4.
# Run in a Kaggle notebook cell or terminal with the GPU accelerator enabled.
#
# First-time setup: attach a Kaggle Dataset containing the GGUF, e.g.
#   dataset slug: <your-user>/qwen3-coder-32b-gguf
#   file:         qwen3-coder-32b-q5_k_m.gguf
# Attaching as a Dataset avoids a ~10min HF download on every session.

set -e
cd /kaggle/working

MODEL_PATH="/kaggle/input/qwen3-coder-32b-gguf/qwen3-coder-32b-q5_k_m.gguf"

if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model not found at $MODEL_PATH"
  echo "Attach your GGUF as a Kaggle Dataset (Add Input -> Datasets) first."
  exit 1
fi

# Fetch prebuilt llama.cpp CUDA binary if not already present in this session
if [ ! -x ./llama-server ]; then
  echo "Fetching llama.cpp CUDA build..."
  URL="https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b-bin-ubuntu-x64.zip"
  curl -sL "$URL" -o llama.zip && unzip -o -j llama.zip '*/llama-server' && rm llama.zip
fi

echo "Pre-warming page cache..."
cat "$MODEL_PATH" > /dev/null

echo "Starting llama-server (32k ctx, q8_0 KV cache, split across both T4s)..."
./llama-server \
  -m "$MODEL_PATH" \
  --split-mode layer \
  --tensor-split 1,1 \
  -ngl 999 \
  --flash-attn \
  -c 32768 \
  --cache-type-k q8_0 \
  --cache-type-v q8_0 \
  --host 0.0.0.0 \
  --port 8080

# REMINDER: stop this process AND end the Kaggle session when done.
# Kaggle bills GPU quota by session wall-clock time; an idle server
# costs exactly as much as an active one.
