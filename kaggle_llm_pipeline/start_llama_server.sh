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

MODEL_DIR="/kaggle/working/models"
MODEL_PATH="$MODEL_DIR/qwen3-coder-32b-q5_k_m.gguf"

# Download model from Hugging Face if not already present this session
if [ ! -f "$MODEL_PATH" ]; then
  echo "Downloading Qwen3-Coder-30B-A3B-Instruct-Q5_K_M from Hugging Face (~10 min)..."
  pip install -q -U huggingface_hub 2>/dev/null || true
  export MODEL_PATH
  python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import shutil, os
path = hf_hub_download(
    repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    filename="Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf",
)
dst = os.environ["MODEL_PATH"]
if os.path.islink(dst) or os.path.exists(dst):
    os.remove(dst)
os.makedirs(os.path.dirname(dst), exist_ok=True)
# Symlink (not copy): /kaggle/working has a ~20GB cap, the model is ~21GB.
# The HF cache lives outside that quota, so we link to it instead.
os.symlink(path, dst)
EOF
fi
if [ ! -f "$MODEL_PATH" ]; then
  echo "ERROR: model download failed."
  exit 1
fi

# Fetch prebuilt llama.cpp CUDA binary if not already present in this session
if [ ! -x ./llama-server ]; then
  echo "Fetching llama.cpp CUDA build..."
  pip install -q llama-cpp-python 2>/dev/null || true
  # Prefer the official release binary; fall back to building from source.
  curl -sL https://github.com/ggml-org/llama.cpp/releases/latest/download/llama-b-bin-ubuntu-x64.zip -o llama.zip \
    && unzip -o -j llama.zip '*/llama-server' && rm llama.zip \
    || { echo "Prebuilt binary unavailable — building from source (~10 min)..."
         git clone --depth 1 https://github.com/ggml-org/llama.cpp.git
         cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DLLAMA_CURL=OFF
         cmake --build llama.cpp/build --config Release -j2 --target llama-server
         cp llama.cpp/build/bin/llama-server .; }
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
