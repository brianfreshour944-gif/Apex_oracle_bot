# =====================================================================
# ⚠️  OBSOLETE — DO NOT USE  ⚠️
# =====================================================================
# This script used Cloudflare Tunnel (cloudflared), which has been
# REPLACED by FRP (Fast Reverse Proxy) as of August 2026.
#
# Reason: Tailscale was tried first for the Oracle<->Kaggle connection
# but gets killed by Kaggle's sandbox within ~5 seconds. FRP was found
# to work reliably instead.
#
# The current, working startup script is NOT in this repo — it lives
# in the user's private notes and is pasted directly into a fresh
# Kaggle notebook cell each session. It:
#   1. Loads models from the cached 'trading-bot-llms' Kaggle Dataset
#   2. Loads dependencies from the cached 'trading-bot-wheels' Dataset
#   3. Starts llama-cpp-python servers on ports 8001/8002
#   4. Connects via FRP client to Oracle's frps server (port 7000),
#      tunneling to fixed ports 7001 (Mistral) and 7002 (DeepSeek)
#
# See CONNECTIONS.md / the Oracle-Kaggle reference doc for full details.
# This file is kept only for historical reference. Do not run it.
# =====================================================================

#!/bin/bash
# =====================================================================
# Kaggle LLM Setup - One-command startup script
# =====================================================================
# Usage: bash kaggle_start.sh
# Run this in a Kaggle notebook cell (%%bash) or terminal
# =====================================================================

set -e

echo "🚀 Starting Kaggle LLM Setup (Qwen3-Coder-30B on 2× T4)"
echo "This will take ~15 minutes on first run (builds CUDA llama.cpp)"

cd /kaggle/working/Apex_oracle_bot/kaggle_llm_pipeline

MODEL="/kaggle/tmp/models/qwen3-coder-32b-q4_k_m.gguf"

# 1. Download model if not cached
if [ ! -f "$MODEL" ]; then
    echo "📥 Downloading 21GB model from Hugging Face..."
    pip install -q -U huggingface_hub
    python3 - <<'EOF'
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(
    repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
    filename="Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf"
)
os.makedirs(os.path.dirname("/kaggle/tmp/models/"), exist_ok=True)
shutil.copy(path, "/kaggle/tmp/models/qwen3-coder-32b-q4_k_m.gguf")
EOF
    echo "✅ Model downloaded"
else
    echo "✅ Model already cached"
fi

# 2. Build CUDA llama-server if not already built
if [ ! -x ./llama-cuda ]; then
    echo "🔨 Building CUDA llama-server (~10-15 min)..."
    
    # Symlink driver lib where CMake expects it
    ln -sf /usr/local/nvidia/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so
    ln -sf /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1
    
    # Build llama.cpp with CUDA
    git clone --depth 1 https://github.com/ggml-org/llama.cpp.git cuda_src 2>/dev/null || true
    
    cmake -S cuda_src -B cuda_build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
          -DCUDAToolkit_ROOT=/usr/local/cuda > build.log 2>&1
    cmake --build cuda_build --config Release -j4 --target llama-server >> build.log 2>&1
    
    cp cuda_build/bin/llama-server ./llama-cuda
    echo "✅ llama-cuda built"
else
    echo "✅ llama-cuda already built"
fi

# 3. Start llama-server (detached)
echo "🚀 Starting llama-server..."
nohup ./llama-cuda \
  -m /kaggle/tmp/models/qwen3-coder-32b-q4_k_m.gguf \
  --split-mode layer --tensor-split 1,1 -ngl 999 \
  -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --host 0.0.0.0 --port 8080 > server.log 2>&1 &

echo "⏳ Waiting for model to load on GPUs (1-2 min)..."
for i in {1..24}; do
    sleep 5
    if grep -q "listening on http://0.0.0.0:8080" server.log; then
        echo "✅ llama-server listening on port 8080"
        break
    fi
    if grep -q "offloaded 49/49 layers to GPU" server.log; then
        echo "  ... GPU offload in progress ($i/24)"
    fi
done

# 4. Start Cloudflare tunnel (detached)
if [ ! -x ./cloudflared ]; then
    echo " Downloading cloudflared..."
    wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O cloudflared
    chmod +x cloudflared
fi

echo "🌐 Starting Cloudflare tunnel..."
nohup ./cloudflared tunnel --url http://localhost:8080 --no-autoupdate > tunnel.log 2>&1 &

echo "⏳ Waiting for tunnel URL..."
for i in {1..20}; do
    sleep 2
    if [ -f tunnel.log ]; then
        URL=$(grep -o "https://[a-z0-9-]*\.trycloudflare\.com" tunnel.log | tail -1)
        if [ -n "$URL" ]; then
            echo ""
            echo "=========================================="
            echo "🎉 TUNNEL URL: $URL"
            echo "=========================================="
            echo ""
            echo "📋 COPY THIS URL AND RUN ON ORACLE:"
            echo "   kaggle-url $URL"
            echo ""
            break
        fi
    fi
done

# Verify
echo "🧪 Testing server health..."
curl -s http://localhost:8080/health && echo " ✅ Server OK"

echo ""
echo "✅ SETUP COMPLETE!"
echo "Server and tunnel running in background."
echo "Check server.log / tunnel.log for logs."
echo "Run 'kaggle-url <URL>' on Oracle to connect opencode."