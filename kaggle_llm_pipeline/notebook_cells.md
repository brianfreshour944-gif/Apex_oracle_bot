# Kaggle LLM — Session Cheat Sheet

Qwen3-Coder-30B on Kaggle 2× T4, served via llama-server + cloudflare tunnel.
Full pipeline lives in this folder. Total setup per session: ~15 min.

## Session start (Kaggle notebook)

1. kaggle.com → Code → **New Notebook**
2. Right panel: **Accelerator → GPU T4 x2**, **Internet → ON**

### Cell 1 — clone repo
```
!git clone https://github.com/brianfreshour944-gif/Apex_oracle_bot.git
%cd Apex_oracle_bot/kaggle_llm_pipeline
```

### Cell 2 — build CUDA llama-server (~12 min, one-time per session)
```
%%bash
MODEL=/kaggle/working/models/qwen3-coder-32b-q5_k_m.gguf

# download model (21GB, ~5-10 min; skipped if cached)
if [ ! -f "$MODEL" ]; then
  pip install -q -U huggingface_hub
  export MODEL_PATH="$MODEL"
  python3 - <<'PYEOF'
from huggingface_hub import hf_hub_download
import os, shutil
path = hf_hub_download(repo_id="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF",
                       filename="Qwen3-Coder-30B-A3B-Instruct-Q5_K_M.gguf")
os.makedirs(os.path.dirname(os.environ["MODEL_PATH"]), exist_ok=True)
shutil.copy(path, os.environ["MODEL_PATH"])
PYEOF
fi

# build llama.cpp with CUDA (skipped if binary exists)
if [ ! -x ./llama-cuda ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git cuda_src
  ln -sf /usr/local/nvidia/lib64/libcuda.so /usr/local/cuda/lib64/libcuda.so
  ln -sf /usr/local/nvidia/lib64/libcuda.so.1 /usr/local/cuda/lib64/libcuda.so.1
  cmake -S cuda_src -B cuda_build -DGGML_CUDA=ON -DLLAMA_CURL=OFF \
        -DCUDAToolkit_ROOT=/usr/local/cuda > build.log 2>&1 &&
  cmake --build cuda_build --config Release -j4 --target llama-server >> build.log 2>&1 &&
  cp cuda_build/bin/llama-server ./llama-cuda && echo BUILD_OK || tail -20 build.log
fi
echo "ready"
```

### Cell 3 — start server + tunnel
```
%%bash
cd /kaggle/working/Apex_oracle_bot/kaggle_llm_pipeline
nohup ./llama-cuda \
  -m /kaggle/working/models/qwen3-coder-32b-q5_k_m.gguf \
  --split-mode layer --tensor-split 1,1 -ngl 999 \
  -c 32768 --cache-type-k q8_0 --cache-type-v q8_0 --jinja \
  --host 0.0.0.0 --port 8080 > server.log 2>&1 &
sleep 3
nohup ./cloudflared tunnel --url http://localhost:8080 --no-autoupdate > tunnel.log 2>&1 &
sleep 10
grep -o "https://[a-z0-9-]*\.trycloudflare\.com" tunnel.log | tail -1
```
Model load takes ~1–2 min. Verify: `!curl -s localhost:8080/health`

## On Oracle (after tunnel prints its URL)

```
kaggle-url https://<the-url>.trycloudflare.com
```

Then in opencode (or JetBrains → opencode agent): model `kaggle/qwen3-coder-30b` is the default.

## Session end

Interrupt cells and **stop the Kaggle session** — quota bills by wall-clock time,
even idle. Everything on Oracle side persists; nothing to save.

## Troubleshooting

| Symptom | Fix |
|---|---|
| opencode errors / no response | Tunnel died: re-run Cell 3's tunnel part, `kaggle-url <new-url>` |
| 502 from trycloudflare | Server dead: re-run Cell 3 |
| Responses slow (>30s) | Check it's the CUDA binary: `!head -3 server.log` should show CUDA devices, not llvmpipe |
| Disk full on /kaggle/working | `!rm -rf cuda_src cuda_build llama.zip` (keeps binary + model) |
| Fresh session = everything gone | Normal — re-run Cells 1–3 |

## Gotchas learned the hard way

- Kaggle wipes ALL files between sessions (model + binary re-download/rebuild each time)
- Kaggle kills idle/background processes after ~30-60 min of browser inactivity — keep tab open
- Cloudflare quick-tunnels have a 100s request timeout; use streaming for long generations
- Don't use the Vulkan prebuilt binary — silently falls back to CPU (7 tok/s)
- The `libcuda.so` symlink into `/usr/local/cuda/lib64/` is REQUIRED for the CUDA build
- Old llama.cpp versions hang on `tools` requests; b7200+ (current build) is fine
