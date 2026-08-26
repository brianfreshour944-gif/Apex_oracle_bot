#!/bin/bash
# tunnel_server.sh — exposes the llama-server (port 8080) via a free
# Cloudflare quick-tunnel. No account needed.
#
# Run in a Kaggle notebook cell AFTER start_llama_server.sh is listening:
#   !bash tunnel_server.sh
#
# It prints a https://xxxx.trycloudflare.com URL — that's your API endpoint.
# Give it an API key requirement? The URL is unguessable but public;
# treat anything sent through it as non-sensitive (code, no credentials).

set -e

if ! curl -s http://localhost:8080/health > /dev/null 2>&1; then
  echo "ERROR: llama-server is not running on port 8080."
  echo "Start it first: bash start_llama_server.sh"
  exit 1
fi

if [ ! -x ./cloudflared ]; then
  echo "Downloading cloudflared..."
  curl -sL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64" -o cloudflared
  chmod +x cloudflared
fi

echo "Starting tunnel... (your URL appears below in 'trycloudflare.com' lines)"
./cloudflared tunnel --url http://localhost:8080 --no-autoupdate 2>&1 | \
  grep --line-buffered -oE "https://[a-z0-9-]+\.trycloudflare\.com" | while read url; do
    echo ""
    echo "=================================================="
    echo "YOUR LLM ENDPOINT: $url"
    echo "Test from anywhere:"
    echo "  curl $url/health"
    echo "=================================================="
    break
done
