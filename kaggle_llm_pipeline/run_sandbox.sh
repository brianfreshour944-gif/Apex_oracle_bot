#!/bin/bash
# run_sandbox.sh
#
# Usage: ./run_sandbox.sh /path/to/patch.diff
#
# Applies a validated patch to an ISOLATED clone of the repo (git clone,
# NOT cp -r — so .env, data/, models/, venv never enter the sandbox),
# runs tests with live credentials stripped, pushes a branch on success.

set -e

PATCH_FILE="$1"
REPO_DIR="$(pwd)"
MODEL_NAME="qwen32b"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
BRANCH_NAME="fix/agent-${TIMESTAMP}-${MODEL_NAME}"
SANDBOX_DIR="/tmp/sandbox_${TIMESTAMP}"

if [ -z "$PATCH_FILE" ] || [ ! -f "$PATCH_FILE" ]; then
  echo "Usage: ./run_sandbox.sh /path/to/patch.diff"
  exit 1
fi

echo "== Creating isolated sandbox at $SANDBOX_DIR =="
git clone --quiet "$REPO_DIR" "$SANDBOX_DIR"
cd "$SANDBOX_DIR"

echo "== Applying patch =="
if ! git apply --check "$REPO_DIR/$PATCH_FILE" 2>/dev/null && \
   ! git apply --check "$PATCH_FILE" 2>/dev/null; then
  echo "Patch does not apply cleanly. Aborting — no changes made."
  exit 1
fi
git apply "$REPO_DIR/$PATCH_FILE" 2>/dev/null || git apply "$PATCH_FILE"

echo "== Credential scrubbing =="
rm -f .env .env.*
unset ALPACA_API_KEY ALPACA_SECRET_KEY ALPACA_API_SECRET \
      APCA_API_KEY_ID APCA_API_SECRET_KEY \
      BINANCE_API_KEY BINANCE_SECRET KAGGLE_API_TOKEN OPENROUTER_API_KEY 2>/dev/null || true
export TRADING_ENV="testnet"

echo "== Running tests with repo venv (excluding live-marked tests) =="
VENV_PY="$REPO_DIR/venv/bin/python"
[ -x "$VENV_PY" ] || VENV_PY=python3

if PYTHONPATH="$SANDBOX_DIR" pytest tests/ -m "not live" --tb=short -q; then
  echo "== Tests passed. Committing branch $BRANCH_NAME =="
  git checkout -b "$BRANCH_NAME"
  git add -A
  git -c user.name="agent-qwen32b" -c user.email="agent@localhost" \
      commit -m "Agent fix: ${TIMESTAMP} (${MODEL_NAME})"
  git push origin "$BRANCH_NAME"
  echo ""
  echo "SUCCESS: Branch '$BRANCH_NAME' pushed."
  echo "Review the PR manually before merging. main was NOT touched."
else
  rc=$?
  echo ""
  echo "FAILURE: tests did not pass (exit $rc). Sandbox kept at $SANDBOX_DIR."
  echo "Feed pytest output back to the model ONCE. If it fails again, stop"
  echo "and review manually. Do not loop retries."
  exit $rc
fi
