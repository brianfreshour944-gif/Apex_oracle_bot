# syntax=docker/dockerfile:1
# Apex Oracle Bot - single-stage image, deps installed with uv.
# ------------------------------------------------------------------------------
# Why single-stage: the old multi-stage layout copied the ~2.5GB .venv between
# two identical python:3.12-slim stages, which took ~14 min on the deploy
# server's slow disk (#15 COPY --from=builder: 832s), and Coolify always builds
# with --no-cache, so nothing was ever reused anyway.
#
# Speed notes:
# - The RUN below uses a BuildKit CACHE MOUNT for uv's download cache. Cache
#   mounts survive `docker build --no-cache` (they are not layers), so repeat
#   builds skip re-downloading ~500MB of wheels; only install/compile re-runs.
# - No curl/apt-get: the healthcheck uses Python's stdlib urllib, removing the
#   apt-get layer that cost ~8 minutes per build on the deploy server.
# ------------------------------------------------------------------------------
FROM python:3.12-slim

# Copy uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Create non-root user and data directory
RUN groupadd -g 10001 botuser && \
    useradd -u 10001 -g botuser -s /bin/bash -m botuser && \
    mkdir -p /app/data && \
    chown -R botuser:botuser /app

# Copy dependency manifests first so this layer stays cacheable
COPY pyproject.toml requirements.txt ./

# Two-step install: torch first, scoped ONLY to the PyTorch CPU index (no
# --extra-index-url here), so uv can't get confused about which index "owns"
# the torch package. Then requirements.txt against default PyPI. Combining
# these into one call with both indexes made uv treat PyPI as authoritative
# for torch and fail to find the +cpu build - see incident 2026-09-03.
# The venv stays root-owned like before; the runtime user only needs
# read+execute.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv /app/.venv && \
    uv pip install --index-url https://download.pytorch.org/whl/cpu "torch==2.14.0+cpu" --python /app/.venv && \
    uv pip install -r requirements.txt --python /app/.venv

# Copy source code (venv stays root-owned/read-only for botuser)
COPY --chown=botuser:botuser . .

USER botuser

EXPOSE 8000

# Healthcheck via stdlib urllib (no curl in the image). start_period gives
# torch import + model loading time to finish before checks count.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["run"]
