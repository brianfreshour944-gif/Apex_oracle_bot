# Modern Multi-Stage Dockerfile for Apex Oracle Bot using Astral UV
# ------------------------------------------------------------------------------
# Build Stage: Install dependencies with uv
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS builder

# Copy uv binary from official Astral image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy configuration files first for Docker layer caching
COPY pyproject.toml requirements.txt ./

# Install dependencies into virtual environment.
# We explicitly install the CPU-only version of PyTorch to prevent Docker OOM
# and save 2.5GB of useless CUDA libraries on GPU-less Oracle servers.
RUN uv venv /app/.venv && \
    uv pip install --no-cache torch --index-url https://download.pytorch.org/whl/cpu --python /app/.venv && \
    uv pip install --no-cache -r requirements.txt --python /app/.venv

# ------------------------------------------------------------------------------
# Production Runner Stage
# ------------------------------------------------------------------------------
FROM python:3.12-slim AS runner

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Create non-root user and data directory
RUN groupadd -g 10001 botuser && \
    useradd -u 10001 -g botuser -s /bin/bash -m botuser && \
    mkdir -p /app/data && \
    chown -R botuser:botuser /app

# Copy virtual environment and source code
COPY --from=builder /app/.venv /app/.venv
COPY --chown=botuser:botuser . .

USER botuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8000/health || exit 1

ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["run"]
