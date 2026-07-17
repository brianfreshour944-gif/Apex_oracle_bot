# Modern Dockerfile for Apex Oracle Bot using Python 3.12

# Use official Python 3.12 slim image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV PIP_NO_CACHE_DIR=off
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_DEFAULT_TIMEOUT=100
ENV POETRY_VERSION=1.7.1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv for modern dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Create and set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY requirements.txt ./
COPY src/ ./src/
COPY test_*.py ./

# Install dependencies using uv
RUN uv pip install --system --no-cache-dir -r requirements.txt

# Create data directory
RUN mkdir -p /app/data

# Expose the API port
EXPOSE 8080

# Set the default command
CMD ["python", "src/main.py"]