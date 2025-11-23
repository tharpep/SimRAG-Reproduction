# Multi-stage Dockerfile for SimRAG Reproduction
# Supports both GPU (CUDA) and CPU execution

# Stage 1: Base image with CUDA support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
# Note: Python 3.12 is not in Ubuntu 22.04 default repos, so we use deadsnakes PPA
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa -y \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# Create symlink for python and install pip for Python 3.12
RUN ln -s /usr/bin/python3.12 /usr/bin/python \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12

# Install Poetry
RUN pip install --no-cache-dir poetry==1.8.3

# Configure Poetry
ENV POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies (without dev dependencies for smaller image)
# Note: Poetry will use the PyTorch CUDA source defined in pyproject.toml
RUN poetry install --no-dev --no-interaction --no-ansi && \
    rm -rf $POETRY_CACHE_DIR

# Stage 2: Final image
FROM base AS final

# Copy application code
COPY . .

# Install the package in development mode
RUN poetry install --no-dev --no-interaction --no-ansi

# Create directories for data persistence
RUN mkdir -p /app/data/qdrant_db \
    /app/tuned_models \
    /app/logs/rag \
    /app/logs/tuning \
    /app/comparison_results

# Set permissions
RUN chmod -R 755 /app

# Default command
CMD ["poetry", "run", "simrag", "--help"]

