
#!/usr/bin/env bash
set -euo pipefail

BASE_COMPOSE="docker-compose.yaml"
GPU_COMPOSE="compose.gpu.yaml"

echo "Checking NVIDIA GPU availability..."

if docker run --rm --gpus all \
    nvidia/cuda:12.8.0-base-ubuntu22.04 \
    nvidia-smi >/dev/null 2>&1; then

    echo "NVIDIA GPU detected."
    echo "Starting GPU-enabled containers."

    docker compose \
        -f "$BASE_COMPOSE" \
        -f "$GPU_COMPOSE" \
        up -d --build --force-recreate

else
    echo "No Docker-accessible NVIDIA GPU detected."
    echo "Starting CPU-compatible containers."

    docker compose \
        -f "$BASE_COMPOSE" \
        up -d --build --force-recreate
fi

echo "Environment ready."
echo "Run: docker compose exec rllib bash"
