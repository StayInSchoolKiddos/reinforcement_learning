# syntax=docker/dockerfile:1
# Ray RLlib development image. Dependencies are managed by uv (pyproject.toml + uv.lock).
FROM python:3.12-slim

ARG USERNAME=dev
ARG UID=1000
ARG GID=1000

COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/local/bin/

# The venv lives outside /workspace so the project bind mount doesn't hide it.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_CACHE_DIR=/home/${USERNAME}/.cache/uv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/workspace/src \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    RAY_RESULTS_DIR=/workspace/results

# Build tools (box2d/swig), ffmpeg (episode videos), GL libs (env rendering).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential swig git curl ffmpeg \
        libgl1 libglib2.0-0 libosmesa6 libglfw3 \
    && rm -rf /var/lib/apt/lists/*

# Non-root user matching the host UID/GID so files written to mounted volumes are owned by you.
RUN groupadd --gid ${GID} ${USERNAME} \
    && useradd --uid ${UID} --gid ${GID} --create-home --shell /bin/bash ${USERNAME} \
    && mkdir -p /opt/venv /workspace /home/${USERNAME}/.cache/uv \
    && chown -R ${UID}:${GID} /opt/venv /workspace /home/${USERNAME}/.cache

USER ${USERNAME}
WORKDIR /workspace

# Only the manifests are copied, so editing code never invalidates this (large) layer.
RUN --mount=type=cache,target=/home/dev/.cache/uv,uid=${UID},gid=${GID} \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    uv sync --frozen --no-install-project

# 8265 = Ray dashboard, 8888 = JupyterLab, 6006 = TensorBoard
EXPOSE 8265 8888 6006

CMD ["bash"]
