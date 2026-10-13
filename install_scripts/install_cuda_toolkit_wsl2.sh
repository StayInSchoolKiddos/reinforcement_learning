#!/usr/bin/env bash
# Configure NVIDIA GPU passthrough for Docker running under WSL2 Ubuntu/Debian.
# Does not install Linux display drivers or a CUDA toolkit.
set -Eeuo pipefail

CUDA_IMAGE="${CUDA_IMAGE:-nvidia/cuda:12.8.0-base-ubuntu22.04}"
log()  { printf '\n[INFO] %s\n' "$*"; }
warn() { printf '\n[WARN] %s\n' "$*" >&2; }
die()  { printf '\n[ERROR] %s\n' "$*" >&2; exit 1; }

command -v docker >/dev/null 2>&1 || die "Docker CLI not installed. Install/configure Docker first."
grep -qi microsoft /proc/sys/kernel/osrelease || die "This script is intended for WSL2."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi || die "NVIDIA GPU not accessible in WSL2. Check your Windows driver and WSL installation."
elif [[ -x /usr/lib/wsl/lib/nvidia-smi ]]; then
  /usr/lib/wsl/lib/nvidia-smi || die "WSL2 GPU query failed."
else
  die "No WSL2 nvidia-smi found; verify Windows GPU drivers and WSL2 first."
fi

docker info >/dev/null 2>&1 || die "Docker daemon not reachable. Start Docker Desktop or the Linux Docker service."

# Avoid changing the wrong Docker daemon (remote hosts / alternate endpoints).
context=$(docker context show)
endpoint=$(docker context inspect "$context" --format '{{ (index .Endpoints "docker").Host }}' 2>/dev/null || true)
os=$(docker info --format '{{.OperatingSystem}}')
log "Docker context: $context"
log "Docker endpoint: $endpoint"
log "Docker OS: $os"

case "$endpoint" in
  unix://*|"") ;;
  *) die "Docker endpoint is not a local Unix socket. Refusing to modify local daemon settings." ;;
esac

# Docker Desktop manages GPU passthrough itself, so never change its daemon config.
if [[ "$os" == *"Docker Desktop"* || "$context" == "desktop-linux" || "$endpoint" == *"docker-desktop"* ]]; then
  log "Docker Desktop detected. No NVIDIA Container Toolkit installation in WSL is needed."
  log "Make sure Docker Desktop uses the WSL2 backend and WSL integration is enabled."
else
  log "Native Linux Docker daemon detected."
  [[ -f /etc/os-release ]] || die "Cannot identify Linux distribution."
  . /etc/os-release
  case "$ID" in
    ubuntu|debian) ;;
    *) die "Only Ubuntu/Debian are supported by this installer (found $ID)." ;;
  esac
  [[ "$endpoint" == "unix:///var/run/docker.sock" || "$endpoint" == "unix:///run/docker.sock" ]] || die "Nonstandard Docker socket; refusing to configure possibly rootless/custom daemon."
  command -v sudo >/dev/null 2>&1 || die "sudo is required for native Docker setup."
  sudo -v
  log "Installing repository prerequisites..."
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends ca-certificates curl gnupg2

  log "Adding official NVIDIA Container Toolkit APT repository..."
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  repo=$(mktemp)
  trap 'rm -f "$repo"' EXIT
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' > "$repo"
  [[ -s "$repo" ]] || die "NVIDIA repository list was empty."
  sudo install -m 0644 "$repo" /etc/apt/sources.list.d/nvidia-container-toolkit.list

  log "Installing NVIDIA Container Toolkit..."
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit

  log "Configuring native Docker's NVIDIA runtime (backs up daemon.json first)..."
  if [[ -f /etc/docker/daemon.json ]]; then
    backup="/etc/docker/daemon.json.backup.$(date +%Y%m%d%H%M%S)"
    sudo cp -a /etc/docker/daemon.json "$backup"
    log "Backup: $backup"
  fi
  sudo nvidia-ctk runtime configure --runtime=docker

  if command -v systemctl >/dev/null 2>&1 && [[ "$(ps -p 1 -o comm=)" == "systemd" ]]; then
    sudo systemctl restart docker
  elif command -v service >/dev/null 2>&1; then
    sudo service docker restart
  else
    die "Toolkit installed, but cannot restart Docker automatically. Restart your native Docker daemon and rerun this script."
  fi
fi

log "Testing Docker GPU passthrough with $CUDA_IMAGE ..."
if docker run --rm --gpus all "$CUDA_IMAGE" nvidia-smi; then
  log "SUCCESS: NVIDIA GPU is available inside Docker."
  log "Your Compose GPU override can now use: gpus: all"
else
  die "Docker GPU test failed. Read the error above. If using Docker Desktop, update WSL with 'wsl --update' in Windows PowerShell and check Docker Desktop WSL2 settings."
fi
