#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

on_error() {
  local exit_code=$?
  local line_no=${1:-unknown}
  echo "[ERROR] install_runpod.sh failed at line ${line_no} with exit code ${exit_code}." >&2
  echo "[ERROR] Resolve the issue above, then re-run this script." >&2
  exit "$exit_code"
}
trap 'on_error $LINENO' ERR

log() {
  echo "[INFO] $*"
}

log "Repository root: $ROOT_DIR"
log "Using python binary: $PYTHON_BIN"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "[ERROR] Python executable '$PYTHON_BIN' was not found in PATH." >&2
  exit 1
fi

if [[ ! -d "$VENV_DIR" ]]; then
  log "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
else
  log "Virtual environment already exists at $VENV_DIR"
fi

# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

log "Upgrading pip/setuptools/wheel"
python -m pip install --upgrade pip setuptools wheel

log "Installing torch/torchvision/torchaudio for CUDA 12.9"
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu129 \
  --extra-index-url https://pypi.org/simple \
  torch==2.8.0+cu129 \
  torchvision==0.23.0+cu129 \
  torchaudio==2.8.0+cu129

log "Installing CUDA/TensorRT support packages"
python -m pip install \
  --extra-index-url https://pypi.nvidia.com \
  -r "$ROOT_DIR/requirements-cuda-cu129.txt"

log "Installing common runtime/UI dependencies"
python -m pip install -r "$ROOT_DIR/requirements-base.txt"

log "Running runtime import smoke test"
python "$ROOT_DIR/scripts/check_runtime_imports.py"

log "Runpod installation completed successfully."
log "Activate later with: source $VENV_DIR/bin/activate"
