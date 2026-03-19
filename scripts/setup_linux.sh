#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VISOMASTER_ENV_NAME:-visomaster}"
CONDA_ROOT="${CONDA_EXE:-}"
PYTHON_VERSION="3.11"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu129"
PYPI_INDEX_URL="https://pypi.org/simple"
NVIDIA_INDEX_URL="https://pypi.nvidia.com"
CUDA_REQUIREMENTS_FILE="$ROOT_DIR/requirements-cuda-cu129.txt"
BASE_REQUIREMENTS_FILE="$ROOT_DIR/requirements-base.txt"
AGG_REQUIREMENTS_FILE="$ROOT_DIR/requirements_cu129.txt"
PRECHECK_SCRIPT="$ROOT_DIR/scripts/preflight_linux.py"
IMPORT_CHECK_SCRIPT="$ROOT_DIR/scripts/check_runtime_imports.py"
SUMMARY_WARNINGS=()
TENSORRT_STATUS="not-checked"
CUDA_STATUS="not-checked"

log() { echo "[setup_linux] $*"; }
warn() { echo "[setup_linux][WARN] $*"; SUMMARY_WARNINGS+=("$*"); }
fail() { echo "[setup_linux][ERROR] $*" >&2; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || fail "Required command '$1' was not found in PATH."
}

host_python_can_run_preflight() {
  python3 - <<'PY' >/dev/null 2>&1
import sys
raise SystemExit(0 if sys.version_info >= (3, 9) else 1)
PY
}

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return 0
  fi
  if [[ -n "$CONDA_ROOT" && -f "$(dirname "$CONDA_ROOT")/../etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1090
    source "$(dirname "$CONDA_ROOT")/../etc/profile.d/conda.sh"
    return 0
  fi
  return 1
}

activate_env() {
  load_conda || fail "conda is required for the Linux bootstrap flow. Install Miniconda/Anaconda first."

  if conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
    log "Reusing conda env '$ENV_NAME'"
  else
    log "Creating conda env '$ENV_NAME' with Python $PYTHON_VERSION"
    conda create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
  fi

  conda activate "$ENV_NAME"
  log "Activated env: $CONDA_DEFAULT_ENV"
}

print_host_summary() {
  log "Host OS: $(uname -srvmo)"
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "GPU / driver / CUDA:"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader || true
  else
    warn "nvidia-smi is missing on PATH; continuing because some pods expose GPU tooling late."
  fi
  if command -v nvcc >/dev/null 2>&1; then
    nvcc --version | tail -n 1 || true
  else
    warn "nvcc is not installed or not on PATH; relying on driver-reported CUDA runtime only."
  fi
}

run_host_preflight() {
  if host_python_can_run_preflight; then
    log "Running non-fatal host preflight (uses host python only for warnings)."
    python3 "$PRECHECK_SCRIPT" --stage host --before-install || true
  else
    warn "Host python3 is too old to run the preflight helper; continuing to the Python 3.11 conda env creation step."
  fi
}

run_env_preflight() {
  log "Running env preflight using the activated Python 3.11 environment."
  python "$PRECHECK_SCRIPT" --stage env --before-install --strict
}

patch_cudnn_pin_if_needed() {
  local target_file="$CUDA_REQUIREMENTS_FILE"
  if [[ ! -f "$target_file" ]]; then
    fail "Missing requirements file: $target_file"
  fi

  if grep -Eq '^nvidia-cudnn-cu12==9\.10\.2\.21$' "$target_file"; then
    log "Replacing strict cuDNN pin with a compatible range to avoid Linux resolver conflicts."
    sed -i 's/^nvidia-cudnn-cu12==9\.10\.2\.21$/nvidia-cudnn-cu12>=9.10,<10/' "$target_file"
  fi
}

pip_install() {
  python -m pip install --upgrade pip setuptools wheel
  python -m pip config set global.index-url "$PYPI_INDEX_URL" >/dev/null
  python -m pip config set global.extra-index-url "$NVIDIA_INDEX_URL $TORCH_INDEX_URL" >/dev/null
}

install_base() {
  log "Installing common runtime/UI dependencies first."
  python -m pip install --index-url "$PYPI_INDEX_URL" -r "$BASE_REQUIREMENTS_FILE"
}

install_torch() {
  log "Installing torch/torchvision/torchaudio from the CUDA 12.9 wheel index."
  python -m pip install \
    --index-url "$TORCH_INDEX_URL" \
    --extra-index-url "$PYPI_INDEX_URL" \
    torch==2.8.0+cu129 \
    torchvision==0.23.0+cu129 \
    torchaudio==2.8.0+cu129
}

install_cuda_stack() {
  log "Installing CUDA/cuDNN/TensorRT runtime packages."
  if ! python -m pip install \
    --index-url "$PYPI_INDEX_URL" \
    --extra-index-url "$NVIDIA_INDEX_URL" \
    -r "$CUDA_REQUIREMENTS_FILE"; then
    warn "Initial CUDA/TensorRT install failed; retrying with a relaxed cuDNN pin."
    python -m pip install --index-url "$PYPI_INDEX_URL" --extra-index-url "$NVIDIA_INDEX_URL" 'nvidia-cudnn-cu12>=9.10,<10'
    python -m pip install \
      --index-url "$PYPI_INDEX_URL" \
      --extra-index-url "$NVIDIA_INDEX_URL" \
      --no-deps \
      tensorrt-cu12==10.9.0.34 \
      tensorrt-cu12_libs==10.9.0.34 \
      tensorrt-cu12_bindings==10.9.0.34 \
      cuda-toolkit==12.9.1
  fi
}

install_runtime_extras() {
  log "Ensuring explicit runtime/UI packages are present."
  python -m pip install --index-url "$PYPI_INDEX_URL" \
    PySide6 pyqtdarktheme pyqt-toast-notification send2trash pyvirtualcam tqdm qdarkstyle
}

run_post_install_checks() {
  log "Running import smoke test."
  python "$IMPORT_CHECK_SCRIPT"

  if python - <<'PY'
import ctypes.util
import tensorrt  # noqa: F401
assert ctypes.util.find_library('nvinfer')
assert ctypes.util.find_library('nvinfer_plugin')
PY
  then
    TENSORRT_STATUS="available"
  else
    TENSORRT_STATUS="missing-runtime-libs"
    warn "TensorRT Python package or shared libraries are still not discoverable; app will use CUDA fallback if available."
  fi

  if python - <<'PY'
import torch
raise SystemExit(0 if torch.cuda.is_available() else 1)
PY
  then
    CUDA_STATUS="available"
  else
    CUDA_STATUS="unavailable"
    warn "torch.cuda.is_available() is false after install."
  fi
}

final_summary() {
  echo
  log "Final summary"
  log "  repo: $ROOT_DIR"
  log "  conda env: $ENV_NAME"
  log "  aggregate requirements file kept for compatibility: $AGG_REQUIREMENTS_FILE"
  log "  CUDA status: $CUDA_STATUS"
  log "  TensorRT status: $TENSORRT_STATUS"
  if ((${#SUMMARY_WARNINGS[@]})); then
    log "  warnings:"
    printf '    - %s\n' "${SUMMARY_WARNINGS[@]}"
  else
    log "  warnings: none"
  fi
  log "Next steps:"
  log "  1. bash scripts/run_linux.sh"
}

main() {
  require_cmd python3
  require_cmd bash
  print_host_summary
  run_host_preflight

  activate_env
  python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
  run_env_preflight
  patch_cudnn_pin_if_needed
  pip_install
  install_base
  install_torch
  install_cuda_stack
  install_runtime_extras

  log "Downloading required models."
  python "$ROOT_DIR/download_models.py"

  run_post_install_checks
  final_summary
}

main "$@"
