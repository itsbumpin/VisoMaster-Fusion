#!/usr/bin/env bash
set -u -o pipefail

echo "[ENV] Linux GPU environment validation"
echo "[ENV] Host: $(uname -a)"

echo "[ENV] NVIDIA GPU"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
  echo "[WARN] nvidia-smi not found"
fi

if ! command -v python >/dev/null 2>&1; then
  echo "[WARN] python not found in PATH"
  exit 0
fi

echo "[ENV] CUDA"
python - <<'PY' || true
import torch
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.version.cuda={torch.version.cuda}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY

echo "[ENV] ONNX Runtime providers"
python - <<'PY' || true
import onnxruntime
print(onnxruntime.get_available_providers())
PY

echo "[ENV] TensorRT Python + shared libraries"
python - <<'PY' || true
import ctypes.util
try:
    import tensorrt as trt
    print(f"tensorrt={trt.__version__}")
except Exception as e:
    print(f"[WARN] TensorRT Python import failed: {e}")
for lib in ("nvinfer", "nvinfer_plugin"):
    print(f"{lib} => {ctypes.util.find_library(lib)}")
PY

PLUGIN_PATH="model_assets/libgrid_sample_3d_plugin.so"
if [[ -f "$PLUGIN_PATH" ]]; then
  echo "[ENV] Inspecting custom TensorRT plugin dependencies: $PLUGIN_PATH"
  ldd "$PLUGIN_PATH" || true
fi
