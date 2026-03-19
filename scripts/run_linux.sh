#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${VISOMASTER_ENV_NAME:-visomaster}"

log() { echo "[run_linux] $*"; }
fail() { echo "[run_linux][ERROR] $*" >&2; exit 1; }

load_conda() {
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    return 0
  fi
  return 1
}

activate_env() {
  load_conda || fail "conda was not found. Run scripts/setup_linux.sh first."
  conda activate "$ENV_NAME" || fail "Conda env '$ENV_NAME' does not exist. Run scripts/setup_linux.sh first."
}

export_runtime_paths() {
  local python_site
  python_site="$(python - <<'PY'
import site
paths = [p for p in site.getsitepackages() if 'site-packages' in p]
print(paths[0] if paths else '')
PY
)"
  [[ -n "$python_site" ]] || fail "Could not determine site-packages path."

  local extra_ld_paths=""
  while IFS= read -r line; do
    [[ -n "$line" ]] || continue
    extra_ld_paths+="$line:"
  done < <(python - <<'PY'
import site
from pathlib import Path
roots = [Path(p) for p in site.getsitepackages()]
patterns = [
    'tensorrt_libs',
    'nvidia/cudnn/lib',
    'nvidia/cublas/lib',
    'nvidia/cuda_runtime/lib',
    'nvidia/cusolver/lib',
    'nvidia/cusparse/lib',
    'nvidia/cufft/lib',
    'nvidia/curand/lib',
]
seen = set()
for root in roots:
    for pattern in patterns:
        path = root / pattern
        if path.exists() and str(path) not in seen:
            seen.add(str(path))
            print(path)
PY
)

  if [[ -n "$extra_ld_paths" ]]; then
    export LD_LIBRARY_PATH="${extra_ld_paths}${LD_LIBRARY_PATH:-}"
  fi

  export QT_QPA_PLATFORM_PLUGIN_PATH="$(python - <<'PY'
from PySide6.QtCore import QLibraryInfo
print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath) + '/platforms')
PY
)"
  export QT_PLUGIN_PATH="$(python - <<'PY'
from PySide6.QtCore import QLibraryInfo
print(QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath))
PY
)"
}

print_runtime_mode() {
  python - <<'PY'
import ctypes.util
import onnxruntime as ort
import torch
mode = 'CPU fallback'
if torch.cuda.is_available():
    mode = 'CUDA fallback'
if ctypes.util.find_library('nvinfer') and ctypes.util.find_library('nvinfer_plugin') and 'TensorrtExecutionProvider' in ort.get_available_providers():
    mode = 'TensorRT'
print(f"[run_linux] execution_mode={mode}")
print(f"[run_linux] onnx_providers={ort.get_available_providers()}")
print(f"[run_linux] torch_cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[run_linux] gpu={torch.cuda.get_device_name(0)}")
PY
}

main() {
  cd "$ROOT_DIR"
  activate_env
  export_runtime_paths
  print_runtime_mode
  exec python "$ROOT_DIR/main.py" "$@"
}

main "$@"
