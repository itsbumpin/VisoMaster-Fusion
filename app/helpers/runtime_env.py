import ctypes.util
import os
import platform
import subprocess
from typing import Any

import onnxruntime
import torch


def _run_command(command: list[str]) -> str:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        return (result.stdout or result.stderr or "").strip()
    except Exception:
        return ""


def _nvidia_smi_query(query: str) -> str:
    output = _run_command(["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"])
    if not output:
        return "unknown"
    return output.splitlines()[0].strip()


def _detect_cuda_version() -> str:
    if torch.version.cuda:
        return torch.version.cuda

    nvcc_version = _run_command(["nvcc", "--version"])
    if nvcc_version:
        for line in nvcc_version.splitlines():
            if "release" in line:
                return line.strip()
    return "unknown"


def _detect_tensorrt_libs() -> dict[str, Any]:
    required_libs = ["nvinfer", "nvinfer_plugin"]
    resolved = {lib: ctypes.util.find_library(lib) for lib in required_libs}
    missing = [lib for lib, path in resolved.items() if not path]
    return {
        "resolved": resolved,
        "missing": missing,
    }


def detect_runtime_environment() -> dict[str, Any]:
    gpu_name = "cpu"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)

    trt_libs = _detect_tensorrt_libs()
    available_providers = onnxruntime.get_available_providers()

    return {
        "os": platform.system(),
        "gpu_name": gpu_name,
        "is_blackwell": "blackwell" in gpu_name.lower() or "rtx pro 6000" in gpu_name.lower(),
        "nvidia_driver": _nvidia_smi_query("driver_version"),
        "cuda_version": _detect_cuda_version(),
        "onnxruntime_providers": available_providers,
        "tensorrt_provider_available": "TensorrtExecutionProvider" in available_providers,
        "tensorrt_python_version": _safe_get_tensorrt_version(),
        "tensorrt_libs": trt_libs,
    }


def _safe_get_tensorrt_version() -> str:
    try:
        import tensorrt as trt

        return trt.__version__
    except Exception:
        return "not-installed"


def configure_qt_environment() -> None:
    """
    Ensure Qt plugins are loaded from PySide6 on Linux and not from OpenCV's bundled Qt.
    """
    if platform.system() != "Linux":
        return

    try:
        from PySide6.QtCore import QLibraryInfo

        pyside_plugin_path = QLibraryInfo.path(QLibraryInfo.LibraryPath.PluginsPath)
    except Exception:
        return

    if pyside_plugin_path:
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(
            pyside_plugin_path, "platforms"
        )
        os.environ["QT_PLUGIN_PATH"] = pyside_plugin_path

    for env_key in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_PLUGIN_PATH"):
        current = os.environ.get(env_key, "")
        if "cv2/qt/plugins" in current:
            os.environ[env_key] = current.replace("cv2/qt/plugins", "").strip(":")


def check_plugin_library_compatibility(plugin_path: str) -> tuple[bool, str]:
    if not plugin_path or not os.path.exists(plugin_path):
        return False, f"TensorRT plugin not found: {plugin_path}"

    if platform.system() != "Linux":
        return True, "Compatibility check skipped (non-Linux platform)."

    ldd_output = _run_command(["ldd", plugin_path])
    if not ldd_output:
        return False, "Unable to inspect TensorRT plugin dependencies via ldd."

    missing_lines = [line for line in ldd_output.splitlines() if "not found" in line]
    if missing_lines:
        return False, "Missing shared libraries: " + "; ".join(missing_lines)

    return True, "Plugin shared-library dependencies look valid."


def should_use_safe_mode(runtime_env: dict[str, Any]) -> bool:
    if runtime_env["os"] != "Linux":
        return False
    if not runtime_env["is_blackwell"]:
        return False

    trt_missing = bool(runtime_env["tensorrt_libs"]["missing"])
    trt_ep_missing = not runtime_env["tensorrt_provider_available"]
    return trt_missing or trt_ep_missing


def format_runtime_warnings(runtime_env: dict[str, Any]) -> list[str]:
    warnings = []
    if runtime_env["nvidia_driver"] == "unknown":
        warnings.append("NVIDIA driver version could not be detected via nvidia-smi.")
    if runtime_env["cuda_version"] == "unknown":
        warnings.append("CUDA version could not be detected.")
    if runtime_env["tensorrt_libs"]["missing"]:
        warnings.append(
            "Missing TensorRT runtime libraries: "
            + ", ".join(runtime_env["tensorrt_libs"]["missing"])
        )
    if not runtime_env["tensorrt_provider_available"]:
        warnings.append("ONNX Runtime TensorRT execution provider is not available.")
    return warnings


def print_runtime_diagnostics(runtime_env: dict[str, Any]) -> None:
    print("[ENV] Runtime environment diagnostics:")
    print(f"[ENV]   OS: {runtime_env['os']}")
    print(f"[ENV]   GPU: {runtime_env['gpu_name']}")
    print(f"[ENV]   NVIDIA Driver: {runtime_env['nvidia_driver']}")
    print(f"[ENV]   CUDA Version: {runtime_env['cuda_version']}")
    print(f"[ENV]   TensorRT Python: {runtime_env['tensorrt_python_version']}")
    print(f"[ENV]   ONNX Providers: {runtime_env['onnxruntime_providers']}")
    print(f"[ENV]   TensorRT libs: {runtime_env['tensorrt_libs']['resolved']}")

    for warning in format_runtime_warnings(runtime_env):
        print(f"[WARN] {warning}")
