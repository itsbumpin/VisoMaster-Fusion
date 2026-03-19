#!/usr/bin/env python3
from __future__ import annotations

import ctypes.util
import importlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 11)
MIN_DRIVER_MAJOR = 576
INDEXES = {
    "pypi": "https://pypi.org/simple/pip/",
    "nvidia": "https://pypi.nvidia.com/simple/tensorrt-cu12/",
    "pytorch-cu129": "https://download.pytorch.org/whl/cu129/torch/",
}
QT_SYSTEM_LIBS = ["libGL.so.1", "libxkbcommon.so.0", "libEGL.so.1"]
RUNTIME_IMPORTS = [
    "PySide6",
    "pyqttoast",
    "qdarktheme",
    "send2trash",
    "pyvirtualcam",
    "tqdm",
    "requests",
    "onnxruntime",
]
OPTIONAL_IMPORTS = ["torch", "torchvision", "torchaudio", "tensorrt"]
TENSORRT_LIBS = ["nvinfer", "nvinfer_plugin"]


def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, ""
    return result.returncode, (result.stdout or result.stderr or "").strip()


def check_index(name: str, url: str) -> tuple[bool, str]:
    try:
        request = urllib.request.Request(url, method="GET", headers={"User-Agent": "VisoMaster-Linux-Preflight"})
        with urllib.request.urlopen(request, timeout=10) as response:
            ok = 200 <= response.status < 400
            return ok, f"HTTP {response.status}"
    except Exception as exc:
        return False, str(exc)


def discover_tensorrt_libs() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for lib in TENSORRT_LIBS:
        found[lib] = ctypes.util.find_library(lib)
    return found


def discover_qt_system_libs() -> dict[str, str | None]:
    return {lib: ctypes.util.find_library(lib.split('.so')[0]) for lib in QT_SYSTEM_LIBS}


def import_status(module_name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(module_name)
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def collect_gpu_info() -> dict[str, str]:
    info = {
        "gpu_name": "not-detected",
        "driver_version": "not-detected",
        "cuda_version": "not-detected",
        "nvidia_smi": shutil.which("nvidia-smi") or "missing",
        "nvcc": shutil.which("nvcc") or "missing",
    }
    code, output = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version",
        "--format=csv,noheader",
    ])
    if code == 0 and output:
        first = output.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 3:
            info["gpu_name"], info["driver_version"], info["cuda_version"] = parts[:3]
    elif code == 127:
        info["error"] = "nvidia-smi missing"
    else:
        info["error"] = output or "nvidia-smi failed"
    return info


def main() -> int:
    failures: list[str] = []
    warnings: list[str] = []

    print("[PRECHECK] VisoMaster Linux preflight")
    print(f"[PRECHECK] repo={ROOT_DIR}")
    print(f"[PRECHECK] platform={platform.platform()}")
    print(f"[PRECHECK] python={sys.version.split()[0]} executable={sys.executable}")

    if platform.system() != "Linux":
        failures.append(f"This script only supports Linux. Detected {platform.system()}.")

    if sys.version_info[:2] != EXPECTED_PYTHON:
        failures.append(
            f"Unsupported Python {sys.version.split()[0]}. Expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]}."
        )

    gpu = collect_gpu_info()
    print(f"[PRECHECK] gpu_name={gpu['gpu_name']}")
    print(f"[PRECHECK] driver_version={gpu['driver_version']}")
    print(f"[PRECHECK] cuda_version={gpu['cuda_version']}")

    if gpu["nvidia_smi"] == "missing":
        failures.append("nvidia-smi is missing; NVIDIA driver/runtime is not discoverable.")
    else:
        driver_major = gpu["driver_version"].split(".", 1)[0]
        if driver_major.isdigit() and int(driver_major) < MIN_DRIVER_MAJOR:
            failures.append(
                f"NVIDIA driver {gpu['driver_version']} is too old for the supported CUDA 12.9 flow. Need >= {MIN_DRIVER_MAJOR}."
            )
        if gpu["cuda_version"] in {"not-detected", "N/A"}:
            failures.append("CUDA runtime version was not reported by nvidia-smi.")

    print("[PRECHECK] package_indexes=")
    for name, url in INDEXES.items():
        ok, detail = check_index(name, url)
        prefix = "OK" if ok else "FAIL"
        print(f"  - {name}: {prefix} ({detail}) {url}")
        if not ok:
            failures.append(f"Required package index '{name}' is not reachable: {detail}")

    trt_libs = discover_tensorrt_libs()
    print(f"[PRECHECK] tensorrt_libs={json.dumps(trt_libs, sort_keys=True)}")
    for lib, path in trt_libs.items():
        if not path:
            warnings.append(f"TensorRT shared library '{lib}' is not currently discoverable.")

    qt_libs = discover_qt_system_libs()
    print(f"[PRECHECK] qt_system_libs={json.dumps(qt_libs, sort_keys=True)}")
    missing_qt_libs = [lib for lib, path in qt_libs.items() if not path]
    if missing_qt_libs:
        failures.append(
            "Qt runtime dependencies are missing from the host: " + ", ".join(missing_qt_libs)
        )

    print("[PRECHECK] import_smoke=")
    for module_name in RUNTIME_IMPORTS + OPTIONAL_IMPORTS:
        ok, detail = import_status(module_name)
        status = "OK" if ok else "MISS"
        print(f"  - {module_name}: {status} ({detail})")
        if module_name in RUNTIME_IMPORTS and not ok:
            warnings.append(f"Runtime import '{module_name}' is unavailable before install: {detail}")
        elif module_name == "tensorrt" and not ok:
            warnings.append(f"TensorRT Python import is unavailable before install: {detail}")

    if failures:
        print("\n[PRECHECK] FAILED")
        for item in failures:
            print(f"  - {item}")
        if warnings:
            print("[PRECHECK] Warnings observed before failure:")
            for item in warnings:
                print(f"  - {item}")
        return 1

    print("\n[PRECHECK] PASS")
    if warnings:
        print("[PRECHECK] Warnings:")
        for item in warnings:
            print(f"  - {item}")
    else:
        print("[PRECHECK] No warnings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
