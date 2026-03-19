#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes.util
import importlib
import json
import platform
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_PYTHON = (3, 11)
MIN_DRIVER_MAJOR = 576
INDEXES = {
    "pypi": {
        "url": "https://pypi.org/simple/pip/",
        "required": True,
    },
    "nvidia": {
        "url": "https://pypi.nvidia.com/simple/",
        "required": False,
    },
    "pytorch-cu129": {
        "url": "https://download.pytorch.org/whl/cu129/torch/",
        "required": True,
    },
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


class Report:
    def __init__(self) -> None:
        self.blockers: list[str] = []
        self.warnings: list[str] = []
        self.notes: list[str] = []

    def blocker(self, message: str) -> None:
        self.blockers.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, ""
    return result.returncode, (result.stdout or result.stderr or "").strip()


def check_index(name: str, url: str) -> tuple[bool, int | None, str]:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "VisoMaster-Linux-Preflight"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return True, response.status, f"HTTP {response.status}"
    except urllib.error.HTTPError as exc:
        if name == "nvidia" and exc.code == 404:
            return True, exc.code, "HTTP 404 from NVIDIA simple index treated as non-fatal"
        return False, exc.code, f"HTTP {exc.code}"
    except Exception as exc:
        return False, None, str(exc)


def discover_tensorrt_libs() -> dict[str, str | None]:
    return {lib: ctypes.util.find_library(lib) for lib in TENSORRT_LIBS}


def discover_qt_system_libs() -> dict[str, str | None]:
    return {lib: ctypes.util.find_library(lib.split(".so")[0]) for lib in QT_SYSTEM_LIBS}


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

    query_attempts = [
        ["nvidia-smi", "--query-gpu=name,driver_version,cuda_version", "--format=csv,noheader"],
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
    ]
    for command in query_attempts:
        code, output = run(command)
        if code == 0 and output:
            parts = [part.strip() for part in output.splitlines()[0].split(",")]
            if len(parts) >= 1 and parts[0]:
                info["gpu_name"] = parts[0]
            if len(parts) >= 2 and parts[1]:
                info["driver_version"] = parts[1]
            if len(parts) >= 3 and parts[2]:
                info["cuda_version"] = parts[2]
            break

    code, output = run(["nvidia-smi", "-L"])
    if code == 0 and output and info["gpu_name"] == "not-detected":
        info["gpu_name"] = output.splitlines()[0].split("(", 1)[0].replace("GPU 0:", "").strip()

    if info["driver_version"] == "not-detected":
        driver_version_file = Path("/proc/driver/nvidia/version")
        if driver_version_file.exists():
            text = driver_version_file.read_text(encoding="utf-8", errors="ignore")
            marker = "Kernel Module"
            if marker in text:
                info["driver_version"] = text.split(marker, 1)[1].strip().split()[0]

    if info["cuda_version"] == "not-detected":
        code, output = run(["nvidia-smi"])
        if code == 0 and output:
            for line in output.splitlines():
                if "CUDA Version:" in line:
                    info["cuda_version"] = line.split("CUDA Version:", 1)[1].split()[0].strip()
                    break

    if info["gpu_name"] == "not-detected" and Path("/dev/nvidia0").exists():
        info["gpu_name"] = "nvidia-device-present"

    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisoMaster Linux preflight")
    parser.add_argument("--stage", choices=["host", "env"], default="env")
    parser.add_argument("--before-install", action="store_true", help="Expected missing Python imports are warnings.")
    parser.add_argument("--strict", action="store_true", help="Treat detected blockers as a non-zero exit.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = Report()

    print("[PRECHECK] VisoMaster Linux preflight")
    print(f"[PRECHECK] stage={args.stage}")
    print(f"[PRECHECK] repo={ROOT_DIR}")
    print(f"[PRECHECK] platform={platform.platform()}")
    print(f"[PRECHECK] python={sys.version.split()[0]} executable={sys.executable}")

    if platform.system() != "Linux":
        report.blocker(f"This script only supports Linux. Detected {platform.system()}.")

    if args.stage == "env" and sys.version_info[:2] != EXPECTED_PYTHON:
        report.blocker(
            f"Unsupported Python {sys.version.split()[0]}. Expected {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} in the bootstrap env."
        )
    elif args.stage == "host" and sys.version_info[:2] != EXPECTED_PYTHON:
        report.note(
            f"Host Python is {sys.version.split()[0]}; setup will create/use a Python {EXPECTED_PYTHON[0]}.{EXPECTED_PYTHON[1]} conda env next."
        )

    gpu = collect_gpu_info()
    print(f"[PRECHECK] gpu_name={gpu['gpu_name']}")
    print(f"[PRECHECK] driver_version={gpu['driver_version']}")
    print(f"[PRECHECK] cuda_version={gpu['cuda_version']}")

    driver_major = gpu["driver_version"].split(".", 1)[0]
    if driver_major.isdigit() and int(driver_major) < MIN_DRIVER_MAJOR:
        report.blocker(
            f"NVIDIA driver {gpu['driver_version']} is too old for the supported CUDA 12.9 flow. Need >= {MIN_DRIVER_MAJOR}."
        )
    elif gpu["driver_version"] == "not-detected":
        report.warn("NVIDIA driver version could not be detected yet. Continuing because some pods expose it late.")

    if gpu["cuda_version"] in {"not-detected", "N/A"}:
        report.warn("CUDA runtime version was not detected. Continuing because some pods expose it late.")

    if gpu["gpu_name"] == "not-detected":
        report.warn("GPU name could not be detected yet. Continuing because detection can be transient during pod startup.")

    print("[PRECHECK] package_indexes=")
    for name, config in INDEXES.items():
        ok, status_code, detail = check_index(name, config["url"])
        prefix = "OK" if ok else "FAIL"
        print(f"  - {name}: {prefix} ({detail}) {config['url']}")
        if not ok:
            message = f"Package index '{name}' is not reachable: {detail}"
            if config["required"] and args.stage == "env":
                report.blocker(message)
            else:
                report.warn(message)
        elif name == "nvidia" and status_code == 404:
            report.note("NVIDIA simple index returned 404 for the probe URL; treating that as non-fatal.")

    trt_libs = discover_tensorrt_libs()
    print(f"[PRECHECK] tensorrt_libs={json.dumps(trt_libs, sort_keys=True)}")
    for lib, path in trt_libs.items():
        if not path:
            report.warn(f"TensorRT shared library '{lib}' is not currently discoverable.")

    qt_libs = discover_qt_system_libs()
    print(f"[PRECHECK] qt_system_libs={json.dumps(qt_libs, sort_keys=True)}")
    missing_qt_libs = [lib for lib, path in qt_libs.items() if not path]
    if missing_qt_libs:
        report.warn(
            "Qt runtime dependencies are not discoverable yet: " + ", ".join(missing_qt_libs)
        )

    print("[PRECHECK] import_smoke=")
    for module_name in RUNTIME_IMPORTS + OPTIONAL_IMPORTS:
        ok, detail = import_status(module_name)
        status = "OK" if ok else "MISS"
        print(f"  - {module_name}: {status} ({detail})")
        if ok:
            continue
        if args.before_install:
            report.note(f"Expected missing import before install: {module_name} ({detail})")
        elif module_name in OPTIONAL_IMPORTS:
            report.warn(f"Optional runtime import '{module_name}' is unavailable: {detail}")
        else:
            report.blocker(f"Required runtime import '{module_name}' failed: {detail}")

    if report.blockers:
        print("\n[PRECHECK] HARD BLOCKERS")
        for item in report.blockers:
            print(f"  - {item}")
    else:
        print("\n[PRECHECK] HARD BLOCKERS: none")

    if report.warnings:
        print("[PRECHECK] WARNINGS")
        for item in report.warnings:
            print(f"  - {item}")
    else:
        print("[PRECHECK] WARNINGS: none")

    if report.notes:
        print("[PRECHECK] NOTES")
        for item in report.notes:
            print(f"  - {item}")
    else:
        print("[PRECHECK] NOTES: none")

    if report.blockers and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
