#!/usr/bin/env python3
"""Runtime import smoke test for VisoMaster-Fusion."""

from __future__ import annotations

import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
CRITICAL_MODULES = [
    "torch",
    "torchvision",
    "torchaudio",
    "onnxruntime",
    "PySide6",
    "pyqttoast",
    "qdarktheme",
    "qdarkstyle",
    "send2trash",
    "pyvirtualcam",
    "tqdm",
    "requests",
    "omegaconf",
    "einops",
    "lightning",
]
OPTIONAL_MODULES = ["tensorrt"]


def run(command: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        return 127, ""
    return result.returncode, (result.stdout or result.stderr or "").strip()


def nvidia_details() -> dict[str, str]:
    code, output = run([
        "nvidia-smi",
        "--query-gpu=name,driver_version,cuda_version",
        "--format=csv,noheader",
    ])
    if code != 0 or not output:
        return {"gpu_name": "not-detected", "driver_version": "not-detected", "cuda_version": "not-detected"}
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    while len(parts) < 3:
        parts.append("unknown")
    return {"gpu_name": parts[0], "driver_version": parts[1], "cuda_version": parts[2]}


def main() -> int:
    failed: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []
    print("[INFO] Checking runtime imports...")
    print(f"[INFO] platform={platform.platform()}")
    print(f"[INFO] python={sys.version.split()[0]}")
    print(f"[INFO] repo={ROOT_DIR}")
    print(f"[INFO] gpu={json.dumps(nvidia_details(), sort_keys=True)}")

    for module_name in CRITICAL_MODULES + OPTIONAL_MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:  # broad on purpose for smoke testing
            target = warnings if module_name in OPTIONAL_MODULES else failed
            target.append((module_name, str(exc)))
            print(f"[{'WARN' if module_name in OPTIONAL_MODULES else 'FAIL'}] {module_name}: {exc}")

    if failed:
        print("\n[ERROR] Missing/broken critical runtime dependencies detected:")
        for module_name, error_text in failed:
            print(f"  - {module_name}: {error_text}")
        if warnings:
            print("[WARN] Optional components:")
            for module_name, error_text in warnings:
                print(f"  - {module_name}: {error_text}")
        return 1

    print("\n[INFO] All critical runtime imports succeeded.")
    if warnings:
        print("[WARN] Optional runtime components unavailable:")
        for module_name, error_text in warnings:
            print(f"  - {module_name}: {error_text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
