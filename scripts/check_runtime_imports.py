#!/usr/bin/env python3
"""Runtime import smoke test for VisoMaster-Fusion."""

from __future__ import annotations

import importlib

MODULES = [
    "torch",
    "torchvision",
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


def main() -> int:
    failed: list[tuple[str, str]] = []
    print("[INFO] Checking runtime imports...")

    for module_name in MODULES:
        try:
            importlib.import_module(module_name)
            print(f"[OK] {module_name}")
        except Exception as exc:  # broad on purpose for smoke testing
            failed.append((module_name, str(exc)))
            print(f"[FAIL] {module_name}: {exc}")

    if failed:
        print("\n[ERROR] Missing/broken runtime dependencies detected:")
        for module_name, error_text in failed:
            print(f"  - {module_name}: {error_text}")
        return 1

    print("\n[INFO] All runtime imports succeeded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
