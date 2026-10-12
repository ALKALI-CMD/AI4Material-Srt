from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any


def collect_environment() -> dict[str, Any]:
    packages = ["atomlens", "ase", "numpy", "scipy", "torch", "mace-torch", "chgnet", "plotly"]
    payload: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {package: _package_version(package) for package in packages},
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch": _torch_info(),
    }
    return payload


def write_json(payload: dict[str, Any], out: Path | None = None) -> str:
    text = json.dumps(payload, indent=2, sort_keys=True)
    if out is not None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    return text


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        module_name = "mace" if package == "mace-torch" else package.replace("-", "_")
        if importlib.util.find_spec(module_name) is None:
            return None
        return "installed"


def _torch_info() -> dict[str, Any]:
    try:
        import torch
    except ImportError:
        return {"available": False}
    return {
        "available": True,
        "version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
    }
