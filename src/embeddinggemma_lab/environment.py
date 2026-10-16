"""Recording the run environment for reproducible evaluation reports."""

from __future__ import annotations

import platform
from datetime import UTC, datetime
from importlib import metadata
from typing import Any

PACKAGES = ("torch", "transformers", "sentence-transformers", "numpy", "streamlit")


def package_versions() -> dict[str, str]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def peak_gpu_memory_mb() -> float | None:
    import torch

    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / 2**20


def collect_environment(encoder: Any) -> dict[str, Any]:
    device_info = getattr(encoder, "device_info", None)
    settings = getattr(encoder, "settings", None)
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "model_id": encoder.model_id,
        "model_revision": encoder.revision,
        "device": getattr(device_info, "device", "n/a"),
        "device_name": getattr(device_info, "name", "n/a"),
        "dtype": getattr(device_info, "dtype_name", "n/a"),
        "batch_size": getattr(settings, "batch_size", None),
        "retrieval": "numpy exact inner product (cosine on L2-normalised vectors)",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "packages": package_versions(),
    }
