"""Device and dtype selection.

The model card forbids float16 (activations overflow to NaN), so only
bfloat16 (when the GPU supports it) or float32 are ever returned.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch

DEVICE_ENV_VAR = "EGLAB_DEVICE"


@dataclass(frozen=True)
class DeviceInfo:
    device: str
    dtype: torch.dtype
    name: str

    @property
    def dtype_name(self) -> str:
        return str(self.dtype).removeprefix("torch.")


def detect_device(preferred: str | None = None) -> str:
    """``preferred`` > ``$EGLAB_DEVICE`` > ``cuda`` when available > ``cpu``."""
    preferred = preferred or os.environ.get(DEVICE_ENV_VAR)
    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def select_dtype(device: str) -> torch.dtype:
    """bfloat16 on GPUs with native support, float32 everywhere else."""
    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float32


def device_name(device: str) -> str:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.cuda.get_device_name(torch.device(device))
    return "CPU"


def resolve_device(preferred: str | None = None) -> DeviceInfo:
    device = detect_device(preferred)
    return DeviceInfo(device=device, dtype=select_dtype(device), name=device_name(device))
