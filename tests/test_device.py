import torch

from embeddinggemma_lab import device


def test_cpu_uses_float32():
    assert device.select_dtype("cpu") == torch.float32


def test_float16_is_never_selected(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)
    assert device.select_dtype("cuda") == torch.float32
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    assert device.select_dtype("cuda") == torch.bfloat16


def test_preferred_device_wins():
    assert device.detect_device("cpu") == "cpu"


def test_resolve_device_cpu():
    info = device.resolve_device("cpu")
    assert (info.device, info.dtype_name, info.name) == ("cpu", "float32", "CPU")


def test_env_var_overrides_auto_detection(monkeypatch):
    monkeypatch.setenv(device.DEVICE_ENV_VAR, "cpu")
    assert device.detect_device() == "cpu"
    assert device.detect_device("cuda:1") == "cuda:1"
