"""Tests for runtime device auto-selection (app/retrieval/device.py)."""

from types import SimpleNamespace

import sys

import app.retrieval.device as device_module


def test_get_device_falls_back_to_cpu_without_torch(monkeypatch):
    """When torch is not installed, the device must fall back to CPU."""
    monkeypatch.setitem(sys.modules, "torch", None)
    assert device_module.get_device() == "cpu"


def test_get_device_selects_cuda_when_available(monkeypatch):
    """When torch reports a usable CUDA GPU, pick cuda."""
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert device_module.get_device() == "cuda"


def test_get_device_selects_cpu_when_cuda_unavailable(monkeypatch):
    """When torch is installed but has no usable CUDA GPU, pick cpu."""
    fake_torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    assert device_module.get_device() == "cpu"