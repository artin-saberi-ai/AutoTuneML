from types import SimpleNamespace

import pytest

import autotuneml.hardware as hardware
from autotuneml.hardware import (
    _nvidia_smi_devices,
    collect_hardware,
    gpu_available,
    parse_nvidia_smi,
    resolve_device,
)


def test_parse_nvidia_smi_output():
    devices = parse_nvidia_smi("NVIDIA A100-SXM4-40GB, 40536\nTesla T4, 15360\n")
    assert [device["name"] for device in devices] == ["NVIDIA A100-SXM4-40GB", "Tesla T4"]
    assert devices[0]["memory_gb"] == pytest.approx(39.6)
    assert devices[1]["memory_gb"] == pytest.approx(15.0)


def test_parse_nvidia_smi_ignores_bad_lines():
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("not a gpu line\n, \n") == []


def test_nvidia_smi_without_binary_reports_no_devices(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: None)
    assert _nvidia_smi_devices() == []


def test_nvidia_smi_parses_subprocess_output(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")
    fake = SimpleNamespace(returncode=0, stdout="Tesla T4, 15360\n")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: fake)
    devices = _nvidia_smi_devices()
    assert len(devices) == 1 and devices[0]["name"] == "Tesla T4"


def test_nvidia_smi_failure_reports_no_devices(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda _: "/usr/bin/nvidia-smi")
    fake = SimpleNamespace(returncode=1, stdout="")
    monkeypatch.setattr("subprocess.run", lambda *args, **kwargs: fake)
    assert _nvidia_smi_devices() == []


def test_resolve_device_always_accepts_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_rejects_unknown_choice():
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_cuda_request_fails_without_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="no GPU"):
        resolve_device("cuda")
    assert resolve_device("auto") == "cpu"


def test_auto_and_cuda_use_gpu_when_present(monkeypatch):
    monkeypatch.setattr(hardware, "gpu_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_device("cuda") == "cuda"


def test_collect_hardware_reports_expected_shape():
    info = collect_hardware()
    assert set(info) == {"cpu", "gpu", "accelerator"}
    assert info["accelerator"] in ("cpu", "cuda")
    assert isinstance(info["gpu"]["present"], bool)
    assert info["gpu"]["present"] == gpu_available()
    assert isinstance(info["cpu"]["count"], int)
