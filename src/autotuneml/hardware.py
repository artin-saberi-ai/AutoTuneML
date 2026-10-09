from __future__ import annotations

import functools
import os
import platform
import shutil
import subprocess

DEVICE_CHOICES = ("auto", "cpu", "cuda")


def cpu_count() -> int | None:
    return os.cpu_count()


def total_memory_gb() -> float | None:
    try:
        import psutil
    except Exception:
        psutil = None
    if psutil is not None:
        try:
            return round(psutil.virtual_memory().total / (1024 ** 3), 1)
        except Exception:
            pass
    try:
        pages = os.sysconf("SC_PHYS_PAGES")
        page_size = os.sysconf("SC_PAGE_SIZE")
        return round(pages * page_size / (1024 ** 3), 1)
    except (ValueError, OSError, AttributeError):
        return None


def _torch_cuda_devices() -> list[dict]:
    try:
        import torch
    except Exception:
        return []
    try:
        if not torch.cuda.is_available():
            return []
        return [
            {
                "name": torch.cuda.get_device_name(index),
                "memory_gb": round(torch.cuda.get_device_properties(index).total_memory / (1024 ** 3), 1),
                "backend": "torch.cuda",
            }
            for index in range(torch.cuda.device_count())
        ]
    except Exception:
        return []


def _nvidia_smi_devices() -> list[dict]:
    binary = shutil.which("nvidia-smi")
    if binary is None:
        return []
    try:
        completed = subprocess.run(
            [binary, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
    except Exception:
        return []
    if completed.returncode != 0:
        return []
    return parse_nvidia_smi(completed.stdout)


def parse_nvidia_smi(output: str) -> list[dict]:
    devices = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 2 or not parts[0]:
            continue
        try:
            memory_gb = round(float(parts[1]) / 1024, 1)
        except ValueError:
            memory_gb = None
        devices.append({"name": parts[0], "memory_gb": memory_gb, "backend": "nvidia-smi"})
    return devices


def _apple_silicon_note() -> dict | None:
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        return {
            "note": "Apple Silicon GPU is only reachable through torch MPS, not CUDA",
        }
    return None


@functools.lru_cache(maxsize=1)
def collect_hardware() -> dict:
    devices = _torch_cuda_devices() or _nvidia_smi_devices()
    gpu: dict = {"present": bool(devices), "devices": devices}
    note = _apple_silicon_note()
    if note is not None:
        gpu.update(note)
    return {
        "cpu": {
            "count": cpu_count(),
            "arch": platform.machine(),
            "system": platform.system(),
            "memory_gb": total_memory_gb(),
        },
        "gpu": gpu,
        "accelerator": "cuda" if devices else "cpu",
    }


def gpu_available() -> bool:
    return bool(collect_hardware()["gpu"]["present"])


def resolve_device(preference: str = "auto") -> str:
    if preference not in DEVICE_CHOICES:
        raise ValueError(f"unknown device '{preference}' (use one of {', '.join(DEVICE_CHOICES)})")
    if preference == "cpu":
        return "cpu"
    if gpu_available():
        return "cuda"
    if preference == "cuda":
        raise RuntimeError(
            "device 'cuda' was requested but no GPU was detected on this machine; "
            "use device 'auto' or 'cpu', or run where a CUDA GPU is present"
        )
    return "cpu"
