"""Shared pieces of the benchmark runners (run_flux.py, run_sdcpp.py).

Nothing here imports torch, so an engine wrapper can use it without creating a CUDA context
of its own (which would add to the device-wide memory it is measuring).
Metric definitions: wiki/methods/baseline-metrics.md.
"""

import ctypes
import datetime as dt
import json
import os
import platform
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
GIB = 1024**3


def now_utc():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def sh(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"<failed: {e}>"


def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None
    return {
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "n": len(values),
    }


def peak_rss_gib():
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / GIB  # Linux reports KiB


# --------------------------------------------------------------------------- device memory (NVML via ctypes)


class _NvmlMemory(ctypes.Structure):
    _fields_ = [("total", ctypes.c_ulonglong), ("free", ctypes.c_ulonglong), ("used", ctypes.c_ulonglong)]


def visible_gpu_uuid():
    """UUID of the single GPU this job can see, without creating a CUDA context."""
    uuids = [u for u in sh(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"]).splitlines() if u.strip()]
    assert len(uuids) == 1, f"expected exactly one visible GPU, nvidia-smi reports {uuids}"
    return uuids[0].strip()


class DeviceMemorySampler:
    """Samples device-wide used memory (NVML) in a background thread.

    Device-wide: includes the CUDA context and any other process on the GPU. Sampling can
    miss spikes shorter than the interval. `uuid` is the GPU's "GPU-..." UUID.
    """

    def __init__(self, interval_ms, uuid):
        self.interval = interval_ms / 1000.0
        self.samples = []  # (perf_counter, used_bytes)
        self.note = ""
        self._stop = threading.Event()
        self._thread = None
        try:
            self.nvml = ctypes.CDLL("libnvidia-ml.so.1")
            assert self.nvml.nvmlInit_v2() == 0
            self.handle = ctypes.c_void_p()
            if self.nvml.nvmlDeviceGetHandleByUUID(uuid.encode(), ctypes.byref(self.handle)) == 0:
                self.note = f"nvml handle by uuid {uuid}"
            else:
                assert self.nvml.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) == 0
                self.note = "nvml handle by index 0 (uuid lookup failed)"
            self.available = True
        except Exception as e:  # noqa: BLE001
            self.available = False
            self.note = f"NVML unavailable: {e!r}"

    def read(self):
        mem = _NvmlMemory()
        self.nvml.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(mem))
        return mem.used

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.read()))
            time.sleep(self.interval)

    def start(self):
        if self.available:
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()

    def stop(self):
        if self._thread:
            self._stop.set()
            self._thread.join()

    def peak_between(self, t0, t1):
        vals = [u for t, u in self.samples if t0 <= t <= t1]
        return max(vals) / GIB if vals else None


# --------------------------------------------------------------------------- environment


SLURM_ENV_VARS = [
    "SLURM_JOB_ID",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_NODELIST",
    "SLURM_CPUS_PER_TASK",
    "CUDA_VISIBLE_DEVICES",
    "HF_HOME",
    "HF_HUB_OFFLINE",
]


def host_environment(extra_env_vars=()):
    """Engine-independent environment facts: host, OS, Python, GPU (via nvidia-smi), git state."""
    gpu = sh(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid,power.limit", "--format=csv,noheader"])
    return {
        "timestamp_utc": now_utc(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "nvidia_smi_gpu": gpu,
        "nvidia_driver": sh(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]),
        "git_commit": sh(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"]),
        "git_dirty_files": sh(["git", "-C", str(REPO_ROOT), "status", "--porcelain"]).splitlines(),
        "env_vars": {k: os.environ.get(k) for k in [*SLURM_ENV_VARS, *extra_env_vars]},
    }
