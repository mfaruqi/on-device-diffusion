"""Device measurements for the sd.cpp child process; no engine imports."""

import importlib.metadata
import hashlib
import json
import os
import platform
import resource
import threading
import time
from pathlib import Path

from benchlib import DeviceMemorySampler, GIB, host_environment, sh, visible_gpu_uuid, write_json
from measurement import write_csv


def child_peak_rss_gib(raw=None, system=None):
    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss if raw is None else raw
    return raw * (1 if (system or platform.system()) == "Darwin" else 1024) / GIB


class MacMemorySampler:
    """Whole-system memory, including unrelated processes; never a GPU allocation counter."""

    note = "psutil system unified memory; GPU allocation/peak unavailable"

    def __init__(self, interval_ms, run_dir):
        import psutil
        self.psutil = psutil
        self.interval = interval_ms / 1000
        assert self.interval > 0
        self.run_dir = run_dir
        self.samples = []
        self.done = threading.Event()
        self.thermal = {}
        self.error = None

    def _sample(self):
        mem, swap = self.psutil.virtual_memory(), self.psutil.swap_memory()
        self.samples.append({"host_monotonic_s": time.monotonic(), "wall_unix_s": time.time(), "system_total_gib": mem.total / GIB,
                             "system_available_gib": mem.available / GIB,
                             "system_total_minus_available_gib": (mem.total - mem.available) / GIB,
                             "system_swap_used_gib": swap.used / GIB})

    def _loop(self):
        try:
            while not self.done.wait(self.interval):
                self._sample()
        except Exception as error:
            self.error = error

    def start(self):
        self.thermal["before"] = sh(["pmset", "-g", "therm"])
        self._sample()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.done.set()
        self.thread.join()
        self._sample()
        self.thermal["after"] = sh(["pmset", "-g", "therm"])
        write_csv(self.run_dir / "system-memory.csv", self.samples)
        write_json(self.run_dir / "thermal.json", self.thermal)
        write_json(self.run_dir / "system-memory-summary.json", self.summary())
        if self.error:
            raise RuntimeError("System memory sampling failed") from self.error

    def peak_between(self, start, end):
        # C++ and Python clock epochs are not assumed equal on macOS. Only whole-run system
        # aggregates are exposed, under distinct names, rather than fabricated per-stage GPU peaks.
        return None

    def summary(self):
        first, last = self.samples[0], self.samples[-1]
        clock_gap = (last["wall_unix_s"] - first["wall_unix_s"]) - (last["host_monotonic_s"] - first["host_monotonic_s"])
        return {"sample_interval_ms": self.interval * 1000,
                "wall_minus_monotonic_elapsed_s": clock_gap,
                "system_total_gib": self.samples[0]["system_total_gib"],
                "system_available_min_gib": min(s["system_available_gib"] for s in self.samples),
                "system_total_minus_available_peak_gib": max(s["system_total_minus_available_gib"] for s in self.samples),
                "system_swap_used_start_gib": self.samples[0]["system_swap_used_gib"],
                "system_swap_used_peak_gib": max(s["system_swap_used_gib"] for s in self.samples),
                "scope": "Whole system across model load and all generations; includes other applications."}


class SdcppDevice:
    def __init__(self, cfg, run_dir, nsys):
        self.cfg = cfg
        self.metal = cfg["engine_settings"]["backend"].lower() in ("metal", "mtl0")
        interval = cfg["protocol"]["device_memory_sample_ms"]
        if self.metal:
            assert platform.system() == "Darwin" and platform.machine() == "arm64", "Apple Silicon required"
            assert not nsys, "Nsight/CUDA profiling is unavailable for Metal"
            self.sampler = MacMemorySampler(interval, run_dir)
        else:
            self.sampler = DeviceMemorySampler(interval, visible_gpu_uuid())

    def wrap_command(self, command):
        if self.metal and self.cfg["protocol"].get("prevent_idle_sleep", False):
            return ["/usr/bin/caffeinate", "-is", *command]
        return command

    def environment(self, binary):
        env = host_environment(extra_env_vars=["GGML_CUDA_CUBLAS_COMPUTE_TYPE", "LD_LIBRARY_PATH"])
        if not self.metal:
            env.update(nvcc=sh(["nvcc", "--version"]).splitlines()[-1:], ldd=sh(["ldd", str(binary)]).splitlines())
            env["device_description"] = env["nvidia_smi_gpu"]
            return env
        data = json.loads(sh(["system_profiler", "SPHardwareDataType", "SPDisplaysDataType", "-json"]))
        hw = data["SPHardwareDataType"][0]
        env["hardware"] = {k: hw.get(k) for k in ["machine_name", "machine_model", "chip_type", "number_processors", "physical_memory"]}
        expected = self.cfg["device_requirements"]
        assert hw["chip_type"] == expected["chip"], f"Unexpected chip: {hw['chip_type']}"
        assert self.sampler.psutil.virtual_memory().total == expected["memory_gib"] * GIB, "Unexpected unified memory capacity"
        env["gpu_devices"] = [{k: d.get(k) for k in ["sppci_model", "sppci_cores", "spdisplays_metal"]} for d in data["SPDisplaysDataType"]]
        env["device_description"] = env["gpu_devices"]
        env.pop("nvidia_smi_gpu", None)
        env.pop("nvidia_driver", None)
        build = Path(self.cfg["engine"]["source_dir"]).expanduser() / "build"
        library = build / "bin/libstable-diffusion.dylib"
        cache = dict(line.split("=", 1) for line in (build / "CMakeCache.txt").read_text().splitlines()
                     if "=" in line and not line.startswith(("//", "#")))
        assert cache.get("SD_METAL:BOOL") == "ON" and cache.get("SD_CUDA:BOOL") == "OFF", "Expected a Metal build"
        env["metal_build"] = {"library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
                              "cmake": {k: v for k, v in cache.items() if k.startswith(("SD_", "GGML_", "CMAKE_OSX_", "CMAKE_BUILD_TYPE:"))}}
        env.update(macos=sh(["sw_vers"]), power_source=sh(["pmset", "-g", "batt"]),
                   host_load_average=os.getloadavg(),
                   toolchain=sh(["clang", "--version"]), dylibs=sh(["otool", "-L", str(binary)]),
                   packages={p: importlib.metadata.version(p) for p in ["psutil", "pillow", "huggingface-hub"]})
        return env

    def summary_fields(self):
        if not self.metal:
            return {}
        memory = self.sampler.summary()
        return {"system_memory": memory, "memory_scope": self.sampler.note,
                "clock_continuity_pass": abs(memory["wall_minus_monotonic_elapsed_s"]) < 5,
                "clock_continuity_note": "Wall/monotonic divergence >=5 s rejects timing; investigate sleep or clock changes."}
