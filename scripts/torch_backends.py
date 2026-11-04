"""Device adapters for the portable PyTorch runner (run_flux_portable.py).

One adapter per backend. Each says how to synchronize, time a generation, seed noise and sample
memory on that device, and states which metrics it cannot provide instead of substituting others.

  cuda    discrete NVIDIA GPU (A100): CUDA events, allocator peaks, device-wide memory from NVML.
  jetson  Jetson Orin (CUDA, shared CPU/GPU memory): CUDA events and allocator peaks as on cuda;
          NVML is unavailable, so device-wide memory is reported as unavailable and whole-system
          RAM/swap comes from tegrastats instead (jetson_device.py).
  mps     Apple silicon (Metal): MPS events; no allocator peak counters, no device-wide sampler.
          Inherits nothing from CUDA; CPU fallback is forbidden.

Compilers: TensorRT is CUDA-only (cuda, jetson). Requesting it on mps is an error, never a fallback.
"""

import os
import time

from benchlib import GIB, DeviceMemorySampler

BACKENDS = ("cuda", "jetson", "mps")
COMPILERS_BY_BACKEND = {
    "cuda": (None, "torch_compile", "tensorrt"),
    "jetson": (None, "torch_compile", "tensorrt"),
    "mps": (None, "torch_compile"),
}


def check_backend(cfg):
    """Validate backend and compiler choices before any library import."""
    backend = cfg.get("backend")
    assert backend in BACKENDS, f"backend must be one of {BACKENDS}, got {backend!r}"
    compiler = cfg["optimizations"].get("compiler")
    name = compiler["name"] if compiler else None
    assert name in COMPILERS_BY_BACKEND[backend], (
        f"compiler {name!r} is not available on backend {backend!r}; allowed: {COMPILERS_BY_BACKEND[backend]}"
    )
    return backend


class _NoDeviceSampler:
    """Explicitly unavailable device-wide memory; never filled with system RAM."""
    available = False

    def __init__(self, note):
        self.note = note

    def start(self):
        pass

    def stop(self):
        pass

    def read(self):
        return None

    def peak_between(self, t0, t1):
        return None


class CudaDevice:
    name = "cuda"
    torch_device = "cuda"

    def __init__(self, cfg, run_dir):
        import torch
        assert torch.cuda.is_available(), "No CUDA device visible"
        self.torch = torch
        uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid)
        self.sampler = DeviceMemorySampler(cfg["protocol"]["device_memory_sample_ms"], uuid)

    def synchronize(self):
        self.torch.cuda.synchronize()

    def recorder(self):
        from torch_stages import StageRecorder
        return StageRecorder(self.torch)

    def generate(self, pipe, recorder, workload):
        from flux_engine import generate_once
        return generate_once(pipe, recorder, workload)

    def allocator_after_load(self):
        c = self.torch.cuda
        return {"allocated_after_load_gib": c.memory_allocated() / GIB,
                "reserved_after_load_gib": c.memory_reserved() / GIB,
                "peak_alloc_during_load_gib": c.max_memory_allocated() / GIB}

    def reset_peaks(self):
        self.torch.cuda.reset_peak_memory_stats()

    def environment(self):
        props = self.torch.cuda.get_device_properties(0)
        return {"backend": self.name, "gpu_name": props.name, "gpu_compute_capability": f"{props.major}.{props.minor}",
                "gpu_total_memory_gib": props.total_memory / GIB, "torch_cuda_runtime": self.torch.version.cuda,
                "cudnn": self.torch.backends.cudnn.version(), "device_memory_sampler": self.sampler.note}

    def system_memory_summary(self, run_dir):
        return None

    def unavailable_metrics(self):
        return {}


class JetsonDevice(CudaDevice):
    name = "jetson"

    def __init__(self, cfg, run_dir):
        import torch
        from jetson_device import TegrastatsMonitor
        assert torch.cuda.is_available(), "No CUDA device visible"
        self.torch = torch
        self.sampler = _NoDeviceSampler("NVML unavailable on Jetson; device_used_peak_gib not reported")
        self.monitor = TegrastatsMonitor(run_dir / "tegrastats.log", cfg["protocol"]["tegrastats_interval_ms"])

    def start_monitor(self):
        self.monitor.start()

    def stop_monitor(self):
        self.monitor.stop()

    def environment(self):
        env = super().environment()
        try:
            env["jetpack"] = open("/etc/nv_tegra_release").readline().strip()
        except OSError as error:
            env["jetpack"] = f"<unavailable: {error}>"
        from benchlib import sh
        env["power_mode"] = sh(["nvpmodel", "-q"])
        return env

    def system_memory_summary(self, run_dir):
        from jetson_device import summarize_monitor
        return summarize_monitor(run_dir)

    def unavailable_metrics(self):
        return {"device_used_peak_gib": "NVML is unavailable on Jetson; whole-system RAM is in system_memory instead."}


class MpsDevice:
    name = "mps"
    torch_device = "mps"

    def __init__(self, cfg, run_dir):
        import torch
        assert torch.backends.mps.is_available(), "Metal MPS device unavailable"
        assert os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK", "0") == "0", "CPU fallback is forbidden"
        self.torch = torch
        self.sampler = _NoDeviceSampler("No device-wide memory sampler on MPS in this runner")

    def synchronize(self):
        self.torch.mps.synchronize()

    def recorder(self):
        return MpsStageRecorder(self.torch)

    def generate(self, pipe, recorder, w):
        torch = self.torch
        # MPS has no device Generator; the CPU generator makes initial noise explicit (not matched to CUDA).
        generator = torch.Generator(device="cpu").manual_seed(w["seed"])
        torch.mps.synchronize()
        recorder.reset()
        a, b = torch.mps.Event(enable_timing=True), torch.mps.Event(enable_timing=True)
        h0 = time.perf_counter()
        a.record()
        image = pipe(prompt=w["prompt"], height=w["height"], width=w["width"],
                     num_inference_steps=w["num_inference_steps"], guidance_scale=w["guidance_scale"],
                     max_sequence_length=w["max_sequence_length"], generator=generator, output_type="pil").images[0]
        b.record()
        torch.mps.synchronize()
        h1 = time.perf_counter()
        stages, peak_alloc, peak_reserved, segments = recorder.finish()
        return image, {"calls_per_step": 2 if pipe.do_classifier_free_guidance else 1,
                       "scheduler_steps": recorder.scheduler_steps, "wall_ms": (h1 - h0) * 1000,
                       "gpu_span_ms": a.elapsed_time(b), "t0": h0, "t1": h1, "stages": stages,
                       "segments": segments, "peak_alloc_gib": peak_alloc, "peak_reserved_gib": peak_reserved}

    def allocator_after_load(self):
        return {"allocated_after_load_gib": self.torch.mps.current_allocated_memory() / GIB,
                "driver_allocated_after_load_gib": self.torch.mps.driver_allocated_memory() / GIB}

    def reset_peaks(self):
        pass

    def environment(self):
        return {"backend": self.name, "recommended_max_memory_gib": self.torch.mps.recommended_max_memory() / GIB,
                "device_memory_sampler": self.sampler.note}

    def system_memory_summary(self, run_dir):
        return None

    def unavailable_metrics(self):
        return {"peak_alloc_gib/peak_reserved_gib": "No MPS allocator peak counters.",
                "device_used_peak_gib": "No device-wide sampler on MPS in this runner."}


class MpsStageRecorder:
    """Same pipeline hooks as torch_stages.StageRecorder, with MPS events and no peak counters."""

    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.stages, self.scheduler_steps, self._open = [], 0, None

    def start(self, name):
        assert self._open is None, f"nested stage {name}"
        ev = self.torch.mps.Event(enable_timing=True)
        ev.record()
        self._open = {"stage": name, "ev0": ev, "host0": time.perf_counter(),
                      "alloc_start_gib": self.torch.mps.current_allocated_memory() / GIB}

    def end(self):
        s = self._open
        ev = self.torch.mps.Event(enable_timing=True)
        ev.record()
        s.update(ev1=ev, host1=time.perf_counter(), alloc_end_gib=self.torch.mps.current_allocated_memory() / GIB)
        self.stages.append(s)
        self._open = None

    def finish(self):
        assert self._open is None
        rows = [{"stage": s["stage"], "gpu_ms": s["ev0"].elapsed_time(s["ev1"]),
                 "host_ms": (s["host1"] - s["host0"]) * 1000, "alloc_start_gib": s["alloc_start_gib"],
                 "alloc_end_gib": s["alloc_end_gib"], "peak_alloc_gib": None, "peak_reserved_gib": None}
                for s in self.stages]
        return rows, None, None, []

    def install(self, pipe):
        from torch_stages import StageRecorder
        StageRecorder.install(self, pipe)


def make_device(cfg, run_dir):
    backend = check_backend(cfg)
    return {"cuda": CudaDevice, "jetson": JetsonDevice, "mps": MpsDevice}[backend](cfg, run_dir)
