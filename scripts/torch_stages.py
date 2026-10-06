"""CUDA stage instrumentation for the PyTorch engine only.

Read StageRecorder.start/end to see the timing boundaries. No stage synchronizes
CUDA; the caller synchronizes once at generation end before reading the events.
"""

import time

from benchlib import GIB


class StageRecorder:
    """Marks pipeline stages with CUDA events, profiler labels and allocator peaks.

    Stages are delimited by wrapping pipe.encode_prompt, each transformer forward,
    pipe.vae.decode and pipe.image_processor.postprocess. The pipeline code itself
    is not modified. CUDA events are recorded on the current stream without
    synchronizing, so the instrumentation does not change the baseline's execution.

    Allocator peaks are tracked per segment: the peak counter is reset at every stage
    boundary, so both stages and the gaps between them ("between:a->b") get a peak,
    and the run peak is the max over all segments.
    """

    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.stages = []  # dicts: name, ev0, ev1, host0, host1, peak_alloc, peak_reserved, alloc0, alloc1
        self.segments = []  # dicts: name, peak_alloc, peak_reserved
        self._open = None
        self._last = "start"
        self.scheduler_steps = 0
        self.torch.cuda.reset_peak_memory_stats()

    def _close_segment(self, name):
        c = self.torch.cuda
        self.segments.append(
            {"name": name, "peak_alloc": c.max_memory_allocated() / GIB, "peak_reserved": c.max_memory_reserved() / GIB}
        )
        c.reset_peak_memory_stats()

    def start(self, name):
        from torch.profiler import record_function

        assert self._open is None, f"nested stage {name} inside {self._open['name']}"
        self._close_segment(f"between:{self._last}->{name}")
        ev0 = self.torch.cuda.Event(enable_timing=True)
        ev0.record()
        rf = record_function(name)
        rf.__enter__()
        self._open = {
            "name": name,
            "ev0": ev0,
            "host0": time.perf_counter(),
            "rf": rf,
            "alloc0": self.torch.cuda.memory_allocated() / GIB,
        }

    def end(self):
        s = self._open
        s["ev1"] = self.torch.cuda.Event(enable_timing=True)
        s["ev1"].record()
        s["host1"] = time.perf_counter()
        s["rf"].__exit__(None, None, None)
        s["alloc1"] = self.torch.cuda.memory_allocated() / GIB
        c = self.torch.cuda
        s["peak_alloc"] = c.max_memory_allocated() / GIB
        s["peak_reserved"] = c.max_memory_reserved() / GIB
        c.reset_peak_memory_stats()
        self.segments.append({"name": s["name"], "peak_alloc": s["peak_alloc"], "peak_reserved": s["peak_reserved"]})
        self.stages.append(s)
        self._last = s["name"]
        self._open = None

    def finish(self):
        """Call after torch.cuda.synchronize(). Returns per-stage rows and run peaks."""
        self._close_segment(f"between:{self._last}->end")
        rows = []
        for s in self.stages:
            rows.append(
                {
                    "stage": s["name"],
                    "gpu_ms": s["ev0"].elapsed_time(s["ev1"]),
                    "host_ms": (s["host1"] - s["host0"]) * 1000,
                    "alloc_start_gib": s["alloc0"],
                    "alloc_end_gib": s["alloc1"],
                    "peak_alloc_gib": s["peak_alloc"],
                    "peak_reserved_gib": s["peak_reserved"],
                }
            )
        peak_alloc = max(seg["peak_alloc"] for seg in self.segments)
        peak_reserved = max(seg["peak_reserved"] for seg in self.segments)
        return rows, peak_alloc, peak_reserved, list(self.segments)

    def install(self, pipe):
        rec = self

        def wrap(obj, attr, name):
            orig = getattr(obj, attr)

            def wrapped(*args, **kwargs):
                rec.start(name)
                try:
                    return orig(*args, **kwargs)
                finally:
                    rec.end()

            setattr(obj, attr, wrapped)

        wrap(pipe, "encode_prompt", "text_encode")
        wrap(pipe.vae, "decode", "vae_decode")
        wrap(pipe.image_processor, "postprocess", "postprocess")
        # CFG makes two forwards before one scheduler update. Both belong to the
        # same logical step; the scheduler itself stays outside the timed stages.
        scheduler_step = pipe.scheduler.step

        def step(*args, **kwargs):
            result = scheduler_step(*args, **kwargs)
            rec.scheduler_steps += 1
            return result

        pipe.scheduler.step = step
        pipe.transformer.register_forward_pre_hook(lambda m, a: rec.start(f"denoise_step_{rec.scheduler_steps}"))
        pipe.transformer.register_forward_hook(lambda m, a, o: rec.end())
