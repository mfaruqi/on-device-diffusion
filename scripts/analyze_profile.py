#!/usr/bin/env python
"""Per-stage GPU kernel breakdown from a run_flux.py --profile trace or a run_sdcpp.py --nsys report.

Usage:
  python scripts/analyze_profile.py --run-dir results/runs/<id>-<stamp>-profile [--stages 'denoise_step_\\d+']

Reads <run-dir>/profile/trace.json (torch.profiler Chrome trace) or trace.nsys-rep (Nsight Systems;
see load_nsys for how stages are delimited there) and writes
<run-dir>/profile/stage_kernels.md and stage_kernels.json.

Method:
- Stage windows are the GPU-side `gpu_user_annotation` spans that run_flux.py's
  record_function labels produce, i.e. when the stage ran on the GPU, not when the
  CPU launched it.
- Every GPU activity (kernel, memcpy, memset) whose start falls inside a window is
  assigned to that stage.
- Each kernel is linked to the CPU launch (`correlation` id) and from there to the
  enclosing aten ops on the launching thread: the outermost op ("top op", e.g.
  aten::linear) and the innermost one ("leaf op", e.g. aten::mm).
- Busy time is the union of captured activity intervals; idle = window - busy.
  Uncovered time alone does not identify its cause (host work, disk reads, etc.).
- GEMM and attention throughput use shapes from `record_shapes=True`.
Kernel durations are GPU timestamps and are only mildly affected by profiler
overhead; compare shares and per-kernel times, not end-to-end latency.
"""

import argparse
import json
from pathlib import Path

from profile_metrics import summarize_stage
from profile_readers import load_nsys, load_torch_trace
from profile_report import render_md


def analyze(trace_path, stage_re, generation=-1):
    """Read one trace format, then apply the same calculations to each stage."""
    trace_path = Path(trace_path)
    is_nsys = trace_path.suffix in (".nsys-rep", ".sqlite")
    if is_nsys:
        stages = load_nsys(trace_path, stage_re, generation)
    else:
        stages = load_torch_trace(trace_path, stage_re)
    result = {"trace": str(trace_path), "stage_regex": stage_re, "stages": {}}
    if is_nsys:
        result["generation"] = generation
    for name, stage in stages.items():
        result["stages"][name] = summarize_stage(stage)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--stages", default=r"denoise_step_\d+", help="regex over stage labels")
    ap.add_argument("--out-name", default="stage_kernels")
    ap.add_argument("--generation", type=int, default=-1,
                    help="nsys reports only: which generation to analyse (index into 'generate' ranges; -1 = last)")
    args = ap.parse_args()

    prof = args.run_dir / "profile"
    trace = next((prof / n for n in ("trace.json", "trace.nsys-rep", "trace.sqlite") if (prof / n).exists()), None)
    assert trace, f"no trace.json / trace.nsys-rep in {prof}"
    res = analyze(trace, args.stages, args.generation)
    (prof / f"{args.out_name}.json").write_text(json.dumps(res, indent=2) + "\n")
    md = render_md(res)
    (prof / f"{args.out_name}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
