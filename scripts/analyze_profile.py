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
- Busy time is the union of kernel intervals; idle = window - busy (GPU waiting
  on the CPU or on synchronization).
- GEMM and attention throughput use shapes from `record_shapes=True`.
Kernel durations are GPU timestamps and are only mildly affected by profiler
overhead; compare shares and per-kernel times, not end-to-end latency.
"""

import argparse
import bisect
import collections
import json
import re
import statistics
from pathlib import Path

GPU_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}


def categorize(kernel_name, leaf_op, cat):
    """Kernel group. leaf_op is the launching aten op name for torch traces, None for ggml (nsys)."""
    n = kernel_name.lower()
    if cat != "kernel":
        return "memcpy/memset"
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "gemm" in n or "cutlass" in n or "xmma" in n:
        return "gemm"
    if leaf_op is None:
        return categorize_ggml(n)
    if "norm" in n or leaf_op in ("aten::native_layer_norm", "aten::native_group_norm", "aten::_fused_rms_norm"):
        return "norm"
    if "catarray" in n:
        return "cat"
    if "copy" in n or leaf_op in ("aten::copy_", "aten::_to_copy", "aten::contiguous", "aten::clone"):
        return "copy/cast"
    if "reduce_kernel" in n:
        return "reduction"
    if "elementwise" in n:
        if leaf_op in ("aten::mul", "aten::mul_"):
            return "elementwise: mul"
        if leaf_op in ("aten::add", "aten::add_"):
            return "elementwise: add"
        return "elementwise: other"
    return "other"


def categorize_ggml(n):
    """Group for a lower-cased ggml CUDA kernel name (demangled, so template ops are visible)."""
    if "mul_mat" in n or re.search(r"\bmm[qf]|mmvf|mmvq", n):
        return "gemm"
    if "norm" in n:
        return "norm"
    if "bin_bcast" in n:
        for op, group in (("op_mul", "elementwise: mul"), ("op_add", "elementwise: add")):
            if op in n:
                return group
        return "elementwise: other"
    if "rope" in n:
        return "elementwise: rope"
    if "concat" in n:
        return "cat"
    if "cpy" in n or "convert" in n or "get_rows" in n or "copy" in n:
        return "copy/cast"
    if "sum_rows" in n or "reduce" in n:
        return "reduction"
    if re.search(r"unary|silu|gelu|scale|soft_max|glu|sqr|clamp|neg|exp", n):
        return "elementwise: other"
    return "other"


def build_op_chains(events):
    """Map each cuda_runtime event (by correlation id) to its enclosing cpu_op chain."""
    by_tid = collections.defaultdict(list)
    for e in events:
        if e.get("cat") == "cpu_op":
            by_tid[e["tid"]].append(("op", e["ts"], -e["dur"], e))
        elif e.get("cat") == "cuda_runtime" and "correlation" in e.get("args", {}):
            by_tid[e["tid"]].append(("rt", e["ts"], 0, e))
    chains = {}
    for items in by_tid.values():
        # ops sort before a launch at the same ts; longer ops first so nesting is outer->inner
        items.sort(key=lambda x: (x[1], 0 if x[0] == "op" else 1, x[2]))
        stack = []
        for kind, ts, _, e in items:
            while stack and stack[-1]["ts"] + stack[-1]["dur"] < ts:
                stack.pop()
            if kind == "op":
                stack.append(e)
            else:
                chains[e["args"]["correlation"]] = list(stack)
    return chains


def union_ms(intervals):
    total, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total / 1000.0


def gemm_mnk(op):
    dims = op.get("args", {}).get("Input Dims", [])
    name = op["name"]
    if name == "aten::mm" and len(dims) >= 2 and len(dims[0]) == 2 and len(dims[1]) == 2:
        (m, k), (_, n) = dims[0], dims[1]
    elif name == "aten::addmm" and len(dims) >= 3 and len(dims[1]) == 2 and len(dims[2]) == 2:
        (m, k), (_, n) = dims[1], dims[2]
    elif name == "aten::bmm" and len(dims) >= 2 and len(dims[0]) == 3:
        (b, m, k), (_, _, n) = dims[0], dims[1]
        return b * m, n, k  # treat batch as extra rows for FLOP counting
    else:
        return None
    return m, n, k


def attn_flops(op):
    """FLOPs of one flash-attention forward: q [B, H, Sq, D], k [B, H, Sk, D]."""
    dims = op.get("args", {}).get("Input Dims", [])
    if len(dims) < 2 or len(dims[0]) != 4 or len(dims[1]) != 4:
        return None, None
    q, k = dims[0], dims[1]
    # _flash_attention_forward takes [B, S, H, D]; sdpa takes [B, H, S, D]
    if op["name"] == "aten::_flash_attention_forward":
        b, sq, h, d = q
        sk = k[1]
    else:
        b, h, sq, d = q
        sk = k[2]
    return 4 * b * h * sq * sk * d, f"B{b} H{h} Sq{sq} Sk{sk} D{d}"


def load_torch_trace(trace_path, stage_re):
    """Stage windows and GPU activities from a torch.profiler Chrome trace (run_flux.py --profile)."""
    events = json.loads(Path(trace_path).read_text())["traceEvents"]
    windows = sorted(
        (e["ts"], e["ts"] + e["dur"], e["name"])
        for e in events
        if e.get("cat") == "gpu_user_annotation" and re.fullmatch(stage_re, e["name"])
    )
    assert windows, f"no gpu_user_annotation matching {stage_re!r} in {trace_path}"
    starts = [w[0] for w in windows]
    chains = build_op_chains(events)

    stages = {name: {"span_ms": (e - s) / 1000.0, "acts": []} for s, e, name in windows}
    for e in events:
        if e.get("cat") not in GPU_CATS:
            continue
        i = bisect.bisect_right(starts, e["ts"]) - 1
        if i < 0 or e["ts"] > windows[i][1]:
            continue
        chain = chains.get(e.get("args", {}).get("correlation"), [])
        aten = [op for op in chain if op["name"].startswith("aten::")]
        top = aten[0]["name"] if aten else "(no aten op)"
        leaf_op = aten[-1] if aten else None
        leaf = leaf_op["name"] if leaf_op else "(no aten op)"
        stages[windows[i][2]]["acts"].append(
            {
                "name": e["name"],
                "cat": e["cat"],
                "ts": e["ts"],
                "dur_us": e["dur"],
                "top": top,
                "leaf": leaf,
                "leaf_op": leaf_op,
                "group": categorize(e["name"], leaf, e["cat"]),
            }
        )
    return stages


def load_nsys(path, stage_re, generation=-1):
    """Stage windows and GPU activities from an Nsight Systems report (run_sdcpp.py --nsys).

    Stage windows are the harness's NVTX ranges. They are CPU-side ranges, but each ends only
    after ggml's synchronous graph compute has finished, so a kernel belongs to the stage whose
    range contains its GPU start time. ggml kernels have no launching aten op, so "top" is the
    kernel group and "leaf" is the kernel's short name.

    The report covers every generation of the run (first, warm-up, measured), each wrapped in a
    `generate` range. Only one is analysed: `generation` indexes those ranges (default -1, the last,
    which is a measured run).
    """
    import sqlite3
    import subprocess

    path = Path(path)
    if path.suffix == ".nsys-rep":
        db = path.with_suffix(".sqlite")
        if not db.exists():
            subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite", "true", "-o", str(db), str(path)],
                           check=True, capture_output=True)
        path = db
    con = sqlite3.connect(str(path))
    tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    nvtx = con.execute(
        "SELECT n.start, n.end, COALESCE(n.text, s.value) FROM NVTX_EVENTS n "
        "LEFT JOIN StringIds s ON n.textId = s.id WHERE n.end IS NOT NULL").fetchall()
    gens = sorted((a / 1000.0, b / 1000.0) for a, b, name in nvtx if name == "generate")
    assert gens, f"no NVTX 'generate' ranges in {path}"
    g0, g1 = gens[generation]
    windows = sorted((a / 1000.0, b / 1000.0, name) for a, b, name in nvtx
                     if name and re.fullmatch(stage_re, name) and g0 <= a / 1000.0 <= g1)
    assert windows, f"no NVTX range matching {stage_re!r} in generation {generation} of {path}"
    assert len({w[2] for w in windows}) == len(windows), "duplicate stage names within one generation"
    starts = [w[0] for w in windows]
    stages = {name: {"span_ms": (e - s) / 1000.0, "acts": []} for s, e, name in windows}

    rows = [("kernel", a, b, short, full) for a, b, short, full in con.execute(
        "SELECT k.start, k.end, s.value, d.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
        "JOIN StringIds s ON k.shortName = s.id JOIN StringIds d ON k.demangledName = d.id")]
    for table, cat in (("CUPTI_ACTIVITY_KIND_MEMCPY", "gpu_memcpy"), ("CUPTI_ACTIVITY_KIND_MEMSET", "gpu_memset")):
        if table in tables:
            rows += [(cat, a, b, cat, cat) for a, b in con.execute(f"SELECT start, end FROM {table}")]
    for cat, a, b, short, full in rows:
        ts = a / 1000.0
        i = bisect.bisect_right(starts, ts) - 1
        if i < 0 or ts > windows[i][1]:
            continue
        group = categorize(full, None, cat)
        stages[windows[i][2]]["acts"].append(
            {"name": full, "cat": cat, "ts": ts, "dur_us": (b - a) / 1000.0,
             "top": group, "leaf": short, "leaf_op": None, "group": group})
    return stages


def analyze(trace_path, stage_re, generation=-1):
    trace_path = Path(trace_path)
    if trace_path.suffix in (".nsys-rep", ".sqlite"):
        stages = load_nsys(trace_path, stage_re, generation)
    else:
        stages = load_torch_trace(trace_path, stage_re)

    out = {"trace": str(trace_path), "stage_regex": stage_re, "stages": {}}
    if trace_path.suffix in (".nsys-rep", ".sqlite"):
        out["generation"] = generation
    for name, st in stages.items():
        acts = st["acts"]
        busy = union_ms([(a["ts"], a["ts"] + a["dur_us"]) for a in acts])
        kernel_sum = sum(a["dur_us"] for a in acts) / 1000.0
        durs = [a["dur_us"] for a in acts]

        def agg(key):
            d = collections.defaultdict(lambda: {"ms": 0.0, "kernels": 0})
            for a in acts:
                d[a[key]]["ms"] += a["dur_us"] / 1000.0
                d[a[key]]["kernels"] += 1
            return dict(sorted(d.items(), key=lambda kv: -kv[1]["ms"]))

        gemm = collections.defaultdict(lambda: {"ms": 0.0, "calls": set(), "flop_per_call": 0})
        attn = collections.defaultdict(lambda: {"ms": 0.0, "calls": set(), "flop_per_call": 0})
        for a in acts:
            op = a["leaf_op"]
            if op is None:
                continue
            if a["group"] == "gemm":
                mnk = gemm_mnk(op)
                if mnk:
                    g = gemm["M{} N{} K{}".format(*mnk)]
                    g["ms"] += a["dur_us"] / 1000.0
                    g["calls"].add(op["ts"])
                    g["flop_per_call"] = 2 * mnk[0] * mnk[1] * mnk[2]
            elif a["group"] == "attention":
                flops, label = attn_flops(op)
                if flops:
                    g = attn[label]
                    g["ms"] += a["dur_us"] / 1000.0
                    g["calls"].add(op["ts"])
                    g["flop_per_call"] = flops

        def finish(d):
            rows = {}
            for k, v in sorted(d.items(), key=lambda kv: -kv[1]["ms"]):
                calls = len(v["calls"])
                rows[k] = {
                    "ms": v["ms"],
                    "calls": calls,
                    "tflops": (v["flop_per_call"] * calls) / (v["ms"] / 1000.0) / 1e12 if v["ms"] else None,
                }
            return rows

        out["stages"][name] = {
            "span_ms": st["span_ms"],
            "busy_ms": busy,
            "idle_ms": st["span_ms"] - busy,
            "kernel_sum_ms": kernel_sum,
            "n_kernels": len(acts),
            "median_kernel_us": statistics.median(durs) if durs else None,
            "frac_kernels_under_10us": sum(d < 10 for d in durs) / len(durs) if durs else None,
            "time_in_kernels_under_10us_ms": sum(d for d in durs if d < 10) / 1000.0,
            "by_group": agg("group"),
            "by_top_op": agg("top"),
            "by_leaf_op": agg("leaf"),
            "gemm_shapes": finish(gemm),
            "attention_shapes": finish(attn),
        }
    return out


def mean_over(stages, path_fn):
    vals = [path_fn(s) for s in stages.values()]
    return statistics.fmean(vals)


def render_md(res, top_n=15):
    st = res["stages"]
    names = list(st)
    n = len(names)
    lines = [f"# Kernel breakdown: `{res['stage_regex']}`", "",
             f"Trace: `{res['trace']}`" + (f" (generation {res['generation']})" if "generation" in res else "")
             + ". Generated by `scripts/analyze_profile.py`; method in its docstring.",
             "Times are GPU kernel durations in ms; shares are of summed kernel time.", "",
             "## Per stage", "",
             "| Stage | GPU span | Busy | Idle | Kernels | Median kernel (µs) | Kernels < 10 µs |",
             "|---|---|---|---|---|---|---|"]
    for k in names:
        s = st[k]
        lines.append(f"| {k} | {s['span_ms']:.1f} | {s['busy_ms']:.1f} | {s['idle_ms']:.1f} | {s['n_kernels']} | "
                     f"{s['median_kernel_us']:.1f} | {100 * s['frac_kernels_under_10us']:.0f}% "
                     f"({s['time_in_kernels_under_10us_ms']:.1f} ms) |")

    def table(title, key):
        allk = collections.OrderedDict()
        for k in names:
            for g in st[k][key]:
                allk.setdefault(g, None)
        total = mean_over(st, lambda s: s["kernel_sum_ms"])
        rows = []
        for g in allk:
            per = [st[k][key].get(g, {"ms": 0.0, "kernels": 0}) for k in names]
            ms = statistics.fmean(p["ms"] for p in per)
            kern = statistics.fmean(p["kernels"] for p in per)
            rows.append((ms, g, kern, [p["ms"] for p in per]))
        rows.sort(reverse=True)
        out = ["", f"## {title} (mean per stage over {n} stages)", "",
               "| Group | ms / stage | Share | Kernels / stage | " + " | ".join(names) + " |",
               "|---|---|---|---|" + "---|" * n]
        for ms, g, kern, per in rows[:top_n]:
            out.append(f"| `{g}` | {ms:.2f} | {100 * ms / total:.1f}% | {kern:.0f} | "
                       + " | ".join(f"{p:.2f}" for p in per) + " |")
        if len(rows) > top_n:
            rest = sum(r[0] for r in rows[top_n:])
            out.append(f"| (other {len(rows) - top_n}) | {rest:.2f} | {100 * rest / total:.1f}% | | " + " | " * (n - 1) + " |")
        return out

    lines += table("By kernel group", "by_group")
    lines += table("By top-level aten op", "by_top_op")
    lines += table("By leaf aten op (the op that launched the kernel)", "by_leaf_op")

    for title, key in (("GEMM shapes", "gemm_shapes"), ("Attention shapes", "attention_shapes")):
        first = st[names[0]][key]
        if not first:
            continue
        lines += ["", f"## {title} (first stage: `{names[0]}`)", "",
                  "| Shape | ms | Calls | Achieved TFLOP/s |", "|---|---|---|---|"]
        for shape, v in list(first.items())[:top_n]:
            tf = f"{v['tflops']:.0f}" if v["tflops"] else "–"
            lines.append(f"| {shape} | {v['ms']:.2f} | {v['calls']} | {tf} |")
    return "\n".join(lines) + "\n"


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
