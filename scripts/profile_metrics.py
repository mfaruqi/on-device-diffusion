"""Pure arithmetic for GPU profile reports. Inputs use microseconds; outputs use ms.

Busy time is the union of captured GPU activity intervals. Its complement does
not identify the cause of inactivity (for example disk reads versus host work).
"""

import collections
import re
import statistics


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


def summarize_stage(stage):
    """Aggregate one stage without reading files or querying hardware."""
    acts = stage["acts"]
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

    return {
        "span_ms": stage["span_ms"],
        "busy_ms": busy,
        "idle_ms": stage["span_ms"] - busy,
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
