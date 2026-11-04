"""Reject paired-case schedules that would retain unintended prior weights."""

import json
from pathlib import Path


def validate_case_sequence(config, resolve=Path):
    cases = config["cases"]
    if (
        not cases
        or cases[0]["name"] != "bf16"
        or cases[0]["artifacts"] is not None
        or cases[0].get("activation_quantization")
    ):
        raise ValueError("First case must be the unmodified BF16 reference")
    names = [c["name"] for c in cases]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate case names overwrite evidence")
    samples = config.get("samples", [])
    if len({s["id"] for s in samples}) != len(samples):
        raise ValueError("Duplicate calibration sample IDs")
    previous = set()
    audit = []
    for case in cases:
        if case["artifacts"] is None:
            if previous:
                raise ValueError("Returning to BF16 requires an explicit model reload")
            current = set()
        else:
            root = resolve(case["artifacts"])
            if json.loads((root / "status.json").read_text())["status"] != "completed":
                raise ValueError("Incomplete artifact bundle")
            summary = json.loads((root / "summary.json").read_text())
            fmt = case.get("weight_format", "sign_planes")
            rows = (
                [{"tensor": r["module"] + ".weight"} for r in summary["storage"]]
                if fmt == "upstream_bcq"
                else summary["results"]
            )
            if fmt == "sign_planes":
                planes = case["planes"]
                rows = [r for r in rows if planes is None or r["planes"] == planes]
            elif fmt == "symmetric_uniform":
                if summary.get("format") != "symmetric_uniform":
                    raise ValueError("Weight format mismatch")
            elif fmt == "upstream_bcq":
                pass
            else:
                raise ValueError("Unknown weight format")
            requested = case.get("tensors")
            if requested is not None:
                rows = [r for r in rows if r["tensor"] in requested]
            current = {r["tensor"] for r in rows}
            if (
                not current
                or len(current) != len(rows)
                or (requested is not None and current != set(requested))
            ):
                raise ValueError("Missing or ambiguous tensor selection")
            if not previous.issubset(current):
                raise ValueError("Non-nested tensor replacements require an explicit model reload")
        previous = current
        audit.append({"case": case["name"], "selected_tensors": sorted(current)})
    return {
        "case_sequence_valid": True,
        "rule": "every candidate overwrites all previously modified parameters",
        "cases": audit,
    }
