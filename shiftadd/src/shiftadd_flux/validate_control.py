"""Validate paired control evidence; this is not a quality acceptance gate."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def validate(run, prior, bundles):
    def read(root, name):
        return json.loads((root / name).read_text())

    if read(run, "status.json")["status"] != "completed":
        raise ValueError("Run incomplete")
    config = read(run, "config.json")
    reference_config = read(prior, "config.json")
    for key in ("model_path", "seed", "prompt", "height", "width"):
        if config[key] != reference_config[key]:
            raise ValueError(f"Unmatched workload: {key}")
    for key in ("torch", "device"):
        if read(run, "environment.json")[key] != read(prior, "environment.json")[key]:
            raise ValueError(f"Unmatched runtime: {key}")
    rows = read(run, "summary.json")["results"]
    cases = config["cases"]
    if (
        [r["case"] for r in rows] != [c["name"] for c in cases]
        or cases[0]["name"] != "bf16"
        or cases[0]["artifacts"] is not None
    ):
        raise ValueError("Missing/reordered cases or invalid BF16 reference")
    baseline = np.load(run / "bf16-latent.npy", allow_pickle=False)
    old = np.load(prior / "bf16-latent.npy", allow_pickle=False)
    if not np.array_equal(baseline, old):
        raise ValueError("BF16 reference differs from earlier run")
    report = []
    for case, row in zip(cases, rows):
        name = case["name"]
        array = np.load(run / (name + "-latent.npy"), allow_pickle=False)
        if array.shape != baseline.shape or not np.isfinite(array).all() or not row["finite"]:
            raise ValueError("Invalid latent")
        relative = float(np.linalg.norm(array - baseline) / np.linalg.norm(baseline))
        if not np.isclose(relative, row["relative_latent_l2"], rtol=1e-6, atol=1e-8):
            raise ValueError("Recorded latent metric disagrees")
        with Image.open(run / (name + ".png")) as image:
            image.load()
            if image.size != (config["width"], config["height"]):
                raise ValueError("Wrong image dimensions")
        count = 0
        activation = case.get("activation_quantization")
        if activation and activation.get("expected_calls_per_module") is not None:
            activation_audit = read(run, name + "-activation-audit.json")
            expected = {
                module: activation["expected_calls_per_module"] for module in activation["modules"]
            }
            if activation_audit.get("calls_by_module") != expected or activation_audit[
                "calls"
            ] != sum(expected.values()):
                raise ValueError("Incomplete activation coverage")
            if activation_audit["bits"] != activation["bits"] or activation_audit.get(
                "group_size"
            ) != activation.get("group_size"):
                raise ValueError("Activation policy audit mismatch")
        if case["artifacts"] is not None:
            local = bundles / Path(case["artifacts"]).name
            manifest = read(local, "summary.json")
            expected = (
                {r["module"] + ".weight" for r in manifest["storage"]}
                if case.get("weight_format") == "upstream_bcq"
                else {r["tensor"] for r in manifest["results"]}
            )
            if case.get("tensors") is not None:
                requested = set(case["tensors"])
                if not requested or not requested.issubset(expected):
                    raise ValueError("Invalid requested tensor selection")
                expected = requested
            replaced = read(run, name + "-audit.json")["replaced_tensors"]
            if set(replaced) != expected or len(replaced) != len(expected) or not expected:
                raise ValueError("Incomplete replacement audit")
            count = len(replaced)
        report.append(
            {
                "case": name,
                "replaced_matrices": count,
                "relative_latent_l2": relative,
                "latent_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
        )
    return {
        "status": "passed",
        "scope": "artifact and paired numerical validation; not quality admission",
        "prior_bf16_exact_match": True,
        "cases": report,
        "performance_comparison_eligible": False,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--prior", required=True)
    p.add_argument("--bundles", default="shiftadd/artifacts")
    a = p.parse_args()
    report = validate(Path(a.run), Path(a.prior), Path(a.bundles))
    with (Path(a.run) / "control-validation.json").open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
