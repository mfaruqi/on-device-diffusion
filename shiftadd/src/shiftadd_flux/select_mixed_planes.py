"""Select a fixed number of smaller fits using fitting-call error only.

This produces a candidate, not a quality-approved plan. The diagnostic proxy
averages independently measured projection errors and ignores error propagation.
"""

import argparse
import json
import math
import platform
import statistics
import traceback
from pathlib import Path

from .assemble_upstream import assemble
from .package_upstream import digest, write_json


def select(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config)
    write_json(output / "status.json", {"status": "running"})
    try:
        fits, summaries, inventories = {}, {}, {}
        for label in ("higher", "lower"):
            root = Path(config[label + "_fit"])
            if json.loads((root / "status.json").read_text())["status"] != "completed":
                raise ValueError("Require completed fits")
            verification = json.loads((root / "reload-verification.json").read_text())
            if (
                not verification["all_metrics_reproduced"]
                or not verification["all_artifact_hashes_match"]
            ):
                raise ValueError("Require independent source verification")
            fits[label] = json.loads((root / "config.json").read_text())
            summaries[label] = json.loads((root / "summary.json").read_text())
            inventories[label] = {r["module"]: r for r in summaries[label]["storage"]}
        keys = (
            "source",
            "source_revision",
            "capture",
            "fit_calls",
            "group_size",
            "scale_format",
            "damping",
            "fit_iters",
            "row_policy",
        )
        if any(fits["higher"][key] != fits["lower"][key] for key in keys):
            raise ValueError("Fit settings differ beyond plane count")
        modules = sorted(inventories["higher"])
        if set(modules) != set(inventories["lower"]) or not modules:
            raise ValueError("Source module coverage differs")
        if not fits["lower"]["planes"] < fits["higher"]["planes"]:
            raise ValueError("Lower fit must use fewer planes")
        count = config["lower_count"]
        if type(count) is not int or not 0 < count < len(modules):
            raise ValueError("Require an interior mixed allocation")
        savings = {
            inventories["higher"][m]["file_bytes"] - inventories["lower"][m]["file_bytes"]
            for m in modules
        }
        if len(savings) != 1 or min(savings) <= 0:
            raise ValueError(
                "This fixed-count selector requires equal positive per-matrix byte savings"
            )
        means = {}
        for label in ("higher", "lower"):
            for module in modules:
                for split in ("fit", "calibration_validation"):
                    values = [
                        r["relative_output_l2"]
                        for r in summaries[label]["results"]
                        if r["module"] == module and r["split"] == split
                    ]
                    if len(values) != 8 or not all(math.isfinite(v) for v in values):
                        raise ValueError("Require eight finite errors per matrix and split")
                    means[label, module, split] = statistics.mean(values)
        ranked = sorted(
            modules, key=lambda m: (means["lower", m, "fit"] - means["higher", m, "fit"], m)
        )
        chosen = set(ranked[:count])
        curve = []
        for k in range(len(modules) + 1):
            smaller = set(ranked[:k])
            row = {
                "lower_count": k,
                "file_bytes_for_variable_matrices": sum(
                    inventories["lower" if m in smaller else "higher"][m]["file_bytes"]
                    for m in modules
                ),
            }
            for split in ("fit", "calibration_validation"):
                row[split + "_mean_projection_error_proxy"] = statistics.mean(
                    means["lower" if m in smaller else "higher", m, split] for m in modules
                )
            curve.append(row)
        assembly = {
            "id": config["id"] + "-assembly",
            "output": config["bundle_output"],
            "description": "Fixed-count mixed-plane candidate selected by fitting-call projection error; no image-quality admission.",
            "selections": list(config.get("fixed_selections", []))
            + [
                {"artifacts": config["lower_fit"], "modules": sorted(chosen)},
                {"artifacts": config["higher_fit"], "modules": sorted(set(modules) - chosen)},
            ],
        }
        write_json(output / "assembly-config.json", assembly)
        assembled = assemble(assembly)
        summary = {
            "scope": "mixed-plane candidate; local projection-error proxy only, no pipeline quality or performance measurement",
            "selection_split": "fit",
            "evaluation_split": "calibration_validation",
            "selection_rule": "choose fixed count with smallest increase in mean fitting-call relative output L2; ties by module name; equal byte savings required",
            "lower_modules": sorted(chosen),
            "ranked_modules": ranked,
            "bytes_saved_per_lower_matrix": next(iter(savings)),
            "candidate_proxy": curve[count],
            "allocation_curve": curve,
            "assembly": assembled,
            "source_summary_sha256": {
                label: digest(Path(config[label + "_fit"]) / "summary.json")
                for label in ("higher", "lower")
            },
        }
        write_json(output / "summary.json", summary)
        write_json(output / "environment.json", {"platform": platform.platform()})
        write_json(output / "status.json", {"status": "completed"})
        return summary
    except BaseException:
        write_json(
            output / "status.json", {"status": "failed", "traceback": traceback.format_exc()}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = select(json.loads(Path(args.config).read_text()))
    print(
        json.dumps(
            {"lower_modules": result["lower_modules"], "assembly": result["assembly"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
