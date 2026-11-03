"""Assemble verified full-row BCQ fits without refitting or changing values."""

import argparse
import copy
import json
import shutil
import traceback
from pathlib import Path

import numpy as np

from .package_upstream import digest, write_json


def assemble(config):
    output = Path(config["output"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "assembly-config.json", config)
    write_json(output / "status.json", {"status": "running"})
    try:
        common_keys = (
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
        base, selected, storage, results, provenance = None, {}, [], [], []
        for selection in config["selections"]:
            source = Path(selection["artifacts"])
            fit = json.loads((source / "config.json").read_text())
            summary = json.loads((source / "summary.json").read_text())
            status = json.loads((source / "status.json").read_text())
            verified = json.loads((source / "reload-verification.json").read_text())
            if status["status"] != "completed" or not verified["all_metrics_reproduced"]:
                raise ValueError("Require completed and independently reloaded fits")
            if not verified["all_artifact_hashes_match"] or fit["row_policy"] != "all":
                raise ValueError("Require verified full-row artifacts")
            if (
                fit["planes"] not in (3, 4, 5)
                or fit["group_size"] != 128
                or fit["scale_format"] not in ("u8", "pot")
            ):
                raise ValueError("Unsupported source fit format")
            if base is None:
                base = copy.deepcopy(fit)
            elif any(base[key] != fit[key] for key in common_keys):
                raise ValueError("Source fits differ in more than plane count or selection")
            inventory = {row["module"]: row for row in summary["storage"]}
            if len(inventory) != len(summary["storage"]) or set(inventory) != set(fit["modules"]):
                raise ValueError("Source inventory disagrees with its fit configuration")
            if verified["verified_output_comparisons"] != len(summary["results"]):
                raise ValueError("Incomplete source reload verification")
            expected_calls = {(module, call) for module in inventory for call in range(16)}
            actual_calls = {(row["module"], row["call"]) for row in summary["results"]}
            if actual_calls != expected_calls or len(actual_calls) != len(summary["results"]):
                raise ValueError("Source fit must cover all 16 calls per module exactly once")
            if not selection["modules"]:
                raise ValueError("Empty source selection")
            for module in selection["modules"]:
                if module in selected or module not in inventory:
                    raise ValueError("Duplicate or missing selected module")
                filename = module + ".npz"
                if Path(filename).name != filename:
                    raise ValueError("Module names must produce relative basenames")
                row = inventory[module]
                path = source / filename
                if (
                    digest(path) != row["artifact_sha256"]
                    or path.stat().st_size != row["file_bytes"]
                ):
                    raise ValueError("Source artifact integrity mismatch")
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(
                        data["output_row_indices"], np.arange(row["output_rows"])
                    ):
                        raise ValueError("Source artifact has partial or reordered rows")
                    shape = data["planes"].shape
                    if (
                        shape[:2] != (fit["planes"], row["output_rows"])
                        or shape[2] * shape[3] * 8 != row["channels"]
                    ):
                        raise ValueError("Source artifact shape disagrees with its fit")
                shutil.copyfile(path, output / filename)
                if digest(output / filename) != row["artifact_sha256"]:
                    raise ValueError("Copied artifact hash mismatch")
                selected[module] = fit["planes"]
                storage.append(row)
                results.extend(r for r in summary["results"] if r["module"] == module)
                provenance.append(
                    {
                        "module": module,
                        "source": str(source),
                        "planes": fit["planes"],
                        "artifact_sha256": row["artifact_sha256"],
                        "fit_config_sha256": digest(source / "config.json"),
                        "fit_summary_sha256": digest(source / "summary.json"),
                    }
                )
        if not selected:
            raise ValueError("Empty assembly")
        planes = set(selected.values())
        base.update(
            id=config["id"],
            output=str(output),
            modules=list(selected),
            planes=next(iter(planes)) if len(planes) == 1 else None,
            planes_by_module=selected,
            description=config["description"],
            artifact_kind="loader_assembly_no_refitting",
        )
        write_json(output / "config.json", base)
        write_json(
            output / "summary.json",
            {
                "scope": "loader assembly of verified fits; no new fitting or image evaluation",
                "storage": storage,
                "results": results,
                "total_file_bytes": sum(row["file_bytes"] for row in storage),
            },
        )
        write_json(output / "assembly-provenance.json", provenance)
        write_json(output / "status.json", {"status": "completed"})
        return {
            "modules": len(selected),
            "packed_file_bytes": sum(row["file_bytes"] for row in storage),
        }
    except BaseException:
        write_json(
            output / "status.json", {"status": "failed", "traceback": traceback.format_exc()}
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(assemble(json.loads(Path(args.config).read_text())), indent=2))


if __name__ == "__main__":
    main()
