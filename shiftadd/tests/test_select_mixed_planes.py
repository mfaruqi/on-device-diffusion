import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shiftadd_flux.package_upstream import write_json
from shiftadd_flux.select_mixed_planes import select


class SelectionTests(unittest.TestCase):
    def test_validation_errors_do_not_select_modules(self):
        root = Path(tempfile.mkdtemp(prefix="shiftadd-selection-test-"))
        for label, planes, size in [("higher", 4, 100), ("lower", 3, 80)]:
            directory = root / label
            directory.mkdir()
            write_json(directory / "status.json", {"status": "completed"})
            write_json(
                directory / "reload-verification.json",
                {
                    "all_metrics_reproduced": True,
                    "all_artifact_hashes_match": True,
                },
            )
            write_json(
                directory / "config.json",
                dict(
                    source="synthetic",
                    source_revision="test",
                    capture="capture",
                    fit_calls=8,
                    group_size=128,
                    scale_format="u8",
                    damping=0.01,
                    fit_iters=15,
                    row_policy="all",
                    planes=planes,
                ),
            )
            rows = []
            for module in ["a", "b"]:
                for call in range(16):
                    split = "fit" if call < 8 else "calibration_validation"
                    error = 1.0
                    if label == "lower":
                        error = (
                            (1.1 if module == "a" else 1.2)
                            if split == "fit"
                            else (100.0 if module == "a" else 1.0)
                        )
                    rows.append(
                        dict(module=module, call=call, split=split, relative_output_l2=error)
                    )
            write_json(
                directory / "summary.json",
                {
                    "storage": [{"module": m, "file_bytes": size} for m in ["a", "b"]],
                    "results": rows,
                },
            )
        config = dict(
            id="test",
            higher_fit=str(root / "higher"),
            lower_fit=str(root / "lower"),
            lower_count=1,
            output=str(root / "report"),
            bundle_output=str(root / "bundle"),
        )
        with patch(
            "shiftadd_flux.select_mixed_planes.assemble", return_value={"modules": 2}
        ) as assembly:
            result = select(config)
        self.assertEqual(result["lower_modules"], ["a"])
        self.assertEqual(
            result["candidate_proxy"]["calibration_validation_mean_projection_error_proxy"], 50.5
        )
        self.assertEqual(result["candidate_proxy"]["file_bytes_for_variable_matrices"], 180)
        self.assertEqual(assembly.call_args.args[0]["selections"][0]["modules"], ["a"])


if __name__ == "__main__":
    unittest.main()
