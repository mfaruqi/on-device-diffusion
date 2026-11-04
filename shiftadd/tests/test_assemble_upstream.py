import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from shiftadd_flux.assemble_upstream import assemble
from shiftadd_flux.package_upstream import digest, write_json


class AssemblyTests(unittest.TestCase):
    def test_mixed_planes_and_fit_mismatch_guard(self):
        root = Path(tempfile.mkdtemp(prefix="shiftadd-assembly-test-"))
        selections = []
        for module, planes in [("a", 1), ("b", 2), ("c", 3), ("d", 4)]:
            directory = root / module
            directory.mkdir()
            path = directory / (module + ".npz")
            np.savez(
                path,
                planes=np.zeros((planes, 2, 2, 16), dtype=np.uint8),
                output_row_indices=np.arange(2),
            )
            fit = dict(
                source="synthetic",
                source_revision="test",
                capture="same-capture",
                fit_calls=8,
                group_size=128,
                scale_format="u8",
                damping=0.01,
                fit_iters=15,
                row_policy="all",
                planes=planes,
                modules=[module],
            )
            write_json(directory / "config.json", fit)
            write_json(
                directory / "summary.json",
                {
                    "storage": [
                        {
                            "module": module,
                            "output_rows": 2,
                            "channels": 256,
                            "artifact_sha256": digest(path),
                            "file_bytes": path.stat().st_size,
                        }
                    ],
                    "results": [{"module": module, "call": call} for call in range(16)],
                },
            )
            write_json(directory / "status.json", {"status": "completed"})
            write_json(
                directory / "reload-verification.json",
                {
                    "all_metrics_reproduced": True,
                    "all_artifact_hashes_match": True,
                    "verified_output_comparisons": 16,
                },
            )
            selections.append({"artifacts": str(directory), "modules": [module]})
        config = dict(
            id="test",
            output=str(root / "assembled"),
            description="Synthetic mixed-plane assembly",
            selections=selections,
        )
        result = assemble(config)
        self.assertEqual(result["modules"], 4)
        saved = json.loads((root / "assembled/config.json").read_text())
        self.assertEqual(saved["planes_by_module"], {"a": 1, "b": 2, "c": 3, "d": 4})
        self.assertIsNone(saved["planes"])
        for module in ["a", "b", "c", "d"]:
            self.assertEqual(
                (root / "assembled" / (module + ".npz")).read_bytes(),
                (root / module / (module + ".npz")).read_bytes(),
            )
        with self.assertRaises(FileExistsError):
            assemble(config)
        changed = root / "b/config.json"
        fit = json.loads(changed.read_text())
        fit["capture"] = "different-capture"
        write_json(changed, fit)
        config["output"] = str(root / "mismatch")
        with self.assertRaisesRegex(ValueError, "differ"):
            assemble(config)
        self.assertEqual(
            json.loads((root / "mismatch/status.json").read_text())["status"], "failed"
        )


if __name__ == "__main__":
    unittest.main()
