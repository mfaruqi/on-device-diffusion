import json
import tempfile
import unittest
from pathlib import Path

from shiftadd_flux.case_validation import validate_case_sequence


class CaseValidationTest(unittest.TestCase):
    def test_nested_cases_and_reject_stale_weights(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "status.json").write_text(json.dumps({"status": "completed"}))
            (root / "summary.json").write_text(
                json.dumps({"results": [{"tensor": n, "planes": 4} for n in ("a", "b")]})
            )
            bf = {"name": "bf16", "artifacts": None}
            first = {
                "name": "one",
                "artifacts": str(root),
                "planes": 4,
                "tensors": ["a"],
            }
            full = dict(first, name="all", tensors=None)
            validate_case_sequence({"cases": [bf, first, full]})
            for cases in (
                [bf, full, first],
                [bf, first, dict(first, name="other", tensors=["b"])],
                [bf, first, dict(bf, name="back")],
                [bf, first, first],
            ):
                with self.assertRaises(ValueError):
                    validate_case_sequence({"cases": cases})
            with self.assertRaises(ValueError):
                validate_case_sequence({"cases": [dict(bf, activation_quantization={"bits": 4})]})
