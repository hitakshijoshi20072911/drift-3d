import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from driftx.ablation import summarize_reports
from driftx.benchmark.runner import _validate_large_model
from driftx.doctor import run_doctor
from driftx.models import detect_model_capabilities, require_gaussian_capability


class SystemContractTests(unittest.TestCase):
    def test_large_is_not_gaussian_capable(self):
        capability = detect_model_capabilities("DA3_LARGE_1.1_SAFE")
        self.assertTrue(capability.supports_depth)
        self.assertTrue(capability.supports_pose)
        self.assertFalse(capability.supports_gaussian)
        with self.assertRaisesRegex(RuntimeError, "does not expose Gaussian"):
            require_gaussian_capability("DA3_LARGE_1.1_SAFE")

    def test_giant_config_is_gaussian_capable(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "da3-giant"
            path.mkdir()
            (path / "config.json").write_text(json.dumps({"gs_head": {"name": "GSDPT"}}))
            self.assertTrue(detect_model_capabilities(path).supports_gaussian)

    def test_giant_is_allowed_only_for_gaussian_modes(self):
        with self.assertRaises(ValueError):
            _validate_large_model("DA3-GIANT-1.1")
        _validate_large_model("DA3-GIANT-1.1", allow_gaussian=True)

    def test_doctor_persists_json_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = run_doctor(output=Path(tmp) / "doctor.json")
            self.assertIn(report["status"], {"PASS", "WARN", "FAIL"})
            self.assertTrue((Path(tmp) / "doctor.json").is_file())
            self.assertEqual(json.loads((Path(tmp) / "doctor.json").read_text())["schema_version"], 1)

    def test_ablation_summary_marks_ground_truth_metrics_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "run" / "run_report.json"
            report.parent.mkdir()
            report.write_text(json.dumps({"status": "completed", "total_runtime_seconds": 2.0}))
            output = root / "summary"
            result = summarize_reports([report], output)
            self.assertEqual(result["status"], "completed")
            row = result["runs"][0]
            self.assertEqual(row["coverage"], "unavailable without reference geometry")
            self.assertTrue((output / "ablation_summary.csv").is_file())


if __name__ == "__main__":
    unittest.main()
