from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path
from E_training import training_reporting

class TrainingReportingTests(unittest.TestCase):
    # Verify reporting profile normalization and legacy switch.
    def test_reporting_profile_normalization_and_legacy_switch(self) -> None:
        self.assertEqual(training_reporting.normalize_reporting_profile(None), "compact")
        self.assertEqual(training_reporting.normalize_reporting_profile(" COMPACT "), "compact")
        self.assertEqual(training_reporting.normalize_reporting_profile("debug"), "debug")
        self.assertFalse(training_reporting.should_write_legacy_reports("compact"))
        self.assertTrue(training_reporting.should_write_legacy_reports("debug"))

    # Verify invalid reporting profile is rejected.
    def test_invalid_reporting_profile_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            training_reporting.normalize_reporting_profile("verbose")

    # Verify save JSON uses stable sorted format.
    def test_save_json_uses_stable_sorted_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "payload.json"
            training_reporting.save_json(path, {"b": 2, "a": {"z": 1}})

            self.assertEqual(path.read_text(encoding="utf-8"), '{\n  "a": {\n    "z": 1\n  },\n  "b": 2\n}')
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["a"]["z"], 1)

    # Verify the build run report preserves nested metric blocks.
    def test_build_run_report_preserves_nested_metric_blocks(self) -> None:
        report = training_reporting.build_run_report(
            script_id="E_05",
            config={"dataset": "joint", "run_name": "iid_6banks", "seed": 42},
            metrics={"test": {"outcome": {"macro_f1": 0.4}}},
            diagnostics={"target": {"remaining_time": {"scaling": "zscore"}}},
            artifacts={"files": ["E_05_model_best.pt"]},
        )

        self.assertEqual(report["schema_version"], 1)
        self.assertEqual(report["script_id"], "E_05")
        self.assertEqual(report["run_identity"]["dataset"], "joint")
        self.assertEqual(report["metrics"]["test"]["outcome"]["macro_f1"], 0.4)
        self.assertEqual(report["diagnostics"]["target"]["remaining_time"]["scaling"], "zscore")
        self.assertIn("created_at", report)
        self.assertIn("timing", report)

    # Verify artifact manifest lists relative files.
    def test_artifact_manifest_lists_relative_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "predictions").mkdir()
            (root / "E_05_run_report.json").write_text("{}", encoding="utf-8")
            (root / "predictions" / "E_05_train_log.csv").write_text("epoch\n1\n", encoding="utf-8")

            manifest = training_reporting.build_artifact_manifest(root)

            self.assertEqual(manifest["file_count"], 2)
            self.assertEqual(
                manifest["files"],
                ["E_05_run_report.json", "predictions/E_05_train_log.csv"],
            )

    # Verify run report carries a populated environment block.
    def test_run_report_carries_a_populated_environment_block(self) -> None:
        report = training_reporting.build_run_report(
            script_id="E_05", config={"dataset": "bpic2017", "resolved_device": "mps"},
            metrics={}, diagnostics={}, artifacts={},
        )
        environment = report["environment"]

        self.assertEqual(environment["resolved_device"], "mps")
        self.assertTrue(environment["python_version"])
        self.assertTrue(environment["platform"])
        self.assertTrue(environment["machine"])
        for package in ("torch", "numpy", "pandas", "scikit-learn"):
            self.assertRegex(environment["packages"][package], r"^\d+\.\d+")

        # Opacus only belongs to the report when the run activates the DP path.
        self.assertNotIn("opacus", environment["packages"])
        dp_report = training_reporting.build_run_report(
            script_id="E_06", config={"use_dp": True}, metrics={}, diagnostics={}, artifacts={},
        )
        self.assertRegex(dp_report["environment"]["packages"]["opacus"], r"^\d+\.\d+")

    # Verify the environment snapshot is written beside the output root.
    def test_environment_snapshot_is_written_beside_the_output_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = training_reporting.write_environment_snapshot(Path(tmp))
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(path.name, "E_environment.json")
        self.assertIn("created_at", payload)
        self.assertRegex(payload["packages"]["torch"], r"^\d+\.\d+")

    # Verify write run report uses script specific filename.
    def test_write_run_report_uses_script_specific_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            path = training_reporting.write_run_report(output_dir, "E_06", {"schema_version": 1})

            self.assertEqual(path, output_dir / "E_06_run_report.json")
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["schema_version"], 1)

if __name__ == "__main__":
    unittest.main()