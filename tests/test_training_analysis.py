from __future__ import annotations
import csv
import json
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from E_training import E_07_generate_training_analysis as analysis
from E_training import training_reporting

# HELPER: Build metric block.
def _metric_block(macro_f1: float = 0.5, next_top1: float = 0.4, rt_mae: float = 100.0) -> dict[str, object]:
    return {
        "n_prefixes": 10,
        "outcome": {
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1 + 0.1,
            "balanced_accuracy": macro_f1 - 0.1,
        },
        "next_activity": {"top1_accuracy": next_top1},
        "remaining_time": {"mae": rt_mae, "rmse": rt_mae + 25.0},
    }

# HELPER: Build the federated report.
def _federated_report(dataset: str, run_name: str, strategy: str, macro_f1: float, rt_mae: float, use_dp: bool = False,
        ) -> dict[str, object]:
    config = {
        "dataset": dataset,
        "run_name": run_name,
        "heterogeneity": run_name.rsplit("_", 1)[0],
        "n_clients": int(run_name.rsplit("_", 1)[1].replace("banks", "")),
        "strategy": strategy,
        "seed": 42,
        "learning_rate": 2.5e-4,
        "max_rounds": 1,
        "local_epochs": 1,
        "fedprox_mu": 1e-4,
        "use_dp": use_dp,
        "dp_target_epsilon": 10.0 if use_dp else "",
        "dp_delta": 1e-5 if use_dp else "",
        "dp_max_grad_norm": 1.0 if use_dp else "",
    }
    diagnostics = {"best_round": {"best_round": 1, "best_validation_loss": 1.0}}
    if use_dp: diagnostics["best_round"]["dp_epsilon_spent"] = 9.5
    return training_reporting.build_run_report(
        "E_06",
        config,
        {
            "validation": _metric_block(macro_f1 - 0.05, 0.4, rt_mae + 10.0),
            "test": _metric_block(macro_f1, 0.42, rt_mae),
            "per_bank_test": {
                "bpic2017:A": _metric_block(macro_f1 + 0.03, 0.5, rt_mae - 5.0),
                "bpic2012:A": _metric_block(macro_f1 - 0.04, 0.25, rt_mae + 20.0),
            },
        },
        diagnostics,
        {"files": ["E_06_run_report.json", "E_06_round_log.csv"]},
    )

class TrainingAnalysisTests(unittest.TestCase):
    # Verify training analysis reads compact and legacy outputs.
    def test_training_analysis_reads_compact_and_legacy_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            analysis_root = root / "analysis"
            compact_dir = root / "baselines" / "joint_iid_6banks" / "centralized_seed_42_lr_0p00025"
            compact_dir.mkdir(parents=True)
            compact_report = training_reporting.build_run_report(
                "E_05",
                {
                    "dataset": "joint",
                    "run_name": "iid_6banks",
                    "heterogeneity": "iid",
                    "n_clients": 6,
                    "regime": "centralized",
                    "seed": 42,
                    "learning_rate": 2.5e-4,
                    "max_epochs": 1,
                },
                {
                    "validation": _metric_block(),
                    "test": _metric_block(0.6, 0.5, 80.0),
                    "per_bank_test": {
                        "bpic2017:A": _metric_block(0.55, 0.5, 90.0),
                        "bpic2012:A": _metric_block(0.45, 0.3, 110.0),
                    },
                },
                {"best_epoch": {"best_total_validation_loss": {"epoch": 1}}},
                {"files": ["E_05_run_report.json"]},
            )
            training_reporting.write_run_report(compact_dir, "E_05", compact_report)

            fedavg_dir = root / "federated" / "joint_iid_6banks" / "fedavg_seed_42_lr_0p00025_rounds_1_le_1"
            fedavg_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                fedavg_dir, "E_06", _federated_report("joint", "iid_6banks", "fedavg", 0.52, 95.0)
            )
            (fedavg_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            fedprox_dir = root / "federated" / "joint_iid_6banks" / "fedprox_seed_42_lr_0p00025_rounds_1_le_1_mu_0p0001"
            fedprox_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                fedprox_dir, "E_06", _federated_report("joint", "iid_6banks", "fedprox", 0.56, 90.0)
            )
            (fedprox_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            dp_dir = (root / "differential_privacy" / "bpic2017_iid_3banks"
                      / "fedavg_seed_42_lr_0p00025_rounds_1_le_1_dp_eps_10_dp_batches_full")
            dp_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                dp_dir, "E_06", _federated_report("bpic2017", "iid_3banks", "fedavg", 0.38, 125.0, use_dp=True)
            )
            (dp_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            legacy_dir = root / "federated" / "bpic2017_iid_3banks" / "fedavg_seed_42_lr_0p00025_rounds_1_le_1"
            legacy_dir.mkdir(parents=True)
            training_reporting.save_json(legacy_dir / "E_06_test_metrics.json", _metric_block(0.4, 0.2, 120.0))
            training_reporting.save_json(legacy_dir / "E_06_validation_metrics.json", _metric_block(0.35, 0.2, 130.0))
            training_reporting.save_json(legacy_dir / "E_06_per_bank_test_metrics.json",
                                         {"A": _metric_block(0.4, 0.2, 120.0)})
            (legacy_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            rc = analysis.main(["--output-root", str(root), "--analysis-root", str(analysis_root)])

            self.assertEqual(rc, 0)
            markdown = (analysis_root / "E_07_training_analysis.md").read_text(encoding="utf-8")
            payload = json.loads((analysis_root / "E_07_training_analysis.json").read_text(encoding="utf-8"))
            with (analysis_root / "E_07_run_summary.csv").open(encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            with (analysis_root / "E_07_per_bank_summary.csv").open(encoding="utf-8") as handle:
                bank_rows = list(csv.DictReader(handle))

            self.assertIn("## Run inventory", markdown)
            self.assertIn("## Artifact index", markdown)
            self.assertIn("## Differential privacy profile", markdown)
            # Every GFM table must have a blank line before its header row so it renders as a table.
            md_lines = markdown.split("\n")
            table_headers = [
                index for index, line in enumerate(md_lines)
                if line.startswith("|") and index + 1 < len(md_lines)
                and set(md_lines[index + 1].replace("|", "").strip()) <= set("-: ") and "-" in md_lines[index + 1]
            ]
            self.assertTrue(table_headers, "expected at least one markdown table")
            for index in table_headers:
                self.assertEqual(md_lines[index - 1].strip(), "",
                                 f"table header at line {index + 1} needs a blank line before it")
            self.assertIn("Dataset balance for joint runs", markdown)
            self.assertIn("FedProx minus FedAvg", markdown)
            self.assertIn("DP-SGD runs are interpreted as privacy accounting experiments", markdown)
            self.assertNotIn("Generated from run summary metrics.", markdown)
            self.assertGreaterEqual(len(rows), 2)
            self.assertTrue(any(row["script_id"] == "E_05" and row["dataset"] == "joint" for row in rows))
            self.assertTrue(any(row["script_id"] == "E_06" and row["dataset"] == "bpic2017" for row in rows))
            self.assertTrue(any(row["use_dp"] == "True" for row in rows))
            self.assertTrue(any(row["bank"] == "bpic2017:A" for row in bank_rows))
            self.assertTrue(any("missing expected matrix run" in warning["message"] for warning in payload["warnings"]))

    # Verify a present DP run cannot fill the matrix slot of the missing no-DP federated run of the same split.
    def test_present_dp_run_does_not_suppress_a_missing_federated_warning(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            dp_dir = root / "differential_privacy" / "bpic2017_iid_3banks" / "fedavg_dp_eps_10"
            dp_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                dp_dir, "E_06", _federated_report("bpic2017", "iid_3banks", "fedavg", 0.38, 125.0, use_dp=True)
            )
            (dp_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            _run_rows, _bank_rows, warnings = analysis.run_analysis(root, root / "analysis")

        missing = [warning["message"] for warning in warnings
                   if warning["message"].startswith("missing expected matrix run")]
        expected = analysis.matrix_key("E_06", "bpic2017", "iid_3banks", "", "fedavg", "")
        self.assertTrue(any(message.endswith(expected) for message in missing),
                        "the no-DP fedavg run must still be reported as missing")

    # Verify the deterministic convergence selection keeps only the runs that reached their own budget ceiling.
    def test_convergence_selection_is_the_budget_ceiling_rule(self) -> None:
        rows = [
            {"dataset": "bpic2012", "run_name": "iid_3banks", "regime": "", "strategy": "fedprox", "bank": "",
             "best_round": 40, "max_rounds": 40, "use_dp": False},
            {"dataset": "bpic2017", "run_name": "iid_3banks", "regime": "centralized", "strategy": "", "bank": "",
             "best_epoch": 4, "max_epochs": 40, "use_dp": False},
            {"dataset": "bpic2017", "run_name": "medium_5banks", "regime": "", "strategy": "fedavg", "bank": "",
             "best_round": 40, "max_rounds": 40, "use_dp": True},
        ]

        selected = analysis.budget_limited_rows(rows)

        self.assertEqual([row["dataset"] for row in selected], ["bpic2012"])

    # Verify the secure-aggregation POC subtree stays outside the matrix analysis.
    def test_secure_aggregation_subtree_is_excluded_from_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            secure_dir = root / analysis.SECURE_AGGREGATION_SUBFOLDER / "bpic2017_medium_3banks" / "fedprox_seed_42"
            secure_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                secure_dir, "E_06", _federated_report("bpic2017", "medium_3banks", "fedprox", 0.5, 100.0)
            )

            run_rows, _bank_rows, _warnings = analysis.run_analysis(root, root / "analysis")
            markdown = (root / "analysis" / "E_07_training_analysis.md").read_text(encoding="utf-8")

        self.assertEqual(run_rows, [])
        self.assertIn(f"The {analysis.SECURE_AGGREGATION_SUBFOLDER}/ subtree is excluded by design", markdown)

    # Verify the identity and count columns are never empty, whichever script produced the report.
    def test_identity_and_count_columns_are_populated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            fed_dir = root / "federated" / "bpic2017_iid_3banks" / "fedprox_seed_42"
            fed_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                fed_dir, "E_06", _federated_report("bpic2017", "iid_3banks", "fedprox", 0.5, 100.0))
            (fed_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")
            dp_dir = root / "differential_privacy" / "bpic2017_iid_3banks" / "fedprox_dp"
            dp_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                dp_dir, "E_06", _federated_report("bpic2017", "iid_3banks", "fedprox", 0.4, 120.0, use_dp=True))
            (dp_dir / "E_06_round_log.csv").write_text("round,val_loss_total\n1,1.0\n", encoding="utf-8")

            run_rows, bank_rows, _warnings = analysis.run_analysis(root, root / "analysis")

        # An E_06 report carries no regime key, so the resolver must fill it for federated and DP rows alike.
        self.assertEqual(sorted(row["regime"] for row in run_rows), ["federated", "federated_dp"])
        for row in run_rows + bank_rows:
            for column in ("script_id", "dataset", "run_name", "regime", "n_prefixes"):
                self.assertNotIn(str(row.get(column, "")).strip(), {""}, f"{column} empty in {row}")

    # Verify the prefix count falls back to the confusion-matrix total for reports without the key.
    def test_prefix_count_falls_back_to_the_confusion_matrix(self) -> None:
        self.assertEqual(analysis.prefix_count({"n_prefixes": 42}), 42)
        self.assertEqual(analysis.prefix_count({"outcome": {"confusion_matrix": [[1, 2], [3, 4]]}}), 10)
        self.assertEqual(analysis.prefix_count({}), "")

    # Verify small magnitudes survive the Markdown formatter instead of collapsing to zero.
    def test_format_cell_keeps_small_magnitudes(self) -> None:
        self.assertEqual(analysis.format_cell(1e-06), "1e-06")
        self.assertEqual(analysis.format_cell("1e-06"), "1e-06")
        self.assertEqual(analysis.format_cell(2.5e-4), "0.00025")
        self.assertEqual(analysis.format_cell(0.5), "0.5")
        self.assertEqual(analysis.format_cell(0.0), "0")
        self.assertEqual(analysis.format_cell(123456.0), "123456.0")

    # Verify the exclusion assertion fires when a future change lets the secure subtree into the discovery roots.
    # The glob roots exclude it today, so this guard is the tripwire that keeps the exclusion deliberate.
    def test_secure_subtree_leak_is_refused_by_the_assertion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            secure_dir = root / analysis.SECURE_AGGREGATION_SUBFOLDER / "bpic2017_medium_3banks" / "fedprox_seed_42"
            secure_dir.mkdir(parents=True)
            training_reporting.write_run_report(
                secure_dir, "E_06", _federated_report("bpic2017", "medium_3banks", "fedprox", 0.5, 100.0))

            leaky_subfolders = analysis.MATRIX_SUBFOLDERS + (analysis.SECURE_AGGREGATION_SUBFOLDER,)
            with mock.patch.object(analysis, "MATRIX_SUBFOLDERS", leaky_subfolders):
                with self.assertRaises(RuntimeError):
                    analysis.discover_runs(root, [])

    # Verify training analysis strict mode fails on warnings.
    def test_training_analysis_strict_mode_fails_on_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            analysis_root = root / "analysis"
            root.mkdir()

            rc = analysis.main(["--output-root", str(root), "--analysis-root", str(analysis_root), "--strict"])

            self.assertEqual(rc, 1)
            self.assertTrue((analysis_root / "E_07_training_analysis.json").exists())

if __name__ == "__main__":
    unittest.main()