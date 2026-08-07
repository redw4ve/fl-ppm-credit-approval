from __future__ import annotations
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import training_reporting

# HELPER: Build one nested metric block with the outcome, next-activity and remaining-time sections a run reports.
def _metric_block(macro_f1: float, top1: float, mae: float) -> dict[str, object]:
    return {
        "loss_total": 1.0,
        "outcome": {
            "macro_f1": macro_f1,
            "weighted_f1": macro_f1 + 0.05,
            "balanced_accuracy": macro_f1 + 0.02,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "per_class": {
                "0": {"precision": 0.5, "recall": 0.6, "f1": 0.55, "support": 10},
                "1": {"precision": 0.4, "recall": 0.3, "f1": 0.34, "support": 20},
                "2": {"precision": 0.9, "recall": 0.8, "f1": 0.85, "support": 30},
            },
        },
        "next_activity": {"top1_accuracy": top1, "top3_accuracy": top1 + 0.1},
        "remaining_time": {"mae": mae, "rmse": mae * 2.0},
    }

# HELPER: Write the compact run report and the test prediction parquet one local bank run leaves behind.
def _write_local_bank_run(output_root: Path, config: baseline.BaselineRunConfig, bank: str,
                          metrics: dict[str, object], n_prefixes: int) -> None:
    bank_config = replace(config, bank=bank)
    run_dir = baseline.output_dir_for_run(output_root, bank_config)
    predictions = baseline.prediction_artifact_path(run_dir, "predictions_test.parquet")
    predictions.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"case_id": [f"case_{index}" for index in range(n_prefixes)]}).to_parquet(predictions, index=False)
    report = training_reporting.build_run_report(
        baseline.SCRIPT_ID, {"dataset": bank_config.dataset, "bank": bank}, {"test": metrics}, {}, {}
    )
    training_reporting.write_run_report(run_dir, baseline.SCRIPT_ID, report)

class CentralizedSummaryTests(unittest.TestCase):
    # Verify the centralized summary carries the global row plus one row per bank without the legacy debug fragment.
    def test_centralized_summary_has_one_row_per_bank_under_the_compact_profile(self) -> None:
        per_bank = {
            "A": _metric_block(0.60, 0.80, 100.0),
            "B": _metric_block(0.55, 0.78, 110.0),
            "C": _metric_block(0.51, 0.76, 120.0),
        }

        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            baseline.write_centralized_summary(output_dir, _metric_block(0.58, 0.79, 105.0), per_bank)

            # The legacy per-bank fragment must stay absent because the compact profile never writes it.
            self.assertFalse(baseline.run_artifact_path(output_dir, "per_bank_test_metrics.json").exists())
            frame = pd.read_csv(baseline.run_artifact_path(output_dir, "centralized_summary_test.csv"))

        self.assertEqual(len(frame), 1 + len(per_bank))
        self.assertEqual(list(frame["scope"]), ["global_test", "bank_A", "bank_B", "bank_C"])
        for column in ("outcome_macro_f1", "outcome_weighted_f1", "outcome_balanced_accuracy",
                       "outcome_class_0_precision", "outcome_class_1_recall", "outcome_class_2_f1"):
            self.assertIn(column, frame.columns)
        self.assertAlmostEqual(float(frame.loc[frame["scope"] == "bank_B", "outcome_macro_f1"].iloc[0]), 0.55)

    # Verify the summary recovers the per-bank rows from an already written run report when no payload is passed.
    def test_centralized_summary_falls_back_to_the_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = Path(tmp)
            report = training_reporting.build_run_report(
                baseline.SCRIPT_ID, {}, {"per_bank_test": {"A": _metric_block(0.61, 0.81, 101.0)}}, {}, {}
            )
            training_reporting.write_run_report(output_dir, baseline.SCRIPT_ID, report)

            baseline.write_centralized_summary(output_dir, _metric_block(0.58, 0.79, 105.0), {})
            frame = pd.read_csv(baseline.run_artifact_path(output_dir, "centralized_summary_test.csv"))

        self.assertEqual(list(frame["scope"]), ["global_test", "bank_A"])

class LocalSummaryTests(unittest.TestCase):
    # Verify the shared local summary appears once every bank run report exists, with one row per bank.
    def test_local_summary_is_written_from_compact_run_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp)
            config = baseline.BaselineRunConfig(
                dataset="bpic2012", heterogeneity="medium", n_clients=3, regime="local", bank="C",
                output_root=output_root,
            )
            _write_local_bank_run(output_root, config, "A", _metric_block(0.60, 0.80, 100.0), 40)
            _write_local_bank_run(output_root, config, "B", _metric_block(0.50, 0.70, 200.0), 60)
            csv_path, json_path = baseline.local_summary_paths(config)

            # An incomplete split writes nothing, because bank C has not reported yet.
            baseline.write_local_summary_if_complete(config)
            self.assertFalse(csv_path.exists())

            _write_local_bank_run(output_root, config, "C", _metric_block(0.40, 0.60, 300.0), 100)
            baseline.write_local_summary_if_complete(config)
            self.assertTrue(json_path.exists())
            frame = pd.read_csv(csv_path)

        self.assertEqual(list(frame["bank"]), ["A", "B", "C", "unweighted_average", "prefix_weighted_average"])
        for column in ("outcome_macro_f1", "outcome_weighted_f1", "outcome_balanced_accuracy",
                       "outcome_class_0_precision", "outcome_class_1_recall", "outcome_class_2_f1"):
            self.assertIn(column, frame.columns)

        # The unweighted row averages the three banks and the weighted row uses the prefix counts as weights.
        unweighted = frame.loc[frame["bank"] == "unweighted_average", "outcome_macro_f1"].iloc[0]
        weighted = frame.loc[frame["bank"] == "prefix_weighted_average", "outcome_macro_f1"].iloc[0]
        self.assertAlmostEqual(float(unweighted), 0.50)
        self.assertAlmostEqual(float(weighted), (0.60 * 40 + 0.50 * 60 + 0.40 * 100) / 200)

if __name__ == "__main__":
    unittest.main()