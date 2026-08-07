from __future__ import annotations
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from E_training import E_05_central_and_local_baselines_final as baseline
from E_training import E_08_outcome_robustness as robustness
from E_training import training_core_final as core

# Three cases of three events each, so every case contributes exactly one final prefix.
CASE_EVENT_COUNTS: dict[str, int] = {"c1": 3, "c2": 3, "c3": 3}

# HELPER: Build one exported test prediction frame with a deterministic outcome column set.
def _prediction_frame() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id, n_events in CASE_EVENT_COUNTS.items():
        for label_pos in range(n_events):
            rows.append({
                "dataset_id": "bpic2017",
                "client_id": "centralized",
                "case_id": case_id,
                "split": "test",
                "prefix_length": label_pos + 1,
                "label_pos": label_pos,
            })
    frame = pd.DataFrame(rows)

    # The label follows the case and the model is wrong on the two earliest prefixes of the minority classes.
    labels = {"c1": 2, "c2": 1, "c3": 0}
    frame["outcome_label"] = [labels[case_id] for case_id in frame["case_id"]]
    frame["outcome_pred"] = [
        2 if label_pos == 0 else label for label, label_pos in zip(frame["outcome_label"], frame["label_pos"])
    ]
    for label in core.OUTCOME_CLASSES:
        frame[f"outcome_prob_{label}"] = [0.9 if label == pred else 0.05 for pred in frame["outcome_pred"]]
    return frame

# HELPER: Write the processed bank split parquets and return the matching dataset mapping.
def _write_processed_splits(root: Path) -> dict[str, Any]:
    split_dir = root / "medium_3banks"
    split_dir.mkdir(parents=True, exist_ok=True)
    for bank, cases in (("A", ["c1"]), ("B", ["c2"]), ("C", ["c3"])):
        events = [{"case": case_id} for case_id in cases for _ in range(CASE_EVENT_COUNTS[case_id])]
        pd.DataFrame(events).to_parquet(split_dir / f"A_02_bank_{bank}_test.parquet", index=False)
    return {"datasets": {"bpic2017": {"input_root": str(root), "split_prefix": "A_02",
                                      "column_mapping": {"case_id": "case"}}}}

# HELPER: Write one compact centralized run report plus its exported test predictions.
def _write_run(output_root: Path) -> Path:
    run_dir = output_root / "baselines" / "bpic2017_medium_3banks" / "centralized_seed_42_lr_0p00025"
    predictions = run_dir / baseline.PREDICTIONS_DIR_NAME / "E_05_predictions_test.parquet"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame().to_parquet(predictions, index=False)
    report = {
        "script_id": "E_05",
        "config": {"dataset": "bpic2017", "run_name": "medium_3banks", "regime": "centralized", "n_clients": 3},
        "metrics": {}, "diagnostics": {},
    }
    (run_dir / "E_05_run_report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir

# HELPER: Write one compact federated run report without a regime key plus its exported test predictions.
def _write_federated_run(output_root: Path) -> Path:
    run_dir = output_root / "federated" / "bpic2017_medium_3banks" / "federated_fedavg_seed_42_lr_0p00025"
    predictions = run_dir / baseline.PREDICTIONS_DIR_NAME / "E_06_predictions_test.parquet"
    predictions.parent.mkdir(parents=True, exist_ok=True)
    _prediction_frame().to_parquet(predictions, index=False)
    report = {
        "script_id": "E_06",
        "config": {"dataset": "bpic2017", "run_name": "medium_3banks", "strategy": "fedavg", "n_clients": 3},
        "metrics": {}, "diagnostics": {},
    }
    (run_dir / "E_06_run_report.json").write_text(json.dumps(report), encoding="utf-8")
    return run_dir

class FinalPrefixMarkingTests(unittest.TestCase):
    # Verify the last prefix of every case at or below the cap is marked and a capped case is not.
    @staticmethod
    def test_only_the_last_prefix_of_an_uncapped_case_is_marked() -> None:
        predictions = pd.DataFrame({
            "dataset_id": ["bpic2017"] * 5,
            "case_id": ["short", "short", "long", "long", "long"],
            "label_pos": [0, 1, 0, 1, 2],
        })
        counts = {("bpic2017", "short"): 2, ("bpic2017", "long"): 9}

        marked = robustness.mark_final_prefixes(predictions, counts)
        np.testing.assert_array_equal(marked, np.array([False, True, False, False, False]))

    # Verify a predicted case without an event count fails loudly instead of silently counting as not final.
    def test_missing_event_count_is_rejected(self) -> None:
        predictions = pd.DataFrame({"dataset_id": ["bpic2017"], "case_id": ["c1"], "label_pos": [0]})

        with self.assertRaises(ValueError): robustness.mark_final_prefixes(predictions, {})

class SignAgreementTests(unittest.TestCase):
    # Verify the sign statement is computed from the rows and names the comparisons that flip.
    def test_sign_agreement_counts_and_names_the_exceptions(self) -> None:
        rows = [
            {"scope": "all", "dataset": "bpic2017", "run_name": "iid_3banks", "regime": "centralized",
             "outcome_macro_f1": 0.60347934, "outcome_macro_f1_excluding_final": 0.59265940},
            {"scope": "all", "dataset": "bpic2017", "run_name": "iid_3banks", "regime": "", "strategy": "fedavg",
             "outcome_macro_f1": 0.60346279, "outcome_macro_f1_excluding_final": 0.59294539},
            {"scope": "all", "dataset": "bpic2017", "run_name": "iid_3banks", "regime": "local", "bank": "A",
             "outcome_macro_f1": 0.50, "outcome_macro_f1_excluding_final": 0.48},
        ]
        agreement = robustness.within_split_sign_agreement(rows)

        # Three runs give three pairs. Only the near-tied centralized against FedAvg pair flips.
        self.assertEqual(agreement["comparisons"], 3)
        self.assertEqual(agreement["sign_kept"], 2)
        self.assertEqual(len(agreement["flipped"]), 1)
        flipped = agreement["flipped"][0]
        self.assertGreater(flipped["raw_delta"], 0.0)
        self.assertLess(flipped["corrected_delta"], 0.0)
        self.assertIn("centralized", flipped["comparison"])

    # Verify a run without a corrected value is skipped instead of producing a bogus comparison.
    def test_rows_without_a_corrected_value_are_skipped(self) -> None:
        rows = [
            {"scope": "all", "dataset": "bpic2012", "run_name": "iid_3banks", "regime": "centralized",
             "outcome_macro_f1": 0.6, "outcome_macro_f1_excluding_final": ""},
            {"scope": "all", "dataset": "bpic2012", "run_name": "iid_3banks", "regime": "local", "bank": "A",
             "outcome_macro_f1": 0.5, "outcome_macro_f1_excluding_final": 0.48},
        ]
        self.assertEqual(robustness.within_split_sign_agreement(rows)["comparisons"], 0)

class RobustnessStageTests(unittest.TestCase):
    # Verify the stage reports the final-prefix share, its accuracy and the corrected metrics for a discovered run.
    def test_stage_reports_share_accuracy_and_corrected_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            mapping = _write_processed_splits(Path(tmp) / "processed")
            _write_run(output_root)

            with mock.patch.object(baseline.encoding, "load_dataset_mapping", return_value=mapping):
                rows, warnings = robustness.run_robustness(output_root, Path(tmp) / "analysis")
            frame = pd.read_csv(Path(tmp) / "analysis" / "E_08_outcome_robustness.csv")

        self.assertEqual(warnings, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(list(frame["scope"]), ["all"])

        # Three cases of three events give nine prefixes, of which three are final and all three are predicted right.
        self.assertEqual(int(frame.loc[0, "n_prefixes"]), 9)
        self.assertEqual(int(frame.loc[0, "n_final_prefixes"]), 3)
        self.assertAlmostEqual(float(frame.loc[0, "final_prefix_share"]), 3.0 / 9.0, places=12)
        self.assertAlmostEqual(float(frame.loc[0, "final_prefix_outcome_accuracy"]), 1.0, places=12)

        # The corrected columns must equal the shared metric helper on the split without its final prefixes.
        predictions = _prediction_frame()
        is_final = predictions["label_pos"].to_numpy() == 2
        expected_all = robustness.outcome_metrics_for(predictions)
        expected_excluding = robustness.outcome_metrics_for(predictions.loc[~is_final])
        for column, key in (("outcome_macro_f1", "macro_f1"), ("outcome_weighted_f1", "weighted_f1"),
                            ("outcome_balanced_accuracy", "balanced_accuracy")):
            self.assertAlmostEqual(float(frame.loc[0, column]), float(expected_all[key]), places=12)
            self.assertAlmostEqual(
                float(frame.loc[0, f"{column}_excluding_final"]), float(expected_excluding[key]), places=12)
            self.assertAlmostEqual(
                float(frame.loc[0, f"{column}_inflation"]),
                float(expected_all[key]) - float(expected_excluding[key]), places=12)

        # Excluding the always-correct final prefixes must lower the reported quality, which is the disclosed bias.
        self.assertGreater(float(frame.loc[0, "outcome_macro_f1_inflation"]), 0.0)

    # Verify a federated report without a regime key derives its regime instead of publishing an empty string.
    def test_federated_report_without_regime_key_derives_federated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            mapping = _write_processed_splits(Path(tmp) / "processed")
            _write_federated_run(output_root)

            with mock.patch.object(baseline.encoding, "load_dataset_mapping", return_value=mapping):
                rows, warnings = robustness.run_robustness(output_root, Path(tmp) / "analysis")

        self.assertEqual(warnings, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["regime"], "federated")
        self.assertEqual(rows[0]["strategy"], "fedavg")

    # Verify the stage, writes the three artifacts and reports a missing prediction export as a warning.
    def test_missing_predictions_are_reported_and_do_not_abort(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output_root = Path(tmp) / "outputs"
            mapping = _write_processed_splits(Path(tmp) / "processed")
            run_dir = _write_run(output_root)
            (run_dir / baseline.PREDICTIONS_DIR_NAME / "E_05_predictions_test.parquet").unlink()

            with mock.patch.object(baseline.encoding, "load_dataset_mapping", return_value=mapping):
                rows, warnings = robustness.run_robustness(output_root, Path(tmp) / "analysis")
            analysis_root = Path(tmp) / "analysis"
            written = sorted(path.name for path in analysis_root.iterdir())

        self.assertEqual(rows, [])
        self.assertEqual(len(warnings), 1)
        self.assertIn("missing test predictions", warnings[0]["message"])
        self.assertEqual(written, ["E_08_outcome_robustness.csv", "E_08_outcome_robustness.json",
                                   "E_08_outcome_robustness.md"])

if __name__ == "__main__":
    unittest.main()